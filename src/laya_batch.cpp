// Laya packed batched forward: any number of (state, question) items from one
// request or many, packed back to back without padding into one encoder +
// head + scorer + act-head pass. See DecisionModel::forward_items.
//
// Per call: one host->device upload (every index buffer in one INT32 block),
// the device work, one device->host readback (scorer logits and act logits in
// one buffer). On CUDA the device work is replayed from a CUDA graph cached
// per (T, N, K) bucket — the token, item and marker counts rounded up so a
// handful of graphs covers a live workload. Padding rows are singleton
// sequences, padding items empty segments; neither touches a real item.
//
// Graph replay needs every device pointer the graph captured to stay put, so
// scratch is sized to a capacity that only grows (powers of two); growing it,
// or the rotary tables, drops every cached graph.

#include "brolm/laya.h"

#include "brolm/detail/compute.h"
#include "brolm/detail/device.h"

#include "brotensor/ops.h"
#include "brotensor/runtime.h"
#if defined(BROTENSOR_HAS_CUDA)
#include "brotensor/cuda_graph.h"
#endif

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace brolm::laya {

namespace bt = ::brotensor;
using Clock = std::chrono::steady_clock;

namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("laya::DecisionModel: " + msg);
}

// Times one stage into `acc` when profiling; syncs so GPU work lands in the
// stage that issued it.
class StageTimer {
public:
    StageTimer(bool on, double& acc) : on_(on), acc_(acc) {
        if (on_) {
            bt::sync_all();
            t0_ = Clock::now();
        }
    }
    ~StageTimer() {
        if (on_) {
            bt::sync_all();
            acc_ += std::chrono::duration<double, std::milli>(Clock::now() - t0_).count();
        }
    }
    StageTimer(const StageTimer&) = delete;
    StageTimer& operator=(const StageTimer&) = delete;

private:
    bool on_;
    double& acc_;
    Clock::time_point t0_;
};

int pow2_at_least(int n, int floor_) {
    int p = floor_;
    while (p < n) p <<= 1;
    return p;
}

// Token-count bucket: 16 steps per power of two (<= 6.25 % padding), 16-row
// granularity at the small end.
int bucket_rows(int n) {
    const int p = pow2_at_least(n, 16);
    const int step = std::max(16, p / 16);
    return (n + step - 1) / step * step;
}

bool env_graphs_enabled() {
    const char* e = std::getenv("BROLM_LAYA_GRAPHS");
    return !(e && e[0] == '0');
}

constexpr std::size_t kMaxGraphs = 96;

}  // namespace

struct DecisionModel::Batch {
    // Capacities every scratch buffer is reserved to (0 = not yet).
    int cap_T = 0, cap_N = 0, cap_K = 0;
    bool graphs_on = env_graphs_enabled();
    bool graphs_broken = false;

    std::vector<int32_t> host_idx;
    bt::Tensor idx;  // INT32 index block: ids | pos | bounds | type | markers | starts | segs
    bt::Tensor v_ids, v_pos, v_bounds, v_type, v_markers, v_starts, v_segs;

    bt::Tensor h, h_norm, qkv, attn, ffn1, type_rows;
    bt::Tensor m, s0, s2, wide, logits, feats, h0, act_in, act_h2, out;
    const void* ws_data = nullptr;  // split-K workspace the cached graphs captured
    std::vector<uint16_t> out_bits;
    std::vector<float> out_f32;

#if defined(BROTENSOR_HAS_CUDA)
    std::map<std::tuple<int, int, int>, bt::CudaGraph> graphs;
#endif

    void clear_graphs() {
#if defined(BROTENSOR_HAS_CUDA)
        graphs.clear();
#endif
    }
};

// Special members live here, where Batch is complete.
DecisionModel::DecisionModel() {
    head_layers_.resize(static_cast<std::size_t>(cfg_.head_layers));
}
DecisionModel::~DecisionModel() = default;
DecisionModel::DecisionModel(DecisionModel&&) noexcept = default;
DecisionModel& DecisionModel::operator=(DecisionModel&&) noexcept = default;

void DecisionModel::reset_batch_() { batch_.reset(); }

DecisionModel::Batch& DecisionModel::batch() {
    if (!batch_) {
        batch_ = std::make_unique<Batch>();
        // FP16 linears use brotensor's hybrid accumulation (16-term FP16
        // steps folded into FP32): ~5-8 % end to end, parity well inside
        // tolerance. BROLM_LAYA_FAST_ACCUM=0 restores pure FP32 accumulation.
        const char* e = std::getenv("BROLM_LAYA_FAST_ACCUM");
        encoder_.set_fast_accum(!(e && e[0] == '0'));
    }
    return *batch_;
}

void DecisionModel::set_graphs_enabled(bool on) {
    Batch& b = batch();
    b.graphs_on = on;
    if (!on) b.clear_graphs();
}

// The device half of forward_items over the index views already set up in
// `b`: T packed rows, N items, K markers. Issues device work only (no host
// transfer, no allocation within capacity) so it can be graph-captured.
void DecisionModel::linear_(const bt::Tensor& W, const bt::Tensor& bias, const bt::Tensor& X, int act,
                            int epilogue, bt::Tensor& Y) {
    const int flags = encoder_.fast_accum() ? bt::kLinearEpiFastAccum : 0;
    bt::linear_forward_batched_ex(W, &bias, X, act, epilogue | flags, &encoder_.gemm_workspace(), Y);
}

void DecisionModel::run_device_(Batch& b, int T, int N, int K) {
    const int D = encoder_.config().hidden_size;
    const int H = std::max(1, D / 64);  // nn.TransformerEncoderLayer(d, d // 64 heads)
    const bt::Dtype dt = brolm::compute_dtype();
    const bool prof = profiling_;
    LayaTimings& Tm = timings_;

    // 1. ModernBERT encoder over the packed rows.
    {
        StageTimer t(prof, Tm.encoder_ms);
        encoder_.forward_packed(modernbert::PackedInputs{&b.v_ids, &b.v_pos, &b.v_bounds}, b.h);
    }

    // 2-3. Question-type embedding on every row (each row its item's type),
    //      then the Pre-LN head layers (MHA with biases, ReLU FFN —
    //      nn.TransformerEncoderLayer defaults), attention per item.
    {
        StageTimer t(prof, Tm.head_ms);
        bt::embedding_lookup_forward(type_emb_, static_cast<const int32_t*>(b.v_type.data), T, b.type_rows);
        bt::add_inplace(b.h, b.type_rows);
        for (TransformerHeadLayer& L : head_layers_) {
            brolm::detail::layernorm_batched(b.h, L.norm1_g, L.norm1_b, b.h_norm, 1e-5f);
            linear_(L.in_proj_W, L.in_proj_b, b.h_norm, bt::kLinearActNone, bt::kLinearEpiStore, b.qkv);
            bt::flash_attention_packed_qkv_forward(b.qkv, b.v_bounds, H, /*window=*/0, b.attn);
            linear_(L.out_proj_W, L.out_proj_b, b.attn, bt::kLinearActNone, bt::kLinearEpiAccumulate, b.h);

            brolm::detail::layernorm_batched(b.h, L.norm2_g, L.norm2_b, b.h_norm, 1e-5f);
            linear_(L.linear1_W, L.linear1_b, b.h_norm, bt::kLinearActRelu, bt::kLinearEpiStore, b.ffn1);
            linear_(L.linear2_W, L.linear2_b, b.ffn1, bt::kLinearActNone, bt::kLinearEpiAccumulate, b.h);
        }
    }

    // 4-5. Gather every item's [MASK] marker rows and score them:
    //      LayerNorm -> Linear -> GELU -> Linear -> raw logits (K, 1).
    {
        StageTimer t(prof, Tm.scorer_ms);
        bt::gather_rows(b.h, b.v_markers, b.m);
        brolm::detail::layernorm_batched(b.m, scorer_ln_g_, scorer_ln_b_, b.s0, 1e-5f);
        linear_(scorer_l1_W_, scorer_l1_b_, b.s0, bt::kLinearActGeluExact, bt::kLinearEpiStore, b.s2);
        // scorer_l2 is zero-padded to kOutPad outputs; column 0 is the logit.
        linear_(scorer_l2_W_, scorer_l2_b_, b.s2, bt::kLinearActNone, bt::kLinearEpiStore, b.wide);
        b.logits.resize(K, 1, dt);
        bt::copy_d2d_strided(b.wide, 0, kOutPad, b.logits, 0, 1, 1, K);
    }

    // 6. Act head per item: [h[item start] | top1, top1 - top2, normalized
    //    entropy, k / 255] of the uncalibrated option distribution, all on
    //    the device; then logits + act logits into one readback buffer.
    {
        StageTimer t(prof, Tm.act_ms);
        bt::segment_softmax_stats(b.logits, b.v_segs, b.feats);
        bt::gather_rows(b.h, b.v_starts, b.h0);
        // act_in is (N, D + kActPad); its pad columns stay zero from allocation.
        const int AP = D + kActPad;
        b.act_in.resize(N, AP, dt);
        bt::copy_d2d_strided(b.h0, 0, D, b.act_in, 0, AP, D, N);
        bt::copy_d2d_strided(b.feats, 0, 4, b.act_in, D, AP, 4, N);
        linear_(act_l1_W_, act_l1_b_, b.act_in, bt::kLinearActGeluExact, bt::kLinearEpiStore, b.act_h2);
        linear_(act_l2_W_, act_l2_b_, b.act_h2, bt::kLinearActNone, bt::kLinearEpiStore, b.wide);
        b.out.resize(K + 2 * N, 1, dt);
        bt::copy_d2d(b.logits, 0, b.out, 0, K);
        bt::copy_d2d_strided(b.wide, 0, kOutPad, b.out, K, 2, 2, N);
    }
}

std::vector<LayaItemLogits> DecisionModel::forward_items(const std::vector<LayaItem>& items) {
    timings_.encoder_ms = timings_.head_ms = timings_.scorer_ms = 0;
    timings_.act_ms = timings_.download_ms = 0;
    timings_.uploads = timings_.downloads = 0;
    if (items.empty()) return {};

    int T = 0, K = 0, max_len = 0;
    for (const LayaItem& it : items) {
        if (it.num_ids <= 0 || !it.input_ids) fail("forward_items: an item has no input ids");
        if (it.num_markers <= 0 || !it.marker_pos) fail("forward_items: an item has no markers");
        if (it.qtype < 0 || it.qtype > 2) fail("forward_items: qtype must be 0, 1 or 2");
        for (int k = 0; k < it.num_markers; ++k) {
            if (it.marker_pos[k] < 0 || it.marker_pos[k] >= it.num_ids) {
                fail("forward_items: marker position outside its sequence");
            }
        }
        T += it.num_ids;
        K += it.num_markers;
        max_len = std::max(max_len, it.num_ids);
    }
    const int N = static_cast<int>(items.size());

    Batch& b = batch();
    const bt::Device dev = bt::default_device();
    const bt::Dtype dt = brolm::compute_dtype();
    const int D = encoder_.config().hidden_size;
#if defined(BROTENSOR_HAS_CUDA)
    const bool use_graphs = dev.type == bt::DeviceType::CUDA && b.graphs_on && !b.graphs_broken && !profiling_;
#else
    const bool use_graphs = false;
#endif

    // Bucketed shapes when replaying graphs; exact shapes otherwise.
    const int Tb = use_graphs ? bucket_rows(T) : T;
    const int Nb = use_graphs ? pow2_at_least(N, 4) : N;
    const int Kb = use_graphs ? pow2_at_least(K, 8) : K;

    if (encoder_.reserve_positions(max_len)) b.clear_graphs();
    if (Tb > b.cap_T || Nb > b.cap_N || Kb > b.cap_K) {
        b.cap_T = pow2_at_least(Tb, std::max(512, b.cap_T));
        b.cap_N = pow2_at_least(Nb, std::max(64, b.cap_N));
        b.cap_K = pow2_at_least(Kb, std::max(256, b.cap_K));
        const int cT = b.cap_T, cN = b.cap_N, cK = b.cap_K;
        encoder_.reserve_rows(cT);
        using brolm::detail::resize_like;
        resize_like(b.idx, 5 * cT + cK + 2 * cN + 1, 1, bt::Dtype::INT32, dev);
        resize_like(b.h, cT, D, dt, dev);
        resize_like(b.h_norm, cT, D, dt, dev);
        resize_like(b.qkv, cT, 3 * D, dt, dev);
        resize_like(b.attn, cT, D, dt, dev);
        resize_like(b.ffn1, cT, 4 * D, dt, dev);
        resize_like(b.type_rows, cT, D, dt, dev);
        resize_like(b.m, cK, D, dt, dev);
        resize_like(b.s0, cK, D, dt, dev);
        resize_like(b.s2, cK, D, dt, dev);
        resize_like(b.logits, cK, 1, dt, dev);
        resize_like(b.feats, cN, 4, dt, dev);
        resize_like(b.h0, cN, D, dt, dev);
        b.act_in = bt::Tensor::zeros_on(dev, cN, D + kActPad, dt);  // pad columns must read zero
        resize_like(b.act_h2, cN, 256, dt, dev);
        resize_like(b.wide, std::max(cK, cN), kOutPad, dt, dev);
        resize_like(b.out, cK + 2 * cN, 1, dt, dev);
        b.clear_graphs();
    }

    // Host index block. Padding rows [T, Tb) are singleton sequences of token
    // 0; padding items have empty marker segments and read row 0.
    const int o_pos = Tb, o_bounds = 2 * Tb, o_type = 4 * Tb, o_markers = 5 * Tb;
    const int o_starts = o_markers + Kb, o_segs = o_starts + Nb, total = o_segs + Nb + 1;
    std::vector<int32_t>& hi = b.host_idx;
    hi.assign(static_cast<std::size_t>(total), 0);
    {
        int row = 0, mk = 0;
        for (int n = 0; n < N; ++n) {
            const LayaItem& it = items[static_cast<std::size_t>(n)];
            std::memcpy(hi.data() + row, it.input_ids, static_cast<std::size_t>(it.num_ids) * sizeof(int32_t));
            for (int i = 0; i < it.num_ids; ++i) {
                hi[static_cast<std::size_t>(o_pos + row + i)] = i;
                hi[static_cast<std::size_t>(o_bounds + 2 * (row + i))] = row;
                hi[static_cast<std::size_t>(o_bounds + 2 * (row + i) + 1)] = row + it.num_ids;
                hi[static_cast<std::size_t>(o_type + row + i)] = it.qtype;
            }
            hi[static_cast<std::size_t>(o_segs + n)] = mk;
            hi[static_cast<std::size_t>(o_starts + n)] = row;
            for (int k = 0; k < it.num_markers; ++k) {
                hi[static_cast<std::size_t>(o_markers + mk + k)] = row + it.marker_pos[k];
            }
            row += it.num_ids;
            mk += it.num_markers;
        }
        for (int r = T; r < Tb; ++r) {
            hi[static_cast<std::size_t>(o_bounds + 2 * r)] = r;
            hi[static_cast<std::size_t>(o_bounds + 2 * r + 1)] = r + 1;
        }
        for (int n = N; n <= Nb; ++n) hi[static_cast<std::size_t>(o_segs + n)] = K;
    }
    b.idx.resize(total, 1, bt::Dtype::INT32);  // within capacity: pointer unchanged
    {
        StageTimer t(profiling_, timings_.encoder_ms);  // upload counts toward the encoder stage
        b.idx.copy_from_host_raw(hi.data(), hi.size() * sizeof(int32_t));
        ++timings_.uploads;
    }
    int32_t* base = static_cast<int32_t*>(b.idx.data);
    b.v_ids = bt::Tensor::view(dev, base, Tb, 1, bt::Dtype::INT32);
    b.v_pos = bt::Tensor::view(dev, base + o_pos, Tb, 1, bt::Dtype::INT32);
    b.v_bounds = bt::Tensor::view(dev, base + o_bounds, Tb, 2, bt::Dtype::INT32);
    b.v_type = bt::Tensor::view(dev, base + o_type, Tb, 1, bt::Dtype::INT32);
    b.v_markers = bt::Tensor::view(dev, base + o_markers, Kb, 1, bt::Dtype::INT32);
    b.v_starts = bt::Tensor::view(dev, base + o_starts, Nb, 1, bt::Dtype::INT32);
    b.v_segs = bt::Tensor::view(dev, base + o_segs, Nb + 1, 1, bt::Dtype::INT32);

    bool ran = false;
#if defined(BROTENSOR_HAS_CUDA)
    if (use_graphs) {
        const auto key = std::make_tuple(Tb, Nb, Kb);
        auto it = b.graphs.find(key);
        if (it != b.graphs.end()) {
            it->second.launch();
            ran = true;
        } else {
            run_device_(b, Tb, Nb, Kb);  // eager: sizes every output, gives this call's result
            ran = true;
            // A grown split-K workspace moved; graphs captured on the old one are stale.
            if (encoder_.workspace_data() != b.ws_data) {
                b.clear_graphs();
                b.ws_data = encoder_.workspace_data();
            }
            try {
                bt::CudaGraph g;
                {
                    bt::CudaGraphCapture cap;
                    run_device_(b, Tb, Nb, Kb);
                    g = cap.finish();
                }
                if (b.graphs.size() >= kMaxGraphs) b.graphs.clear();
                b.graphs.emplace(key, std::move(g));
            } catch (const std::exception& e) {
                std::fprintf(stderr, "laya: CUDA graph capture failed, running eagerly: %s\n", e.what());
                b.graphs_broken = true;
                b.clear_graphs();
            }
        }
    }
#endif
    if (!ran) run_device_(b, Tb, Nb, Kb);

    // One readback: logits (Kb) then act logits (2 * Nb).
    const int n_out = Kb + 2 * Nb;
    {
        StageTimer t(profiling_, timings_.download_ms);
        b.out_f32.resize(static_cast<std::size_t>(n_out));
        if (b.out.dtype == bt::Dtype::FP32) {
            b.out.copy_to_host_raw(b.out_f32.data(), b.out_f32.size() * sizeof(float));
        } else {
            b.out_bits.resize(static_cast<std::size_t>(n_out));
            b.out.copy_to_host_raw(b.out_bits.data(), b.out_bits.size() * sizeof(uint16_t));
            const bool bf = b.out.dtype == bt::Dtype::BF16;
            for (int i = 0; i < n_out; ++i) {
                const uint16_t v = b.out_bits[static_cast<std::size_t>(i)];
                b.out_f32[static_cast<std::size_t>(i)] = bf ? bt::bf16_bits_to_fp32(v) : bt::fp16_bits_to_fp32(v);
            }
        }
        ++timings_.downloads;
    }

    std::vector<LayaItemLogits> res(static_cast<std::size_t>(N));
    int mk = 0;
    for (int n = 0; n < N; ++n) {
        const LayaItem& it = items[static_cast<std::size_t>(n)];
        LayaItemLogits& r = res[static_cast<std::size_t>(n)];
        r.logits.assign(b.out_f32.begin() + mk, b.out_f32.begin() + mk + it.num_markers);
        r.act_logits[0] = b.out_f32[static_cast<std::size_t>(Kb + 2 * n)];
        r.act_logits[1] = b.out_f32[static_cast<std::size_t>(Kb + 2 * n + 1)];
        mk += it.num_markers;
    }
    return res;
}

}  // namespace brolm::laya
