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
// Graph replay needs every device pointer the graph captured to stay put.
// Scratch lives in ARENAS: one set of activation buffers (the encoder's
// included) and index / readback buffers at fixed capacities. A forward that
// fits the current arena runs there; one that does not (a per-call max_len
// above what pre-warm sized) opens a new, larger arena and becomes current,
// while the older arena stays alive, owned by the graphs captured on it — so
// a single oversized call costs one capture of its own bucket, not a
// recapture of every pre-warmed one. Growing the rotary tables (positions
// past 8192) still drops every graph.

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
#include <memory>
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

// Pre-warm captures every token bucket up to the scheduler's budget (64 up to
// 2048 rows, 80 up to 4096), so the cache must hold that many plus the odd
// live shape whose item / marker counts overflow their floors.
constexpr std::size_t kMaxGraphs = 192;

}  // namespace

// One scratch set at fixed capacities (see the file comment).
struct DecisionModel::Arena {
    int cap_T = 0, cap_N = 0, cap_K = 0;
    std::shared_ptr<modernbert::ModernBertModel::Scratch> enc;
    const void* ws_data = nullptr;  // the encoder split-K workspace graphs on this arena captured

    bt::Tensor idx;  // INT32 index block: ids | pos | bounds | type | markers | starts | segs
    bt::Tensor v_ids, v_pos, v_bounds, v_type, v_markers, v_starts, v_segs;
    bt::Tensor h, h_norm, qkv, attn, ffn1, type_rows;
    bt::Tensor m, s0, s2, wide, logits, feats, h0, act_in, act_h2, out;

    bool fits(int T, int N, int K) const { return T <= cap_T && N <= cap_N && K <= cap_K; }
};

struct DecisionModel::Batch {
    bool graphs_on = env_graphs_enabled();
    bool graphs_broken = false;

    std::shared_ptr<Arena> cur;  // where new work and new captures go

    std::vector<int32_t> host_idx;
    std::vector<uint16_t> out_bits;
    std::vector<float> out_f32;

#if defined(BROTENSOR_HAS_CUDA)
    struct Cached {
        bt::CudaGraph graph;
        std::shared_ptr<Arena> arena;  // keeps the buffers it captured alive
    };
    std::map<std::tuple<int, int, int>, Cached> graphs;
#endif

    void clear_graphs() {
#if defined(BROTENSOR_HAS_CUDA)
        graphs.clear();
#endif
    }
    // Drop the graphs captured on `a` (its buffers moved under them).
    void clear_graphs_on(const Arena* a) {
#if defined(BROTENSOR_HAS_CUDA)
        for (auto it = graphs.begin(); it != graphs.end();) {
            if (it->second.arena.get() == a) it = graphs.erase(it);
            else ++it;
        }
#else
        (void)a;
#endif
    }
    std::size_t arenas() const {
        std::vector<const Arena*> seen;
        if (cur) seen.push_back(cur.get());
#if defined(BROTENSOR_HAS_CUDA)
        for (const auto& [k, c] : graphs) {
            if (std::find(seen.begin(), seen.end(), c.arena.get()) == seen.end()) seen.push_back(c.arena.get());
        }
#endif
        return seen.size();
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

bool DecisionModel::graphs_enabled() const {
    return batch_ ? (batch_->graphs_on && !batch_->graphs_broken) : env_graphs_enabled();
}

std::size_t DecisionModel::cached_graphs() const {
#if defined(BROTENSOR_HAS_CUDA)
    return batch_ ? batch_->graphs.size() : 0;
#else
    return 0;
#endif
}

std::size_t DecisionModel::scratch_arenas() const { return batch_ ? batch_->arenas() : 0; }

int DecisionModel::scratch_rows() const { return batch_ && batch_->cur ? batch_->cur->cap_T : 0; }

int DecisionModel::token_bucket(int tokens) { return bucket_rows(std::max(1, tokens)); }

std::vector<DecisionModel::WarmPoint> DecisionModel::prewarm_graphs(int max_tokens) {
    const int top = bucket_rows(std::max(16, max_tokens));
    const int item_len = std::max(1, std::min(cfg_.max_len, 128));
    const bool graphs = bt::default_device().type == bt::DeviceType::CUDA && graphs_enabled() && !profiling_;

    // Sizes to warm: every bucket with graphs, a handful for the cost model without.
    std::vector<int> sizes;
    if (graphs) {
        for (int t = 1; t <= top;) {
            const int b = bucket_rows(t);
            sizes.push_back(b);
            t = b + 1;
        }
    } else {
        for (int t : {64, 256, 1024, top}) {
            if (t <= top) sizes.push_back(bucket_rows(t));
        }
    }
    std::sort(sizes.rbegin(), sizes.rend());
    sizes.erase(std::unique(sizes.begin(), sizes.end()), sizes.end());

    // Rotary tables for ModernBERT's full 8192 positions (a few MB), so a
    // live request — even one overriding max_len upward — never regrows them,
    // which would drop every graph.
    if (encoder_.reserve_positions(std::max({cfg_.max_len, item_len, 8192}))) batch().clear_graphs();
    // One arena for the whole range, with headroom on items (one per 8 rows)
    // and markers (one per 2 rows), and rows for at least two full-length
    // items (the checkpoint's max_len: 1024 on the multilingual and
    // typed-decisions checkpoints), so no live batch within the budget opens
    // another. A few MB.
    reserve_batch_(std::max(top, 2 * bucket_rows(cfg_.max_len)), std::max(64, top / 8), std::max(256, top / 2));

    // Synthetic items exactly filling `rows`: item_len-row items with 4
    // markers each (inside the item floors forward_items buckets to), the
    // remainder in a last shorter item. Token values do not matter to a graph.
    std::vector<int32_t> ids(static_cast<std::size_t>(top), 100);
    std::vector<int32_t> markers = {0, 1, 2, 3};
    auto items_for = [&](int rows) {
        std::vector<LayaItem> items;
        for (int at = 0; at < rows; at += item_len) {
            const int len = std::min(item_len, rows - at);
            LayaItem it;
            it.input_ids = ids.data() + at;
            it.num_ids = len;
            it.marker_pos = markers.data();
            it.num_markers = std::min(4, len);
            it.qtype = 0;
            items.push_back(it);
        }
        return items;
    };

    std::vector<WarmPoint> pts;
    pts.reserve(sizes.size());
    for (int rows : sizes) {
        const std::vector<LayaItem> items = items_for(rows);
        forward_items(items);  // captures (graphs) or warms (eager)
        double best = 1e30;
        for (int rep = 0; rep < 2; ++rep) {
            const auto t0 = Clock::now();
            forward_items(items);
            best = std::min(best, std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
        }
        pts.push_back(WarmPoint{rows, best});
    }
    std::sort(pts.begin(), pts.end(), [](const WarmPoint& a, const WarmPoint& b) { return a.tokens < b.tokens; });
    return pts;
}

void DecisionModel::linear_(const bt::Tensor& W, const bt::Tensor& bias, const bt::Tensor& X, int act,
                            int epilogue, bt::Tensor& Y) {
    const int flags = encoder_.fast_accum() ? bt::kLinearEpiFastAccum : 0;
    bt::linear_forward_batched_ex(W, &bias, X, act, epilogue | flags, &encoder_.gemm_workspace(), Y);
}

// The device half of forward_items over the index views already set up in
// arena `a` (whose scratch the encoder is pointed at): T packed rows, N items,
// K markers. Issues device work only (no host transfer, no allocation within
// capacity) so it can be graph-captured.
void DecisionModel::run_device_(Arena& b, int T, int N, int K) {
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

    // 4-5. Gather every item's mask-marker rows and score them:
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

// The arena for T rows, N items and K markers: the current one when it fits;
// otherwise a new one with every capacity grown to a power of two covering
// both the request and the old arena, which becomes current. The old arena is
// freed when no cached graph holds it.
DecisionModel::Arena& DecisionModel::reserve_batch_(int T, int N, int K) {
    Batch& b = batch();
    if (b.cur && b.cur->fits(T, N, K)) return *b.cur;
    const bt::Device dev = bt::default_device();
    const bt::Dtype dt = brolm::compute_dtype();
    const int D = encoder_.config().hidden_size;
    auto a = std::make_shared<Arena>();
    const Arena* old = b.cur.get();
    a->cap_T = pow2_at_least(T, std::max(512, old ? old->cap_T : 0));
    a->cap_N = pow2_at_least(N, std::max(64, old ? old->cap_N : 0));
    a->cap_K = pow2_at_least(K, std::max(256, old ? old->cap_K : 0));
    const int cT = a->cap_T, cN = a->cap_N, cK = a->cap_K;
    a->enc = std::make_shared<modernbert::ModernBertModel::Scratch>();
    encoder_.set_scratch(a->enc);
    encoder_.reserve_rows(cT);
    a->ws_data = encoder_.workspace_data();
    using brolm::detail::resize_like;
    resize_like(a->idx, 5 * cT + cK + 2 * cN + 1, 1, bt::Dtype::INT32, dev);
    resize_like(a->h, cT, D, dt, dev);
    resize_like(a->h_norm, cT, D, dt, dev);
    resize_like(a->qkv, cT, 3 * D, dt, dev);
    resize_like(a->attn, cT, D, dt, dev);
    resize_like(a->ffn1, cT, 4 * D, dt, dev);
    resize_like(a->type_rows, cT, D, dt, dev);
    resize_like(a->m, cK, D, dt, dev);
    resize_like(a->s0, cK, D, dt, dev);
    resize_like(a->s2, cK, D, dt, dev);
    resize_like(a->logits, cK, 1, dt, dev);
    resize_like(a->feats, cN, 4, dt, dev);
    resize_like(a->h0, cN, D, dt, dev);
    a->act_in = bt::Tensor::zeros_on(dev, cN, D + kActPad, dt);  // pad columns must read zero
    resize_like(a->act_h2, cN, 256, dt, dev);
    resize_like(a->wide, std::max(cK, cN), kOutPad, dt, dev);
    resize_like(a->out, cK + 2 * cN, 1, dt, dev);
    b.cur = std::move(a);
    return *b.cur;
}

std::vector<LayaItemLogits> DecisionModel::forward_items(const std::vector<LayaItem>& items) {
    timings_.encoder_ms = timings_.head_ms = timings_.scorer_ms = 0;
    timings_.act_ms = timings_.download_ms = 0;
    timings_.uploads = timings_.downloads = 0;
    if (items.empty()) return {};

    int T = 0, K = 0, max_len = 0;
    const int32_t vocab = encoder_.config().vocab_size;
    for (const LayaItem& it : items) {
        if (it.num_ids <= 0 || !it.input_ids) fail("forward_items: an item has no input ids");
        for (int i = 0; i < it.num_ids; ++i) {
            if (static_cast<uint32_t>(it.input_ids[i]) >= static_cast<uint32_t>(vocab)) {
                fail("forward_items: token id " + std::to_string(it.input_ids[i]) + " outside the vocabulary");
            }
        }
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
#if defined(BROTENSOR_HAS_CUDA)
    const bool use_graphs = dev.type == bt::DeviceType::CUDA && b.graphs_on && !b.graphs_broken && !profiling_;
#else
    const bool use_graphs = false;
#endif

    // Bucketed shapes when replaying graphs; exact shapes otherwise. The item
    // and marker buckets have floors that follow the token bucket (one item
    // per 64 rows, one marker per 8), so a token bucket almost always maps to
    // ONE graph whatever the item mix: padding items / markers cost only the
    // tiny scorer and act-head rows, and it is what lets load-time pre-warm
    // (prewarm_graphs) cover a live workload.
    const int Tb = use_graphs ? bucket_rows(T) : T;
    const int Nb = use_graphs ? pow2_at_least(std::max(N, (Tb + 63) / 64), 4) : N;
    const int Kb = use_graphs ? pow2_at_least(std::max(K, (Tb + 7) / 8), 8) : K;

    if (encoder_.reserve_positions(max_len)) b.clear_graphs();

    // The arena: a cached graph's own when this bucket was captured (it may
    // be an older, smaller arena), else the current one (grown if short).
    std::shared_ptr<Arena> arena;
#if defined(BROTENSOR_HAS_CUDA)
    const auto key = std::make_tuple(Tb, Nb, Kb);
    auto cached = use_graphs ? b.graphs.find(key) : b.graphs.end();
    const bool replay = cached != b.graphs.end();
    if (replay) arena = cached->second.arena;
#else
    const bool replay = false;
#endif
    if (!arena) {
        reserve_batch_(Tb, Nb, Kb);
        arena = b.cur;
    }
    Arena& A = *arena;
    encoder_.set_scratch(A.enc);

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
    A.idx.resize(total, 1, bt::Dtype::INT32);  // within capacity: pointer unchanged
    {
        StageTimer t(profiling_, timings_.encoder_ms);  // upload counts toward the encoder stage
        A.idx.copy_from_host_raw(hi.data(), hi.size() * sizeof(int32_t));
        ++timings_.uploads;
    }
    int32_t* base = static_cast<int32_t*>(A.idx.data);
    A.v_ids = bt::Tensor::view(dev, base, Tb, 1, bt::Dtype::INT32);
    A.v_pos = bt::Tensor::view(dev, base + o_pos, Tb, 1, bt::Dtype::INT32);
    A.v_bounds = bt::Tensor::view(dev, base + o_bounds, Tb, 2, bt::Dtype::INT32);
    A.v_type = bt::Tensor::view(dev, base + o_type, Tb, 1, bt::Dtype::INT32);
    A.v_markers = bt::Tensor::view(dev, base + o_markers, Kb, 1, bt::Dtype::INT32);
    A.v_starts = bt::Tensor::view(dev, base + o_starts, Nb, 1, bt::Dtype::INT32);
    A.v_segs = bt::Tensor::view(dev, base + o_segs, Nb + 1, 1, bt::Dtype::INT32);

    bool ran = false;
#if defined(BROTENSOR_HAS_CUDA)
    if (replay) {
        cached->second.graph.launch();
        ran = true;
    } else if (use_graphs) {
        run_device_(A, Tb, Nb, Kb);  // eager: sizes every output, gives this call's result
        ran = true;
        // A grown split-K workspace moved; graphs captured on the old one are stale.
        if (encoder_.workspace_data() != A.ws_data) {
            b.clear_graphs_on(&A);
            A.ws_data = encoder_.workspace_data();
        }
        try {
            bt::CudaGraph g;
            {
                bt::CudaGraphCapture cap;
                run_device_(A, Tb, Nb, Kb);
                g = cap.finish();
            }
            if (b.graphs.size() >= kMaxGraphs) b.graphs.clear();
            b.graphs.emplace(key, Batch::Cached{std::move(g), arena});
        } catch (const std::exception& e) {
            std::fprintf(stderr, "laya: CUDA graph capture failed, running eagerly: %s\n", e.what());
            b.graphs_broken = true;
            b.clear_graphs();
        }
    }
#endif
    if (!ran) run_device_(A, Tb, Nb, Kb);

    // One readback: logits (Kb) then act logits (2 * Nb). A replayed graph
    // skips run_device_, so `out` still has the shape of the last EAGER run;
    // set this bucket's shape (within capacity: the pointer the graph
    // writes through is unchanged).
    const int n_out = Kb + 2 * Nb;
    A.out.resize(n_out, 1, dt);
    {
        StageTimer t(profiling_, timings_.download_ms);
        b.out_f32.resize(static_cast<std::size_t>(n_out));
        if (A.out.dtype == bt::Dtype::FP32) {
            A.out.copy_to_host_raw(b.out_f32.data(), b.out_f32.size() * sizeof(float));
        } else {
            b.out_bits.resize(static_cast<std::size_t>(n_out));
            A.out.copy_to_host_raw(b.out_bits.data(), b.out_bits.size() * sizeof(uint16_t));
            const bool bf = A.out.dtype == bt::Dtype::BF16;
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
