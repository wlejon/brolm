#include "brolm/qwen35_text.h"
#include "qwen35_detail.h"

#include "brolm/detail/compute.h"
#include "brolm/detail/device.h"
#include "brolm/detail/profile.h"
#include "brotensor/ops.h"
#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace brolm::qwen35 {

namespace bt = ::brotensor;

namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("qwen35::TextModel: " + msg);
}

bt::Tensor make_idx_device(const int32_t* host, int n, bt::Device dev = bt::default_device()) {
    bt::Tensor cpu = bt::Tensor::empty_on(bt::Device::CPU, n, 1, bt::Dtype::INT32);
    std::memcpy(cpu.host_raw_mut(), host,
                static_cast<std::size_t>(n) * sizeof(int32_t));
    return cpu.to(dev);
}

// Build a per-axis sin/cos table (max_pos+1, d_axis) using base rope_theta and
// a list of GLOBAL inv_freq indices (one per axis slot). The HF schedule is
// inv_freq[j] = 1 / theta^(2j / rotary_dim) over j in [0, rotary_dim/2). For
// the chunked-axis layout brotensor expects, the per-axis tables index this
// schedule at the HF pair-indices owned by that axis (see `mrope_pairing`).
void build_axis_tables(int max_pos_inclusive, int d_axis, int rotary_dim,
                       float rope_theta,
                       const std::vector<int>& freq_indices,
                       bt::Tensor& cos_t, bt::Tensor& sin_t,
                       bt::Device dev = bt::default_device()) {
    const int rows = std::max(1, max_pos_inclusive + 1);
    if (d_axis <= 0) {
        // Degenerate axis: brotensor still accepts a (rows, 0)-ish layout via
        // an empty (rows, 1) placeholder; we pass empty Tensors. The mrope op
        // skips them when d_axis==0.
        cos_t = bt::Tensor::zeros_on(dev, rows, 1, bt::Dtype::FP32);
        sin_t = bt::Tensor::zeros_on(dev, rows, 1, bt::Dtype::FP32);
        return;
    }
    if (static_cast<int>(freq_indices.size()) != d_axis) {
        fail("build_axis_tables: freq_indices.size() != d_axis");
    }
    std::vector<float> cos_h(static_cast<std::size_t>(rows) * d_axis);
    std::vector<float> sin_h(static_cast<std::size_t>(rows) * d_axis);
    std::vector<float> inv_freq(static_cast<std::size_t>(d_axis));
    for (int i = 0; i < d_axis; ++i) {
        const int gj = freq_indices[static_cast<std::size_t>(i)];
        const float exp = 2.0f * static_cast<float>(gj) /
                          static_cast<float>(rotary_dim);
        inv_freq[static_cast<std::size_t>(i)] =
            1.0f / std::pow(rope_theta, exp);
    }
    for (int p = 0; p < rows; ++p) {
        for (int i = 0; i < d_axis; ++i) {
            const float angle = static_cast<float>(p) *
                                inv_freq[static_cast<std::size_t>(i)];
            cos_h[static_cast<std::size_t>(p) * d_axis + i] = std::cos(angle);
            sin_h[static_cast<std::size_t>(p) * d_axis + i] = std::sin(angle);
        }
    }
    cos_t = bt::Tensor::from_host_on(dev, cos_h.data(), rows, d_axis);
    sin_t = bt::Tensor::from_host_on(dev, sin_h.data(), rows, d_axis);
}

}  // namespace

// ─── ctor / dtor ───────────────────────────────────────────────────────────

TextModel::TextModel(const Qwen35Config::Text& cfg) : cfg_(cfg) {
    if (cfg_.hidden_size <= 0 || cfg_.intermediate_size <= 0 ||
        cfg_.num_hidden_layers <= 0 || cfg_.vocab_size <= 0 ||
        cfg_.head_dim <= 0) {
        fail("config has non-positive dimension");
    }
    if (cfg_.num_attention_heads <= 0 || cfg_.num_key_value_heads <= 0) {
        fail("num_attention_heads / num_key_value_heads must be positive");
    }
    if (cfg_.num_attention_heads % cfg_.num_key_value_heads != 0) {
        fail("num_attention_heads must be a multiple of num_key_value_heads");
    }
    if (cfg_.head_dim % 2 != 0) {
        fail("head_dim must be even");
    }
    if (cfg_.layer_types.size() !=
        static_cast<std::size_t>(cfg_.num_hidden_layers)) {
        fail("layer_types.size() != num_hidden_layers");
    }
    rotary_dim_ = cfg_.rotary_dim();
    if (rotary_dim_ <= 0 || rotary_dim_ > cfg_.head_dim || rotary_dim_ % 2 != 0) {
        fail("rotary_dim must be even and in (0, head_dim]");
    }
    if (cfg_.rope.mrope_section.size() != 3) {
        fail("mrope_section must have 3 entries (t,h,w)");
    }
    d_t_ = cfg_.rope.mrope_section[0];
    d_h_ = cfg_.rope.mrope_section[1];
    d_w_ = cfg_.rope.mrope_section[2];
    if (2 * (d_t_ + d_h_ + d_w_) != rotary_dim_) {
        fail("2*sum(mrope_section) != rotary_dim");
    }
    layers_.resize(static_cast<std::size_t>(cfg_.num_hidden_layers));
    for (int i = 0; i < cfg_.num_hidden_layers; ++i) {
        layers_[static_cast<std::size_t>(i)].type = cfg_.layer_types[static_cast<std::size_t>(i)];
    }
}

TextModel::~TextModel() = default;

bt::Device TextModel::stage_device(int stage_idx) const {
    if (cfg_.pipeline_devices.empty()) return bt::default_device();
    if (stage_idx < 0) stage_idx = 0;
    if (stage_idx >= static_cast<int>(cfg_.pipeline_devices.size())) {
        stage_idx = static_cast<int>(cfg_.pipeline_devices.size()) - 1;
    }
    return cfg_.pipeline_devices[static_cast<std::size_t>(stage_idx)];
}

bt::Device TextModel::layer_device(int layer_idx) const {
    if (cfg_.pipeline_devices.empty()) return bt::default_device();
    const int num_stages = static_cast<int>(cfg_.pipeline_devices.size());
    if (num_stages <= 1) return cfg_.pipeline_devices[0];
    const int total_layers = cfg_.num_hidden_layers;
    int stage = (layer_idx * num_stages) / total_layers;
    if (stage >= num_stages) stage = num_stages - 1;
    return cfg_.pipeline_devices[static_cast<std::size_t>(stage)];
}

bt::Device TextModel::embed_device() const {
    return stage_device(0);
}

bt::Device TextModel::final_device() const {
    if (cfg_.pipeline_devices.empty()) return bt::default_device();
    return cfg_.pipeline_devices.back();
}


// ─── cache ─────────────────────────────────────────────────────────────────

std::vector<LayerCache> TextModel::make_cache(int max_seq) const {
    if (max_seq <= 0) fail("make_cache: max_seq must be positive");
    const int n_kv = cfg_.num_key_value_heads;
    const int cache_cols = n_kv * cfg_.head_dim;  // true KV width; decode does GQA
    const bt::Dtype dt = brolm::compute_dtype();

    // Linear-attn shapes.
    const int lin_h_v  = cfg_.linear_num_value_heads;
    const int lin_h_k  = cfg_.linear_num_key_heads;
    const int lin_d_v  = cfg_.linear_value_head_dim;
    const int lin_d_k  = cfg_.linear_key_head_dim;
    const int kdim     = lin_h_k * lin_d_k;
    const int vdim     = lin_h_v * lin_d_v;
    const int qkv_ch   = 2 * kdim + vdim;
    const int conv_st_cols = qkv_ch * (cfg_.linear_conv_kernel_dim - 1);
    const int state_cols   = lin_d_v * lin_d_k;

    std::vector<LayerCache> out(static_cast<std::size_t>(cfg_.num_hidden_layers));
    for (int i = 0; i < cfg_.num_hidden_layers; ++i) {
        const bt::Device dev = layer_device(i);
        LayerCache& c = out[static_cast<std::size_t>(i)];
        if (cfg_.layer_types[static_cast<std::size_t>(i)] == LayerType::Full) {
            brolm::detail::resize_like(c.full.k, max_seq, cache_cols, dt, dev);
            brolm::detail::resize_like(c.full.v, max_seq, cache_cols, dt, dev);
            c.full.len = 0;
        } else {
            // Recurrent state + conv shift register, both FP32 (the recurrence
            // op and causal_conv1d_update are FP32 on CPU).
            c.lin.recurrent = bt::Tensor::zeros_on(dev, lin_h_v, state_cols,
                                                   bt::Dtype::FP32);
            c.lin.conv_state = bt::Tensor::zeros_on(dev, 1, conv_st_cols,
                                                    bt::Dtype::FP32);
            c.lin.len = 0;
            c.lin.initialized = true;
        }
    }
    return out;
}

void TextModel::truncate_cache(std::vector<LayerCache>& cache, int len) const {
    if (len < 0) fail("truncate_cache: len must be non-negative");
    if (cache.size() != static_cast<std::size_t>(cfg_.num_hidden_layers)) {
        fail("truncate_cache: cache size mismatch");
    }
    // Validate everything BEFORE mutating anything, so a throw leaves the
    // cache untouched. Linear-attention layers fold every absorbed token into
    // a fixed-size running state (S and the conv shift register) — there is
    // no per-token history to truncate back to, so any genuine rollback must
    // fail loudly rather than silently decode from a state that still
    // contains the rolled-back tokens.
    for (int i = 0; i < cfg_.num_hidden_layers; ++i) {
        const LayerCache& c = cache[static_cast<std::size_t>(i)];
        if (cfg_.layer_types[static_cast<std::size_t>(i)] == LayerType::Linear) {
            if (c.lin.len != len) {
                fail("truncate_cache: cannot roll back linear-attention state "
                     "at layer " + std::to_string(i) + " (state holds " +
                     std::to_string(c.lin.len) + " tokens, requested " +
                     std::to_string(len) + "). Gated DeltaNet state is a "
                     "lossy running summary with no per-token snapshots; "
                     "speculative rollback across linear layers is "
                     "unsupported — re-prefill instead.");
            }
        } else if (len > c.full.len) {
            fail("truncate_cache: len " + std::to_string(len) +
                 " exceeds current cache len " + std::to_string(c.full.len) +
                 " at layer " + std::to_string(i));
        }
    }
    for (int i = 0; i < cfg_.num_hidden_layers; ++i) {
        LayerCache& c = cache[static_cast<std::size_t>(i)];
        if (cfg_.layer_types[static_cast<std::size_t>(i)] == LayerType::Full) {
            c.full.len = len;
        }
    }
}

// ─── partial M-RoPE ────────────────────────────────────────────────────────
//
// `qk` is (L, num_heads * head_dim). Extract the rotary subrange (first
// rotary_dim columns of each head) into a contiguous (L, num_heads*rotary_dim)
// scratch tensor, run rope_apply_mrope on it (sees a head_dim==rotary_dim
// space), and copy back. Pass-through columns are untouched.
//
// pos_t/h/w must be already in length L. The peak position across the three
// streams determines the per-axis table height.

void TextModel::prepare_mrope_(const std::vector<int32_t>& pos_t,
                               const std::vector<int32_t>& pos_h,
                               const std::vector<int32_t>& pos_w, int L) {
    const int rd = rotary_dim_;
    if (rd == 0) return;

    pos_t_host_ = pos_t;
    pos_h_host_ = pos_h;
    pos_w_host_ = pos_w;

    int max_pos = 0;
    auto upd = [&](const std::vector<int32_t>& v) {
        for (int32_t p : v) if (p > max_pos) max_pos = p;
    };
    upd(pos_t); upd(pos_h); upd(pos_w);
    mrope_max_pos_ = max_pos;
}

void TextModel::apply_partial_mrope_(bt::Tensor& qk, int num_heads, int L) {
    const int HD = cfg_.head_dim;
    const int rd = rotary_dim_;
    if (rd == 0) return;
    const int rot_cols = num_heads * rd;

    const bt::Device dev = qk.device;

    MRopeDeviceState* dev_st = nullptr;
    for (auto& s : mrope_states_) {
        if (s.device == dev) {
            dev_st = &s;
            break;
        }
    }
    if (!dev_st) {
        mrope_states_.push_back(MRopeDeviceState{});
        dev_st = &mrope_states_.back();
        dev_st->device = dev;
    }

    if (mrope_max_pos_ > dev_st->tbl_max_pos) {
        const int cap = std::max({mrope_max_pos_, 2 * dev_st->tbl_max_pos, 1023});
        internal::MRopePairing pairing = internal::mrope_pairing(d_t_, d_h_, d_w_);
        build_axis_tables(cap, d_t_, rd, cfg_.rope.rope_theta,
                          pairing.t_pairs, dev_st->cos_t, dev_st->sin_t, dev);
        build_axis_tables(cap, d_h_, rd, cfg_.rope.rope_theta,
                          pairing.h_pairs, dev_st->cos_h, dev_st->sin_h, dev);
        build_axis_tables(cap, d_w_, rd, cfg_.rope.rope_theta,
                          pairing.w_pairs, dev_st->cos_w, dev_st->sin_w, dev);
        dev_st->tbl_max_pos = cap;
    }

    dev_st->pos_t_dev = make_idx_device(pos_t_host_.data(), L, dev);
    dev_st->pos_h_dev = make_idx_device(pos_h_host_.data(), L, dev);
    dev_st->pos_w_dev = make_idx_device(pos_w_host_.data(), L, dev);

    // Choose a scratch tensor: q_rot_ or k_rot_
    bt::Tensor& scratch = (num_heads == cfg_.num_attention_heads) ? q_rot_ : k_rot_;
    brolm::detail::resize_like(scratch, L, rot_cols, qk.dtype, dev);

    // Rotary subrange qk -> scratch: the first rd columns of each head
    bt::copy_d2d_strided(qk, 0, HD, scratch, 0, rd,
                         /*width=*/rd, /*height=*/L * num_heads);

    bt::rope_apply_mrope(
        scratch,
        dev_st->cos_t, dev_st->sin_t,
        dev_st->cos_h, dev_st->sin_h,
        dev_st->cos_w, dev_st->sin_w,
        static_cast<const int32_t*>(dev_st->pos_t_dev.data),
        static_cast<const int32_t*>(dev_st->pos_h_dev.data),
        static_cast<const int32_t*>(dev_st->pos_w_dev.data),
        /*head_dim=*/rd,
        /*num_heads=*/num_heads,
        /*d_t=*/d_t_, /*d_h=*/d_h_, /*d_w=*/d_w_,
        scratch);

    // Rotated subrange back into qk; pass-through columns untouched.
    bt::copy_d2d_strided(scratch, 0, rd, qk, 0, HD,
                         /*width=*/rd, /*height=*/L * num_heads);
}

// ─── MLP block ─────────────────────────────────────────────────────────────

void TextModel::mlp_block_(const MLP& mlp, int L) {
    (void)L;
    if (mlp.gate_up_W.size() > 0) {
        {
            brolm::detail::profile::ScopedStage ps(brolm::detail::profile::Stage::mlp_proj);
            bt::linear_forward_batched_ex(mlp.gate_up_W, /*bias=*/nullptr, norm_, 0,
                                          bt::kLinearEpiSwiglu, nullptr, mlp_gate_);
        }
    } else {
        {
            brolm::detail::profile::ScopedStage ps(brolm::detail::profile::Stage::mlp_proj);
            detail::linear_batched(mlp.gate_W, /*bias=*/nullptr, norm_, mlp_gate_);
            detail::linear_batched(mlp.up_W,   /*bias=*/nullptr, norm_, mlp_up_);
        }

        // SwiGLU without concat staging: gate <- silu(gate) * up, in place
        // (silu_forward allows aliasing), then down-project.
        {
            brolm::detail::profile::ScopedStage ps(brolm::detail::profile::Stage::swiglu);
            bt::silu_forward(mlp_gate_, mlp_gate_);
            bt::mul_inplace(mlp_gate_, mlp_up_);
        }
    }
    {
        brolm::detail::profile::ScopedStage ps(brolm::detail::profile::Stage::mlp_proj);
        detail::linear_batched_accumulate(mlp.down_W, /*bias=*/nullptr, mlp_gate_, h_);
    }
}

// ─── forward ───────────────────────────────────────────────────────────────

bt::Tensor TextModel::embed_tokens(const std::vector<int>& token_ids) const {
    if (token_ids.empty()) fail("embed_tokens: token_ids empty");
    if (embed_.size() == 0) fail("embed_tokens: weights not loaded");
    const int L = static_cast<int>(token_ids.size());
    std::vector<int32_t> ids32(L);
    for (int i = 0; i < L; ++i) {
        ids32[static_cast<std::size_t>(i)] =
            static_cast<int32_t>(token_ids[static_cast<std::size_t>(i)]);
    }
    bt::Tensor ids_dev;
    {
        brolm::detail::profile::ScopedStage ps(brolm::detail::profile::Stage::idx_upload);
        ids_dev = make_idx_device(ids32.data(), L, embed_device());
    }
    bt::Tensor out;
    {
        brolm::detail::profile::ScopedStage ps(brolm::detail::profile::Stage::embed);
        bt::embedding_lookup_forward(
            embed_, static_cast<const int32_t*>(ids_dev.data), L, out);
    }
    // Clone so the result owns its own storage (embedding_lookup_forward may
    // alias internal scratch on some backends).
    return out.clone();
}

void TextModel::forward(const std::vector<int>& token_ids,
                        const std::vector<int64_t>& mrope_t,
                        const std::vector<int64_t>& mrope_h,
                        const std::vector<int64_t>& mrope_w,
                        std::vector<LayerCache>& cache,
                        bt::Tensor& logits_out) {
    // Embed then delegate — keeps the token-id and pre-embedded paths sharing
    // the exact same compute kernel below.
    bt::Tensor embeds = embed_tokens(token_ids);
    forward_embeds(embeds, mrope_t, mrope_h, mrope_w, cache, logits_out);
}

void TextModel::forward_embeds(const bt::Tensor& embeds,
                               const std::vector<int64_t>& mrope_t,
                               const std::vector<int64_t>& mrope_h,
                               const std::vector<int64_t>& mrope_w,
                               std::vector<LayerCache>& cache,
                               bt::Tensor& logits_out) {
    if (embeds.rows <= 0) fail("forward_embeds: empty input");
    const int L = embeds.rows;
    if (embeds.cols != cfg_.hidden_size) {
        fail("forward_embeds: embeds.cols != hidden_size");
    }
    if (static_cast<int>(mrope_t.size()) != L ||
        static_cast<int>(mrope_h.size()) != L ||
        static_cast<int>(mrope_w.size()) != L) {
        fail("forward_embeds: mrope_{t,h,w} length must match embeds.rows");
    }
    if (cache.size() != static_cast<std::size_t>(cfg_.num_hidden_layers)) {
        fail("forward_embeds: cache size mismatch");
    }
    if (embed_.size() == 0) fail("forward_embeds: weights not loaded");

    const int HD   = cfg_.head_dim;
    const int n_q  = cfg_.num_attention_heads;
    const int n_kv = cfg_.num_key_value_heads;
    const float eps = cfg_.rms_norm_eps;
    const int q_dim  = n_q  * HD;

    // Convert positions to int32 (brotensor expects int32 streams).
    std::vector<int32_t> pos_t(L), pos_h(L), pos_w(L);
    for (int i = 0; i < L; ++i) {
        pos_t[static_cast<std::size_t>(i)] = static_cast<int32_t>(mrope_t[static_cast<std::size_t>(i)]);
        pos_h[static_cast<std::size_t>(i)] = static_cast<int32_t>(mrope_h[static_cast<std::size_t>(i)]);
        pos_w[static_cast<std::size_t>(i)] = static_cast<int32_t>(mrope_w[static_cast<std::size_t>(i)]);
    }

    // Stage the M-RoPE position streams + tables once for the whole layer
    // stack.
    prepare_mrope_(pos_t, pos_h, pos_w, L);

    // Initialise the residual stream from the supplied embeddings. Clone so we
    // don't mutate the caller's tensor across the layer stack.
    h_ = embeds.clone();

    bt::Device prev_dev = bt::Device::CPU;

    for (int li = 0; li < cfg_.num_hidden_layers; ++li) {
        LayerSlot& layer = layers_[static_cast<std::size_t>(li)];
        LayerCache& c    = cache[static_cast<std::size_t>(li)];
        const bt::Device cur_dev = layer_device(li);

        if (h_.device != cur_dev) {
            h_ = h_.to(cur_dev);
        }
        if (cur_dev != prev_dev) {
            norm_ = bt::Tensor();
            q_ = bt::Tensor(); k_ = bt::Tensor(); v_ = bt::Tensor(); gate_ = bt::Tensor();
            qn_ = bt::Tensor(); kn_ = bt::Tensor(); q_rot_ = bt::Tensor(); k_rot_ = bt::Tensor();
            attn_ = bt::Tensor(); gate_sig_ = bt::Tensor(); proj_ = bt::Tensor();
            mlp_gate_ = bt::Tensor(); mlp_up_ = bt::Tensor(); mlp_gate_up_ = bt::Tensor();
            lin_qkv_ = bt::Tensor(); lin_qkv_ncl_ = bt::Tensor(); lin_conv_ncl_ = bt::Tensor();
            lin_qkv_conv_ = bt::Tensor(); lin_q_ = bt::Tensor(); lin_k_ = bt::Tensor(); lin_v_ = bt::Tensor();
            lin_a_raw_ = bt::Tensor(); lin_beta_ = bt::Tensor(); lin_z_ = bt::Tensor(); lin_zsilu_ = bt::Tensor();
            lin_O_ = bt::Tensor(); lin_O_norm_ = bt::Tensor(); lin_log_A_ = bt::Tensor();
            lin_x_fp32_ = bt::Tensor(); lin_proj_cast_ = bt::Tensor();
            lin_q_exp_ = bt::Tensor(); lin_k_exp_ = bt::Tensor();
            lin_z_fp32_ = bt::Tensor(); lin_O_cast_ = bt::Tensor(); lin_qkv_fp32_ = bt::Tensor();
            prev_dev = cur_dev;
        }

        // ── attention sub-layer ───────────────────────────────────────────
        {
            brolm::detail::profile::ScopedStage ps(brolm::detail::profile::Stage::rms_norm);
            bt::rms_norm_forward(h_, layer.in_norm, eps, norm_);
        }

        if (layer.type == LayerType::Full) {
            FullAttnKVCache& kvc = c.full;
            if (kvc.k.size() == 0) fail("forward: full-attn cache not allocated");
            const int max_seq = kvc.k.rows;
            if (kvc.len + L > max_seq) {
                fail("forward: cache_len + L exceeds allocated capacity");
            }

            {
                brolm::detail::profile::ScopedStage ps(brolm::detail::profile::Stage::qkv_proj);
                detail::linear_batched(layer.full.Wq, nullptr, norm_, q_);
                detail::linear_batched(layer.full.Wg, nullptr, norm_, gate_);
                detail::linear_batched(layer.full.Wk, nullptr, norm_, k_);
                detail::linear_batched(layer.full.Wv, nullptr, norm_, v_);
            }

            // Per-head RMSNorm (q/k only; full head_dim including pass-through).
            {
                brolm::detail::profile::ScopedStage ps(brolm::detail::profile::Stage::qk_norm);
                auto headnorm = [&](const bt::Tensor& src, int num_heads,
                                    const bt::Tensor& gain, bt::Tensor& dst) {
                    const int rows = src.rows;
                    bt::Tensor src_v = bt::Tensor::view(
                        src.device, src.data, rows * num_heads, HD, src.dtype);
                    bt::rms_norm_forward(src_v, gain, eps, dst);
                    dst.rows = rows;
                    dst.cols = num_heads * HD;
                };
                headnorm(q_, n_q,  layer.full.q_norm, qn_);
                headnorm(k_, n_kv, layer.full.k_norm, kn_);
            }

            // Partial M-RoPE on q and k.
            {
                brolm::detail::profile::ScopedStage ps(brolm::detail::profile::Stage::rope);
                apply_partial_mrope_(qn_, n_q,  L);
                apply_partial_mrope_(kn_, n_kv, L);
            }

            // GQA: append the n_kv-width k/v straight to the cache; the decode
            // op maps query head h to KV head h/(n_q/n_kv).
            {
                brolm::detail::profile::ScopedStage ps(brolm::detail::profile::Stage::kv_append);
                bt::kv_cache_append(kn_, v_, kvc.len, kvc.k, kvc.v);
            }

            // Causal attention against the populated cache.
            {
                brolm::detail::profile::ScopedStage ps(brolm::detail::profile::Stage::attention);
                bt::flash_attention_decode(qn_, kvc.k, kvc.v,
                                           kvc.len + L, n_q, n_kv, attn_);
                kvc.len += L;

                // attn_output_gate: attn = attn * sigmoid(gate) BEFORE o_proj.
                bt::sigmoid_forward(gate_, gate_sig_);
                bt::mul_inplace(attn_, gate_sig_);
            }

            {
                brolm::detail::profile::ScopedStage ps(brolm::detail::profile::Stage::o_proj);
                detail::linear_batched_accumulate(layer.full.Wo, nullptr, attn_, h_);
            }
        } else {
            // ── Gated DeltaNet (linear-attention) sub-layer ───────────────
            const int lin_h_v = cfg_.linear_num_value_heads;
            const int lin_h_k = cfg_.linear_num_key_heads;
            const int lin_d_k = cfg_.linear_key_head_dim;
            const int lin_d_v = cfg_.linear_value_head_dim;
            const int kdim    = lin_h_k * lin_d_k;     // per-stream cols
            const int vdim    = lin_h_v * lin_d_v;
            const int qkv_ch  = 2 * kdim + vdim;
            const int kK      = cfg_.linear_conv_kernel_dim;

            if (!c.lin.initialized) fail("forward: linear-attn cache not allocated");
            if (c.lin.recurrent.dtype != bt::Dtype::FP32 ||
                c.lin.conv_state.dtype != bt::Dtype::FP32) {
                fail("forward: linear-attn state must be FP32");
            }

            const bool has_fused_in_proj = (layer.lin.in_proj_all.size() > 0);
            const bt::Tensor* lin_x = &norm_;

            // The qkv / z projections. Normally the owned scratch members; on
            // the fused L == 1 decode path they are zero-copy views into
            // lin_in_all_ held in these locals instead. The views must never
            // be assigned into lin_qkv_ / lin_z_: a later multi-row call
            // resize()s those members, and resizing a non-owning view throws.
            bt::Tensor lin_qkv_view, lin_z_view;
            const bt::Tensor* lin_qkv = &lin_qkv_;
            const bt::Tensor* lin_z   = &lin_z_;

            if (has_fused_in_proj) {
                {
                    brolm::detail::profile::ScopedStage ps(brolm::detail::profile::Stage::lin_proj);
                    detail::linear_batched(layer.lin.in_proj_all, nullptr, norm_, lin_in_all_);
                }
                if (L == 1) {
                    char* all_bytes = static_cast<char*>(lin_in_all_.data);
                    const std::size_t es = bt::dtype_size_bytes(lin_in_all_.dtype);
                    lin_qkv_view = bt::Tensor::view(lin_in_all_.device, all_bytes, 1, qkv_ch, lin_in_all_.dtype);
                    lin_z_view   = bt::Tensor::view(lin_in_all_.device, all_bytes + qkv_ch * es, 1, vdim, lin_in_all_.dtype);
                    lin_qkv = &lin_qkv_view;
                    lin_z   = &lin_z_view;
                    bt::Tensor lin_a_sl = bt::Tensor::view(lin_in_all_.device, all_bytes + (qkv_ch + vdim) * es, 1, lin_h_v, lin_in_all_.dtype);
                    bt::Tensor lin_b_sl = bt::Tensor::view(lin_in_all_.device, all_bytes + (qkv_ch + vdim + lin_h_v) * es, 1, lin_h_v, lin_in_all_.dtype);

                    bt::cast(lin_a_sl, lin_a_raw_, bt::Dtype::FP32);
                    bt::add_inplace(lin_a_raw_, layer.lin.dt_bias);
                    bt::cast(lin_b_sl, lin_beta_, bt::Dtype::FP32);
                } else {
                    const int total_in_ch = qkv_ch + vdim + 2 * lin_h_v;
                    brolm::detail::resize_like(lin_qkv_, L, qkv_ch, lin_in_all_.dtype, lin_in_all_.device);
                    brolm::detail::resize_like(lin_z_,   L, vdim,   lin_in_all_.dtype, lin_in_all_.device);
                    bt::Tensor lin_a_sl, lin_b_sl;
                    brolm::detail::resize_like(lin_a_sl, L, lin_h_v, lin_in_all_.dtype, lin_in_all_.device);
                    brolm::detail::resize_like(lin_b_sl, L, lin_h_v, lin_in_all_.dtype, lin_in_all_.device);

                    bt::copy_d2d_strided(lin_in_all_, 0, total_in_ch,
                                         lin_qkv_, 0, qkv_ch, qkv_ch, L);
                    bt::copy_d2d_strided(lin_in_all_, qkv_ch, total_in_ch,
                                         lin_z_, 0, vdim, vdim, L);
                    bt::copy_d2d_strided(lin_in_all_, qkv_ch + vdim, total_in_ch,
                                         lin_a_sl, 0, lin_h_v, lin_h_v, L);
                    bt::copy_d2d_strided(lin_in_all_, qkv_ch + vdim + lin_h_v, total_in_ch,
                                         lin_b_sl, 0, lin_h_v, lin_h_v, L);

                    bt::cast(lin_a_sl, lin_a_raw_, bt::Dtype::FP32);
                    for (int l = 0; l < L; ++l) {
                        bt::Tensor row_a = bt::Tensor::view(lin_a_raw_.device,
                            static_cast<float*>(lin_a_raw_.data) + l * lin_h_v, 1, lin_h_v, bt::Dtype::FP32);
                        bt::add_inplace(row_a, layer.lin.dt_bias);
                    }
                    bt::cast(lin_b_sl, lin_beta_, bt::Dtype::FP32);
                }
                lin_log_A_ = layer.lin.A_log;
                lin_log_A_.rows = lin_h_v;
                lin_log_A_.cols = 1;
            } else {
                if (norm_.dtype != bt::Dtype::FP32) {
                    bt::cast(norm_, lin_x_fp32_, bt::Dtype::FP32);
                    lin_x = &lin_x_fp32_;
                }

                // 1) in_proj_qkv -> (L, qkv_ch)
                {
                    brolm::detail::profile::ScopedStage ps(brolm::detail::profile::Stage::lin_proj);
                    const bt::Tensor* qkv_in = (layer.lin.in_proj_qkv.dtype == bt::Dtype::FP32) ? lin_x : &norm_;
                    detail::linear_batched(layer.lin.in_proj_qkv, nullptr, *qkv_in, lin_qkv_);
                }
            }

            const bt::Tensor* qkv_conv_in = lin_qkv;
            if (lin_qkv->dtype != bt::Dtype::FP32) {
                bt::cast(*lin_qkv, lin_qkv_fp32_, bt::Dtype::FP32);
                qkv_conv_in = &lin_qkv_fp32_;
            }

            // 2) Depthwise causal conv1d against the rolling conv_state, then SiLU
            {
                brolm::detail::profile::ScopedStage ps(brolm::detail::profile::Stage::lin_conv);
                if (L == 1) {
                    bt::causal_conv1d_update(*qkv_conv_in, layer.lin.conv1d,
                                             /*bias=*/nullptr,
                                             /*N=*/1, /*C=*/qkv_ch, /*L_step=*/1,
                                             /*kL=*/kK, /*dilation=*/1,
                                             c.lin.conv_state, lin_qkv_conv_);
                } else {
                    bt::sequence_to_nchw(*qkv_conv_in, /*N=*/1, /*C=*/qkv_ch,
                                         /*H=*/1, /*W=*/L, lin_qkv_ncl_);
                    bt::causal_conv1d_update(lin_qkv_ncl_, layer.lin.conv1d,
                                             /*bias=*/nullptr,
                                             /*N=*/1, /*C=*/qkv_ch, /*L_step=*/L,
                                             /*kL=*/kK, /*dilation=*/1,
                                             c.lin.conv_state, lin_conv_ncl_);
                    bt::nchw_to_sequence(lin_conv_ncl_, /*N=*/1, /*C=*/qkv_ch,
                                         /*H=*/1, /*W=*/L, lin_qkv_conv_);
                }
                bt::silu_forward(lin_qkv_conv_, lin_qkv_conv_);
            }

            // 3) Split qkv_conv into q,k,v
            {
                brolm::detail::profile::ScopedStage ps(brolm::detail::profile::Stage::lin_split);
                brolm::detail::resize_like(lin_q_, L, kdim, lin_qkv_conv_.dtype, lin_qkv_conv_.device);
                brolm::detail::resize_like(lin_k_, L, kdim, lin_qkv_conv_.dtype, lin_qkv_conv_.device);
                brolm::detail::resize_like(lin_v_, L, vdim, lin_qkv_conv_.dtype, lin_qkv_conv_.device);
                bt::copy_d2d_strided(lin_qkv_conv_, 0 * kdim, qkv_ch,
                                     lin_q_, 0, kdim, kdim, L);
                bt::copy_d2d_strided(lin_qkv_conv_, 1 * kdim, qkv_ch,
                                     lin_k_, 0, kdim, kdim, L);
                bt::copy_d2d_strided(lin_qkv_conv_, 2 * kdim, qkv_ch,
                                     lin_v_, 0, vdim, vdim, L);
            }

            // 4) z, a_raw, beta, log_A (only if unfused)
            if (!has_fused_in_proj) {
                brolm::detail::profile::ScopedStage ps(brolm::detail::profile::Stage::lin_z_ab);
                const bt::Tensor* z_in = (layer.lin.in_proj_z.dtype == bt::Dtype::FP32) ? lin_x : &norm_;
                detail::linear_batched(layer.lin.in_proj_z, nullptr, *z_in, lin_z_);

                // 5) a_raw = in_proj_a @ x + dt_bias    (T, H).
                detail::linear_batched(layer.lin.in_proj_a, &layer.lin.dt_bias,
                                       *lin_x, lin_a_raw_);
                detail::linear_batched(layer.lin.in_proj_b, nullptr,
                                       *lin_x, lin_beta_);

                // 6) log_A as (num_heads, 1) FP32
                lin_log_A_ = layer.lin.A_log;
                lin_log_A_.rows = lin_h_v;
                lin_log_A_.cols = 1;
            }

            // 7) Pre-recurrence q/k transforms + GQA expansion
            const bt::Tensor* q_rec = &lin_q_;
            const bt::Tensor* k_rec = &lin_k_;
            {
                brolm::detail::profile::ScopedStage ps(brolm::detail::profile::Stage::lin_l2);
                bt::l2_norm_forward(lin_q_, /*head_dim=*/lin_d_k,
                                    /*num_heads=*/lin_h_k, /*eps=*/1e-6f, lin_q_);
                bt::l2_norm_forward(lin_k_, /*head_dim=*/lin_d_k,
                                    /*num_heads=*/lin_h_k, /*eps=*/1e-6f, lin_k_);
                bt::scale_inplace(lin_q_, 1.0f / std::sqrt(static_cast<float>(lin_d_k)));

                if (lin_h_k != lin_h_v) {
                    const int gqa_ratio = lin_h_v / lin_h_k;
                    const int v_kdim = lin_h_v * lin_d_k;
                    brolm::detail::resize_like(lin_q_exp_, L, v_kdim, lin_q_.dtype, lin_q_.device);
                    brolm::detail::resize_like(lin_k_exp_, L, v_kdim, lin_k_.dtype, lin_k_.device);
                    for (int hk = 0; hk < lin_h_k; ++hk) {
                        for (int r = 0; r < gqa_ratio; ++r) {
                            const int hv = hk * gqa_ratio + r;
                            bt::copy_d2d_strided(lin_q_, hk * lin_d_k, kdim,
                                                 lin_q_exp_, hv * lin_d_k, v_kdim, lin_d_k, L);
                            bt::copy_d2d_strided(lin_k_, hk * lin_d_k, kdim,
                                                 lin_k_exp_, hv * lin_d_k, v_kdim, lin_d_k, L);
                        }
                    }
                    q_rec = &lin_q_exp_;
                    k_rec = &lin_k_exp_;
                }
            }

            // 9) Recurrence: updates c.lin.recurrent in place; writes O.
            {
                brolm::detail::profile::ScopedStage ps(brolm::detail::profile::Stage::lin_delta_step);
                if (L == 1) {
                    bt::gated_delta_rule_step(*q_rec, *k_rec, lin_v_,
                                              lin_a_raw_, lin_beta_, lin_log_A_,
                                              /*num_heads=*/lin_h_v,
                                              /*d_k=*/lin_d_k, /*d_v=*/lin_d_v,
                                              c.lin.recurrent, lin_O_);
                } else {
                    bt::gated_delta_rule_chunked(*q_rec, *k_rec, lin_v_,
                                                 lin_a_raw_, lin_beta_, lin_log_A_,
                                                 /*num_heads=*/lin_h_v,
                                                 /*d_k=*/lin_d_k, /*d_v=*/lin_d_v,
                                                 c.lin.recurrent, lin_O_);
                }
                c.lin.len += L;
            }

            const bt::Tensor* z_fp32 = lin_z;
            if (lin_z->dtype != bt::Dtype::FP32) {
                bt::cast(*lin_z, lin_z_fp32_, bt::Dtype::FP32);
                z_fp32 = &lin_z_fp32_;
            }

            // 10) Per-head RMSNorm with norm.weight, then multiply by silu(z)
            {
                brolm::detail::profile::ScopedStage ps(brolm::detail::profile::Stage::lin_norm_gate);
                bt::Tensor o_view = bt::Tensor::view(
                    lin_O_.device, lin_O_.data,
                    L * lin_h_v, lin_d_v, lin_O_.dtype);
                bt::rms_norm_forward(o_view, layer.lin.norm, eps, lin_O_norm_);
                lin_O_norm_.rows = L;
                lin_O_norm_.cols = vdim;
                bt::silu_forward(*z_fp32, lin_zsilu_);
                bt::mul_inplace(lin_O_norm_, lin_zsilu_);
            }

            // 11) out_proj back to hidden, residual add.
            {
                brolm::detail::profile::ScopedStage ps(brolm::detail::profile::Stage::o_proj);
                const bt::Tensor* out_proj_in = &lin_O_norm_;
                if (lin_O_norm_.dtype != layer.lin.out_proj.dtype) {
                    bt::cast(lin_O_norm_, lin_O_cast_, layer.lin.out_proj.dtype);
                    out_proj_in = &lin_O_cast_;
                }
                detail::linear_batched_accumulate(layer.lin.out_proj, nullptr, *out_proj_in, h_);
            }
            (void)n_kv;
        }

        // ── MLP sub-layer ─────────────────────────────────────────────────
        {
            brolm::detail::profile::ScopedStage ps(brolm::detail::profile::Stage::rms_norm);
            bt::rms_norm_forward(h_, layer.post_attn_norm, eps, norm_);
        }
        mlp_block_(layer.mlp, L);
    }

    // Final RMSNorm + tied LM head on final_device().
    if (h_.device != final_device()) {
        h_ = h_.to(final_device());
    }
    if (final_device() != prev_dev) {
        norm_ = bt::Tensor();
    }
    {
        brolm::detail::profile::ScopedStage ps(brolm::detail::profile::Stage::final_norm);
        bt::rms_norm_forward(h_, final_norm_, eps, norm_);
    }
    {
        brolm::detail::profile::ScopedStage ps(brolm::detail::profile::Stage::lm_head);
        detail::linear_batched(lm_head_, /*bias=*/nullptr, norm_, logits_out);
    }
    (void)q_dim;
}

}  // namespace brolm::qwen35
