#include "brolm/qwen35_text.h"

#include "brolm/detail/compute.h"
#include "brolm/detail/device.h"
#include "brotensor/ops.h"
#include "brotensor/runtime.h"
#include "brotensor/safetensors.h"
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
namespace st = ::brotensor::safetensors;

using st::upload_compute_checked;

namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("qwen35::TextModel: " + msg);
}

const st::TensorView& need(const std::vector<const st::File*>& shards,
                           const std::string& key) {
    for (const st::File* f : shards) {
        if (const auto* v = f->find(key)) return *v;
    }
    fail("missing tensor '" + key + "'");
}

const st::TensorView* find_in(const std::vector<const st::File*>& shards,
                              const std::string& key) {
    for (const st::File* f : shards) {
        if (const auto* v = f->find(key)) return v;
    }
    return nullptr;
}

bt::Tensor make_idx_device(const int32_t* host, int n) {
    bt::Tensor cpu = bt::Tensor::empty_on(bt::Device::CPU, n, 1, bt::Dtype::INT32);
    std::memcpy(cpu.host_raw_mut(), host,
                static_cast<std::size_t>(n) * sizeof(int32_t));
    return cpu.to(bt::default_device());
}

// HF's Qwen3_5RMSNorm applies `(1 + weight)` as the gain (init zeros, centred
// on identity). brotensor's rms_norm_forward expects the raw gain, so we add
// 1.0 to every Qwen3_5RMSNorm weight at load time. Qwen3_5RMSNormGated (the
// linear-attn `linear_attn.norm`) uses plain `weight` (init ones) and is
// EXCLUDED from this transform. See HF transformers
// `Qwen3_5RMSNorm.forward` and the `_init_weights` comment "We initialize
// with 0s to be 1 centered as the RMSNorm here does (1 + weight)".
void add_one_to_norm_weight(bt::Tensor& t) {
    // Stage on host (FP32), add 1, re-upload at the original compute dtype.
    // Load-path only — runs once per layer.
    std::vector<float> h(static_cast<std::size_t>(t.size()));
    if (t.dtype == bt::Dtype::FP16) {
        std::vector<std::uint16_t> bits(h.size());
        t.copy_to_host_fp16(bits.data());
        for (std::size_t i = 0; i < h.size(); ++i)
            h[i] = bt::fp16_bits_to_fp32(bits[i]) + 1.0f;
    } else {
        h = t.to_host_vector();
        for (float& v : h) v += 1.0f;
    }
    const int r = t.rows;
    const int c = t.cols;
    t = brolm::detail::upload_host(h.data(), r, c);
}

std::vector<float> download_fp32(const bt::Tensor& t) {
    const std::size_t n = static_cast<std::size_t>(t.size());
    if (t.dtype == bt::Dtype::FP16) {
        std::vector<std::uint16_t> bits(n);
        t.copy_to_host_fp16(bits.data());
        std::vector<float> out(n);
        for (std::size_t i = 0; i < n; ++i) {
            out[i] = bt::fp16_bits_to_fp32(bits[i]);
        }
        return out;
    }
    return t.to_host_vector();
}

// HF M-RoPE interleaves three position streams (t,h,w) at the pair-index level.
// `apply_interleaved_mrope` (modeling_qwen3_5.py) overwrites the T-axis freq
// vector with H-axis values at slice(1, d_h*3, 3) and W-axis values at
// slice(2, d_w*3, 3); the remaining indices stay T. Concretely, with
// mrope_section=[d_t,d_h,d_w] and rotary_dim/2=d_t+d_h+d_w:
//   axis owner of HF pair-index j is:
//     'H' if j in {1, 4, 7, ..., 1 + 3*(d_h-1)}
//     'W' if j in {2, 5, 8, ..., 2 + 3*(d_w-1)}
//     'T' otherwise
// inv_freq[j] (from the global rotary_dim/2 schedule) is always used at HF
// pair-index j, regardless of axis ownership.
//
// brotensor's `rope_apply_mrope` assumes the chunked-axis layout: T owns the
// FIRST d_t pairs, H the next d_h, W the last d_w. To reconcile, we permute
// q_proj/k_proj rows (and q_norm/k_norm gains) at load time so that HF's
// scattered-by-axis ordering becomes brotensor's chunked ordering. The same
// permutation, plus the original rotate_half->pair conversion, is folded into
// a single row-index map below; `build_axis_tables` consumes the matching
// inv_freq index list per axis so that frequencies stay aligned.
//
// Returns, for each axis A in (T,H,W), the list of HF pair-indices owned by A
// in ascending pair-index order. Sum of sizes == rotary_dim/2.
struct MRopePairing {
    std::vector<int> t_pairs, h_pairs, w_pairs;
};

MRopePairing mrope_pairing(int d_t, int d_h, int d_w) {
    const int half = d_t + d_h + d_w;
    std::vector<char> owner(static_cast<std::size_t>(half), 'T');
    for (int k = 0; k < d_h; ++k) owner[static_cast<std::size_t>(1 + 3*k)] = 'H';
    for (int k = 0; k < d_w; ++k) owner[static_cast<std::size_t>(2 + 3*k)] = 'W';
    MRopePairing p;
    for (int j = 0; j < half; ++j) {
        switch (owner[static_cast<std::size_t>(j)]) {
            case 'T': p.t_pairs.push_back(j); break;
            case 'H': p.h_pairs.push_back(j); break;
            case 'W': p.w_pairs.push_back(j); break;
        }
    }
    return p;
}

// Build the brolm-dim -> HF-dim permutation for the rotary subrange of one
// head. For brolm pair b in [0, half), the source HF pair is
//   T_pairs[b]                              if b in [0, d_t)
//   H_pairs[b - d_t]                        if b in [d_t, d_t+d_h)
//   W_pairs[b - d_t - d_h]                  if b in [d_t+d_h, half)
// The HF dim for pair p slot s in {0,1} is `p + s*half` (rotate_half layout).
std::vector<int> rotary_row_perm(int rotary_dim, int d_t, int d_h, int d_w) {
    const int half = rotary_dim / 2;
    MRopePairing P = mrope_pairing(d_t, d_h, d_w);
    std::vector<int> hf_pair_for_brolm;
    hf_pair_for_brolm.reserve(static_cast<std::size_t>(half));
    hf_pair_for_brolm.insert(hf_pair_for_brolm.end(), P.t_pairs.begin(), P.t_pairs.end());
    hf_pair_for_brolm.insert(hf_pair_for_brolm.end(), P.h_pairs.begin(), P.h_pairs.end());
    hf_pair_for_brolm.insert(hf_pair_for_brolm.end(), P.w_pairs.begin(), P.w_pairs.end());
    std::vector<int> brolm_to_hf(static_cast<std::size_t>(rotary_dim));
    for (int b = 0; b < half; ++b) {
        const int hp = hf_pair_for_brolm[static_cast<std::size_t>(b)];
        brolm_to_hf[static_cast<std::size_t>(2*b)]     = hp;
        brolm_to_hf[static_cast<std::size_t>(2*b + 1)] = hp + half;
    }
    return brolm_to_hf;
}

// Permute the first `rotary_dim` rows within each head's `head_dim` row block
// from HF rotate_half order into brotensor's chunked-axis interleaved-pair
// order. Rows [rotary_dim, head_dim) are pass-through.
// src/dst are (num_heads*head_dim, cols) host buffers.
std::vector<float> permute_rotary_rows(const std::vector<float>& src,
                                       int num_heads, int head_dim,
                                       int rotary_dim, int cols,
                                       int d_t, int d_h, int d_w) {
    std::vector<int> brolm_to_hf = rotary_row_perm(rotary_dim, d_t, d_h, d_w);
    std::vector<float> dst(src.size());
    const std::size_t row_bytes = static_cast<std::size_t>(cols) * sizeof(float);
    for (int h = 0; h < num_heads; ++h) {
        const std::size_t base =
            static_cast<std::size_t>(h) * head_dim *
            static_cast<std::size_t>(cols);
        // Rotated subrange [0, rotary_dim): dst[b] <- src[brolm_to_hf[b]].
        for (int b = 0; b < rotary_dim; ++b) {
            const std::size_t doff = base + static_cast<std::size_t>(b) * cols;
            const std::size_t soff = base +
                static_cast<std::size_t>(brolm_to_hf[static_cast<std::size_t>(b)]) * cols;
            std::memcpy(&dst[doff], &src[soff], row_bytes);
        }
        // Pass-through tail [rotary_dim, head_dim): copy as is.
        for (int r = rotary_dim; r < head_dim; ++r) {
            const std::size_t off = base + static_cast<std::size_t>(r) * cols;
            std::memcpy(&dst[off], &src[off], row_bytes);
        }
    }
    return dst;
}

// Split q_proj's fanned-out weight (rows organized per-head as
// [head_h q (head_dim rows), head_h gate (head_dim rows), ...]) into two
// separate (n_q*head_dim, hidden) matrices `q_rows` and `g_rows`.
void split_q_gate_rows(const std::vector<float>& src,
                       int n_q, int head_dim, int cols,
                       std::vector<float>& q_rows,
                       std::vector<float>& g_rows) {
    const std::size_t per_head_rows = static_cast<std::size_t>(2 * head_dim);
    const std::size_t row_bytes = static_cast<std::size_t>(cols) * sizeof(float);
    const std::size_t single_dim = static_cast<std::size_t>(n_q) *
                                   static_cast<std::size_t>(head_dim) *
                                   static_cast<std::size_t>(cols);
    q_rows.assign(single_dim, 0.0f);
    g_rows.assign(single_dim, 0.0f);
    for (int h = 0; h < n_q; ++h) {
        const std::size_t src_base = static_cast<std::size_t>(h) *
                                     per_head_rows *
                                     static_cast<std::size_t>(cols);
        const std::size_t dst_base = static_cast<std::size_t>(h) *
                                     static_cast<std::size_t>(head_dim) *
                                     static_cast<std::size_t>(cols);
        for (int r = 0; r < head_dim; ++r) {
            const std::size_t soff = src_base + static_cast<std::size_t>(r) * cols;
            const std::size_t goff = src_base + static_cast<std::size_t>(head_dim + r) * cols;
            const std::size_t doff = dst_base + static_cast<std::size_t>(r) * cols;
            std::memcpy(&q_rows[doff], &src[soff], row_bytes);
            std::memcpy(&g_rows[doff], &src[goff], row_bytes);
        }
    }
}

// Build a per-axis sin/cos table (max_pos+1, d_axis) using base rope_theta and
// a list of GLOBAL inv_freq indices (one per axis slot). The HF schedule is
// inv_freq[j] = 1 / theta^(2j / rotary_dim) over j in [0, rotary_dim/2). For
// the chunked-axis layout brotensor expects, the per-axis tables index this
// schedule at the HF pair-indices owned by that axis (see `mrope_pairing`).
void build_axis_tables(int max_pos_inclusive, int d_axis, int rotary_dim,
                       float rope_theta,
                       const std::vector<int>& freq_indices,
                       bt::Tensor& cos_t, bt::Tensor& sin_t) {
    const int rows = std::max(1, max_pos_inclusive + 1);
    if (d_axis <= 0) {
        // Degenerate axis: brotensor still accepts a (rows, 0)-ish layout via
        // an empty (rows, 1) placeholder; we pass empty Tensors. The mrope op
        // skips them when d_axis==0.
        cos_t = bt::Tensor::zeros_on(bt::default_device(), rows, 1, bt::Dtype::FP32);
        sin_t = bt::Tensor::zeros_on(bt::default_device(), rows, 1, bt::Dtype::FP32);
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
    cos_t = bt::Tensor::from_host_on(bt::default_device(), cos_h.data(), rows, d_axis);
    sin_t = bt::Tensor::from_host_on(bt::default_device(), sin_h.data(), rows, d_axis);
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

// ─── load_weights ──────────────────────────────────────────────────────────

void TextModel::load_weights(const st::File& f, const std::string& prefix) {
    const std::vector<const st::File*> shards = {&f};
    load_weights_impl_(shards, prefix);
}

void TextModel::load_weights(const std::vector<const st::File*>& shards,
                             const std::string& prefix) {
    load_weights_impl_(shards, prefix);
}

void TextModel::load_weights_impl_(
    const std::vector<const st::File*>& shards, const std::string& prefix) {
    if (shards.empty()) fail("load_weights: no safetensors shards");

    const int V    = cfg_.vocab_size;
    const int H    = cfg_.hidden_size;
    const int Fm   = cfg_.intermediate_size;
    const int HD   = cfg_.head_dim;
    const int n_q  = cfg_.num_attention_heads;
    const int n_kv = cfg_.num_key_value_heads;
    const int q_dim    = n_q  * HD;
    const int kv_dim   = n_kv * HD;
    const int q_dim2   = 2 * q_dim;        // q_proj output width (q + gate)

    upload_compute_checked(need(shards, prefix + "embed_tokens.weight"),
                           V, H, embed_, "embed_tokens.weight");

    upload_compute_checked(need(shards, prefix + "norm.weight"),
                           H, 1, final_norm_, "language_model.norm.weight");
    add_one_to_norm_weight(final_norm_);

    for (int i = 0; i < cfg_.num_hidden_layers; ++i) {
        const std::string p =
            prefix + "layers." + std::to_string(i) + ".";
        LayerSlot& L = layers_[static_cast<std::size_t>(i)];

        upload_compute_checked(need(shards, p + "input_layernorm.weight"),
                               H, 1, L.in_norm, "input_layernorm.weight");
        add_one_to_norm_weight(L.in_norm);
        upload_compute_checked(
            need(shards, p + "post_attention_layernorm.weight"),
            H, 1, L.post_attn_norm, "post_attention_layernorm.weight");
        add_one_to_norm_weight(L.post_attn_norm);

        // MLP — same shape on every layer.
        upload_compute_checked(need(shards, p + "mlp.gate_proj.weight"),
                               Fm, H, L.mlp.gate_W, "mlp.gate_proj.weight");
        upload_compute_checked(need(shards, p + "mlp.up_proj.weight"),
                               Fm, H, L.mlp.up_W, "mlp.up_proj.weight");
        upload_compute_checked(need(shards, p + "mlp.down_proj.weight"),
                               H, Fm, L.mlp.down_W, "mlp.down_proj.weight");

        if (L.type == LayerType::Full) {
            // q_proj: (2*q_dim, hidden), per-head layout [q, gate]. Load raw,
            // split into Wq/Wg, then permute the rotary subrange of Wq's rows
            // (per-head, only the q half) and of Wk's rows for interleaved RoPE.
            bt::Tensor q_raw;
            upload_compute_checked(need(shards, p + "self_attn.q_proj.weight"),
                                   q_dim2, H, q_raw, "self_attn.q_proj.weight");
            std::vector<float> q_host = download_fp32(q_raw);
            std::vector<float> q_rows, g_rows;
            split_q_gate_rows(q_host, n_q, HD, H, q_rows, g_rows);
            std::vector<float> q_perm =
                permute_rotary_rows(q_rows, n_q, HD, rotary_dim_, H,
                                    d_t_, d_h_, d_w_);
            L.full.Wq = brolm::detail::upload_host(q_perm.data(), q_dim, H);
            L.full.Wg = brolm::detail::upload_host(g_rows.data(), q_dim, H);

            bt::Tensor k_raw;
            upload_compute_checked(need(shards, p + "self_attn.k_proj.weight"),
                                   kv_dim, H, k_raw, "self_attn.k_proj.weight");
            std::vector<float> k_host = download_fp32(k_raw);
            std::vector<float> k_perm =
                permute_rotary_rows(k_host, n_kv, HD, rotary_dim_, H,
                                    d_t_, d_h_, d_w_);
            L.full.Wk = brolm::detail::upload_host(k_perm.data(), kv_dim, H);

            upload_compute_checked(need(shards, p + "self_attn.v_proj.weight"),
                                   kv_dim, H, L.full.Wv, "self_attn.v_proj.weight");
            upload_compute_checked(need(shards, p + "self_attn.o_proj.weight"),
                                   H, q_dim, L.full.Wo, "self_attn.o_proj.weight");

            // Per-head norms: permute the rotary subrange [0, rotary_dim).
            bt::Tensor qn_raw, kn_raw;
            upload_compute_checked(need(shards, p + "self_attn.q_norm.weight"),
                                   HD, 1, qn_raw, "self_attn.q_norm.weight");
            upload_compute_checked(need(shards, p + "self_attn.k_norm.weight"),
                                   HD, 1, kn_raw, "self_attn.k_norm.weight");
            std::vector<float> qn_host = download_fp32(qn_raw);
            std::vector<float> kn_host = download_fp32(kn_raw);
            // HF Qwen3_5RMSNorm applies (1 + weight) as gain — add 1 here.
            for (float& v : qn_host) v += 1.0f;
            for (float& v : kn_host) v += 1.0f;
            std::vector<float> qn_perm =
                permute_rotary_rows(qn_host, /*num_heads=*/1, HD, rotary_dim_, 1,
                                    d_t_, d_h_, d_w_);
            std::vector<float> kn_perm =
                permute_rotary_rows(kn_host, /*num_heads=*/1, HD, rotary_dim_, 1,
                                    d_t_, d_h_, d_w_);
            L.full.q_norm = brolm::detail::upload_host(qn_perm.data(), HD, 1);
            L.full.k_norm = brolm::detail::upload_host(kn_perm.data(), HD, 1);
        } else {
            // Linear-attn layer — load tensors so the safetensors check holds
            // and Stage 3b can read them. Forward path stubs to identity.
            const std::string lp = p + "linear_attn.";
            // Shapes derived from cfg_. All Linear at this point only loaded.
            // (We don't strictly need to load these for Stage 3a, but the spec
            // requires the loader to accept them — so resolve each tensor by
            // name and copy its bytes through upload_compute. Shapes use
            // safetensors view->shape directly via upload_compute (no shape
            // check) to stay robust to future config drift.)
            // Force-upload every linear-attn tensor as FP32 regardless of the
            // pipeline compute dtype. The Gated DeltaNet recurrence
            // (gated_delta_rule_step) and causal_conv1d_update kernels are
            // FP32-only on both CPU and CUDA, so the entire linear-attn block
            // runs FP32 — the forward casts norm_ to FP32 on entry and casts
            // the residual back to compute_dtype before adding to h_. Loading
            // these weights at compute_dtype (FP16 on GPU) would force a
            // per-step dtype mismatch in every kernel of the block.
            auto load_fp32 = [&](const std::string& key, bt::Tensor& dst) {
                const auto& v = need(shards, key);
                int rows = 0, cols = 1;
                if (v.shape.size() == 1) {
                    rows = static_cast<int>(v.shape[0]);
                } else if (v.shape.size() == 2) {
                    rows = static_cast<int>(v.shape[0]);
                    cols = static_cast<int>(v.shape[1]);
                } else if (v.shape.size() == 3) {
                    rows = static_cast<int>(v.shape[0]);
                    cols = static_cast<int>(v.shape[1]) *
                           static_cast<int>(v.shape[2]);
                } else {
                    fail("linear_attn '" + key + "': unsupported rank");
                }
                const std::size_t n =
                    static_cast<std::size_t>(rows) * cols;
                std::vector<float> tmp(n);
                if (v.dtype == st::Dtype::F32) {
                    std::memcpy(tmp.data(), v.data, n * sizeof(float));
                } else if (v.dtype == st::Dtype::F16) {
                    const uint16_t* src =
                        reinterpret_cast<const uint16_t*>(v.data);
                    for (std::size_t i = 0; i < n; ++i)
                        tmp[i] = bt::fp16_bits_to_fp32(src[i]);
                } else if (v.dtype == st::Dtype::BF16) {
                    const uint16_t* src =
                        reinterpret_cast<const uint16_t*>(v.data);
                    for (std::size_t i = 0; i < n; ++i)
                        tmp[i] = bt::bf16_bits_to_fp32(src[i]);
                } else {
                    fail("linear_attn '" + key + "': unsupported dtype");
                }
                dst = bt::Tensor::from_host(tmp.data(), rows, cols);
            };
            load_fp32(lp + "A_log",             L.lin.A_log);
            load_fp32(lp + "conv1d.weight",     L.lin.conv1d);
            load_fp32(lp + "dt_bias",           L.lin.dt_bias);
            load_fp32(lp + "in_proj_a.weight",  L.lin.in_proj_a);
            load_fp32(lp + "in_proj_b.weight",  L.lin.in_proj_b);
            load_fp32(lp + "in_proj_qkv.weight", L.lin.in_proj_qkv);
            load_fp32(lp + "in_proj_z.weight",  L.lin.in_proj_z);
            load_fp32(lp + "norm.weight",       L.lin.norm);
            load_fp32(lp + "out_proj.weight",   L.lin.out_proj);
        }
    }

    // tie_word_embeddings: lm_head is the same matrix as embed_tokens.
    // No explicit lm_head load — the forward computes logits = embed @ h.
    if (!cfg_.tie_word_embeddings) {
        if (find_in(shards, prefix + "lm_head.weight") == nullptr) {
            fail("tie_word_embeddings=false but lm_head.weight missing");
        }
        // Untied path not exercised by Qwen3.5 releases; skip for now.
        fail("tie_word_embeddings=false not supported in this chunk");
    }
}

// ─── cache ─────────────────────────────────────────────────────────────────

std::vector<LayerCache> TextModel::make_cache(int max_seq) const {
    if (max_seq <= 0) fail("make_cache: max_seq must be positive");
    const int n_kv = cfg_.num_key_value_heads;
    const int cache_cols = n_kv * cfg_.head_dim;  // true KV width; decode does GQA
    const bt::Dtype dt = brolm::compute_dtype();
    const bt::Device dev = bt::default_device();

    // Linear-attn shapes.
    const int lin_h    = cfg_.linear_num_value_heads;
    const int lin_d_v  = cfg_.linear_value_head_dim;
    const int lin_d_k  = cfg_.linear_key_head_dim;
    const int qkv_ch   = 3 * lin_h * lin_d_k;             // 3 * heads * d_k
    const int conv_st_cols = qkv_ch * (cfg_.linear_conv_kernel_dim - 1);
    const int state_cols   = lin_d_v * lin_d_k;

    std::vector<LayerCache> out(static_cast<std::size_t>(cfg_.num_hidden_layers));
    for (int i = 0; i < cfg_.num_hidden_layers; ++i) {
        LayerCache& c = out[static_cast<std::size_t>(i)];
        if (cfg_.layer_types[static_cast<std::size_t>(i)] == LayerType::Full) {
            brolm::detail::resize_like(c.full.k, max_seq, cache_cols, dt, dev);
            brolm::detail::resize_like(c.full.v, max_seq, cache_cols, dt, dev);
            c.full.len = 0;
        } else {
            // Recurrent state + conv shift register, both FP32 (the recurrence
            // op and causal_conv1d_update are FP32 on CPU).
            c.lin.recurrent = bt::Tensor::zeros_on(dev, lin_h, state_cols,
                                                   bt::Dtype::FP32);
            c.lin.conv_state = bt::Tensor::zeros_on(dev, 1, conv_st_cols,
                                                    bt::Dtype::FP32);
            c.lin.initialized = true;
        }
    }
    return out;
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

    int max_pos = 0;
    auto upd = [&](const std::vector<int32_t>& v) {
        for (int32_t p : v) if (p > max_pos) max_pos = p;
    };
    upd(pos_t); upd(pos_h); upd(pos_w);

    // (Re)build the per-axis tables only when the cached ones are too short.
    // Bucketed growth: a decode loop advances max_pos by one per token, and
    // rebuilding per token is the per-layer host-rebuild pathology this cache
    // exists to kill. rope_apply_mrope indexes rows by position, so taller
    // tables serve smaller positions unchanged.
    if (max_pos > mrope_tbl_max_pos_) {
        const int cap = std::max({max_pos, 2 * mrope_tbl_max_pos_, 1023});
        // Per-axis global inv_freq indices follow HF's apply_interleaved_mrope.
        MRopePairing pairing = mrope_pairing(d_t_, d_h_, d_w_);
        build_axis_tables(cap, d_t_, rd, cfg_.rope.rope_theta,
                          pairing.t_pairs, mrope_cos_t_, mrope_sin_t_);
        build_axis_tables(cap, d_h_, rd, cfg_.rope.rope_theta,
                          pairing.h_pairs, mrope_cos_h_, mrope_sin_h_);
        build_axis_tables(cap, d_w_, rd, cfg_.rope.rope_theta,
                          pairing.w_pairs, mrope_cos_w_, mrope_sin_w_);
        mrope_tbl_max_pos_ = cap;
    }

    // brotensor's mrope op accepts host pointers on CPU, device on CUDA/Metal.
    // Upload each stream once per forward; every layer reads the same buffers.
    pos_t_dev_ = make_idx_device(pos_t.data(), L);
    pos_h_dev_ = make_idx_device(pos_h.data(), L);
    pos_w_dev_ = make_idx_device(pos_w.data(), L);
}

void TextModel::apply_partial_mrope_(bt::Tensor& qk, int num_heads, int L) {
    const int HD = cfg_.head_dim;
    const int rd = rotary_dim_;
    if (rd == 0) return;
    const int rot_cols = num_heads * rd;

    // Choose a scratch tensor: q_rot_ or k_rot_ — caller passes us qk by ref
    // so we route through a single internal scratch. We pick based on the
    // num_heads value (n_q vs n_kv) — q_rot_ for the larger head count, k_rot_
    // for the smaller. Match against cfg_ so we use the right buffer.
    bt::Tensor& scratch = (num_heads == cfg_.num_attention_heads) ? q_rot_ : k_rot_;
    brolm::detail::resize_like(scratch, L, rot_cols, qk.dtype, qk.device);

    // Rotary subrange qk -> scratch: the first rd columns of each head, one
    // strided device copy over L*num_heads rows (was a copy_d2d per row per
    // head).
    bt::copy_d2d_strided(qk, 0, HD, scratch, 0, rd,
                         /*width=*/rd, /*height=*/L * num_heads);

    bt::rope_apply_mrope(
        scratch,
        mrope_cos_t_, mrope_sin_t_,
        mrope_cos_h_, mrope_sin_h_,
        mrope_cos_w_, mrope_sin_w_,
        static_cast<const int32_t*>(pos_t_dev_.data),
        static_cast<const int32_t*>(pos_h_dev_.data),
        static_cast<const int32_t*>(pos_w_dev_.data),
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
    detail::linear_batched(mlp.gate_W, /*bias=*/nullptr, norm_, mlp_gate_);
    detail::linear_batched(mlp.up_W,   /*bias=*/nullptr, norm_, mlp_up_);

    // SwiGLU without concat staging: gate <- silu(gate) * up, in place
    // (silu_forward allows aliasing), then down-project.
    bt::silu_forward(mlp_gate_, mlp_gate_);
    bt::mul_inplace(mlp_gate_, mlp_up_);
    detail::linear_batched(mlp.down_W, /*bias=*/nullptr, mlp_gate_, proj_);
    bt::add_inplace(h_, proj_);
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
    bt::Tensor ids_dev = make_idx_device(ids32.data(), L);
    bt::Tensor out;
    bt::embedding_lookup_forward(
        embed_, static_cast<const int32_t*>(ids_dev.data), L, out);
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

    for (int li = 0; li < cfg_.num_hidden_layers; ++li) {
        LayerSlot& layer = layers_[static_cast<std::size_t>(li)];
        LayerCache& c    = cache[static_cast<std::size_t>(li)];

        // ── attention sub-layer ───────────────────────────────────────────
        bt::rms_norm_forward(h_, layer.in_norm, eps, norm_);

        if (layer.type == LayerType::Full) {
            FullAttnKVCache& kvc = c.full;
            if (kvc.k.size() == 0) fail("forward: full-attn cache not allocated");
            const int max_seq = kvc.k.rows;
            if (kvc.len + L > max_seq) {
                fail("forward: cache_len + L exceeds allocated capacity");
            }

            detail::linear_batched(layer.full.Wq, nullptr, norm_, q_);
            detail::linear_batched(layer.full.Wg, nullptr, norm_, gate_);
            detail::linear_batched(layer.full.Wk, nullptr, norm_, k_);
            detail::linear_batched(layer.full.Wv, nullptr, norm_, v_);

            // Per-head RMSNorm (q/k only; full head_dim including pass-through).
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

            // Partial M-RoPE on q and k.
            apply_partial_mrope_(qn_, n_q,  L);
            apply_partial_mrope_(kn_, n_kv, L);

            // GQA: append the n_kv-width k/v straight to the cache; the decode
            // op maps query head h to KV head h/(n_q/n_kv).
            bt::kv_cache_append(kn_, v_, kvc.len, kvc.k, kvc.v);

            // Causal attention against the populated cache.
            bt::flash_attention_decode(qn_, kvc.k, kvc.v,
                                       kvc.len + L, n_q, n_kv, attn_);
            kvc.len += L;

            // attn_output_gate: attn = attn * sigmoid(gate) BEFORE o_proj.
            bt::sigmoid_forward(gate_, gate_sig_);
            bt::mul_inplace(attn_, gate_sig_);

            detail::linear_batched(layer.full.Wo, nullptr, attn_, proj_);
            bt::add_inplace(h_, proj_);
        } else {
            // ── Gated DeltaNet (linear-attention) sub-layer ───────────────
            //
            // Faithful port of HF transformers' Qwen3NextGatedDeltaNet.forward
            // (https://raw.githubusercontent.com/huggingface/transformers/main/
            //  src/transformers/models/qwen3_next/modeling_qwen3_next.py).
            //
            // Per-step rule, applied serially over the T new tokens (also when
            // T > 1 in prefill — this is a known perf tradeoff vs. HF's
            // chunked WY/UT formulation; see the note at the top of this
            // function for Stage 5 follow-up):
            //
            //   x      = rms_norm(h, in_norm)                  [already done]
            //   qkv    = in_proj_qkv @ x                       (T, 3*H*D_k)
            //   qkv    = silu( causal_conv1d_update(qkv, conv_state) )
            //   q,k,v  = split qkv  -> each (T, H*D_k or H*D_v)
            //   z      = in_proj_z @ x                         (T, H*D_v)
            //   a_raw  = in_proj_a @ x + dt_bias               (T, H) FP32
            //   b_raw  = in_proj_b @ x                         (T, H) FP32
            //   O, S'  = gated_delta_rule_step(q,k,v,a_raw,b_raw,log_A,S)
            //     (per HF: g = -A_log.exp() * softplus(a + dt_bias);
            //              beta = sigmoid(b); applied inside brotensor's op.)
            //   O'     = rms_norm_per_head(O, norm.weight)     [per HF: norm
            //              before gate]
            //   O'     = O' * silu(z)                          (Qwen3NextRMSNormGated)
            //   out    = out_proj @ O'                         (T, hidden)
            //   h     += out
            const int lin_h   = cfg_.linear_num_value_heads;
            const int lin_d_k = cfg_.linear_key_head_dim;
            const int lin_d_v = cfg_.linear_value_head_dim;
            const int qkv_ch  = 3 * lin_h * lin_d_k;
            const int kdim    = lin_h * lin_d_k;     // per-stream cols
            const int vdim    = lin_h * lin_d_v;
            const int kK      = cfg_.linear_conv_kernel_dim;

            if (!c.lin.initialized) fail("forward: linear-attn cache not allocated");
            if (c.lin.recurrent.dtype != bt::Dtype::FP32 ||
                c.lin.conv_state.dtype != bt::Dtype::FP32) {
                fail("forward: linear-attn state must be FP32");
            }

            // The Gated DeltaNet kernels (causal_conv1d_update,
            // gated_delta_rule_step, the per-head RMSNormGated) are FP32-only.
            // The weights for this block are uploaded as FP32 at load time;
            // here we cast the FP16 activation input to FP32 on entry so every
            // matmul in the block produces FP32 directly, and cast the
            // out_proj result back to compute_dtype before the residual add.
            const bt::Tensor* lin_x = &norm_;
            if (norm_.dtype != bt::Dtype::FP32) {
                bt::cast(norm_, lin_x_fp32_, bt::Dtype::FP32);
                lin_x = &lin_x_fp32_;
            }

            // 1) in_proj_qkv -> (L, qkv_ch)
            detail::linear_batched(layer.lin.in_proj_qkv, nullptr, *lin_x, lin_qkv_);

            // 2) Depthwise causal conv1d with state, then SiLU (HF activation).
            //    brotensor's causal_conv1d_update lays X out as (N, C * L_step)
            //    with the L_step samples per channel CONTIGUOUS (channel-major).
            //    Our `lin_qkv_` is (T, C) token-major, so the obvious bulk call
            //    would reorder the layout. We instead apply the op per token
            //    (L_step=1) — each call is a length-C row, which matches both
            //    layouts trivially and keeps the recurrence-equivalence
            //    between prefill (T>1) and decode (T=1) exact.
            //
            //    Perf note: this serial-per-token pass is O(T * C * kL); a
            //    chunked-parallel implementation matching HF's `causal_conv1d_fn`
            //    would be a Stage 5 follow-up. For the prefill lengths brolm
            //    targets (hundreds of tokens) the difference is negligible.
            brolm::detail::resize_like(lin_qkv_conv_, L, qkv_ch,
                                       bt::Dtype::FP32, lin_qkv_.device);
            {
                bt::Tensor row_in, row_out;
                for (int t = 0; t < L; ++t) {
                    row_in = bt::Tensor::view(lin_qkv_.device,
                        static_cast<char*>(lin_qkv_.data) +
                            static_cast<std::size_t>(t) * qkv_ch * sizeof(float),
                        1, qkv_ch, bt::Dtype::FP32);
                    bt::causal_conv1d_update(row_in, layer.lin.conv1d,
                                             /*bias=*/nullptr,
                                             /*N=*/1, /*C=*/qkv_ch, /*L_step=*/1,
                                             /*kL=*/kK, /*dilation=*/1,
                                             c.lin.conv_state, row_out);
                    // Copy row_out (1, qkv_ch) into lin_qkv_conv_ at row t.
                    bt::copy_d2d(row_out, 0, lin_qkv_conv_, t * qkv_ch, qkv_ch);
                }
            }
            bt::silu_forward(lin_qkv_conv_, lin_qkv_conv_);

            // 3) Split qkv_conv into q,k,v. Layout per HF: dim 0 is q (kdim),
            //    then k (kdim), then v (vdim). With d_k == d_v this is just
            //    three equal slabs of `kdim` columns.
            brolm::detail::resize_like(lin_q_, L, kdim, lin_qkv_conv_.dtype, lin_qkv_conv_.device);
            brolm::detail::resize_like(lin_k_, L, kdim, lin_qkv_conv_.dtype, lin_qkv_conv_.device);
            brolm::detail::resize_like(lin_v_, L, vdim, lin_qkv_conv_.dtype, lin_qkv_conv_.device);
            bt::copy_d2d_strided(lin_qkv_conv_, 0 * kdim, qkv_ch,
                                 lin_q_, 0, kdim, kdim, L);
            bt::copy_d2d_strided(lin_qkv_conv_, 1 * kdim, qkv_ch,
                                 lin_k_, 0, kdim, kdim, L);
            bt::copy_d2d_strided(lin_qkv_conv_, 2 * kdim, qkv_ch,
                                 lin_v_, 0, vdim, vdim, L);

            // 4) z = in_proj_z @ x  (T, H*D_v)
            detail::linear_batched(layer.lin.in_proj_z, nullptr, *lin_x, lin_z_);

            // 5) a_raw = in_proj_a @ x + dt_bias    (T, H). dt_bias is the
            //    linear's bias; brotensor's FP32 batched linear broadcasts
            //    (out, 1) over the row axis. Output is already FP32 since W
            //    and *lin_x are FP32; brotensor's recurrence op does softplus
            //    internally so we hand it the SUM.
            detail::linear_batched(layer.lin.in_proj_a, &layer.lin.dt_bias,
                                   *lin_x, lin_a_raw_);
            detail::linear_batched(layer.lin.in_proj_b, nullptr,
                                   *lin_x, lin_beta_);

            // 6) log_A as (num_heads, 1) FP32 — weight is already FP32, just
            //    rewrite the shape for the recurrence op.
            lin_log_A_ = layer.lin.A_log;
            lin_log_A_.rows = lin_h;
            lin_log_A_.cols = 1;

            // 7) Pre-recurrence q/k transforms that HF folds into its
            //    `chunk_gated_delta_rule` / `recurrent_gated_delta_rule` kernels
            //    when `use_qk_l2norm_in_kernel=True`, but which brotensor's
            //    `gated_delta_rule_step` does NOT apply internally:
            //      - L2-normalize q,k along the per-head d_k axis (eps=1e-6).
            //      - Scale q by 1/sqrt(d_k) (HF: `scale = query.shape[-1]**-0.5`).
            //    See HF transformers Qwen3_5 `torch_recurrent_gated_delta_rule`
            //    (l2norm @ line ~331, scale @ line ~340-341) and the chunked
            //    counterpart for the same two steps.
            bt::l2_norm_forward(lin_q_, /*head_dim=*/lin_d_k,
                                /*num_heads=*/lin_h, /*eps=*/1e-6f, lin_q_);
            bt::l2_norm_forward(lin_k_, /*head_dim=*/lin_d_k,
                                /*num_heads=*/lin_h, /*eps=*/1e-6f, lin_k_);
            bt::scale_inplace(lin_q_, 1.0f / std::sqrt(static_cast<float>(lin_d_k)));

            // 8) Recurrence: updates c.lin.recurrent in place; writes O.
            bt::gated_delta_rule_step(lin_q_, lin_k_, lin_v_,
                                       lin_a_raw_, lin_beta_, lin_log_A_,
                                       /*num_heads=*/lin_h,
                                       /*d_k=*/lin_d_k, /*d_v=*/lin_d_v,
                                       c.lin.recurrent, lin_O_);

            // 9) Per-head RMSNorm with norm.weight (size value_head_dim), then
            //    multiply by silu(z). Order: HF's Qwen3NextRMSNormGated does
            //    "norm before gate" — rms_norm(h)*weight, then * silu(gate).
            {
                bt::Tensor o_view = bt::Tensor::view(
                    lin_O_.device, lin_O_.data,
                    L * lin_h, lin_d_v, lin_O_.dtype);
                bt::rms_norm_forward(o_view, layer.lin.norm, eps, lin_O_norm_);
                lin_O_norm_.rows = L;
                lin_O_norm_.cols = vdim;
            }
            bt::silu_forward(lin_z_, lin_zsilu_);
            bt::mul_inplace(lin_O_norm_, lin_zsilu_);

            // 10) out_proj back to hidden, residual add. proj_ is FP32 here
            //     (FP32 weight × FP32 input); cast back to compute_dtype so it
            //     can be added to the FP16 residual stream on GPU builds.
            detail::linear_batched(layer.lin.out_proj, nullptr, lin_O_norm_, proj_);
            const bt::Tensor* proj_to_add = &proj_;
            if (proj_.dtype != h_.dtype) {
                bt::cast(proj_, lin_proj_cast_, h_.dtype);
                proj_to_add = &lin_proj_cast_;
            }
            bt::add_inplace(h_, *proj_to_add);
            (void)n_kv;
        }

        // ── MLP sub-layer ─────────────────────────────────────────────────
        bt::rms_norm_forward(h_, layer.post_attn_norm, eps, norm_);
        mlp_block_(layer.mlp, L);
    }

    // Final RMSNorm + tied LM head.
    bt::rms_norm_forward(h_, final_norm_, eps, norm_);
    detail::linear_batched(embed_, /*bias=*/nullptr, norm_, logits_out);
    (void)q_dim;
}

}  // namespace brolm::qwen35
