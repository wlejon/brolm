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

// Permute the first `rotary_dim` rows within each head's `head_dim` row block
// from HF rotate_half order into brotensor's interleaved-pair order. Rows
// `[rotary_dim, head_dim)` are pass-through (copied verbatim).
//
// HF dim i (i in [0, rotary_dim/2)) pairs with HF dim i + rotary_dim/2.
// brotensor wants:
//   interleaved index 2i   <- HF index i
//   interleaved index 2i+1 <- HF index i + rotary_dim/2
//
// src/dst are (num_heads*head_dim, cols) host buffers.
std::vector<float> permute_rotary_rows(const std::vector<float>& src,
                                       int num_heads, int head_dim,
                                       int rotary_dim, int cols) {
    const int half = rotary_dim / 2;
    std::vector<float> dst(src.size());
    for (int h = 0; h < num_heads; ++h) {
        const std::size_t base =
            static_cast<std::size_t>(h) * head_dim *
            static_cast<std::size_t>(cols);
        // Rotated subrange [0, rotary_dim): interleave halves.
        for (int i = 0; i < half; ++i) {
            const std::size_t d0 = base + static_cast<std::size_t>(2 * i) * cols;
            const std::size_t d1 = base + static_cast<std::size_t>(2 * i + 1) * cols;
            const std::size_t s0 = base + static_cast<std::size_t>(i) * cols;
            const std::size_t s1 = base + static_cast<std::size_t>(i + half) * cols;
            std::memcpy(&dst[d0], &src[s0],
                        static_cast<std::size_t>(cols) * sizeof(float));
            std::memcpy(&dst[d1], &src[s1],
                        static_cast<std::size_t>(cols) * sizeof(float));
        }
        // Pass-through tail [rotary_dim, head_dim): copy as is.
        for (int r = rotary_dim; r < head_dim; ++r) {
            const std::size_t off = base + static_cast<std::size_t>(r) * cols;
            std::memcpy(&dst[off], &src[off],
                        static_cast<std::size_t>(cols) * sizeof(float));
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

// Build a per-axis sin/cos table (max_pos+1, d) using base rope_theta and the
// inv_freq schedule used by Qwen3.5: freqs_i = pos / theta^(2i / rotary_dim)
// for i in [0, d). brotensor's rope_apply_mrope expects (max_pos_a, d_a)
// FP32 tables, one cos and one sin per axis, indexed by the per-row pos.
void build_axis_tables(int max_pos_inclusive, int d_axis, int rotary_dim,
                       float rope_theta, bt::Tensor& cos_t, bt::Tensor& sin_t) {
    const int rows = std::max(1, max_pos_inclusive + 1);
    if (d_axis <= 0) {
        // Degenerate axis: brotensor still accepts a (rows, 0)-ish layout via
        // an empty (rows, 1) placeholder; we pass empty Tensors. The mrope op
        // skips them when d_axis==0.
        cos_t = bt::Tensor::zeros_on(bt::default_device(), rows, 1, bt::Dtype::FP32);
        sin_t = bt::Tensor::zeros_on(bt::default_device(), rows, 1, bt::Dtype::FP32);
        return;
    }
    std::vector<float> cos_h(static_cast<std::size_t>(rows) * d_axis);
    std::vector<float> sin_h(static_cast<std::size_t>(rows) * d_axis);
    // inv_freq[i] = 1 / theta^(2i / rotary_dim)
    std::vector<float> inv_freq(static_cast<std::size_t>(d_axis));
    for (int i = 0; i < d_axis; ++i) {
        const float exp = 2.0f * static_cast<float>(i) /
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
                permute_rotary_rows(q_rows, n_q, HD, rotary_dim_, H);
            L.full.Wq = brolm::detail::upload_host(q_perm.data(), q_dim, H);
            L.full.Wg = brolm::detail::upload_host(g_rows.data(), q_dim, H);

            bt::Tensor k_raw;
            upload_compute_checked(need(shards, p + "self_attn.k_proj.weight"),
                                   kv_dim, H, k_raw, "self_attn.k_proj.weight");
            std::vector<float> k_host = download_fp32(k_raw);
            std::vector<float> k_perm =
                permute_rotary_rows(k_host, n_kv, HD, rotary_dim_, H);
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
                permute_rotary_rows(qn_host, /*num_heads=*/1, HD, rotary_dim_, 1);
            std::vector<float> kn_perm =
                permute_rotary_rows(kn_host, /*num_heads=*/1, HD, rotary_dim_, 1);
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
            auto load_any = [&](const std::string& key, bt::Tensor& dst) {
                const auto& v = need(shards, key);
                // Infer rows/cols from the view shape. 1-D and 2-D supported.
                int rows = 0, cols = 1;
                if (v.shape.size() == 1) {
                    rows = static_cast<int>(v.shape[0]);
                } else if (v.shape.size() == 2) {
                    rows = static_cast<int>(v.shape[0]);
                    cols = static_cast<int>(v.shape[1]);
                } else if (v.shape.size() == 3) {
                    // e.g. conv1d.weight (channels, in_per_group, kernel).
                    // Flatten into (channels, in_per_group*kernel) — Stage 3b
                    // will reshape as needed.
                    rows = static_cast<int>(v.shape[0]);
                    cols = static_cast<int>(v.shape[1]) *
                           static_cast<int>(v.shape[2]);
                } else {
                    fail("linear_attn tensor has unsupported rank: " + key);
                }
                upload_compute_checked(v, rows, cols, dst, key);
            };
            load_any(lp + "A_log",          L.lin.A_log);
            load_any(lp + "conv1d.weight",  L.lin.conv1d);
            load_any(lp + "dt_bias",        L.lin.dt_bias);
            load_any(lp + "in_proj_a.weight",   L.lin.in_proj_a);
            load_any(lp + "in_proj_b.weight",   L.lin.in_proj_b);
            load_any(lp + "in_proj_qkv.weight", L.lin.in_proj_qkv);
            load_any(lp + "in_proj_z.weight",   L.lin.in_proj_z);
            load_any(lp + "norm.weight",        L.lin.norm);
            load_any(lp + "out_proj.weight",    L.lin.out_proj);
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
    const int n_q = cfg_.num_attention_heads;
    const int cache_cols = n_q * cfg_.head_dim;
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

// ─── GQA head expansion ────────────────────────────────────────────────────

void TextModel::expand_kv_heads_(const bt::Tensor& src, bt::Tensor& dst) {
    const int HD   = cfg_.head_dim;
    const int n_q  = cfg_.num_attention_heads;
    const int n_kv = cfg_.num_key_value_heads;
    const int group = n_q / n_kv;
    const int L = src.rows;             // src: (L, n_kv*HD)
    const int q_dim = n_q * HD;

    brolm::detail::resize_like(dst, L, q_dim, src.dtype, src.device);
    for (int r = 0; r < L; ++r) {
        for (int h = 0; h < n_q; ++h) {
            const int kv_h = h / group;
            const int src_off = r * (n_kv * HD) + kv_h * HD;
            const int dst_off = r * q_dim + h * HD;
            bt::copy_d2d(src, src_off, dst, dst_off, HD);
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

void TextModel::apply_partial_mrope_(bt::Tensor& qk, int num_heads, int L,
                                     const std::vector<int32_t>& pos_t,
                                     const std::vector<int32_t>& pos_h,
                                     const std::vector<int32_t>& pos_w) {
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

    // Copy rotary subrange from qk -> scratch (per head, per row).
    for (int r = 0; r < L; ++r) {
        for (int h = 0; h < num_heads; ++h) {
            const int src_off = r * (num_heads * HD) + h * HD;
            const int dst_off = r * rot_cols + h * rd;
            bt::copy_d2d(qk, src_off, scratch, dst_off, rd);
        }
    }

    // Build per-axis tables for the maximum position present.
    int max_pos = 0;
    auto upd = [&](const std::vector<int32_t>& v) {
        for (int32_t p : v) if (p > max_pos) max_pos = p;
    };
    upd(pos_t); upd(pos_h); upd(pos_w);

    bt::Tensor cos_t, sin_t, cos_h_tbl, sin_h_tbl, cos_w, sin_w;
    build_axis_tables(max_pos, d_t_, rd, cfg_.rope.rope_theta, cos_t, sin_t);
    build_axis_tables(max_pos, d_h_, rd, cfg_.rope.rope_theta, cos_h_tbl, sin_h_tbl);
    build_axis_tables(max_pos, d_w_, rd, cfg_.rope.rope_theta, cos_w, sin_w);

    // brotensor's mrope op accepts host pointers on CPU, device on CUDA/Metal.
    // We stage on host then optionally migrate. For simplicity match the
    // qwen.cpp convention: build device-resident int32 buffers via make_idx_device.
    bt::Tensor pos_t_dev = make_idx_device(pos_t.data(), L);
    bt::Tensor pos_h_dev = make_idx_device(pos_h.data(), L);
    bt::Tensor pos_w_dev = make_idx_device(pos_w.data(), L);

    bt::rope_apply_mrope(
        scratch,
        cos_t, sin_t,
        cos_h_tbl, sin_h_tbl,
        cos_w, sin_w,
        static_cast<const int32_t*>(pos_t_dev.data),
        static_cast<const int32_t*>(pos_h_dev.data),
        static_cast<const int32_t*>(pos_w_dev.data),
        /*head_dim=*/rd,
        /*num_heads=*/num_heads,
        /*d_t=*/d_t_, /*d_h=*/d_h_, /*d_w=*/d_w_,
        scratch);

    // Copy rotated subrange back into qk.
    for (int r = 0; r < L; ++r) {
        for (int h = 0; h < num_heads; ++h) {
            const int dst_off = r * (num_heads * HD) + h * HD;
            const int src_off = r * rot_cols + h * rd;
            bt::copy_d2d(scratch, src_off, qk, dst_off, rd);
        }
    }
}

// ─── MLP block ─────────────────────────────────────────────────────────────

void TextModel::mlp_block_(const MLP& mlp, int L) {
    const int Fm = cfg_.intermediate_size;
    detail::linear_batched(mlp.gate_W, /*bias=*/nullptr, norm_, mlp_gate_);
    detail::linear_batched(mlp.up_W,   /*bias=*/nullptr, norm_, mlp_up_);

    brolm::detail::resize_like(swiglu_in_, L, 2 * Fm, mlp_gate_.dtype,
                               mlp_gate_.device);
    for (int r = 0; r < L; ++r) {
        bt::copy_d2d(mlp_gate_, r * Fm, swiglu_in_, r * (2 * Fm), Fm);
        bt::copy_d2d(mlp_up_,   r * Fm, swiglu_in_, r * (2 * Fm) + Fm, Fm);
    }
    bt::swiglu_forward(swiglu_in_, mlp_act_);
    detail::linear_batched(mlp.down_W, /*bias=*/nullptr, mlp_act_, proj_);
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
            apply_partial_mrope_(qn_, n_q,  L, pos_t, pos_h, pos_w);
            apply_partial_mrope_(kn_, n_kv, L, pos_t, pos_h, pos_w);

            // GQA: widen k/v to n_q heads, append to cache.
            expand_kv_heads_(kn_, k_exp_);
            expand_kv_heads_(v_,  v_exp_);
            bt::kv_cache_append(k_exp_, v_exp_, kvc.len, kvc.k, kvc.v);

            // Causal attention against the populated cache.
            bt::flash_attention_decode(qn_, kvc.k, kvc.v,
                                       kvc.len + L, n_q, attn_);
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

            // 1) in_proj_qkv -> (L, qkv_ch)
            detail::linear_batched(layer.lin.in_proj_qkv, nullptr, norm_, lin_qkv_);

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
            for (int r = 0; r < L; ++r) {
                bt::copy_d2d(lin_qkv_conv_, r * qkv_ch + 0 * kdim, lin_q_, r * kdim, kdim);
                bt::copy_d2d(lin_qkv_conv_, r * qkv_ch + 1 * kdim, lin_k_, r * kdim, kdim);
                bt::copy_d2d(lin_qkv_conv_, r * qkv_ch + 2 * kdim, lin_v_, r * vdim, vdim);
            }

            // 4) z = in_proj_z @ x  (T, H*D_v)
            detail::linear_batched(layer.lin.in_proj_z, nullptr, norm_, lin_z_);

            // 5) a_raw = in_proj_a @ x + dt_bias    (T, H), FP32 required by the
            //    recurrence op. We compute at compute_dtype then convert to FP32.
            bt::Tensor a_tmp, b_tmp;
            detail::linear_batched(layer.lin.in_proj_a, nullptr, norm_, a_tmp);
            detail::linear_batched(layer.lin.in_proj_b, nullptr, norm_, b_tmp);
            std::vector<float> a_host = download_fp32(a_tmp);
            std::vector<float> b_host = download_fp32(b_tmp);
            std::vector<float> dt_host = download_fp32(layer.lin.dt_bias);
            if (static_cast<int>(dt_host.size()) != lin_h) {
                fail("forward: dt_bias size mismatch");
            }
            // Add dt_bias per row (the bias is broadcast across the T axis;
            // brotensor's op does softplus internally so we hand it the SUM).
            for (int r = 0; r < L; ++r) {
                for (int h = 0; h < lin_h; ++h) {
                    a_host[static_cast<std::size_t>(r) * lin_h + h] +=
                        dt_host[static_cast<std::size_t>(h)];
                }
            }
            lin_a_raw_ = bt::Tensor::from_host_on(bt::default_device(),
                                                  a_host.data(), L, lin_h);
            lin_beta_  = bt::Tensor::from_host_on(bt::default_device(),
                                                  b_host.data(), L, lin_h);

            // 6) log_A as (num_heads, 1) FP32 (forces conversion if loaded
            //    at FP16 on a GPU build; on CPU it's already FP32).
            if (layer.lin.A_log.dtype != bt::Dtype::FP32) {
                std::vector<float> la = download_fp32(layer.lin.A_log);
                lin_log_A_ = bt::Tensor::from_host_on(bt::default_device(),
                                                     la.data(), lin_h, 1);
            } else {
                lin_log_A_ = layer.lin.A_log;
                lin_log_A_.rows = lin_h;
                lin_log_A_.cols = 1;
            }

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

            // 10) out_proj back to hidden, residual add.
            detail::linear_batched(layer.lin.out_proj, nullptr, lin_O_norm_, proj_);
            bt::add_inplace(h_, proj_);
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
