#include "brolm/qwen.h"

#include "brotensor/safetensors.h"
#include "brolm/detail/compute.h"
#include "brolm/detail/device.h"

#include "brotensor/ops.h"
#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace brolm::qwen {

namespace bt = ::brotensor;
namespace st = ::brotensor::safetensors;

using st::upload_compute_checked;

// ─── helpers ───────────────────────────────────────────────────────────────

namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("qwen::Qwen3Model: " + msg);
}

// Find a tensor by name across one or more shards; first match wins.
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

// Build a device-resident INT32 buffer holding `n` token ids. brotensor has
// no from_host path for INT32, so stage on the host then migrate to the
// default device.
bt::Tensor make_idx_device(const int32_t* host, int n) {
    bt::Tensor cpu = bt::Tensor::empty_on(bt::Device::CPU, n, 1, bt::Dtype::INT32);
    std::memcpy(cpu.host_raw_mut(), host,
                static_cast<std::size_t>(n) * sizeof(int32_t));
    return cpu.to(bt::default_device());
}

// Download a loaded weight tensor into a host FP32 vector, handling both the
// FP16 (GPU) and FP32 (CPU) compute-dtype cases.
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

}  // namespace

// ─── RoPE weight permutation ───────────────────────────────────────────────
//
// brotensor's rope_forward rotates ADJACENT pairs (x_{2i}, x_{2i+1}) — the
// GPT-J / interleaved convention. Hugging Face Qwen3 is trained with the
// rotate_half convention, where head_dim index i pairs with i + head_dim/2.
// The two are NOT equivalent.
//
// To run brotensor's fast rope_forward on real HF weights we permute the
// per-head head_dim ordering of q_proj/k_proj (their OUTPUT rows) and of
// q_norm/k_norm (length head_dim) at load time, so that after projection +
// QK-norm the head_dim already lands in interleaved-pair order:
//
//   interleaved index 2i   <- HF head-dim index i
//   interleaved index 2i+1 <- HF head-dim index i + hd/2     for i in [0, hd/2)
//
// i.e. perm[2i] = i, perm[2i+1] = i + hd/2: row r of the permuted weight is
// row `perm[r]` of the HF weight, applied independently within each head's
// head_dim block.
//
// V is not roped, so v_proj is left alone. o_proj is left alone too: permuting
// q and k consistently leaves Q·Kᵀ invariant (the per-head dot product sums
// over head_dim and is permutation-invariant), and the attention output is the
// same consistent permutation of V which... is NOT permuted — so attention
// output stays in HF head order and o_proj sees HF-order activations. This is
// the standard HF→interleaved trick (llama.cpp does the same).

namespace {

// Permute `head_dim` consecutive rows within each of `num_heads` blocks of an
// (num_heads*head_dim, cols) host buffer, mapping interleaved -> HF order.
std::vector<float> permute_rope_rows(const std::vector<float>& src,
                                     int num_heads, int head_dim, int cols) {
    const int half = head_dim / 2;
    std::vector<float> dst(src.size());
    for (int h = 0; h < num_heads; ++h) {
        const std::size_t base =
            static_cast<std::size_t>(h) * head_dim *
            static_cast<std::size_t>(cols);
        for (int i = 0; i < half; ++i) {
            // dst rows 2i, 2i+1  <-  src rows i, i+half
            const std::size_t d0 = base + static_cast<std::size_t>(2 * i) * cols;
            const std::size_t d1 = base + static_cast<std::size_t>(2 * i + 1) * cols;
            const std::size_t s0 = base + static_cast<std::size_t>(i) * cols;
            const std::size_t s1 = base + static_cast<std::size_t>(i + half) * cols;
            std::memcpy(&dst[d0], &src[s0],
                        static_cast<std::size_t>(cols) * sizeof(float));
            std::memcpy(&dst[d1], &src[s1],
                        static_cast<std::size_t>(cols) * sizeof(float));
        }
    }
    return dst;
}

}  // namespace

// ─── ctor / dtor ───────────────────────────────────────────────────────────

Qwen3Model::Qwen3Model(const Qwen3Config& cfg) : cfg_(cfg) {
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
        fail("head_dim must be even (RoPE rotates dimension pairs)");
    }
    layers_.resize(static_cast<std::size_t>(cfg_.num_hidden_layers));
}

Qwen3Model::~Qwen3Model() = default;

// ─── load_weights ──────────────────────────────────────────────────────────

void Qwen3Model::load_weights(const st::File& f, const std::string& prefix) {
    const std::vector<const st::File*> shards = {&f};
    load_weights_impl_(shards, prefix);
}

void Qwen3Model::load_weights(const std::vector<const st::File*>& shards,
                              const std::string& prefix) {
    load_weights_impl_(shards, prefix);
}

void Qwen3Model::load_weights_impl_(
    const std::vector<const st::File*>& shards, const std::string& prefix) {
    if (shards.empty()) fail("load_weights: no safetensors shards");

    const int V    = cfg_.vocab_size;
    const int H    = cfg_.hidden_size;
    const int F    = cfg_.intermediate_size;
    const int HD   = cfg_.head_dim;
    const int n_q  = cfg_.num_attention_heads;
    const int n_kv = cfg_.num_key_value_heads;
    const int q_dim  = n_q  * HD;
    const int kv_dim = n_kv * HD;

    upload_compute_checked(need(shards, prefix + "model.embed_tokens.weight"),
                           V, H, embed_tokens_, "embed_tokens.weight");

    // Load q_proj / k_proj, then permute their per-head head_dim row ordering
    // from HF rotate_half order into brotensor's interleaved-pair order.
    auto load_rope_proj = [&](const std::string& key, int out_dim,
                              int num_heads, bt::Tensor& dst,
                              const std::string& name) {
        bt::Tensor raw;
        upload_compute_checked(need(shards, key), out_dim, H, raw, name);
        std::vector<float> host = download_fp32(raw);
        std::vector<float> perm = permute_rope_rows(host, num_heads, HD, H);
        dst = brolm::detail::upload_host(perm.data(), out_dim, H);
    };

    // Load q_norm / k_norm (length head_dim) with the same interleaved permute.
    auto load_rope_norm = [&](const std::string& key, bt::Tensor& dst,
                              const std::string& name) {
        bt::Tensor raw;
        upload_compute_checked(need(shards, key), HD, 1, raw, name);
        std::vector<float> host = download_fp32(raw);
        std::vector<float> perm = permute_rope_rows(host, /*num_heads=*/1, HD, 1);
        dst = brolm::detail::upload_host(perm.data(), HD, 1);
    };

    for (int i = 0; i < cfg_.num_hidden_layers; ++i) {
        const std::string p =
            prefix + "model.layers." + std::to_string(i) + ".";
        Layer& L = layers_[static_cast<std::size_t>(i)];

        upload_compute_checked(need(shards, p + "input_layernorm.weight"),
                               H, 1, L.input_ln, "input_layernorm.weight");

        load_rope_proj(p + "self_attn.q_proj.weight", q_dim, n_q,
                       L.Wq, "q_proj.weight");
        load_rope_proj(p + "self_attn.k_proj.weight", kv_dim, n_kv,
                       L.Wk, "k_proj.weight");
        upload_compute_checked(need(shards, p + "self_attn.v_proj.weight"),
                               kv_dim, H, L.Wv, "v_proj.weight");
        upload_compute_checked(need(shards, p + "self_attn.o_proj.weight"),
                               H, q_dim, L.Wo, "o_proj.weight");

        load_rope_norm(p + "self_attn.q_norm.weight", L.q_norm, "q_norm.weight");
        load_rope_norm(p + "self_attn.k_norm.weight", L.k_norm, "k_norm.weight");

        upload_compute_checked(
            need(shards, p + "post_attention_layernorm.weight"),
            H, 1, L.post_attn_ln, "post_attention_layernorm.weight");

        upload_compute_checked(need(shards, p + "mlp.gate_proj.weight"),
                               F, H, L.gate_W, "gate_proj.weight");
        upload_compute_checked(need(shards, p + "mlp.up_proj.weight"),
                               F, H, L.up_W, "up_proj.weight");
        upload_compute_checked(need(shards, p + "mlp.down_proj.weight"),
                               H, F, L.down_W, "down_proj.weight");
    }

    upload_compute_checked(need(shards, prefix + "model.norm.weight"),
                           H, 1, final_norm_, "model.norm.weight");

    // lm_head: absent iff weights are tied to embed_tokens.
    const auto* lm = find_in(shards, prefix + "lm_head.weight");
    if (lm != nullptr) {
        upload_compute_checked(*lm, V, H, lm_head_, "lm_head.weight");
    } else {
        if (!cfg_.tie_word_embeddings) {
            fail("load_weights: lm_head.weight missing and "
                 "tie_word_embeddings is false");
        }
        // Tied: lm_head shares the embedding matrix. Clone so the two tensors
        // are independent storage (cheap; avoids aliasing surprises).
        lm_head_ = embed_tokens_.clone();
    }
}

// ─── cache management ──────────────────────────────────────────────────────

void Qwen3Model::allocate_cache(int max_seq_len) {
    if (max_seq_len <= 0) fail("allocate_cache: max_seq_len must be positive");
    const int n_q = cfg_.num_attention_heads;
    const int cache_cols = n_q * cfg_.head_dim;
    const bt::Dtype dt = brolm::compute_dtype();
    const bt::Device dev = bt::default_device();
    for (Layer& L : layers_) {
        // KV cache stores K/V already expanded to n_q heads (see GQA notes).
        brolm::detail::resize_like(L.K_cache, max_seq_len, cache_cols, dt, dev);
        brolm::detail::resize_like(L.V_cache, max_seq_len, cache_cols, dt, dev);
    }
    max_seq_len_ = max_seq_len;
    cache_len_   = 0;
}

void Qwen3Model::reset_cache() { cache_len_ = 0; }

// ─── GQA head expansion ────────────────────────────────────────────────────
//
// flash_attention_decode takes a single num_heads and requires K/V cache cols
// to equal Q cols. Qwen3 has n_kv < n_q, so K and V must be widened from n_kv
// heads to n_q heads before they are appended to the cache: with
// group = n_q / n_kv, query head h reads KV head h / group.
//
// Implemented as a per-(row, query-head) copy_d2d. This is a non-optimal path
// — correctness over speed; a fused gather kernel would be the optimisation.

void Qwen3Model::expand_kv_heads_(const bt::Tensor& src, bt::Tensor& dst) {
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

// ─── forward ───────────────────────────────────────────────────────────────

void Qwen3Model::forward(const int32_t* ids, int L, bt::Tensor& logits_out) {
    if (!ids) fail("forward: ids pointer is null");
    if (L <= 0) fail("forward: L must be positive");
    if (embed_tokens_.size() == 0) fail("forward: weights not loaded");
    if (max_seq_len_ == 0) fail("forward: cache not allocated");
    if (cache_len_ + L > max_seq_len_) {
        fail("forward: cache_len + L exceeds allocated cache capacity");
    }

    const int HD   = cfg_.head_dim;
    const int n_q  = cfg_.num_attention_heads;
    const int n_kv = cfg_.num_key_value_heads;
    const float eps = cfg_.rms_norm_eps;
    // seq_offset for RoPE = absolute position of the first new token, which is
    // the cache length BEFORE this forward.
    const int seq_offset = cache_len_;

    // Per-head RMSNorm: view (L, num_heads*HD) as (L*num_heads, HD), run
    // rms_norm with a (HD,1) gain shared across heads into `dst`, then reshape
    // `dst` back to (L, num_heads*HD). `src` and `dst` are distinct buffers.
    // Mirrors brodiffusion/src/dit/flux.cpp headnorm.
    auto headnorm = [&](const bt::Tensor& src, int num_heads,
                        const bt::Tensor& gain, bt::Tensor& dst) -> void {
        const int rows = src.rows;
        bt::Tensor src_v = bt::Tensor::view(
            src.device, src.data, rows * num_heads, HD, src.dtype);
        bt::rms_norm_forward(src_v, gain, eps, dst);
        // rms_norm_forward sized dst to (rows*num_heads, HD); the storage is
        // identical to (rows, num_heads*HD) row-major. Reshape in place.
        dst.rows = rows;
        dst.cols = num_heads * HD;
    };

    // Embedding lookup -> own a stable residual stream in h_.
    ids_dev_ = make_idx_device(ids, L);
    bt::embedding_lookup_forward(
        embed_tokens_, static_cast<const int32_t*>(ids_dev_.data), L, h_);
    h_ = h_.clone();

    for (Layer& layer : layers_) {
        // ── self-attention sub-layer ──────────────────────────────────────
        bt::rms_norm_forward(h_, layer.input_ln, eps, norm_);

        detail::linear_batched(layer.Wq, /*bias=*/nullptr, norm_, q_);
        detail::linear_batched(layer.Wk, /*bias=*/nullptr, norm_, k_);
        detail::linear_batched(layer.Wv, /*bias=*/nullptr, norm_, v_);

        // QK-norm: per-head RMSNorm over head_dim, into distinct buffers.
        headnorm(q_, n_q,  layer.q_norm, qn_);
        headnorm(k_, n_kv, layer.k_norm, kn_);

        // RoPE on the QK-normed q/k. Weights were permuted at load so this
        // interleaved-pair rotation reproduces HF's rotate_half.
        bt::rope_forward(qn_, HD, n_q,  seq_offset, cfg_.rope_theta, qn_);
        bt::rope_forward(kn_, HD, n_kv, seq_offset, cfg_.rope_theta, kn_);

        // GQA: widen k/v from n_kv heads to n_q heads, then append the
        // n_q-head-width K/V to the cache.
        expand_kv_heads_(kn_, k_exp_);
        expand_kv_heads_(v_, v_exp_);
        bt::kv_cache_append(k_exp_, v_exp_, cache_len_,
                            layer.K_cache, layer.V_cache);

        // Causal flash-attention against the populated cache. valid_len is the
        // total cache length after the append; the op applies causal masking
        // and the 1/sqrt(head_dim) scale internally.
        bt::flash_attention_decode(qn_, layer.K_cache, layer.V_cache,
                                   cache_len_ + L, n_q, attn_);

        detail::linear_batched(layer.Wo, /*bias=*/nullptr, attn_, proj_);
        bt::add_inplace(h_, proj_);

        // ── MLP sub-layer (SwiGLU) ────────────────────────────────────────
        bt::rms_norm_forward(h_, layer.post_attn_ln, eps, norm_);

        detail::linear_batched(layer.gate_W, /*bias=*/nullptr, norm_, gate_);
        detail::linear_batched(layer.up_W,   /*bias=*/nullptr, norm_, up_);

        // swiglu_forward expects (L, 2*F) = concat(gate, up) along columns and
        // computes silu(gate) * up.
        const int F = cfg_.intermediate_size;
        brolm::detail::resize_like(swiglu_in_, L, 2 * F, gate_.dtype,
                                   gate_.device);
        for (int r = 0; r < L; ++r) {
            bt::copy_d2d(gate_, r * F, swiglu_in_, r * (2 * F), F);
            bt::copy_d2d(up_,   r * F, swiglu_in_, r * (2 * F) + F, F);
        }
        bt::swiglu_forward(swiglu_in_, mlp_act_);
        detail::linear_batched(layer.down_W, /*bias=*/nullptr, mlp_act_, proj_);
        bt::add_inplace(h_, proj_);
    }

    // Final norm + LM head.
    bt::rms_norm_forward(h_, final_norm_, eps, norm_);
    detail::linear_batched(lm_head_, /*bias=*/nullptr, norm_, logits_out);

    cache_len_ += L;
}

}  // namespace brolm::qwen
