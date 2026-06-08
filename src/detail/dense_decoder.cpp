#include "brolm/detail/dense_decoder.h"

#include "brolm/detail/compute.h"
#include "brolm/detail/device.h"
#include "brolm/detail/weights.h"

#include "brotensor/ops.h"
#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace brolm::detail {

namespace bt = ::brotensor;

namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("DenseDecoder: " + msg);
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

}  // namespace

// ─── ctor / dtor ─────────────────────────────────────────────────────────────

DenseDecoder::DenseDecoder(const DenseDecoderConfig& cfg) : cfg_(cfg) {
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

DenseDecoder::~DenseDecoder() = default;

// ─── load_weights ────────────────────────────────────────────────────────────

void DenseDecoder::load_weights(const brolm::detail::weights::Source& src) {
    const int V    = cfg_.vocab_size;
    const int H    = cfg_.hidden_size;
    const int F    = cfg_.intermediate_size;
    const int HD   = cfg_.head_dim;
    const int n_q  = cfg_.num_attention_heads;
    const int n_kv = cfg_.num_key_value_heads;
    const int q_dim  = n_q  * HD;
    const int kv_dim = n_kv * HD;

    src.upload_compute_dequant("model.embed_tokens.weight",
                               V, H, embed_tokens_, "embed_tokens.weight");

    for (int i = 0; i < cfg_.num_hidden_layers; ++i) {
        const std::string p = "model.layers." + std::to_string(i) + ".";
        Layer& L = layers_[static_cast<std::size_t>(i)];

        src.upload_compute_dequant(p + "input_layernorm.weight",
                                   H, 1, L.input_ln, "input_layernorm.weight");

        src.upload_compute_rope_permuted(p + "self_attn.q_proj.weight",
                                         q_dim, H, n_q, HD, L.Wq, "q_proj.weight");
        src.upload_compute_rope_permuted(p + "self_attn.k_proj.weight",
                                         kv_dim, H, n_kv, HD, L.Wk, "k_proj.weight");
        src.upload_compute_checked(p + "self_attn.v_proj.weight",
                                   kv_dim, H, L.Wv, "v_proj.weight");
        src.upload_compute_checked(p + "self_attn.o_proj.weight",
                                   H, q_dim, L.Wo, "o_proj.weight");

        if (cfg_.use_qk_norm) {
            // q_norm / k_norm are length head_dim (cols == 1); permute as if a
            // single head over head_dim entries so the rotary subrange matches
            // the rope-permuted q_proj / k_proj rows.
            src.upload_compute_rope_permuted(p + "self_attn.q_norm.weight",
                                             HD, 1, /*num_heads=*/1, HD,
                                             L.q_norm, "q_norm.weight");
            src.upload_compute_rope_permuted(p + "self_attn.k_norm.weight",
                                             HD, 1, /*num_heads=*/1, HD,
                                             L.k_norm, "k_norm.weight");
        }

        src.upload_compute_dequant(p + "post_attention_layernorm.weight",
                                   H, 1, L.post_attn_ln,
                                   "post_attention_layernorm.weight");

        src.upload_compute_checked(p + "mlp.gate_proj.weight",
                                   F, H, L.gate_W, "gate_proj.weight");
        src.upload_compute_checked(p + "mlp.up_proj.weight",
                                   F, H, L.up_W, "up_proj.weight");
        src.upload_compute_checked(p + "mlp.down_proj.weight",
                                   H, F, L.down_W, "down_proj.weight");
    }

    src.upload_compute_dequant("model.norm.weight",
                               H, 1, final_norm_, "model.norm.weight");

    if (src.has("lm_head.weight")) {
        src.upload_compute_checked("lm_head.weight",
                                   V, H, lm_head_, "lm_head.weight");
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

// ─── cache management ────────────────────────────────────────────────────────

void DenseDecoder::allocate_cache(int max_seq_len) {
    if (max_seq_len <= 0) fail("allocate_cache: max_seq_len must be positive");
    const int n_kv = cfg_.num_key_value_heads;
    const int cache_cols = n_kv * cfg_.head_dim;
    const bt::Dtype dt = brolm::compute_dtype();
    const bt::Device dev = bt::default_device();
    for (Layer& L : layers_) {
        // KV cache stores true n_kv-width K/V; flash_attention_decode does the
        // GQA head-mapping internally, so no per-head widening is needed.
        brolm::detail::resize_like(L.K_cache, max_seq_len, cache_cols, dt, dev);
        brolm::detail::resize_like(L.V_cache, max_seq_len, cache_cols, dt, dev);
    }
    max_seq_len_ = max_seq_len;
    cache_len_   = 0;
}

void DenseDecoder::reset_cache() { cache_len_ = 0; }

// ─── forward ─────────────────────────────────────────────────────────────────

void DenseDecoder::embed_tokens(const int32_t* ids, int L, bt::Tensor& out) {
    if (!ids) fail("embed_tokens: ids pointer is null");
    if (L <= 0) fail("embed_tokens: L must be positive");
    if (embed_tokens_.size() == 0) fail("embed_tokens: weights not loaded");

    bt::Tensor idx = make_idx_device(ids, L);
    bt::embedding_lookup_forward(
        embed_tokens_, static_cast<const int32_t*>(idx.data), L, out);
    out = out.clone();
}

void DenseDecoder::forward(const int32_t* ids, int L, bt::Tensor& logits_out) {
    if (!ids) fail("forward: ids pointer is null");
    if (L <= 0) fail("forward: L must be positive");
    if (embed_tokens_.size() == 0) fail("forward: weights not loaded");
    if (max_seq_len_ == 0) fail("forward: cache not allocated");
    if (cache_len_ + L > max_seq_len_) {
        fail("forward: cache_len + L exceeds allocated cache capacity");
    }

    // Embedding lookup -> own a stable residual stream in h_.
    ids_dev_ = make_idx_device(ids, L);
    bt::embedding_lookup_forward(
        embed_tokens_, static_cast<const int32_t*>(ids_dev_.data), L, h_);
    h_ = h_.clone();

    run_layers_(L, logits_out);
}

void DenseDecoder::forward_embeds(const bt::Tensor& embeds, int L,
                                  bt::Tensor& logits_out) {
    if (L <= 0) fail("forward_embeds: L must be positive");
    if (embed_tokens_.size() == 0) fail("forward_embeds: weights not loaded");
    if (max_seq_len_ == 0) fail("forward_embeds: cache not allocated");
    if (cache_len_ + L > max_seq_len_) {
        fail("forward_embeds: cache_len + L exceeds allocated cache capacity");
    }
    if (embeds.rows != L || embeds.cols != cfg_.hidden_size) {
        fail("forward_embeds: embeds must be (L, hidden_size)");
    }

    // Own a stable residual stream in h_ (the layer loop mutates it in place).
    h_ = embeds.clone();

    run_layers_(L, logits_out);
}

void DenseDecoder::run_layers_(int L, bt::Tensor& logits_out) {
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

    for (Layer& layer : layers_) {
        // ── self-attention sub-layer ──────────────────────────────────────
        bt::rms_norm_forward(h_, layer.input_ln, eps, norm_);

        detail::linear_batched(layer.Wq, /*bias=*/nullptr, norm_, q_);
        detail::linear_batched(layer.Wk, /*bias=*/nullptr, norm_, k_);
        detail::linear_batched(layer.Wv, /*bias=*/nullptr, norm_, v_);

        // QK-norm (Qwen3): per-head RMSNorm over head_dim into distinct
        // buffers. Mistral has none — q_/k_ feed RoPE directly. `q_attn`/
        // `k_attn` select the active query/key tensors for the rest of the
        // attention path.
        bt::Tensor* q_attn = &q_;
        bt::Tensor* k_attn = &k_;
        if (cfg_.use_qk_norm) {
            headnorm(q_, n_q,  layer.q_norm, qn_);
            headnorm(k_, n_kv, layer.k_norm, kn_);
            q_attn = &qn_;
            k_attn = &kn_;
        }

        // RoPE on q/k. Weights were permuted at load so this interleaved-pair
        // rotation reproduces HF's rotate_half.
        bt::rope_forward(*q_attn, HD, n_q,  seq_offset, cfg_.rope_theta, *q_attn);
        bt::rope_forward(*k_attn, HD, n_kv, seq_offset, cfg_.rope_theta, *k_attn);

        // GQA: append the n_kv-width K/V straight to the cache. No widening —
        // flash_attention_decode maps query head h to KV head h/(n_q/n_kv).
        bt::kv_cache_append(*k_attn, v_, cache_len_,
                            layer.K_cache, layer.V_cache);

        // Causal flash-attention against the populated cache. valid_len is the
        // total cache length after the append; the op applies causal masking
        // and the 1/sqrt(head_dim) scale internally.
        bt::flash_attention_decode(*q_attn, layer.K_cache, layer.V_cache,
                                   cache_len_ + L, n_q, n_kv, attn_);

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

}  // namespace brolm::detail
