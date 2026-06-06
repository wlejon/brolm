#pragma once

// Shared dense (non-hybrid) transformer decoder core.
//
// Powers brolm's LLaMA-family causal decoders — Qwen3 and Mistral 3 today.
// Both are the same pre-norm, GQA, SwiGLU, plain-1-D-RoPE causal decoder; the
// one architectural axis that varies is per-head QK-norm (Qwen3 applies a
// per-head RMSNorm to q/k, Mistral does not). DenseDecoderConfig::use_qk_norm
// selects it. Everything else — GQA head expansion, the HF rotate_half →
// interleaved-pair RoPE weight permute, SwiGLU MLP, RMSNorm, the KV cache, and
// the tie/untie lm_head rule — is identical across both families.
//
// This is an internal primitive: the public model classes (brolm::qwen::
// Qwen3Model, brolm::mistral::MistralModel) own a DenseDecoder and supply only
// the things that genuinely differ between checkpoints — the typed config, the
// on-disk container (safetensors vs gguf, via brolm::detail::weights::Source),
// and any gguf tensor-name map. Forward-only, KV-cached. Runs on whichever
// backend brotensor resolves at runtime — FP32 on CPU, FP16 on a GPU backend.
//
// Architecture (one decoder layer, residual stream `h` of width hidden_size):
//
//   residual = h
//   h = rms_norm(h, input_layernorm.weight, eps)
//     q = q_proj(h)                                  (L, n_q  * head_dim), no bias
//     k = k_proj(h)                                  (L, n_kv * head_dim), no bias
//     v = v_proj(h)                                  (L, n_kv * head_dim), no bias
//     if use_qk_norm:                                Qwen3 only
//       q = per_head_rmsnorm(q, q_norm.weight, head_dim)
//       k = per_head_rmsnorm(k, k_norm.weight, head_dim)
//     q = rope(q, head_dim, n_q,  seq_offset, rope_theta)
//     k = rope(k, head_dim, n_kv, seq_offset, rope_theta)
//     k_exp = expand_kv_heads(k, n_kv -> n_q)        GQA head expansion
//     v_exp = expand_kv_heads(v, n_kv -> n_q)
//     kv_cache_append(k_exp, v_exp, cache_len, K_cache, V_cache)
//     attn = flash_attention_decode(q, K_cache, V_cache, cache_len + L, n_q)
//     attn = o_proj(attn)                            (L, hidden), no bias
//   h = residual + attn
//   residual = h
//   h = rms_norm(h, post_attention_layernorm.weight, eps)
//     gate = gate_proj(h);  up = up_proj(h)
//     h_mlp = down_proj( silu(gate) * up )           SwiGLU
//   h = residual + h_mlp
//
// After all layers: h = rms_norm(h, model.norm.weight, eps); logits = lm_head(h).
// There are no biases on any projection.
//
// RoPE convention: brotensor's rope_forward rotates adjacent pairs (GPT-J /
// interleaved). HF LLaMA-family models use rotate_half (dim i pairs with
// i + head_dim/2). The weight Source permutes the per-head head_dim ordering of
// q_proj / k_proj (and, when present, q_norm / k_norm) so projected q/k land in
// interleaved-pair order — see brolm::detail::weights::Source.

#include "brotensor/tensor.h"

#include <cstdint>
#include <vector>

namespace brolm::detail::weights { class Source; }

namespace brolm::detail {

// Architectural dimensions of a dense causal decoder. Every public model class
// translates its typed HF config into this struct.
struct DenseDecoderConfig {
    int   vocab_size          = 0;
    int   hidden_size         = 0;
    int   intermediate_size   = 0;
    int   num_hidden_layers   = 0;
    int   num_attention_heads = 0;   // query heads
    int   num_key_value_heads = 0;   // KV heads (GQA)
    int   head_dim            = 0;   // independent of hidden_size/num_heads
    float rms_norm_eps        = 1e-6f;
    float rope_theta          = 1000000.0f;
    bool  use_qk_norm         = false;  // Qwen3: true; Mistral 3: false
    bool  tie_word_embeddings = true;
};

class DenseDecoder {
public:
    explicit DenseDecoder(const DenseDecoderConfig& cfg);
    ~DenseDecoder();

    DenseDecoder(const DenseDecoder&)            = delete;
    DenseDecoder& operator=(const DenseDecoder&) = delete;
    DenseDecoder(DenseDecoder&&) noexcept            = default;
    DenseDecoder& operator=(DenseDecoder&&) noexcept = default;

    // Load every weight through the container-agnostic Source. Names follow the
    // HF convention (`model.embed_tokens.weight`,
    // `model.layers.N.self_attn.q_proj.weight`, ...); the Source applies any
    // prefix and gguf name map. q_norm / k_norm are read only when
    // cfg.use_qk_norm. lm_head.weight is loaded when the source has it,
    // otherwise tied to the embedding matrix (requires tie_word_embeddings).
    // Throws std::runtime_error on a missing name or shape mismatch.
    void load_weights(const brolm::detail::weights::Source& src);

    // Allocate the per-layer K/V cache for sequences up to `max_seq_len`
    // tokens. Sized once; resets cache_len to 0.
    void allocate_cache(int max_seq_len);

    // Reset the cache length to 0, keeping the allocation.
    void reset_cache();

    int cache_len() const { return cache_len_; }

    // Append L tokens at absolute positions [cache_len, cache_len + L), run the
    // decoder, and write `logits_out` := (L, vocab_size) at the compute dtype.
    // Advances cache_len by L. brotensor::init() must have been called once
    // before any forward.
    void forward(const int32_t* ids, int L, brotensor::Tensor& logits_out);

    // Embedding lookup only: write `out` := (L, hidden_size) input embeddings
    // for `ids` at the compute dtype. Does NOT touch the KV cache. Used by
    // multimodal callers that need to splice non-text embeddings (e.g. image
    // tokens) into the input stream before running forward_embeds.
    void embed_tokens(const int32_t* ids, int L, brotensor::Tensor& out);

    // Like forward(), but starts from precomputed input embeddings rather than
    // token ids — `embeds` must be (L, hidden_size) at the compute dtype (e.g.
    // the output of embed_tokens with selected rows overwritten by image
    // embeddings). Appends at [cache_len, cache_len + L), writes
    // `logits_out` := (L, vocab_size), and advances cache_len by L.
    void forward_embeds(const brotensor::Tensor& embeds, int L,
                        brotensor::Tensor& logits_out);

    const DenseDecoderConfig& config() const { return cfg_; }

private:
    struct Layer {
        brotensor::Tensor input_ln;     // (hidden,)
        brotensor::Tensor Wq, Wk, Wv;   // (n_q*hd,h) (n_kv*hd,h) (n_kv*hd,h)
        brotensor::Tensor Wo;           // (hidden, n_q*hd)
        brotensor::Tensor q_norm, k_norm;          // (head_dim,) — QK-norm only
        brotensor::Tensor post_attn_ln;            // (hidden,)
        brotensor::Tensor gate_W, up_W, down_W;    // MLP projections
        // Per-layer KV cache, expanded to n_q heads: (max_seq, n_q*head_dim).
        brotensor::Tensor K_cache, V_cache;
    };

    // GQA head expansion: copy each KV head's head_dim block out to the group
    // of query heads it serves. `src` is (L, n_kv*head_dim); `dst` becomes
    // (L, n_q*head_dim).
    void expand_kv_heads_(const brotensor::Tensor& src, brotensor::Tensor& dst);

    // Run all decoder layers + final norm + lm_head over the residual stream
    // already populated in h_ (L rows at the compute dtype), writing
    // `logits_out` := (L, vocab_size) and advancing cache_len by L. Shared by
    // forward() (h_ from embedding lookup) and forward_embeds() (h_ supplied).
    void run_layers_(int L, brotensor::Tensor& logits_out);

    DenseDecoderConfig cfg_;

    // Weights.
    brotensor::Tensor embed_tokens_;   // (vocab, hidden)
    std::vector<Layer> layers_;
    brotensor::Tensor final_norm_;     // (hidden,)
    brotensor::Tensor lm_head_;        // (vocab, hidden) — aliases embed when tied

    // Cache state.
    int cache_len_   = 0;
    int max_seq_len_ = 0;

    // Per-call scratch — kept alive across forward() calls to avoid realloc.
    brotensor::Tensor ids_dev_;        // device INT32 token buffer
    brotensor::Tensor h_;              // residual stream (L, hidden)
    brotensor::Tensor norm_;           // rms_norm output
    brotensor::Tensor q_, k_, v_;      // projected q/k/v
    brotensor::Tensor qn_, kn_;        // QK-normed q/k (distinct from q_/k_)
    brotensor::Tensor k_exp_, v_exp_;  // GQA-expanded k/v
    brotensor::Tensor attn_;           // flash-attention output
    brotensor::Tensor proj_;           // o_proj / down_proj output
    brotensor::Tensor gate_, up_;      // MLP gate / up
    brotensor::Tensor swiglu_in_;      // concat(gate, up) for swiglu_forward
    brotensor::Tensor mlp_act_;        // swiglu output
};

}  // namespace brolm::detail
