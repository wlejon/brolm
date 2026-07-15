#pragma once

// Shared dense (non-hybrid) transformer decoder core.
//
// Powers brolm's LLaMA-family causal decoders — Qwen3 and Mistral 3 today.
// Both are the same pre-norm, GQA, SwiGLU, plain-1-D-RoPE causal decoder; the
// one architectural axis that varies is per-head QK-norm (Qwen3 applies a
// per-head RMSNorm to q/k, Mistral does not). DenseDecoderConfig::use_qk_norm
// selects it. Everything else — GQA (handled natively by
// flash_attention_decode), the HF rotate_half → interleaved-pair RoPE weight
// permute, SwiGLU MLP, RMSNorm, the KV cache, and the tie/untie lm_head rule —
// is identical across both families.
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
//     kv_cache_append(k, v, cache_len, K_cache, V_cache)   n_kv-width cache
//     attn = flash_attention_decode(q, K_cache, V_cache, cache_len + L, n_q, n_kv)
//                                                    GQA mapping inside the op
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
#include <memory>
#include <vector>

namespace brolm::detail::weights { class Source; }

namespace brolm::detail {

// CUDA-graph decode session (defined in dense_decoder.cpp). Holds the
// persistent fixed-shape buffers + captured graph for the single-token
// forward_last step; absent / inert on CPU builds.
struct DecodeGraphSession;

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
    // Defaulted in dense_decoder.cpp — DecodeGraphSession is incomplete here.
    DenseDecoder(DenseDecoder&&) noexcept;
    DenseDecoder& operator=(DenseDecoder&&) noexcept;

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

    // Roll the cache length back to `len` (0 <= len <= cache_len). The next
    // forward then appends at [len, len + L), overwriting any rows previously
    // written past `len` — no buffer invalidation is needed because those rows
    // are dead until rewritten, and RoPE positions restart from `len`. This is
    // the speculative-draft seam: draft K tokens, inspect, truncate_cache back,
    // and continue from `len` reproduces the never-drafted continuation exactly.
    // Throws if len is negative or exceeds the current cache_len.
    void truncate_cache(int len);

    int cache_len() const { return cache_len_; }

    // Append L tokens at absolute positions [cache_len, cache_len + L), run the
    // decoder, and write `logits_out` := (L, vocab_size) at the compute dtype.
    // Advances cache_len by L. brotensor::init() must have been called once
    // before any forward. When `hidden_out` is non-null it receives the
    // (L, hidden_size) post-final-norm hidden states (the rows lm_head consumes)
    // — the control/bridge readout needs the last row alongside the logits.
    void forward(const int32_t* ids, int L, brotensor::Tensor& logits_out,
                 brotensor::Tensor* hidden_out = nullptr);

    // Like forward(), but `logits_out` := (1, vocab_size) — logits for the
    // LAST of the L appended tokens only. The KV cache still ingests all L
    // tokens, so generation continues exactly as after forward(); only the
    // lm_head matmul over the L-1 intermediate rows (which a sampler never
    // reads) is skipped. This is the prefill fast path of the generate loop:
    // at large L the full-row lm_head costs more than the rest of the layer
    // stack combined (vocab >> hidden).
    void forward_last(const int32_t* ids, int L, brotensor::Tensor& logits_out);

    // Embedding lookup only: write `out` := (L, hidden_size) input embeddings
    // for `ids` at the compute dtype. Does NOT touch the KV cache. Used by
    // multimodal callers that need to splice non-text embeddings (e.g. image
    // tokens) into the input stream before running forward_embeds.
    void embed_tokens(const int32_t* ids, int L, brotensor::Tensor& out);

    // Like forward(), but starts from precomputed input embeddings rather than
    // token ids — `embeds` must be (L, hidden_size) at the compute dtype (e.g.
    // the output of embed_tokens with selected rows overwritten by image
    // embeddings). Appends at [cache_len, cache_len + L), writes
    // `logits_out` := (L, vocab_size), and advances cache_len by L. When
    // `hidden_out` is non-null it receives the (L, hidden_size) post-final-norm
    // hidden states, exactly as forward() above.
    void forward_embeds(const brotensor::Tensor& embeds, int L,
                        brotensor::Tensor& logits_out,
                        brotensor::Tensor* hidden_out = nullptr);

    // Bidirectional, cacheless prefill for ENCODER use. LLM2Vec turns this
    // LLaMA-family decoder into a text encoder by dropping the causal mask;
    // forward_encode embeds the L tokens, runs every layer with FULL grouped-
    // query attention (flash_attention_gqa_forward, causal=false, no KV cache,
    // RoPE positions from 0), and writes `hidden_out` := (L, hidden_size)
    // post-final-norm hidden states — the contextualised per-token embeddings a
    // pooling stage reduces to a sentence vector. lm_head is never run and
    // allocate_cache is NOT required; the KV cache is untouched. Reuses the same
    // RoPE / RMSNorm / SwiGLU / projection ops as forward() — only the attention
    // call and the absence of the cache differ. Encodes one sequence at a time
    // (no padding mask); batching/padding is a caller concern.
    void forward_encode(const int32_t* ids, int L,
                        brotensor::Tensor& hidden_out);

    const DenseDecoderConfig& config() const { return cfg_; }

private:
    struct Layer {
        brotensor::Tensor input_ln;     // (hidden,)
        brotensor::Tensor Wq, Wk, Wv;   // (n_q*hd,h) (n_kv*hd,h) (n_kv*hd,h)
        brotensor::Tensor Wo;           // (hidden, n_q*hd)
        brotensor::Tensor q_norm, k_norm;          // (head_dim,) — QK-norm only
        brotensor::Tensor post_attn_ln;            // (hidden,)
        brotensor::Tensor gate_W, up_W, down_W;    // MLP projections
        // Per-layer KV cache at true KV width: (max_seq, n_kv*head_dim).
        brotensor::Tensor K_cache, V_cache;
    };

    // Run all decoder layers + final norm + lm_head over the residual stream
    // already populated in h_ (L rows at the compute dtype), writing
    // `logits_out` := (L, vocab_size) and advancing cache_len by L. Shared by
    // forward() (h_ from embedding lookup) and forward_embeds() (h_ supplied).
    // When `hidden_out` is non-null it receives a clone of the (L, hidden_size)
    // post-final-norm hidden states (the lm_head input).
    // When `logits_last_row_only`, lm_head consumes only the final row of the
    // post-final-norm hidden states and logits_out is (1, vocab_size).
    void run_layers_(int L, brotensor::Tensor& logits_out,
                     brotensor::Tensor* hidden_out,
                     bool logits_last_row_only = false);

    // Bidirectional (non-causal), cacheless layer stack for forward_encode.
    // Mirrors run_layers_'s per-layer math but attends all L keys via
    // flash_attention_gqa_forward (causal=false) with no KV cache, and writes
    // the (L, hidden_size) post-final-norm hidden states to `hidden_out` (no
    // lm_head). h_ holds the L-row residual stream on entry.
    void run_layers_encode_(int L, brotensor::Tensor& hidden_out);

    // Graph-captured single-token decode (CUDA only). try_graph_step_ runs
    // one forward_last(ids, 1, ...) step through the captured session and
    // returns true, or returns false (CPU backend, BROLM_NO_GRAPH,
    // BROLM_PROFILE, or capture unavailable) so the caller falls back to the
    // eager path. invalidate_graph_ drops the session whenever weights or
    // cache storage are replaced.
    bool try_graph_step_(int32_t token, brotensor::Tensor& logits_out);
    void invalidate_graph_();

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
    brotensor::Tensor attn_;           // flash-attention output
    brotensor::Tensor proj_;           // o_proj / down_proj output
    brotensor::Tensor gate_, up_;      // MLP gate / up; gate_ becomes the
                                       // SwiGLU activation in place

    // CUDA-graph decode session; null until the first graph-eligible
    // forward_last after weights + cache are in place.
    std::unique_ptr<DecodeGraphSession> graph_;
};

}  // namespace brolm::detail
