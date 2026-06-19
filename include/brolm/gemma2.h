#pragma once

// Gemma-2 decoder transformer — inference-only, KV-cached.
//
// Faithful port of Hugging Face `Gemma2ForCausalLM`. Forward-only. Runs on
// whichever backend brotensor resolves at runtime — CPU (FP32) by default,
// CUDA (FP16) when available.
//
// Gemma-2 is the same pre-norm GQA / RoPE causal decoder as the LLaMA family
// (Qwen3 / Mistral 3, see detail/dense_decoder.h) with several deltas that make
// it cheaper to own a dedicated eager core than to bolt flags onto the shared
// DenseDecoder:
//
//   * Embedding scale: the residual stream is multiplied by sqrt(hidden_size)
//     after the embedding lookup (folded in at the head of the layer loop so
//     forward(), forward_last() and forward_embeds() are all consistent).
//   * RMSNorm gain is (1 + weight): the loader reads each norm weight to an
//     FP32 host buffer, adds 1.0f in FP32, and re-uploads at the compute dtype,
//     so rms_norm_forward's plain `x * gamma` reproduces HF's
//     `_norm(x) * (1 + weight)`. Applies to all four per-layer norms and the
//     final norm.
//   * Four norms per layer with post-norms applied to the SUBLAYER OUTPUT
//     before the residual add:
//         residual = h
//         x = rms_norm(h, input_layernorm);  x = attention(x)
//         x = rms_norm(x, post_attention_layernorm);    h = residual + x
//         residual = h
//         x = rms_norm(h, pre_feedforward_layernorm);   x = mlp(x)
//         x = rms_norm(x, post_feedforward_layernorm);  h = residual + x
//   * Attention alternates sliding-window (even layers, window == sliding_window)
//     and global (odd layers) masking, with tanh logit soft-capping
//     (attn_logit_softcapping) — both passed straight to
//     brotensor::flash_attention_decode.
//   * MLP is GeGLU with the gelu-tanh activation (gelu_pytorch_tanh).
//   * Final logits get a tanh soft-cap: logits = c * tanh(logits / c) with
//     c == final_logit_softcapping.
//
// RoPE convention and the HF rotate_half -> interleaved-pair weight permute are
// identical to dense_decoder.h: q_proj / k_proj rows are permuted at load by the
// weights::Source. There is no QK-norm and no projection bias in Gemma-2.
//
// The CUDA-graph single-token decode path that dense_decoder.cpp ships is NOT
// implemented here yet (eager forward only) — a follow-up.

#include "brolm/gemma2_config.h"
#include "brotensor/tensor.h"

#include <cstdint>
#include <string>
#include <vector>

namespace brotensor::safetensors { class File; }

namespace brolm::detail::weights { class Source; }

namespace brolm::gemma {

class Gemma2Model {
public:
    explicit Gemma2Model(const Gemma2Config& cfg);

    // Non-copyable; movable.
    Gemma2Model(const Gemma2Model&)            = delete;
    Gemma2Model& operator=(const Gemma2Model&) = delete;
    Gemma2Model(Gemma2Model&&) noexcept        = default;
    Gemma2Model& operator=(Gemma2Model&&) noexcept = default;

    // Load all weights from a single safetensors file under `prefix`. Tensor
    // names follow the HF convention (`model.embed_tokens.weight`,
    // `model.layers.N.self_attn.{q,k,v,o}_proj.weight`, the four
    // layernorms, `model.layers.N.mlp.{gate,up,down}_proj.weight`,
    // `model.norm.weight`). Source tensors may be F16/F32/BF16. When
    // tie_word_embeddings is true, `lm_head.weight` is expected to be absent and
    // the embedding matrix is reused. Throws on a missing name or shape
    // mismatch.
    void load_weights(const brotensor::safetensors::File& f,
                      const std::string& prefix = "");

    // Sharded overload: a tensor is resolved by scanning the shards in order,
    // first match wins.
    void load_weights(
        const std::vector<const brotensor::safetensors::File*>& shards,
        const std::string& prefix = "");

    // Allocate the per-layer K/V cache for sequences up to `max_seq_len`
    // tokens. Sized once; resets cache_len to 0.
    void allocate_cache(int max_seq_len);

    // Reset the cache length to 0, keeping the allocation.
    void reset_cache() { cache_len_ = 0; }

    // Roll the cache length back to `len` (0 <= len <= cache_len). The next
    // forward appends at [len, len + L); RoPE positions restart from `len`.
    // Throws if len is out of range.
    void truncate_cache(int len);

    int cache_len() const { return cache_len_; }

    // Append L tokens at absolute positions [cache_len, cache_len + L), run the
    // decoder, and write `logits_out` := (L, vocab_size) at the compute dtype.
    // Advances cache_len by L. When `hidden_out` is non-null it receives the
    // (L, hidden_size) post-final-norm hidden states. brotensor::init() must
    // have been called once before any forward.
    void forward(const int32_t* ids, int L, brotensor::Tensor& logits_out,
                 brotensor::Tensor* hidden_out = nullptr);

    // Like forward(), but `logits_out` := (1, vocab_size) — logits for the last
    // appended token only. The KV cache still ingests all L tokens.
    void forward_last(const int32_t* ids, int L, brotensor::Tensor& logits_out);

    // Embedding lookup only (NO embedding scale, NO cache touch): write
    // `out` := (L, hidden_size) raw input embeddings for `ids` at the compute
    // dtype. The sqrt(hidden_size) scale is applied inside the layer loop, so
    // forward(ids) == forward_embeds(embed_tokens(ids)).
    void embed_tokens(const int32_t* ids, int L, brotensor::Tensor& out);

    // Like forward(), but starts from precomputed input embeddings rather than
    // token ids — `embeds` must be (L, hidden_size) at the compute dtype.
    void forward_embeds(const brotensor::Tensor& embeds, int L,
                        brotensor::Tensor& logits_out,
                        brotensor::Tensor* hidden_out = nullptr);

    const Gemma2Config& config() const { return cfg_; }

private:
    struct Layer {
        brotensor::Tensor input_ln;       // (hidden,) — (1 + weight) folded
        brotensor::Tensor Wq, Wk, Wv;     // (n_q*hd,h) (n_kv*hd,h) (n_kv*hd,h)
        brotensor::Tensor Wo;             // (hidden, n_q*hd)
        brotensor::Tensor post_attn_ln;   // (hidden,)
        brotensor::Tensor pre_ffn_ln;     // (hidden,)
        brotensor::Tensor post_ffn_ln;    // (hidden,)
        brotensor::Tensor gate_W, up_W, down_W;
        // Per-layer KV cache at true KV width: (max_seq, n_kv*head_dim).
        brotensor::Tensor K_cache, V_cache;
    };

    void load_weights_(const brolm::detail::weights::Source& src);

    // Run all decoder layers + final norm + lm_head over the residual stream in
    // h_ (L rows at the compute dtype). Applies the embedding scale, writes
    // `logits_out` := (L, vocab_size) (or (1, vocab_size) when
    // logits_last_row_only) with the final tanh soft-cap, and advances cache_len
    // by L. When `hidden_out` is non-null it receives a clone of the
    // (L, hidden_size) post-final-norm hidden states.
    void run_layers_(int L, brotensor::Tensor& logits_out,
                     brotensor::Tensor* hidden_out,
                     bool logits_last_row_only = false);

    Gemma2Config cfg_;

    // Weights.
    brotensor::Tensor embed_tokens_;   // (vocab, hidden)
    std::vector<Layer> layers_;
    brotensor::Tensor final_norm_;     // (hidden,) — (1 + weight) folded
    brotensor::Tensor lm_head_;        // (vocab, hidden) — clone of embed if tied

    // Cache state.
    int cache_len_   = 0;
    int max_seq_len_ = 0;

    // Per-call scratch — kept alive across forward() calls to avoid realloc.
    brotensor::Tensor ids_dev_;  // device INT32 token buffer
    brotensor::Tensor h_;        // residual stream (L, hidden)
    brotensor::Tensor norm_;     // pre-norm (input / pre-ffn) output
    brotensor::Tensor sub_;      // post-norm of a sublayer output
    brotensor::Tensor q_, k_, v_;
    brotensor::Tensor attn_;
    brotensor::Tensor proj_;     // o_proj / down_proj output
    brotensor::Tensor gate_, up_;
};

}  // namespace brolm::gemma
