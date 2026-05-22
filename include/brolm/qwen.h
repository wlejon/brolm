#pragma once

// Qwen3 decoder transformer — inference-only, KV-cached.
//
// Faithful port of Hugging Face `Qwen3ForCausalLM`. Forward-only. Runs on
// whichever backend brotensor resolves at runtime — CPU by default, CUDA when
// available — at that backend's compute dtype (FP32 on CPU, FP16 on a GPU).
//
// Architecture (one decoder layer, residual stream `h` of width hidden_size):
//
//   residual = h
//   h = rms_norm(h, input_layernorm.weight, eps)
//     q = q_proj(h)                                  (L, n_q  * head_dim), no bias
//     k = k_proj(h)                                  (L, n_kv * head_dim), no bias
//     v = v_proj(h)                                  (L, n_kv * head_dim), no bias
//     q = per_head_rmsnorm(q, q_norm.weight, head_dim)   QK-norm, per head
//     k = per_head_rmsnorm(k, k_norm.weight, head_dim)
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
// There are no biases on any projection in Qwen3.
//
// RoPE convention: brotensor's rope_forward rotates adjacent pairs (GPT-J /
// interleaved). HF Qwen3 uses rotate_half (dim i pairs with i + head_dim/2).
// load_weights() permutes the per-head head_dim ordering of q_proj / k_proj /
// q_norm / k_norm so projected q/k land in interleaved-pair order — see the
// comment on permute_rope_weight_() in qwen.cpp.

#include "brotensor/tensor.h"

#include <cstdint>
#include <string>
#include <vector>

namespace brotensor::safetensors { class File; struct TensorView; }

namespace brolm::qwen {

struct Qwen3Config {
    int   vocab_size            = 151936;
    int   hidden_size           = 1024;
    int   intermediate_size     = 3072;
    int   num_hidden_layers     = 28;
    int   num_attention_heads   = 16;     // query heads
    int   num_key_value_heads   = 8;      // KV heads (GQA)
    int   head_dim              = 128;    // independent of hidden_size/num_heads
    float rms_norm_eps          = 1e-6f;
    float rope_theta            = 1000000.0f;
    bool  tie_word_embeddings   = true;
    int   max_position_embeddings = 40960;
};

class Qwen3Model {
public:
    explicit Qwen3Model(const Qwen3Config& cfg);
    ~Qwen3Model();

    // Non-copyable; movable.
    Qwen3Model(const Qwen3Model&) = delete;
    Qwen3Model& operator=(const Qwen3Model&) = delete;
    Qwen3Model(Qwen3Model&&) noexcept = default;
    Qwen3Model& operator=(Qwen3Model&&) noexcept = default;

    // Load all weights from a single safetensors file under `prefix`. Tensor
    // names follow the HF convention; see qwen.cpp for the full list. Source
    // tensors may be F16, F32, or BF16. When tie_word_embeddings is true,
    // `lm_head.weight` is expected to be absent and equals embed_tokens.weight.
    // Throws std::runtime_error on a missing name or shape mismatch.
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
    void reset_cache();

    int cache_len() const { return cache_len_; }

    // Append L tokens at absolute positions [cache_len, cache_len + L), run the
    // decoder, and write `logits_out` := (L, vocab_size) at the compute dtype.
    // Advances cache_len by L. Prefill = one call with the whole prompt;
    // decode = a call with L == 1.
    //   ids: host pointer to L int32 token IDs in [0, vocab_size).
    // brotensor::init() must have been called once before any forward.
    void forward(const int32_t* ids, int L, brotensor::Tensor& logits_out);

    const Qwen3Config& config() const { return cfg_; }

private:
    struct Layer {
        brotensor::Tensor input_ln;     // (hidden,)
        brotensor::Tensor Wq, Wk, Wv;   // (n_q*hd,h) (n_kv*hd,h) (n_kv*hd,h)
        brotensor::Tensor Wo;           // (hidden, n_q*hd)
        brotensor::Tensor q_norm, k_norm;          // (head_dim,) — RoPE-permuted
        brotensor::Tensor post_attn_ln;            // (hidden,)
        brotensor::Tensor gate_W, up_W, down_W;    // MLP projections
        // Per-layer KV cache, expanded to n_q heads: (max_seq, n_q*head_dim).
        brotensor::Tensor K_cache, V_cache;
    };

    void load_weights_impl_(
        const std::vector<const brotensor::safetensors::File*>& shards,
        const std::string& prefix);

    // GQA head expansion: copy each KV head's head_dim block out to the
    // group of query heads it serves. `src` is (L, n_kv*head_dim); `dst`
    // becomes (L, n_q*head_dim).
    void expand_kv_heads_(const brotensor::Tensor& src, brotensor::Tensor& dst);

    Qwen3Config cfg_;

    // Weights.
    brotensor::Tensor embed_tokens_;   // (vocab, hidden)
    std::vector<Layer> layers_;
    brotensor::Tensor final_norm_;     // (hidden,)
    brotensor::Tensor lm_head_;        // (vocab, hidden) — aliases embed when tied

    // Cache state.
    int cache_len_     = 0;
    int max_seq_len_   = 0;

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

}  // namespace brolm::qwen
