#pragma once

// LLM2Vec text encoder — a decoder-only LLaMA-family model turned into a text
// encoder (inference-only). Runs on whichever backend brotensor resolves at
// runtime — FP32 on CPU, FP16 on a GPU backend.
//
// LLM2Vec (McGill-NLP) converts a causal decoder into a sentence/token encoder
// with exactly one architectural change — attention runs BIDIRECTIONAL (the
// causal mask is dropped) — plus weights fine-tuned for that regime (MNTP +
// supervised LoRA). Everything else is the shared brolm::detail::DenseDecoder:
// the same dense GQA / SwiGLU / plain-1-D-RoPE / RMSNorm stack as Mistral 3 and
// Qwen3, with per-head QK-norm DISABLED. This class is a thin wrapper that owns
// the typed Config and a DenseDecoder and drives its bidirectional, cacheless
// forward_encode() readout; the layer math and weight load live in
// dense_decoder.{h,cpp}.
//
// The reference checkpoint is LLM2Vec over Meta-Llama-3-8B-Instruct
// (McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp + the -supervised adapter):
// 32 layers, hidden 4096, 32 q / 8 kv heads (GQA 4:1), head_dim 128, SiLU
// SwiGLU, RMSNorm eps 1e-5, rope_theta 5e5, vocab 128256 → a 4096-dim
// embedding. The two LoRA adapters are merged into the base weights offline by
// scripts/convert-llm2vec.py, so this loader reads a plain HF Llama safetensors
// checkpoint directly.
//
// Tensor names follow the HF LLaMA convention rooted at `model.`
// (`model.embed_tokens.weight`, `model.layers.N.self_attn.q_proj.weight`,
// `model.norm.weight`); pass a `prefix` to rebase. The encoder never runs an
// lm_head, so `lm_head.weight` is optional in the checkpoint.

#include "brolm/detail/dense_decoder.h"
#include "brotensor/tensor.h"

#include <cstdint>
#include <string>
#include <vector>

namespace brotensor::safetensors { class File; }

namespace brolm::llm2vec {

// Typed mirror of the (flat) HF LLaMA `config.json`. Defaults match
// Meta-Llama-3-8B-Instruct — the LLM2Vec base — so a checkpoint that omits a
// key (HF Llama configs usually omit `head_dim`) still resolves correctly.
struct Config {
    int   vocab_size          = 128256;
    int   hidden_size         = 4096;
    int   intermediate_size   = 14336;
    int   num_hidden_layers   = 32;
    int   num_attention_heads = 32;   // query heads
    int   num_key_value_heads = 8;    // KV heads (GQA 4:1)
    int   head_dim            = 128;  // Llama-3 omits it → hidden/heads
    float rms_norm_eps        = 1e-5f;
    float rope_theta          = 5.0e5f;

    // Parse a LLaMA `config.json` from disk / from memory. Throws
    // std::runtime_error with a "llm2vec::Config: ..." prefix on I/O, parse, or
    // schema errors. `head_dim` falls back to hidden_size/num_attention_heads
    // when the key is absent.
    static Config load(const std::string& config_json_path);
    static Config from_json_text(const std::string& json_text);

    // Validate cross-field invariants (positive dims, GQA divisibility, even
    // head_dim). Throws on inconsistency.
    void validate() const;
};

class Encoder {
public:
    explicit Encoder(const Config& cfg);
    ~Encoder();

    Encoder(const Encoder&)            = delete;
    Encoder& operator=(const Encoder&) = delete;
    Encoder(Encoder&&) noexcept            = default;
    Encoder& operator=(Encoder&&) noexcept = default;

    // Load all weights from a single safetensors file under `prefix`. Names
    // follow the HF LLaMA convention; source tensors may be F16, F32, or BF16.
    // `lm_head.weight` is not required (the encoder never runs it). Throws
    // std::runtime_error on a missing name or shape mismatch.
    void load_weights(const brotensor::safetensors::File& f,
                      const std::string& prefix = "");

    // Sharded overload: a tensor is resolved by scanning the shards in order,
    // first match wins (Llama-3-8B ships several safetensors shards).
    void load_weights(
        const std::vector<const brotensor::safetensors::File*>& shards,
        const std::string& prefix = "");

    // Per-token contextualised embeddings: run the bidirectional stack over the
    // L token ids and write `hidden_out` := (L, hidden_size) post-final-norm
    // hidden states at the compute dtype. This is the conditioning sequence a
    // cross-attention consumer (e.g. ARDY's motion denoiser) attends to.
    //   ids: host pointer to L int32 token IDs in [0, vocab_size).
    // brotensor::init() must have been called once before any encode.
    void encode(const int32_t* ids, int L, brotensor::Tensor& hidden_out);

    // Sentence embedding: mean-pool the per-token hidden states over the tokens
    // marked valid in `pool_mask` and write `emb_out` := (hidden_size, 1) at the
    // compute dtype. `pool_mask` is a host array of L floats (1 = include /
    // 0 = exclude); null means pool over every token. This is LLM2Vec's masked
    // mean pooling — pass a mask that zeroes an instruction prefix to reproduce
    // its instruction-exclusion behaviour.
    void encode_pooled(const int32_t* ids, int L, brotensor::Tensor& emb_out,
                       const float* pool_mask = nullptr);

    const Config& config() const { return cfg_; }

private:
    Config cfg_;
    brolm::detail::DenseDecoder core_;
};

}  // namespace brolm::llm2vec
