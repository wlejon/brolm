#pragma once

// Gemma-2 decoder configuration — typed mirror of the HF `config.json` for the
// `Gemma2ForCausalLM` family (google/gemma-2-2b / -9b / -27b).
//
// Gemma-2 is a pre-norm GQA / RoPE causal decoder very close to the LLaMA
// family, with a handful of architectural deltas the brolm core implements
// (see gemma2.h): an embedding scale of sqrt(hidden_size), four RMSNorms per
// layer (the post-attention and post-feedforward norms apply to the SUBLAYER
// OUTPUT before the residual add), RMSNorm computed as (1 + weight), a
// GeGLU MLP with the gelu-tanh activation, per-layer alternating
// sliding-window / global attention, tanh logit soft-capping inside attention,
// and a final tanh soft-cap on the output logits.
//
// Field naming follows the HF JSON key names verbatim. Defaults are the
// gemma-2-2b values. GGUF metadata parsing is a separate follow-up: the
// from_json / from_safetensors_dir seam here is the only config entry point
// today, and a from_gguf static can slot in alongside it without touching the
// model class.

#include <string>

namespace brolm::gemma {

struct Gemma2Config {
    int   vocab_size            = 256000;
    int   hidden_size           = 2304;
    int   intermediate_size     = 9216;
    int   num_hidden_layers     = 26;
    int   num_attention_heads   = 8;        // query heads
    int   num_key_value_heads   = 4;        // KV heads (GQA)
    int   head_dim              = 256;      // independent of hidden_size/heads
    float rms_norm_eps          = 1e-6f;
    float rope_theta            = 10000.0f;
    bool  tie_word_embeddings   = true;

    // Gemma-2 specifics.
    float query_pre_attn_scalar   = 256.0f; // must equal head_dim (see note)
    int   sliding_window          = 4096;   // even layers; odd layers global
    float attn_logit_softcapping  = 50.0f;  // tanh soft-cap inside attention
    float final_logit_softcapping = 30.0f;  // tanh soft-cap on output logits
    int   max_position_embeddings = 8192;
    // hidden_activation is fixed to "gelu_pytorch_tanh" in every Gemma-2
    // release; the MLP forward hard-codes brotensor::gelu_forward (the
    // tanh-approx GELU) to match.

    // Parse a Gemma-2 `config.json` from disk. Throws std::runtime_error with a
    // "gemma::Gemma2Config: ..." prefix on I/O, parse, or schema errors.
    static Gemma2Config from_json(const std::string& config_json_path);

    // Parse `<dir>/config.json`.
    static Gemma2Config from_safetensors_dir(const std::string& dir);

    // Parse from an in-memory JSON document (useful for tests).
    static Gemma2Config from_json_text(const std::string& json_text);

    // Validate cross-field invariants. Notably query_pre_attn_scalar must equal
    // head_dim: brotensor's flash_attention_decode applies the built-in
    // 1/sqrt(head_dim) score scale, which is correct only when the model's
    // query_pre_attn_scalar matches head_dim (true for 2B/9B; the 27B uses a
    // distinct scalar and would silently mis-scale, so we reject it here until
    // a rescale path lands). Throws std::runtime_error on inconsistency.
    void validate() const;
};

}  // namespace brolm::gemma
