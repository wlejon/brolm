#pragma once

// Mistral 3.1 VLM configuration — typed mirror of the HF `config.json` for the
// `Mistral3ForConditionalGeneration` family (`model_type: mistral3`), e.g.
// Mistral-Small-3.1-24B-Instruct-2503.
//
// Mistral 3.1 is a VLM = a Mistral text decoder + a Pixtral vision encoder +
// a small patch-merger / linear projector. The HF config has two logical
// sub-objects plus a thin top-level multimodal glue:
//
//   * `text_config`   (Mistral dense decoder) → Mistral3Config::Text
//   * `vision_config` (Pixtral ViT tower)     → Mistral3Config::Vision
//   * top-level projector + image-token glue  → Mistral3Config
//
// Field naming follows the HF JSON keys verbatim where possible. Defaults match
// the real Mistral-Small-3.1-24B-Instruct-2503 config.json. The Vision section
// is parsed in full now but only consumed once the Pixtral tower lands; the
// Text section drives the brolm::detail::DenseDecoder text backbone today.
//
// Key text-arch facts (verified against the real config.json): hidden 5120,
// intermediate 32768, 40 layers, 32 q / 8 kv heads (GQA 4:1), head_dim 128,
// SiLU SwiGLU, RMSNorm eps 1e-5, rope_theta 1e9, vocab 131072, 128k context,
// no sliding window, NO QK-norm, and an UNTIED lm_head (HF Mistral defaults
// `tie_word_embeddings` to false; the key is absent from the config).

#include <string>
#include <vector>

namespace brolm::mistral3 {

struct Mistral3Config {
    // ── text_config (Mistral dense decoder) ─────────────────────────────────
    struct Text {
        int   vocab_size            = 131072;
        int   hidden_size           = 5120;
        int   intermediate_size     = 32768;
        int   num_hidden_layers     = 40;
        int   num_attention_heads   = 32;    // query heads
        int   num_key_value_heads   = 8;     // KV heads (GQA 4:1)
        int   head_dim              = 128;
        float rms_norm_eps          = 1e-5f;
        float rope_theta            = 1.0e9f;
        // Mistral has no QK-norm and (unlike Qwen3-0.6B) does not tie the
        // embedding and lm_head. The HF default for `tie_word_embeddings` is
        // false and the key is absent from the mistral3 config, so we default
        // it false too.
        bool  tie_word_embeddings   = false;
        int   max_position_embeddings = 131072;
        // `sliding_window: null` in the real config → full causal attention.
        // Carried so a future windowed Mistral (e.g. 0.3) can set it; unused
        // by the decoder today.
        bool  has_sliding_window    = false;
        int   sliding_window        = 0;
    } text;

    // ── vision_config (Pixtral ViT tower) ───────────────────────────────────
    // Parsed now, consumed when the Pixtral tower lands. Pixtral is a 2-D-RoPE
    // ViT with gated (SiLU SwiGLU) MLP and RMSNorm, native variable image
    // sizes (no learned position table), patch 14, rope_theta 1e4.
    struct Vision {
        int   hidden_size          = 1024;
        int   num_attention_heads  = 16;
        int   head_dim             = 64;
        int   intermediate_size    = 4096;
        int   num_hidden_layers    = 24;
        int   num_channels         = 3;
        int   patch_size           = 14;
        int   image_size           = 1540;  // longest supported square side
        float rope_theta           = 1.0e4f;
        // hidden_act is "silu" (gated SwiGLU) in every Mistral 3.1 release;
        // hard-coded in the eventual vision-block forward.
    } vision;

    // ── top-level multimodal glue ───────────────────────────────────────────
    int  image_token_index        = 10;     // [IMG] in the Tekken v7 table
    int  spatial_merge_size       = 2;       // patch-merger 2x2 → 1 token
    bool multimodal_projector_bias = false;  // projector Linears are bias-free
    // projector_hidden_act is "gelu" in every Mistral 3.1 release; hard-coded
    // in the eventual projector forward.

    // Parse a Mistral 3.1 `config.json` from disk. Throws std::runtime_error
    // with a "mistral3::Mistral3Config: ..." prefix on I/O, parse, or schema
    // errors.
    static Mistral3Config load(const std::string& config_json_path);

    // Parse from an in-memory JSON document (useful for tests).
    static Mistral3Config from_json_text(const std::string& json_text);

    // Validate cross-field invariants (positive dims, GQA divisibility, even
    // head_dim). Called by both loaders; throws on inconsistency.
    void validate() const;
};

}  // namespace brolm::mistral3
