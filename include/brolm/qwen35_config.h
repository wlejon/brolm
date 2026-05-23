#pragma once

// Qwen3.5 VLM configuration — typed mirror of the HF `config.json` for the
// `Qwen3_5ForConditionalGeneration` family (the official Qwen3.5 VLM, sizes
// 0.8B/2B/4B/9B/35B-A3B/122B). All Qwen3.5 checkpoints are multimodal; there is
// no text-only Qwen3.5.
//
// The HF config has three logical sections, mirrored here as nested structs:
//
//   * top-level multimodal glue          → Qwen35Config
//   * `text_config`  (hybrid LLM)        → Qwen35Config::Text
//   * `vision_config` (ViT vision tower) → Qwen35Config::Vision
//
// Field naming follows the HF JSON key names verbatim where possible. The few
// places we diverge are commented at the field.

#include <string>
#include <vector>

namespace brolm::qwen35 {

// Per-layer attention type, parallel to HF's `text_config.layer_types` list of
// strings.  The hybrid pattern alternates several "linear" (Gated DeltaNet)
// layers with one "full" attention layer; for 0.8B it is [L,L,L,F] × 6.
enum class LayerType {
    Linear,   // "linear_attention" — Gated DeltaNet (Mamba-2-family recurrence)
    Full,     // "full_attention"   — standard softmax MHA with attn_output_gate
};

struct Qwen35Config {
    // ── text_config ────────────────────────────────────────────────────────
    struct Text {
        int   vocab_size            = 248320;
        int   hidden_size           = 1024;
        int   intermediate_size     = 3584;
        int   num_hidden_layers     = 24;

        // Full-attention block (`self_attn.*`)
        int   num_attention_heads   = 8;     // query heads
        int   num_key_value_heads   = 2;     // KV heads (GQA)
        int   head_dim              = 256;
        bool  attention_bias        = false; // Qwen3.5 has no qkv/o bias
        bool  attn_output_gate      = true;  // q_proj fans out to 2*q_dim;
                                             // gate = sigmoid(second half).

        // Linear-attention block (`linear_attn.*`) — Gated DeltaNet
        int   linear_num_key_heads   = 16;   // == linear_num_value_heads in
        int   linear_num_value_heads = 16;   // every Qwen3.5 release
        int   linear_key_head_dim    = 128;
        int   linear_value_head_dim  = 128;
        int   linear_conv_kernel_dim = 4;    // causal conv1d before the rule

        float rms_norm_eps          = 1e-6f;

        // Hybrid layer schedule (length == num_hidden_layers). Derived from
        // the HF `layer_types` JSON array.
        std::vector<LayerType> layer_types;
        int   full_attention_interval = 4;   // documented period; redundant
                                             // with layer_types, kept for sanity.

        int   max_position_embeddings = 262144;
        bool  tie_word_embeddings     = true;

        // ── rope_parameters ────────────────────────────────────────────────
        // Qwen3.5 uses interleaved multi-axis RoPE (M-RoPE) and rotates only a
        // fraction of head_dim (partial_rotary_factor).
        struct Rope {
            bool  mrope_interleaved      = true;
            // Per-axis sub-range widths in PAIRS (sum * 2 == rotary_dim, where
            // rotary_dim == round_even(head_dim * partial_rotary_factor)).
            // Order: [t (temporal), h (height), w (width)].  HF: [11, 11, 10].
            std::vector<int> mrope_section = {11, 11, 10};
            float rope_theta             = 1.0e7f;
            float partial_rotary_factor  = 0.25f;
        } rope;

        // Multi-Token Prediction head (`mtp.*`). Forward-only; brolm runs it
        // optionally for speculative decoding.
        int   mtp_num_hidden_layers      = 1;
        bool  mtp_use_dedicated_embeddings = false;

        int rotary_dim() const;     // == round_even(head_dim * partial_rotary_factor)
    } text;

    // ── vision_config ──────────────────────────────────────────────────────
    struct Vision {
        int   depth                = 12;   // number of transformer blocks
        int   hidden_size          = 768;
        int   num_heads            = 12;
        int   intermediate_size    = 3072; // MLP fan-out
        int   in_channels          = 3;
        int   patch_size           = 16;   // spatial patch (per side)
        int   temporal_patch_size  = 2;    // patches span 2 frames
        int   spatial_merge_size   = 2;    // patch merger 2x2 → 1
        int   out_hidden_size      = 1024; // == text.hidden_size after merger
        int   num_position_embeddings = 2304; // learned posembed table size
        // hidden_act is fixed to "gelu_pytorch_tanh" in every Qwen3.5 release;
        // hard-coded in the vision-block forward.
        // `deepstack_visual_indexes` is empty for 0.8B and unused at inference
        // time in our path; carried verbatim in case a future size populates it.
        std::vector<int> deepstack_visual_indexes;
    } vision;

    // ── top-level multimodal token IDs ────────────────────────────────────
    int  image_token_id        = 248056; // <|image_pad|>
    int  video_token_id        = 248057; // <|video_pad|>
    int  vision_start_token_id = 248053; // <|vision_start|>
    int  vision_end_token_id   = 248054; // <|vision_end|>
    bool tie_word_embeddings_top = true; // duplicated at top level in HF JSON

    // Parse a Qwen3.5 `config.json` from disk. Throws std::runtime_error with
    // a "qwen35::Qwen35Config: ..." prefix on I/O, parse, or schema errors.
    static Qwen35Config load(const std::string& config_json_path);

    // Parse from an in-memory JSON document (useful for tests).
    static Qwen35Config from_json_text(const std::string& json_text);

    // Validate cross-field invariants (layer_types.size == num_hidden_layers,
    // head_dim % 2 == 0, mrope_section sums to rotary_dim/2, ...). Called by
    // both loaders; throws on inconsistency.
    void validate() const;
};

}  // namespace brolm::qwen35
