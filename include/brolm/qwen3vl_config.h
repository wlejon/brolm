#pragma once

// Qwen3-VL configuration — typed mirror of the HF `config.json` for the
// `Qwen3VLForConditionalGeneration` family (sizes 2B/4B/8B/32B; the 30B-A3B/
// 235B-A22B releases are MoE variants — `Qwen3VLMoeForConditionalGeneration`
// — and are out of scope here).
//
// Unlike Qwen3.5-VL, Qwen3-VL's text backbone is a plain DENSE Qwen3 decoder:
// every layer is standard GQA full attention (no Gated DeltaNet linear-attn
// layers, no attn_output_gate), and RoPE is full-rotary M-RoPE (the rotary
// subrange spans the whole head_dim, not a partial_rotary_factor fraction).
// The vision tower adds a mechanism Qwen3.5-VL's doesn't have: DeepStack —
// at `vision.deepstack_visual_indexes` layers, intermediate hidden states are
// merged through a second (post-shuffle-norm) merger and injected additively
// into the first few text-decoder layers.
//
// The HF config has two logical sections, mirrored here as nested structs:
//
//   * `text_config`   (dense LLM)          → Qwen3VLConfig::Text
//   * `vision_config` (ViT + DeepStack)    → Qwen3VLConfig::Vision
//
// Field naming follows the HF JSON key names verbatim where possible.

#include <string>
#include <vector>

namespace brolm::qwen3vl {

struct Qwen3VLConfig {
    // ── text_config ────────────────────────────────────────────────────────
    struct Text {
        int   vocab_size            = 151936;
        int   hidden_size           = 2560;
        int   intermediate_size     = 9728;
        int   num_hidden_layers     = 36;

        int   num_attention_heads   = 32;    // query heads
        int   num_key_value_heads   = 8;     // KV heads (GQA)
        int   head_dim              = 128;
        bool  attention_bias        = false;

        float rms_norm_eps          = 1e-6f;

        int   max_position_embeddings = 262144;
        bool  tie_word_embeddings     = true;

        // ── rope_scaling ───────────────────────────────────────────────────
        // Qwen3-VL uses full-rotary (every head_dim column rotated, unlike
        // Qwen3.5-VL's partial_rotary_factor) interleaved multi-axis RoPE.
        struct Rope {
            bool  mrope_interleaved = true;
            // Per-axis sub-range widths in PAIRS (sum * 2 == head_dim). Order:
            // [t (temporal), h (height), w (width)]. HF 4B: [24, 20, 20].
            std::vector<int> mrope_section = {24, 20, 20};
            float rope_theta        = 5.0e6f;
        } rope;

        // Always full rotation for Qwen3-VL: rotary_dim == head_dim.
        int rotary_dim() const { return head_dim; }
    } text;

    // ── vision_config ──────────────────────────────────────────────────────
    struct Vision {
        int   depth                = 24;   // number of transformer blocks
        int   hidden_size          = 1024;
        int   num_heads            = 16;
        int   intermediate_size    = 4096; // MLP fan-out
        int   in_channels          = 3;
        int   patch_size           = 16;   // spatial patch (per side)
        int   temporal_patch_size  = 2;    // patches span 2 frames
        int   spatial_merge_size   = 2;    // patch merger 2x2 → 1
        int   out_hidden_size      = 2560; // == text.hidden_size after merger
        int   num_position_embeddings = 2304; // learned posembed table size
        // hidden_act is fixed to "gelu_pytorch_tanh" in every Qwen3-VL
        // release; hard-coded in the vision-block forward.

        // Block indices (0-based, into the `depth` blocks) after which an
        // extra post-shuffle-norm merger produces a DeepStack feature that
        // gets injected into the corresponding text-decoder layer. HF 4B:
        // [5, 11, 17] (3 entries → injected into decoder layers 0, 1, 2).
        std::vector<int> deepstack_visual_indexes = {5, 11, 17};
    } vision;

    // ── top-level multimodal token IDs ────────────────────────────────────
    int  image_token_id        = 151655; // <|image_pad|>
    int  video_token_id        = 151656; // <|video_pad|>
    int  vision_start_token_id = 151652; // <|vision_start|>
    int  vision_end_token_id   = 151653; // <|vision_end|>
    bool tie_word_embeddings_top = true; // duplicated at top level in HF JSON

    // Parse a Qwen3-VL `config.json` from disk. Throws std::runtime_error
    // with a "qwen3vl::Qwen3VLConfig: ..." prefix on I/O, parse, or schema
    // errors.
    static Qwen3VLConfig load(const std::string& config_json_path);

    // Parse from an in-memory JSON document (useful for tests).
    static Qwen3VLConfig from_json_text(const std::string& json_text);

    // Validate cross-field invariants (head_dim % 2 == 0, mrope_section sums
    // to head_dim/2, vision.out_hidden_size == text.hidden_size, DeepStack
    // indexes are in range and strictly increasing, ...). Called by both
    // loaders; throws on inconsistency.
    void validate() const;
};

}  // namespace brolm::qwen3vl
