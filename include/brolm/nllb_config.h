#pragma once

// NLLB-200 configuration — typed mirror of the HF `config.json` for the
// `M2M100ForConditionalGeneration` family (`model_type: m2m_100`), e.g.
// facebook/nllb-200-distilled-600M.
//
// NLLB-200 is a vanilla encoder-decoder translation transformer (the fairseq /
// BART lineage): learned token embedding scaled by sqrt(d_model), COMPUTED
// sinusoidal positions (nothing to load), pre-norm LayerNorm-with-bias blocks,
// standard multi-head attention with q/k/v/out biases, ReLU FFN, a final
// LayerNorm on each stack, cross-attention in the decoder, and a tied lm_head.
//
// Field naming follows the HF JSON keys verbatim. Defaults match the real
// nllb-200-distilled-600M config.json. M2M-100 hard-codes the LayerNorm epsilon
// at 1e-5 (it is not a config field).

#include <string>

namespace brolm::nllb {

struct NllbConfig {
    int d_model                 = 1024;
    int encoder_layers          = 12;
    int decoder_layers          = 12;
    int encoder_attention_heads = 16;
    int decoder_attention_heads = 16;
    int encoder_ffn_dim         = 4096;
    int decoder_ffn_dim         = 4096;
    int vocab_size              = 256206;
    int max_position_embeddings = 1024;

    // Embeddings are multiplied by sqrt(d_model) (M2M-100 scale_embedding).
    bool scale_embedding = true;

    // M2M-100 LayerNorm epsilon — hard-coded upstream, not a config key.
    float layer_norm_eps = 1e-5f;

    // Special tokens (from config.json / generation_config.json).
    int pad_token_id           = 1;   // <pad> — also the sinusoidal padding_idx
    int bos_token_id           = 0;   // <s>
    int eos_token_id           = 2;   // </s>
    int decoder_start_token_id = 2;   // decoder is seeded with </s>

    // Per-head dimension (q/k/v projections split d_model across heads).
    int head_dim() const { return d_model / encoder_attention_heads; }

    // Parse an NLLB `config.json` from disk. Throws std::runtime_error with a
    // "nllb::NllbConfig: ..." prefix on I/O, parse, or schema errors.
    static NllbConfig load(const std::string& config_json_path);

    // Parse from an in-memory JSON document (useful for tests).
    static NllbConfig from_json_text(const std::string& json_text);

    // Validate cross-field invariants (positive dims, head divisibility).
    // Throws std::runtime_error on inconsistency.
    void validate() const;
};

}  // namespace brolm::nllb
