#pragma once

// Qwen3.5-VL top-level inference driver.
//
// Glue layer that stitches the Stage 1-3 pieces (tokenizer, preprocessor,
// vision tower, hybrid text decoder) into a single `prompt + images -> text`
// API. No new neural-network layers live here — everything below is splice
// logic plus an autoregressive sampling loop.
//
// Pipeline per generate() call:
//   1. BPE-encode the prompt with the Qwen3.5-VL tokenizer (already knows the
//      <|vision_start|> / <|image_pad|> / <|vision_end|> specials).
//   2. Preprocess each input image into HF-shaped patches + grid_thw.
//   3. Expand every single <|image_pad|> placeholder into N copies of the
//      same id, where N == PreprocessedImage::num_image_tokens() for the
//      corresponding image. The chat template emits exactly one image_pad per
//      image; the expansion is what gives M-RoPE and the splice loop a slot
//      to write each post-merger vision token into.
//   4. Embed the expanded token sequence through the text model's tied
//      embedding table; run each image through the vision tower; overwrite
//      the i-th image_pad run with the i-th image's (num_image_tokens,
//      text_hidden) embeddings.
//   5. Build per-axis M-RoPE position streams (text-axis positions for text
//      tokens; t/h/w grid offsets inside each vision span; cumulative
//      advancement matching HF get_rope_index).
//   6. Prefill via TextModel::forward_embeds, sample the next token from the
//      last logits row, then run a one-token-at-a-time decode loop using
//      forward_embeds again on the embed table lookup of the sampled id.
//   7. Stop on EOS / <|im_end|> / max_new_tokens, and decode the new ids.
//
// The MTP head present in the checkpoint (`mtp.*` keys) is *out of scope* —
// this driver only emits one token per step using the main head.
//
// VLM owns the tokenizer, vision tower, text model, and per-layer KV/state
// cache. The cache is allocated once at load-time and reused across
// generate() calls; the caller is responsible for not exceeding
// VLMConfig::max_seq_len in a single call.

#include "brolm/qwen35_config.h"
#include "brolm/qwen35_preprocessor.h"
#include "brolm/qwen35_text.h"
#include "brolm/qwen35_tokenizer.h"
#include "brolm/qwen35_vision.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace brolm::qwen35 {

// A single input image, in CHW float layout normalised to [0, 1]. The caller
// owns `pixels` (a contiguous 3*H*W block). The driver makes its own copy
// during preprocess_image so the buffer can be released after generate()
// returns.
struct ImageInput {
    const float* pixels = nullptr;  // (3, H, W), [0, 1] range
    int H = 0;
    int W = 0;
};

// Driver configuration. `model_cfg` carries every architectural dim; `pp`
// drives the image preprocessor (patch size / pixel-count budget / normalise
// stats). The sampling block matches qwen_generate's SamplingParams.
struct VLMConfig {
    Qwen35Config     model_cfg;
    PreprocessConfig pp;
    int              max_seq_len    = 4096;
    // Sampling.
    int              max_new_tokens = 64;
    float            temperature    = 0.0f;   // 0 => greedy (argmax)
    int              top_k          = 0;      // 0 => disabled
    float            top_p          = 1.0f;
    uint64_t         seed           = 0;
};

class VLM {
public:
    explicit VLM(const VLMConfig& cfg);
    ~VLM();
    VLM(const VLM&)            = delete;
    VLM& operator=(const VLM&) = delete;

    // Load tokenizer + vision tower + text model from a Qwen3.5-VL checkpoint
    // directory. The directory must contain:
    //   - config.json
    //   - tokenizer.json or (vocab.json + merges.txt)
    //   - preprocessor_config.json (optional — defaults from VLMConfig.pp used otherwise)
    //   - one or more model.safetensors* shards
    // Throws std::runtime_error with "qwen35::VLM: " prefix on missing files
    // or shape/dtype mismatch.
    void load_from_directory(const std::string& dir);

    // Single-turn generation. The prompt is interpreted verbatim — caller is
    // responsible for any ChatML / chat-template wrapping. For each image in
    // `images`, the prompt must already contain the placeholder triple
    //   <|vision_start|><|image_pad|><|vision_end|>
    // in the position the image should appear. Images are consumed in order.
    //
    // Returns the decoded string of the newly-generated tokens (prompt
    // excluded). Stops on EOS, <|im_end|>, or VLMConfig::max_new_tokens —
    // whichever comes first.
    std::string generate(const std::string& prompt,
                         const std::vector<ImageInput>& images);

    // Lower-level entry point: returns the raw newly-generated token IDs.
    // Same generation logic as generate(); the string entry just wraps this
    // in a tokenizer.decode() call.
    std::vector<int> generate_tokens(const std::string& prompt,
                                     const std::vector<ImageInput>& images);

    // Accessors. The tokenizer's underlying Qwen3 BPE handle is exposed so
    // callers can encode/decode arbitrary text outside the generate() path.
    const Tokenizer&    tokenizer() const;
    const Qwen35Config& config()    const { return cfg_.model_cfg; }

private:
    // The three model components hold non-default-constructible state
    // (tokenizer's private ctor, vision/text's required-config ctor), so we
    // hold them via unique_ptr and instantiate from load_from_directory.
    VLMConfig                       cfg_;
    std::unique_ptr<Tokenizer>      tokenizer_;
    std::unique_ptr<VisionTower>    vision_;
    std::unique_ptr<TextModel>      text_;
    std::vector<LayerCache>         cache_;
    bool                            cache_allocated_ = false;
};

}  // namespace brolm::qwen35
