#pragma once

// Qwen3-VL top-level inference driver.
//
// Glue layer that stitches the pieces (tokenizer, preprocessor, vision
// tower, text decoder) into a single `prompt + images -> text` API. No new
// neural-network layers live here — everything below is splice logic plus
// an autoregressive sampling loop. Mirrors qwen35::VLM closely; the one
// addition is threading each image's DeepStack features from the vision
// tower into the text decoder's forward_embeds alongside the main
// embedding splice.
//
// Pipeline per generate() call:
//   1. BPE-encode the prompt with the Qwen3-VL tokenizer (already knows the
//      <|vision_start|> / <|image_pad|> / <|vision_end|> specials).
//   2. Preprocess each input image into HF-shaped patches + grid_thw.
//   3. Expand every single <|image_pad|> placeholder into N copies of the
//      same id, where N == PreprocessedImage::num_image_tokens() for the
//      corresponding image.
//   4. Embed the expanded token sequence through the text model's tied
//      embedding table; run each image through the vision tower (main
//      merger output + DeepStack feature list); overwrite the i-th
//      image_pad run with the i-th image's main-merger embeddings, and
//      record the DeepStack features + row range for injection.
//   5. Build per-axis M-RoPE position streams (reusing qwen35's
//      get_rope_index port — see qwen3vl_preprocessor.h).
//   6. Prefill via TextModel::forward_embeds (passing the DeepstackSplice
//      list), sample the next token, then decode one token at a time
//      (no DeepStack splices past the prefill — those rows don't recur).
//   7. Stop on EOS / <|im_end|> / max_new_tokens, and decode the new ids.
//
// VLM owns the tokenizer, vision tower, text model, and per-layer KV cache.
// The cache is allocated once at load-time and reused across generate()
// calls; the caller is responsible for not exceeding VLMConfig::max_seq_len
// in a single call.

#include "brolm/qwen3vl_config.h"
#include "brolm/qwen3vl_preprocessor.h"
#include "brolm/qwen3vl_text.h"
#include "brolm/qwen3vl_tokenizer.h"
#include "brolm/qwen3vl_vision.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace brolm::qwen3vl {

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
    Qwen3VLConfig    model_cfg;
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

    // Load tokenizer + vision tower + text model from a Qwen3-VL checkpoint
    // directory. The directory must contain:
    //   - config.json
    //   - vocab.json + merges.txt
    //   - preprocessor_config.json (optional — defaults from VLMConfig.pp used otherwise)
    //   - one or more model*.safetensors shards
    // Throws std::runtime_error with "qwen3vl::VLM: " prefix on missing
    // files or shape/dtype mismatch.
    void load_from_directory(const std::string& dir);

    // Single-turn generation. The prompt is interpreted verbatim — caller is
    // responsible for any ChatML / chat-template wrapping. For each image in
    // `images`, the prompt must already contain the placeholder triple
    //   <|vision_start|><|image_pad|><|vision_end|>
    // in the position the image should appear. Images are consumed in
    // order.
    //
    // Returns the decoded string of the newly-generated tokens (prompt
    // excluded). Stops on EOS, <|im_end|>, or VLMConfig::max_new_tokens —
    // whichever comes first.
    std::string generate(const std::string& prompt,
                         const std::vector<ImageInput>& images);

    // Lower-level entry point: returns the raw newly-generated token IDs.
    std::vector<int> generate_tokens(const std::string& prompt,
                                     const std::vector<ImageInput>& images);

    // Per-token streaming/cancel hook: invoked once per newly generated
    // token, in decode order. Return false to halt generation after this
    // token.
    using TokenCallback = std::function<bool(int token_id)>;

    std::vector<int> generate_tokens(const std::string& prompt,
                                     const std::vector<ImageInput>& images,
                                     const TokenCallback& on_token);

    // Adjust the sampling/budget knobs between generate() calls.
    void set_generation(int max_new_tokens, float temperature, int top_k,
                        float top_p, uint64_t seed);

    const Tokenizer&     tokenizer() const;
    const Qwen3VLConfig& config()    const { return cfg_.model_cfg; }

private:
    VLMConfig                       cfg_;
    std::unique_ptr<Tokenizer>      tokenizer_;
    std::unique_ptr<VisionTower>    vision_;
    std::unique_ptr<TextModel>      text_;
    std::vector<LayerCache>         cache_;
    bool                            cache_allocated_ = false;
};

}  // namespace brolm::qwen3vl
