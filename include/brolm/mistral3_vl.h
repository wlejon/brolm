#pragma once

// Mistral 3.1 vision-language model — the fusion layer that ties the Pixtral
// vision tower, the multimodal projector, and the Mistral text decoder into one
// image→text pipeline (HF `Mistral3ForConditionalGeneration`).
//
// Given a prompt token sequence that already carries each image's inline
// `[IMG]`/`[IMG_BREAK]`/`[IMG_END]` span (see mistral3_preprocessor.h's
// build_image_token_span) plus the preprocessed images in prompt order, it:
//   1. embeds the full token sequence (text embeddings for every token);
//   2. runs tower → projector per image to get (num_image_tokens, text_hidden)
//      image embeddings;
//   3. overwrites the `[IMG]` token rows of the embedding stream with the image
//      embeddings, in order (HF masked_scatter);
//   4. prefills the decoder on the fused stream and autoregressively decodes.
//
// Mistral uses plain 1-D RoPE, so the fused stream feeds the decoder with
// ordinary sequential positions — no M-RoPE bookkeeping.
//
// Weight loading here targets a single-file (e.g. synthetic / consolidated)
// checkpoint: the text backbone under `language_model.`, the tower under
// `vision_tower.`, the projector under `multi_modal_projector.`. Loading the
// real sharded safetensors (or a quantized VLM GGUF with an mmproj vision map)
// is future work tied to the real-weights parity effort — and is gated on
// hardware anyway (the FP16 24B VLM exceeds a single 24 GB GPU).

#include "brolm/mistral3_config.h"
#include "brolm/mistral3_text.h"
#include "brolm/mistral3_vision.h"
#include "brolm/mistral3_projector.h"
#include "brolm/mistral3_preprocessor.h"
#include "brolm/detail/generate.h"

#include "brotensor/tensor.h"

#include <cstdint>
#include <vector>

namespace brotensor::safetensors { class File; }

namespace brolm::mistral3 {

class VLModel {
public:
    explicit VLModel(const Mistral3Config& cfg);
    ~VLModel();

    VLModel(const VLModel&)            = delete;
    VLModel& operator=(const VLModel&) = delete;
    VLModel(VLModel&&) noexcept            = default;
    VLModel& operator=(VLModel&&) noexcept = default;

    // Load text (prefix `language_model.`), vision (`vision_tower.`), and
    // projector (`multi_modal_projector.`) weights from one safetensors file.
    void load_weights(const brotensor::safetensors::File& f);

    // Load from the two llama.cpp GGUFs Mistral 3.1 ships: the (quantized) text
    // decoder gguf and the (F16) mmproj/clip gguf carrying the vision tower +
    // projector. The text quant path is GPU-only; the mmproj is dense F16.
    void load_weights(const brotensor::gguf::File& text_gguf,
                      const brotensor::gguf::File& mmproj_gguf);

    void allocate_cache(int max_seq_len);
    void reset_cache();
    int  cache_len() const;

    // tower → projector for one preprocessed image:
    //   patches_host : the preprocessor's CPU FP32 patch tensor
    //                  (grid_h*grid_w, C*patch²); uploaded internally.
    //   out          : (num_image_tokens, text_hidden) on the compute device.
    void image_embeddings(const brotensor::Tensor& patches_host,
                          int grid_h, int grid_w, brotensor::Tensor& out);

    // Build the fused (L, text_hidden) input-embedding stream for `prompt_ids`
    // with each image's projector embeddings spliced onto the image_token_id
    // positions, in prompt order. The total number of image tokens across
    // `images` must equal the count of image_token_id in `prompt_ids`. Exposed
    // so callers (and tests) can inspect the fused stream directly.
    void fuse_embeds(const std::vector<int32_t>& prompt_ids,
                     const std::vector<PreprocessedImage>& images,
                     int image_token_id, brotensor::Tensor& fused_out);

    // Autoregressive generation for an image+text prompt. Sizes the KV cache for
    // (prompt + max_new_tokens), prefills the fused stream, then decodes token
    // by token (text-only forward). Returns ONLY the newly generated ids.
    std::vector<int32_t> generate(const std::vector<int32_t>& prompt_ids,
                                  const std::vector<PreprocessedImage>& images,
                                  int image_token_id, int eos_id,
                                  const brolm::detail::GenerateOptions& opts);

    TextModel&             text()      { return text_; }
    VisionTower&           vision()    { return vision_; }
    MultiModalProjector&   projector() { return projector_; }
    const Mistral3Config&  config() const { return cfg_; }

private:
    Mistral3Config cfg_;
    TextModel text_;
    VisionTower vision_;
    MultiModalProjector projector_;

    // Scratch reused across image_embeddings calls.
    brotensor::Tensor vis_feat_;   // (grid_h*grid_w, vision_hidden)
};

}  // namespace brolm::mistral3
