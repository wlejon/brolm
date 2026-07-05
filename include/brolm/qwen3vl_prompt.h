#pragma once

// Shared prompt-assembly helpers for driving Qwen3-VL's vision tower + text
// decoder from a token sequence containing <|image_pad|> placeholders.
//
// Factored out of qwen3vl_vl.cpp (VLM::generate_tokens) so other drivers that
// sit on top of this same backbone — e.g. brodiffusion's Krea 2 image-as-
// prompt encoder, which taps the text decoder's hidden states instead of
// sampling from it — can assemble an image-conditioned input sequence
// without reimplementing this glue. VLM::generate_tokens itself calls these
// same functions (no behavior change, just a shared home).

#include "brolm/qwen3vl_preprocessor.h"  // PreprocessedImage, PreprocessConfig
#include "brolm/qwen3vl_vision.h"        // VisionTower
#include "brotensor/tensor.h"

#include <vector>

namespace brolm::qwen3vl {

struct ImageInput;  // qwen3vl_vl.h

// Expand each single <|image_pad|> token in `tokens` into N copies, N = the
// i-th image's num_image_tokens() (images taken in the order their
// placeholders appear in `tokens`). Throws std::runtime_error if the
// <|image_pad|> count doesn't match images.size().
std::vector<int> expand_image_pad(const std::vector<int>& tokens,
                                  const std::vector<PreprocessedImage>& images,
                                  int image_pad_id);

// Locate the start row of each image_pad run in `expanded` (post-expansion).
// Throws on a run-count / run-length mismatch against `images`.
std::vector<int> image_pad_run_starts(
    const std::vector<int>& expanded,
    const std::vector<PreprocessedImage>& images, int image_pad_id);

// Splice the (n, hidden) vision-tower output into `embeds` starting at row
// `dst_row`. Throws on a hidden-width or dtype mismatch.
void splice_vision(brotensor::Tensor& embeds, int dst_row,
                   const brotensor::Tensor& vision_out);

// Upload an FP32 host (3, H, W) image, preprocess it, and run it through
// `tower`, returning the post-merger token tensor; `pp_out`/`deepstack_out`
// receive the preprocessor result and the per-layer DeepStack feature list.
// Throws std::runtime_error on an invalid ImageInput.
brotensor::Tensor run_vision_one(VisionTower& tower, const PreprocessConfig& pp,
                                 const ImageInput& img,
                                 PreprocessedImage& pp_out,
                                 std::vector<brotensor::Tensor>& deepstack_out);

}  // namespace brolm::qwen3vl
