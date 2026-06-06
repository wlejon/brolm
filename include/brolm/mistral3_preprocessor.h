#pragma once

// Mistral 3.1 (Pixtral) image preprocessor — port of HF `PixtralImageProcessor`
// + the Mistral 3.1 image-token expansion. Produces the flat patch tensor the
// Pixtral vision tower consumes, plus the inline `[IMG]`/`[IMG_BREAK]`/
// `[IMG_END]` token span that stands in for the image in the prompt.
//
// Pipeline (HF `PixtralImageProcessor`):
//   1. Resize so the longest side fits `longest_edge` (scale DOWN by ceil only
//      when larger), then round each side up to a whole number of patches.
//      brolm rounds to a multiple of patch_size * spatial_merge_size so the
//      post-patch grid is in turn divisible by the spatial merge — keeping the
//      tower, the projector, and the token span mutually consistent. (Pixtral's
//      raw rule rounds to whole patches; for images whose natural patch grid is
//      odd this differs by at most one merge block — a real-weights-parity
//      detail, flagged in the .cpp.)
//   2. Bicubic resample (PIL BICUBIC ≈ broimage Catmull-Rom).
//   3. Normalise (x - mean) / std per channel (CLIP stats), input already [0,1].
//   4. Patchify to (num_patches, C*patch²) with the channel-major within-patch
//      layout the tower's flattened patch_conv expects:
//        feature f = c*patch² + ph*patch + pw,  patches row-major (h, w).
//
// Mistral uses plain 1-D RoPE on the text side (no M-RoPE), so — unlike the
// Qwen3.5 preprocessor — there is no position-id builder here: the fused token
// sequence feeds the decoder with ordinary sequential positions.
//
// Pure host-side; patches are emitted as CPU FP32 and the caller uploads them
// to the compute device before invoking the vision tower.

#include "brolm/mistral3_config.h"
#include "brotensor/tensor.h"

#include <vector>

namespace brolm::mistral3 {

// Mirrors the relevant Pixtral preprocessor_config.json keys. Defaults match
// Mistral-Small-3.1-2503 (CLIP normalisation, patch 14, longest_edge 1540).
struct PreprocessConfig {
    int   patch_size         = 14;
    int   spatial_merge_size = 2;
    int   longest_edge       = 1540;   // size.longest_edge
    int   num_channels       = 3;
    float mean[3]            = {0.48145466f, 0.4578275f, 0.40821073f};
    float std_[3]            = {0.26862954f, 0.26130258f, 0.27577711f};

    // Resize rounds each side to a multiple of this so the patch grid divides
    // the spatial merge.
    int factor() const { return patch_size * spatial_merge_size; }

    // Build a preprocessor config from a parsed Mistral 3.1 config (patch size,
    // merge, longest edge from vision.image_size). Normalisation stats are not
    // in config.json — they stay at the CLIP defaults above.
    static PreprocessConfig from_config(const Mistral3Config& cfg);
};

// Result of preprocessing a single image.
struct PreprocessedImage {
    // (num_patches, num_channels * patch²) FP32, host-resident; num_patches ==
    // grid_h * grid_w, in row-major (h, w) patch order.
    brotensor::Tensor patches;

    int grid_h = 0;   // resized_height / patch_size
    int grid_w = 0;   // resized_width  / patch_size
    int merge  = 2;   // spatial_merge_size, copied for downstream use

    // Merged-token grid (what the projector emits and the [IMG] span counts).
    int tokens_h() const { return grid_h / merge; }
    int tokens_w() const { return grid_w / merge; }
    int num_image_tokens() const { return tokens_h() * tokens_w(); }
};

// Compute the Pixtral resized (height, width) for an input image. Both outputs
// are multiples of `factor` (patch_size * spatial_merge_size). Throws on a
// non-positive dimension.
void compute_resized_size(int H, int W, int longest_edge, int factor,
                          int& resized_h, int& resized_w);

// Resize + normalise + patchify one static RGB image.
//   image_chw : float CHW, shape (num_channels, H, W), values in [0, 1].
// Throws std::runtime_error ("mistral3::preprocess_image: ...") on bad input.
PreprocessedImage preprocess_image(const float* image_chw, int H, int W,
                                   const PreprocessConfig& cfg);

// Build the inline image token span for one image: for each of tokens_h() rows,
// tokens_w() × `image_token_id`, then `image_break_id`; the final row ends with
// `image_end_id` instead. The number of image_token_id entries equals
// num_image_tokens() — i.e. the projector's output token count — so the caller
// can map the projector embeddings onto the [IMG] positions in order.
std::vector<int> build_image_token_span(const PreprocessedImage& img,
                                        int image_token_id,
                                        int image_break_id,
                                        int image_end_id);

}  // namespace brolm::mistral3
