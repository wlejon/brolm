#pragma once

// Qwen3.5-VL image preprocessor — port of HuggingFace `Qwen2VLImageProcessorFast`
// (which Qwen3.5-VL reuses byte-for-byte for static images). Produces the flat
// patch tensor + `grid_thw` triple that the vision tower consumes, plus the
// M-RoPE position-id streams (t, h, w) for a token sequence that contains image
// placeholders.
//
// Pipeline, matching HF `_preprocess` in
// transformers/models/qwen2_vl/image_processing_qwen2_vl.py:
//
//   1. smart_resize(H, W, factor = patch_size * merge_size, min_pixels,
//                   max_pixels) — keep aspect ratio, round each side to a
//                   multiple of `factor`, scale down/up if total pixel count
//                   leaves [min, max].
//   2. Bilinear resize (per-channel, align_corners=False / center-pixel rule).
//   3. Normalise: (x - mean) / std per channel.
//   4. Patchify with reshape + permute (see qwen35_preprocessor.cpp for the
//      exact axis order — must match HF or the vision tower silently misreads
//      pixels).
//
// For a single static RGB image grid_t = 1, but the channel-temporal axis is
// duplicated `temporal_patch_size` times when flattening (HF
// `unsqueeze(6).expand(..., temporal_patch_size, ...)` in `_preprocess`).
//
// The preprocessor emits *no* tokens; the caller is responsible for splicing
// `<|vision_start|>` + N × `<|image_pad|>` + `<|vision_end|>` into the prompt,
// where N == PreprocessedImage::num_image_tokens().
//
// Pure host-side; no GPU kernels. Output patches live on the CPU as FP32 and
// the caller uploads them to the compute device before invoking the vision
// tower.

#include "brotensor/tensor.h"

#include <cstdint>
#include <vector>

namespace brolm::qwen35 {

// Configuration mirroring `preprocessor_config.json` keys.  Defaults match the
// official Qwen/Qwen3.5-0.8B checkpoint.  Note `min_pixels` / `max_pixels` are
// *total pixel counts* — they correspond to HF's `size.shortest_edge` /
// `size.longest_edge` despite the naming (verified against
// image_processing_qwen2_vl.py: `size = {"shortest_edge": 56*56,
// "longest_edge": 28*28*1280}` and smart_resize's pixel-count comparisons).
struct PreprocessConfig {
    int   patch_size           = 16;
    int   temporal_patch_size  = 2;
    int   merge_size           = 2;
    float mean[3]              = {0.5f, 0.5f, 0.5f};
    float std_[3]              = {0.5f, 0.5f, 0.5f};
    int   min_pixels           = 65536;     // size.shortest_edge
    int   max_pixels           = 16777216;  // size.longest_edge

    // The "factor" smart_resize rounds each side to: every spatial dim must be
    // a multiple of `patch_size * merge_size` so the post-patchify grid is in
    // turn a multiple of `merge_size`.
    int factor() const { return patch_size * merge_size; }
};

// Result of preprocessing a single image.
struct PreprocessedImage {
    // Flat patch tensor in HF's exact order:
    //   shape = (num_patches, channels * temporal_patch_size * patch_size * patch_size)
    // where num_patches == grid_t * grid_h * grid_w (a single static image has
    // grid_t = 1).  FP32, host-resident.
    brotensor::Tensor patches;

    int grid_t      = 1;     // ==1 for a static image
    int grid_h      = 0;     // resized_height / patch_size
    int grid_w      = 0;     // resized_width  / patch_size
    int merge_size  = 2;     // copied from PreprocessConfig for downstream use

    // Number of `<|image_pad|>` placeholder tokens this image expands into in
    // the prompt — equals `grid_t * (grid_h/merge) * (grid_w/merge)` (the
    // post-patch-merger token count).
    int num_image_tokens() const {
        return grid_t * (grid_h / merge_size) * (grid_w / merge_size);
    }
};

// Smart-resize a single static RGB image and emit patches + grid.
//
//   image_chw : float CHW, shape (3, H, W), values in [0, 1] (caller scales
//               from whatever its source range was).
//   H, W      : input height / width.
//   cfg       : pixel range, patch sizes, normalisation stats.
//
// Returns an owning PreprocessedImage. Throws std::runtime_error with the
// "qwen35::preprocess_image: " prefix on bad inputs (e.g. zero-size image,
// extreme aspect ratio rejected by smart_resize).
PreprocessedImage preprocess_image(const float* image_chw, int H, int W,
                                   const PreprocessConfig& cfg);

// HF `smart_resize`. Public so callers can pre-compute the resized shape (e.g.
// to budget context for `<|image_pad|>` tokens) without doing the resize+
// patchify. Throws on aspect ratio > 200 or zero dims, matching HF.
//   factor : patch_size * merge_size (== PreprocessConfig::factor()).
// Out parameters are the resized height and width (both multiples of factor).
void smart_resize(int height, int width, int factor,
                  int min_pixels, int max_pixels,
                  int& resized_height, int& resized_width);

// Three int64 streams of M-RoPE position IDs, one entry per token in the input
// sequence. Mirrors HF `Qwen2VLForConditionalGeneration.get_rope_index` output
// (which Qwen3-VL/3.5-VL reuse): `position_ids[3, seq_len]` plus the scalar
// `mrope_position_delta = max(positions) + 1 - seq_len` used to advance
// positions during decoding.
struct MRopePositions {
    std::vector<int64_t> t;    // temporal axis
    std::vector<int64_t> h;    // height axis
    std::vector<int64_t> w;    // width axis
    int64_t delta = 0;         // mrope_position_delta for the decode step
};

// Build M-RoPE position streams for a flat token sequence containing image
// placeholders.
//
//   tokens                 : full prompt token IDs (incl. each image's
//                            <|vision_start|> + N×<|image_pad|> + <|vision_end|>).
//   images                 : ordered grid_thw / merge info for each image in
//                            the prompt, in the same order the placeholder
//                            spans appear in `tokens`.
//   image_token_id         : Qwen35Config::image_token_id (the <|image_pad|>
//                            placeholder, NOT <|vision_start|>).
//   vision_start_token_id  : Qwen35Config::vision_start_token_id (currently
//                            unused for position-ID assignment — kept for
//                            parity with HF's signature and for future
//                            interleaved-modality logic).
//
// Throws std::runtime_error with the "qwen35::build_mrope_position_ids: "
// prefix when the count of <|image_pad|> runs in `tokens` does not match
// `images.size()`, or when a run's length does not equal
// `images[i].num_image_tokens()`.
MRopePositions build_mrope_position_ids(
    const std::vector<int>& tokens,
    const std::vector<PreprocessedImage>& images,
    int image_token_id, int vision_start_token_id);

}  // namespace brolm::qwen35
