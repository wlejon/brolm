#pragma once

// Qwen3-VL image preprocessor.
//
// HF ships the exact same `Qwen2VLImageProcessorFast` for Qwen3-VL as it does
// for Qwen3.5-VL (and Qwen2-VL/2.5-VL before it) — smart_resize with the same
// factor rule, the same patchify layout, and the same `get_rope_index`
// M-RoPE position-id algorithm. Qwen3-VL's vision_config even uses the exact
// same patch_size=16 / temporal_patch_size=2 / spatial_merge_size=2 numbers
// as Qwen3.5-VL. Rather than re-implement byte-for-byte identical host-side
// math under a new name, this header re-exports brolm::qwen35's
// preprocessor types and functions into brolm::qwen3vl.
//
// This is a deliberate, intentional reuse — not a shortcut for missing
// Qwen3-VL-specific behavior. If a future Qwen3-VL release changes any of
// this preprocessing math, split it out into its own implementation then.

#include "brolm/qwen35_preprocessor.h"

namespace brolm::qwen3vl {

using PreprocessConfig   = brolm::qwen35::PreprocessConfig;
using PreprocessedImage  = brolm::qwen35::PreprocessedImage;
using MRopePositions     = brolm::qwen35::MRopePositions;

using brolm::qwen35::preprocess_image;
using brolm::qwen35::smart_resize;
using brolm::qwen35::build_mrope_position_ids;

}  // namespace brolm::qwen3vl
