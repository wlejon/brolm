// Qwen3.5-VL image preprocessor.
//
// Port of HuggingFace `Qwen2VLImageProcessorFast` (Qwen3.5-VL reuses it
// verbatim for static images).
// Reference: transformers/models/qwen2_vl/image_processing_qwen2_vl.py
//   - smart_resize           (lines 61-86 of the upstream file at fetch time)
//   - Qwen2VLImageProcessor._preprocess.reshape/permute/expand chain
//                            (the patchify block; see comments inline)
//
// And for M-RoPE positions:
//   transformers/models/qwen2_vl/modeling_qwen2_vl.py
//     Qwen2VLModel.get_rope_index           (line 957)
//     Qwen2VLModel.get_vision_position_ids  (line 899)
// Qwen3-VL inherits the same logic from Qwen3VLModel; the static-image
// behaviour (grid_t=1, image-only) is identical.

#include "brolm/qwen35_preprocessor.h"

#include "broimage/geometric.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace brolm::qwen35 {

namespace bt = ::brotensor;

namespace {

[[noreturn]] void fail_pre(const std::string& msg) {
    throw std::runtime_error("qwen35::preprocess_image: " + msg);
}
[[noreturn]] void fail_rope(const std::string& msg) {
    throw std::runtime_error("qwen35::build_mrope_position_ids: " + msg);
}

// HF `round` for positive floats uses banker's rounding; here all our inputs
// are H, W >= 1 and factor >= 1 so the half-away-from-zero `std::lround`
// matches in practice.  We keep it conservative and use `std::lround`.
inline int round_to_multiple(int x, int factor) {
    const long r = std::lround(static_cast<double>(x) / factor);
    return static_cast<int>(r) * factor;
}

}  // namespace

// ──────────────────────────────────────────────────────────────────────────
// smart_resize — verbatim translation of HF (image_processing_qwen2_vl.py).
// ──────────────────────────────────────────────────────────────────────────
void smart_resize(int height, int width, int factor,
                  int min_pixels, int max_pixels,
                  int& resized_height, int& resized_width) {
    if (height <= 0 || width <= 0) {
        fail_pre("smart_resize: non-positive image dimension (" +
                 std::to_string(height) + "x" + std::to_string(width) + ")");
    }
    if (factor <= 0) {
        fail_pre("smart_resize: factor must be positive");
    }
    const int hi = std::max(height, width);
    const int lo = std::min(height, width);
    if (static_cast<double>(hi) / static_cast<double>(lo) > 200.0) {
        fail_pre("smart_resize: absolute aspect ratio must be smaller than 200");
    }

    int h_bar = round_to_multiple(height, factor);
    int w_bar = round_to_multiple(width,  factor);
    const std::int64_t hw = static_cast<std::int64_t>(h_bar) * w_bar;

    if (hw > max_pixels) {
        const double beta = std::sqrt(
            (static_cast<double>(height) * width) / max_pixels);
        h_bar = std::max(factor,
            static_cast<int>(std::floor(height / beta / factor)) * factor);
        w_bar = std::max(factor,
            static_cast<int>(std::floor(width  / beta / factor)) * factor);
    } else if (hw < min_pixels) {
        const double beta = std::sqrt(
            static_cast<double>(min_pixels) / (height * width));
        h_bar = static_cast<int>(std::ceil(height * beta / factor)) * factor;
        w_bar = static_cast<int>(std::ceil(width  * beta / factor)) * factor;
    }
    resized_height = h_bar;
    resized_width  = w_bar;
}

// ──────────────────────────────────────────────────────────────────────────
// preprocess_image
// ──────────────────────────────────────────────────────────────────────────
PreprocessedImage preprocess_image(const float* image_chw, int H, int W,
                                    const PreprocessConfig& cfg) {
    if (!image_chw) fail_pre("null image pointer");
    if (H <= 0 || W <= 0) fail_pre("non-positive image dimension");

    const int C   = 3;
    const int P   = cfg.patch_size;
    const int m   = cfg.merge_size;
    const int tps = cfg.temporal_patch_size;
    if (P <= 0 || m <= 0 || tps <= 0) fail_pre("non-positive patch config");

    // 1. smart_resize.
    int rH = 0, rW = 0;
    smart_resize(H, W, cfg.factor(), cfg.min_pixels, cfg.max_pixels, rH, rW);

    // 2. bilinear resize (per-channel, align_corners=False center-pixel rule).
    std::vector<float> resized(
        static_cast<std::size_t>(C) * rH * rW);
    broimage::resize_chw_f32(image_chw, W, H, C,
                             resized.data(), rW, rH,
                             broimage::Filter::Bilinear);

    // 3. normalise: (x - mean) / std per channel.
    {
        const std::size_t plane = static_cast<std::size_t>(rH) * rW;
        for (int c = 0; c < C; ++c) {
            const float mean    = cfg.mean[c];
            const float inv_std = 1.0f / cfg.std_[c];
            float* p = resized.data() + static_cast<std::size_t>(c) * plane;
            for (std::size_t i = 0; i < plane; ++i) {
                p[i] = (p[i] - mean) * inv_std;
            }
        }
    }

    // 4. patchify.
    //
    // HF order (image_processing_qwen2_vl.py, Qwen2VLImageProcessor._preprocess):
    //   patches = patches.reshape(B, C, gH/m, m, P, gW/m, m, P)
    //   patches = patches.permute(0, 2, 5, 3, 6, 1, 4, 7)
    //        => (B, gH/m, gW/m, m, m, C, P, P)
    //   flatten_patches = patches.unsqueeze(6).expand(
    //                       -1,-1,-1,-1,-1,-1, tps, -1, -1)
    //                   .reshape(B, gH*gW, C*tps*P*P)
    //
    // For B = 1 the outermost flatten goes (gH/m, gW/m, m, m) — merger-block-
    // major.  Per-patch innermost layout is (C, tps, P, P) row-major, with
    // every `t` slice a duplicate of the single static frame.
    const int gH = rH / P;
    const int gW = rW / P;
    if (gH % m != 0 || gW % m != 0) {
        // smart_resize's factor == patch*merge guarantees this; assert anyway.
        fail_pre("internal: grid not divisible by merge_size after smart_resize");
    }
    const int num_patches      = gH * gW;
    const int per_patch_floats = C * tps * P * P;

    bt::Tensor patches = bt::Tensor::empty_on(
        bt::Device::CPU, num_patches, per_patch_floats, bt::Dtype::FP32);
    float* out = patches.host_f32_mut();

    const int gH_m = gH / m;
    const int gW_m = gW / m;

    // src(c, py_g, px_g) where py_g = block_row*m*P + mi*P + py
    //                   px_g = block_col*m*P + mj*P + px
    //                 = resized[c, py_g, px_g] (CHW row-major over rH*rW).
    const std::size_t plane = static_cast<std::size_t>(rH) * rW;
    const float* src = resized.data();

    std::size_t patch_idx = 0;
    for (int br = 0; br < gH_m; ++br) {
        for (int bc = 0; bc < gW_m; ++bc) {
            for (int mi = 0; mi < m; ++mi) {
                for (int mj = 0; mj < m; ++mj) {
                    float* dst = out + patch_idx * per_patch_floats;
                    const int py_base = (br * m + mi) * P;
                    const int px_base = (bc * m + mj) * P;

                    for (int c = 0; c < C; ++c) {
                        const float* plane_c =
                            src + static_cast<std::size_t>(c) * plane;
                        // tps duplicates of the same frame.
                        for (int t = 0; t < tps; ++t) {
                            for (int py = 0; py < P; ++py) {
                                const float* row =
                                    plane_c + (py_base + py) * rW + px_base;
                                for (int px = 0; px < P; ++px) {
                                    *dst++ = row[px];
                                }
                            }
                        }
                    }
                    ++patch_idx;
                }
            }
        }
    }

    PreprocessedImage img;
    img.patches    = std::move(patches);
    img.grid_t     = 1;
    img.grid_h     = gH;
    img.grid_w     = gW;
    img.merge_size = m;
    return img;
}

// ──────────────────────────────────────────────────────────────────────────
// build_mrope_position_ids
// ──────────────────────────────────────────────────────────────────────────
//
// Translation of Qwen2VLModel.get_rope_index for the single-batch, image-only
// case. The HF function takes `mm_token_type_ids`; we recover the same
// modality grouping by scanning for runs of `image_token_id` in the prompt.
//
// Per-image vision sub-positions follow get_vision_position_ids:
//   llm_grid_t = grid_t = 1
//   llm_grid_h = grid_h / merge_size
//   llm_grid_w = grid_w / merge_size
//   position_w = arange(llm_grid_w).repeat(llm_grid_h * llm_grid_t)
//   position_h = arange(llm_grid_h).repeat_interleave(llm_grid_w).repeat(llm_grid_t)
//   position_t = arange(llm_grid_t).repeat_interleave(llm_grid_h*llm_grid_w)
// All three are offset by `current_pos`. After the image span advances
// `current_pos += max(llm_grid_h, llm_grid_w)`.
MRopePositions build_mrope_position_ids(
    const std::vector<int>& tokens,
    const std::vector<PreprocessedImage>& images,
    int image_token_id, int /*vision_start_token_id*/) {

    MRopePositions out;
    out.t.reserve(tokens.size());
    out.h.reserve(tokens.size());
    out.w.reserve(tokens.size());

    int64_t current_pos = 0;
    std::size_t img_idx = 0;
    const std::size_t N = tokens.size();
    std::size_t i = 0;
    while (i < N) {
        if (tokens[i] == image_token_id) {
            // Find run length.
            std::size_t j = i;
            while (j < N && tokens[j] == image_token_id) ++j;
            const std::size_t run_len = j - i;

            if (img_idx >= images.size()) {
                fail_rope("more <|image_pad|> runs in tokens than images supplied");
            }
            const PreprocessedImage& im = images[img_idx++];
            const int llm_grid_t = im.grid_t;
            const int llm_grid_h = im.grid_h / im.merge_size;
            const int llm_grid_w = im.grid_w / im.merge_size;
            const std::size_t expected =
                static_cast<std::size_t>(llm_grid_t) * llm_grid_h * llm_grid_w;
            if (run_len != expected) {
                fail_rope("image_token run length " + std::to_string(run_len) +
                          " != grid_t*(grid_h/m)*(grid_w/m) = " +
                          std::to_string(expected));
            }

            // Emit vision positions.
            for (int t = 0; t < llm_grid_t; ++t) {
                for (int h = 0; h < llm_grid_h; ++h) {
                    for (int w = 0; w < llm_grid_w; ++w) {
                        out.t.push_back(current_pos + t);
                        out.h.push_back(current_pos + h);
                        out.w.push_back(current_pos + w);
                    }
                }
            }
            current_pos += std::max(llm_grid_h, llm_grid_w);
            i = j;
        } else {
            // Text run: append until the next image_token_id or end.
            std::size_t j = i;
            while (j < N && tokens[j] != image_token_id) ++j;
            const std::size_t run_len = j - i;
            for (std::size_t k = 0; k < run_len; ++k) {
                const int64_t p = current_pos + static_cast<int64_t>(k);
                out.t.push_back(p);
                out.h.push_back(p);
                out.w.push_back(p);
            }
            current_pos += static_cast<int64_t>(run_len);
            i = j;
        }
    }

    if (img_idx != images.size()) {
        fail_rope("token sequence contains fewer image runs (" +
                  std::to_string(img_idx) + ") than images supplied (" +
                  std::to_string(images.size()) + ")");
    }

    // mrope_position_delta = max(positions) + 1 - seq_len.
    int64_t max_pos = 0;
    for (std::size_t k = 0; k < out.t.size(); ++k) {
        max_pos = std::max(max_pos,
            std::max(out.t[k], std::max(out.h[k], out.w[k])));
    }
    out.delta = max_pos + 1 - static_cast<int64_t>(N);
    return out;
}

}  // namespace brolm::qwen35
