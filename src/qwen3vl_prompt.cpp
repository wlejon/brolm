// Shared Qwen3-VL prompt-assembly helpers. See qwen3vl_prompt.h.

#include "brolm/qwen3vl_prompt.h"

#include "brolm/detail/compute.h"
#include "brolm/qwen3vl_vl.h"  // ImageInput's complete definition

#include "brotensor/ops.h"
#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace brolm::qwen3vl {

namespace bt = ::brotensor;

std::vector<int> expand_image_pad(const std::vector<int>& tokens,
                                  const std::vector<PreprocessedImage>& images,
                                  int image_pad_id) {
    std::size_t pad_count = 0;
    for (int t : tokens) if (t == image_pad_id) ++pad_count;
    if (pad_count != images.size()) {
        throw std::runtime_error(
            "qwen3vl::expand_image_pad: prompt has " +
            std::to_string(pad_count) + " <|image_pad|> token(s) but " +
            std::to_string(images.size()) + " image(s) supplied");
    }

    std::vector<int> out;
    std::size_t expanded_extra = 0;
    for (const auto& im : images) {
        expanded_extra += static_cast<std::size_t>(im.num_image_tokens());
    }
    out.reserve(tokens.size() + expanded_extra);

    std::size_t img_idx = 0;
    for (int t : tokens) {
        if (t == image_pad_id) {
            const int n = images[img_idx++].num_image_tokens();
            for (int k = 0; k < n; ++k) out.push_back(image_pad_id);
        } else {
            out.push_back(t);
        }
    }
    return out;
}

std::vector<int> image_pad_run_starts(
    const std::vector<int>& expanded,
    const std::vector<PreprocessedImage>& images, int image_pad_id) {
    std::vector<int> starts;
    starts.reserve(images.size());
    std::size_t img_idx = 0;
    const int T = static_cast<int>(expanded.size());
    int i = 0;
    while (i < T) {
        if (expanded[static_cast<std::size_t>(i)] == image_pad_id) {
            int j = i;
            while (j < T &&
                   expanded[static_cast<std::size_t>(j)] == image_pad_id) {
                ++j;
            }
            if (img_idx >= images.size()) {
                throw std::runtime_error(
                    "qwen3vl::image_pad_run_starts: image_pad run count "
                    "mismatch (internal)");
            }
            const int expected_len = images[img_idx++].num_image_tokens();
            if (j - i != expected_len) {
                throw std::runtime_error(
                    "qwen3vl::image_pad_run_starts: image_pad run length " +
                    std::to_string(j - i) + " != num_image_tokens " +
                    std::to_string(expected_len));
            }
            starts.push_back(i);
            i = j;
        } else {
            ++i;
        }
    }
    if (img_idx != images.size()) {
        throw std::runtime_error(
            "qwen3vl::image_pad_run_starts: fewer image_pad runs than images "
            "(internal)");
    }
    return starts;
}

void splice_vision(bt::Tensor& embeds, int dst_row,
                   const bt::Tensor& vision_out) {
    const int n      = vision_out.rows;
    const int hidden = vision_out.cols;
    if (embeds.cols != hidden) {
        throw std::runtime_error(
            "qwen3vl::splice_vision: hidden mismatch " +
            std::to_string(embeds.cols) + " vs " + std::to_string(hidden));
    }
    if (embeds.dtype != vision_out.dtype) {
        throw std::runtime_error("qwen3vl::splice_vision: dtype mismatch");
    }
    bt::copy_d2d(vision_out, /*src_off=*/0,
                embeds, /*dst_off=*/dst_row * hidden,
                n * hidden);
}

bt::Tensor run_vision_one(VisionTower& tower, const PreprocessConfig& pp,
                          const ImageInput& img, PreprocessedImage& pp_out,
                          std::vector<bt::Tensor>& deepstack_out) {
    if (!img.pixels || img.H <= 0 || img.W <= 0) {
        throw std::runtime_error("qwen3vl::run_vision_one: invalid ImageInput");
    }
    pp_out = preprocess_image(img.pixels, img.H, img.W, pp);

    const float* patch_src = pp_out.patches.host_f32();
    const int N          = pp_out.patches.rows;
    const int patch_cols = pp_out.patches.cols;
    bt::Tensor patches_dev;
    if (brolm::compute_dtype() == bt::Dtype::FP16) {
        std::vector<std::uint16_t> bits(
            static_cast<std::size_t>(N) * patch_cols);
        for (std::size_t i = 0; i < bits.size(); ++i) {
            bits[i] = bt::fp32_to_fp16_bits(patch_src[i]);
        }
        patches_dev = bt::Tensor::from_host_fp16(bits.data(), N, patch_cols);
    } else {
        patches_dev = bt::Tensor::from_host(patch_src, N, patch_cols);
    }

    bt::Tensor out;
    tower.forward(patches_dev, pp_out.grid_t, pp_out.grid_h, pp_out.grid_w,
                 out, deepstack_out);
    return out;
}

}  // namespace brolm::qwen3vl
