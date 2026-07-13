#include "brolm/clip_score.h"
#include "brotensor/safetensors.h"
#include "brolm/detail/device.h"
#include "brolm/detail/compute.h"
#include "brolm/detail/weights.h"

#include "brotensor/ops.h"
#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

#include "broimage/geometric.h"
#include "broimage/normalize.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace brolm::clip_score {

namespace bt = ::brotensor;
namespace st = ::brotensor::safetensors;

namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("clip_score::CLIPScorer: " + msg);
}

// Download a compute-dtype tensor's contents into a host FP32 vector,
// converting from FP16 bits on a GPU backend.
std::vector<float> download_compute(const bt::Tensor& t) {
    if (t.dtype == bt::Dtype::FP16) {
        std::vector<std::uint16_t> bits(static_cast<std::size_t>(t.size()));
        t.copy_to_host_fp16(bits.data());
        std::vector<float> out(bits.size());
        for (std::size_t i = 0; i < bits.size(); ++i) {
            out[i] = bt::fp16_bits_to_fp32(bits[i]);
        }
        return out;
    }
    return t.to_host_vector();
}

float l2_normalise_in_place(std::vector<float>& v) {
    double s = 0.0;
    for (float x : v) s += static_cast<double>(x) * static_cast<double>(x);
    const float n = static_cast<float>(std::sqrt(s));
    if (n > 0.0f) {
        const float inv = 1.0f / n;
        for (float& x : v) x *= inv;
    }
    return n;
}

}  // namespace

CLIPScorer::CLIPScorer(const clip::Tokenizer& tokenizer,
                       clip::TextEncoder& text_encoder,
                       clip_image::ImageEncoder& image_encoder,
                       Config cfg)
    : tok_(tokenizer), text_enc_(text_encoder), image_enc_(image_encoder),
      cfg_(cfg) {}

void CLIPScorer::load_projections(const st::File& f, const std::string& prefix) {
    const int P  = cfg_.projection_dim;
    const int Dv = image_enc_.config().hidden_dim;
    const int Dt = text_enc_.config().hidden_dim;

    brolm::detail::weights::SafetensorsSource src({&f}, prefix);
    src.upload_compute_checked("visual_projection.weight",
                               P, Dv, visual_proj_W_, "visual_projection.weight");
    src.upload_compute_checked("text_projection.weight",
                               P, Dt, text_proj_W_, "text_projection.weight");
}

void CLIPScorer::set_prompt(std::string_view prompt) {
    if (visual_proj_W_.size() == 0 || text_proj_W_.size() == 0) {
        fail("set_prompt: projections not loaded — call load_projections first");
    }

    auto ids = tok_.encode(prompt);
    const int L  = static_cast<int>(ids.size());
    const int Dt = text_enc_.config().hidden_dim;
    if (L != text_enc_.config().max_position) {
        fail("set_prompt: tokenizer produced unexpected sequence length");
    }

    // CLIP pooling rule: take the EOS token's hidden state. CLIP pads with
    // EOS (49407), and BOS is 49406; argmax over IDs therefore lands on
    // the first EOS-padded position — i.e. the token *just after* the last
    // real content token. This matches the official CLIP pooling logic.
    int eos_idx = 0;
    int32_t best = -1;
    for (int i = 0; i < L; ++i) {
        if (ids[static_cast<std::size_t>(i)] > best) {
            best = ids[static_cast<std::size_t>(i)];
            eos_idx = i;
        }
    }

    text_enc_.forward(ids.data(), text_hidden_);  // (L, Dt) at compute dtype

    // Pool: copy row eos_idx into text_pooled_ (1, Dt).
    detail::resize_like(text_pooled_, 1, Dt, compute_dtype(), text_hidden_.device);
    bt::copy_d2d(text_hidden_, /*src_off=*/eos_idx * Dt,
                     text_pooled_,  /*dst_off=*/0,
                     /*count=*/Dt);

    // Project to shared space: (1, P) = text_pooled_ @ text_proj_W_.T
    detail::linear_batched(text_proj_W_, /*bias=*/nullptr,
                           text_pooled_, text_proj_);

    bt::sync_all();
    text_feat_ = download_compute(text_proj_);
    l2_normalise_in_place(text_feat_);
}

std::vector<float> CLIPScorer::encode_image(const std::vector<float>& image,
                                            int H, int W) {
    auto pixel_vals = preprocess_(image, H, W);
    const int S  = image_enc_.config().image_size;
    const int C  = image_enc_.config().in_channels;

    pixels_dev_ = detail::upload_host(pixel_vals.data(), 1, C * S * S);
    image_enc_.forward(pixels_dev_, img_cls_);

    // (1, P) = img_cls_ @ visual_proj_W_.T
    detail::linear_batched(visual_proj_W_, /*bias=*/nullptr,
                           img_cls_, img_proj_);

    bt::sync_all();
    std::vector<float> img_feat = download_compute(img_proj_);
    l2_normalise_in_place(img_feat);
    return img_feat;
}

float CLIPScorer::score(const std::vector<float>& image, int H, int W) {
    if (text_feat_.empty()) {
        fail("score: set_prompt was not called");
    }

    const std::vector<float> img_feat = encode_image(image, H, W);
    const int P = cfg_.projection_dim;

    double dot = 0.0;
    for (int i = 0; i < P; ++i) {
        dot += static_cast<double>(img_feat[static_cast<std::size_t>(i)]) *
               static_cast<double>(text_feat_[static_cast<std::size_t>(i)]);
    }
    return static_cast<float>(dot);
}

std::vector<float> CLIPScorer::preprocess_(
    const std::vector<float>& image, int H, int W) const {

    const int S = image_enc_.config().image_size;       // 224
    const int C = image_enc_.config().in_channels;      // 3
    if (static_cast<int>(image.size()) != C * H * W) {
        fail("preprocess: image size mismatch (expected " +
             std::to_string(C * H * W) + ", got " +
             std::to_string(image.size()) + ")");
    }

    // Resize per channel (bilinear, align_corners=False) then CLIP-normalise.
    // Input is in [-1, 1] (the SD pipeline output); fold the [-1,1] -> [0,1]
    // rescale into the per-channel mean/std:
    //   y = (((x + 1) * 0.5) - mean) / std
    //     = (x - (2*mean - 1)) / (2*std)
    // so image_normalize_nchw_f32 with adjusted stats produces the same result.
    std::vector<float> out(static_cast<std::size_t>(C) *
                           static_cast<std::size_t>(S) * S);
    broimage::resize_chw_f32(image.data(), W, H, C,
                             out.data(),   S, S,
                             broimage::Filter::Bilinear);

    float mean_eff[3], std_eff[3];
    for (int c = 0; c < C; ++c) {
        mean_eff[c] = 2.0f * cfg_.mean[c] - 1.0f;
        std_eff[c]  = 2.0f * cfg_.std_[c];
    }
    broimage::image_normalize_nchw_f32(out.data(), mean_eff, std_eff,
                                       /*N=*/1, C, S, S, out.data());
    return out;
}

}  // namespace brolm::clip_score
