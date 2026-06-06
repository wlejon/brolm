#include "brolm/mistral3_preprocessor.h"

#include "broimage/geometric.h"
#include "brotensor/tensor.h"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace brolm::mistral3 {

namespace bt = ::brotensor;

namespace {
[[noreturn]] void fail_pre(const std::string& msg) {
    throw std::runtime_error("mistral3::preprocess_image: " + msg);
}
}  // namespace

PreprocessConfig PreprocessConfig::from_config(const Mistral3Config& cfg) {
    PreprocessConfig p;
    p.patch_size         = cfg.vision.patch_size;
    p.spatial_merge_size = cfg.spatial_merge_size;
    p.longest_edge       = cfg.vision.image_size;
    p.num_channels       = cfg.vision.num_channels;
    return p;
}

// ─── resize geometry ────────────────────────────────────────────────────────
//
// Pixtral `get_resize_output_image_size`: scale the image down (never up) so the
// longest side fits `longest_edge`, rounding the scaled dimensions UP to whole
// patches. brolm rounds to whole `factor` (= patch * merge) units instead, so
// the resulting patch grid is divisible by the spatial merge (the projector and
// the [IMG] span both require that). For images whose natural patch grid is even
// this is identical to Pixtral; otherwise it differs by at most one merge block.
// TODO: confirm Mistral 3.1's processor rounding against the real checkpoint.
void compute_resized_size(int H, int W, int longest_edge, int factor,
                          int& resized_h, int& resized_w) {
    if (H <= 0 || W <= 0) fail_pre("non-positive image dimension");
    if (longest_edge <= 0 || factor <= 0) fail_pre("longest_edge and factor must be positive");

    double h = static_cast<double>(H);
    double w = static_cast<double>(W);
    const double ratio = std::max(h / longest_edge, w / longest_edge);
    if (ratio > 1.0) {
        h = std::ceil(h / ratio);
        w = std::ceil(w / ratio);
    }

    auto round_up = [&](double v) {
        int blocks = static_cast<int>(std::ceil(v / static_cast<double>(factor)));
        if (blocks < 1) blocks = 1;   // at least one merge block per side
        return blocks * factor;
    };
    resized_h = round_up(h);
    resized_w = round_up(w);
}

// ─── preprocess ─────────────────────────────────────────────────────────────

PreprocessedImage preprocess_image(const float* image_chw, int H, int W,
                                   const PreprocessConfig& cfg) {
    if (image_chw == nullptr) fail_pre("null image pointer");
    const int C = cfg.num_channels;
    const int P = cfg.patch_size;
    const int m = cfg.spatial_merge_size;
    if (C <= 0 || P <= 0 || m <= 0) fail_pre("non-positive patch/channel config");

    // 1. resized geometry (multiple of patch*merge).
    int rH = 0, rW = 0;
    compute_resized_size(H, W, cfg.longest_edge, cfg.factor(), rH, rW);

    // 2. bicubic resize, per-channel (CHW). broimage Catmull-Rom ≈ PIL BICUBIC.
    std::vector<float> resized(
        static_cast<std::size_t>(C) * static_cast<std::size_t>(rH) * static_cast<std::size_t>(rW));
    broimage::resize_chw_f32(image_chw, W, H, C,
                             resized.data(), rW, rH,
                             broimage::Filter::Bicubic);

    // 3. normalise (x - mean)/std per channel, in place.
    const std::size_t plane = static_cast<std::size_t>(rH) * static_cast<std::size_t>(rW);
    for (int c = 0; c < C; ++c) {
        const float mean = cfg.mean[c];
        const float inv_std = 1.0f / cfg.std_[c];
        float* p = resized.data() + static_cast<std::size_t>(c) * plane;
        for (std::size_t i = 0; i < plane; ++i) p[i] = (p[i] - mean) * inv_std;
    }

    // 4. patchify. grid is row-major (h, w); per-patch layout is (C, P, P)
    //    row-major: feature f = c*P*P + ph*P + pw. This matches the tower's
    //    flattened patch_conv weight (Conv2d (D, C, P, P) → (D, C*P*P)).
    const int gH = rH / P;
    const int gW = rW / P;
    if (gH % m != 0 || gW % m != 0) {
        fail_pre("internal: grid not divisible by merge_size after resize");
    }
    const int num_patches = gH * gW;
    const int per_patch   = C * P * P;

    bt::Tensor patches = bt::Tensor::empty_on(bt::Device::CPU, num_patches, per_patch, bt::Dtype::FP32);
    float* out = patches.host_f32_mut();
    const float* src = resized.data();   // CHW row-major over (C, rH, rW)

    for (int pr = 0; pr < gH; ++pr) {
        for (int pc = 0; pc < gW; ++pc) {
            const std::size_t pidx = static_cast<std::size_t>(pr) * gW + static_cast<std::size_t>(pc);
            float* dst = out + pidx * per_patch;
            for (int c = 0; c < C; ++c) {
                const float* cplane = src + static_cast<std::size_t>(c) * plane;
                for (int ph = 0; ph < P; ++ph) {
                    const int y = pr * P + ph;
                    const float* row = cplane + static_cast<std::size_t>(y) * rW;
                    float* drow = dst + static_cast<std::size_t>(c) * P * P + static_cast<std::size_t>(ph) * P;
                    for (int pw = 0; pw < P; ++pw) {
                        drow[pw] = row[pc * P + pw];
                    }
                }
            }
        }
    }

    PreprocessedImage img;
    img.patches = std::move(patches);
    img.grid_h  = gH;
    img.grid_w  = gW;
    img.merge   = m;
    return img;
}

// ─── token span ─────────────────────────────────────────────────────────────

std::vector<int> build_image_token_span(const PreprocessedImage& img,
                                        int image_token_id,
                                        int image_break_id,
                                        int image_end_id) {
    const int Lh = img.tokens_h();
    const int Lw = img.tokens_w();
    if (Lh <= 0 || Lw <= 0) fail_pre("build_image_token_span: empty token grid");

    std::vector<int> out;
    out.reserve(static_cast<std::size_t>(Lh) * (Lw + 1));
    for (int r = 0; r < Lh; ++r) {
        for (int c = 0; c < Lw; ++c) out.push_back(image_token_id);
        out.push_back(r + 1 < Lh ? image_break_id : image_end_id);
    }
    return out;
}

}  // namespace brolm::mistral3
