// Tests for brolm::qwen35 image preprocessor.
//
// Covers:
//   1. smart_resize against hand-computed cases (cited inline).
//   2. preprocess_image on a synthetic 3×64×64 gradient: validate grid shape,
//      patch tensor shape, and that mean is roughly zero after (x-0.5)/0.5
//      normalisation of a centred gradient.
//   3. build_mrope_position_ids on a small synthetic prompt: validate stream
//      lengths and that the text token after the image span jumps to
//      max+1 on all three axes.

#include "brolm/qwen35_preprocessor.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

using brolm::qwen35::MRopePositions;
using brolm::qwen35::PreprocessConfig;
using brolm::qwen35::PreprocessedImage;
using brolm::qwen35::preprocess_image;
using brolm::qwen35::smart_resize;
using brolm::qwen35::build_mrope_position_ids;

namespace {

// HF smart_resize hand-computed expectations. Factor = patch*merge = 16*2 = 32.
//   100 x 200: round_to_multiple(100,32)=96, (200,32)=192. Total 96*192=18432.
//     min_pixels=65536, so we scale up. beta = sqrt(65536 / (100*200))
//                                             = sqrt(3.2768)   = 1.8102...
//     h = ceil(100 * beta / 32) * 32 = ceil(5.657) * 32 = 6*32 = 192
//     w = ceil(200 * beta / 32) * 32 = ceil(11.31) * 32 = 12*32 = 384
//     Total 192*384=73728 in [65536, 16777216]. OK.
//
//   1024 x 1024: rounded already; 1024*1024 = 1048576 in range; unchanged.
//
//   8192 x 8192: rounded already; 8192*8192 = 67108864 > 16777216.
//     beta = sqrt((8192*8192)/16777216) = sqrt(4) = 2
//     h = max(32, floor(8192/2/32)*32) = floor(128)*32 = 128*32 = 4096
//     w = 4096. Total 4096*4096 = 16777216. OK.
void test_smart_resize_basic() {
    const PreprocessConfig cfg;          // factor=32, min=65536, max=16777216
    int rH = 0, rW = 0;

    smart_resize(100, 200, cfg.factor(), cfg.min_pixels, cfg.max_pixels, rH, rW);
    assert(rH == 192);
    assert(rW == 384);

    smart_resize(1024, 1024, cfg.factor(), cfg.min_pixels, cfg.max_pixels, rH, rW);
    assert(rH == 1024);
    assert(rW == 1024);

    smart_resize(8192, 8192, cfg.factor(), cfg.min_pixels, cfg.max_pixels, rH, rW);
    assert(rH == 4096);
    assert(rW == 4096);

    // Aspect ratio > 200 must throw.
    bool threw = false;
    try {
        smart_resize(10, 5000, cfg.factor(), cfg.min_pixels, cfg.max_pixels, rH, rW);
    } catch (const std::runtime_error&) { threw = true; }
    assert(threw);

    std::printf("[ok] smart_resize basic cases\n");
}

void test_preprocess_image_gradient() {
    // Build a 3×64×64 gradient image with values in [0, 1].
    //   plane[c, y, x] = (x + y) / (2*(W-1))    →   range exactly [0, 1].
    const int H = 64, W = 64, C = 3;
    std::vector<float> img(static_cast<std::size_t>(C) * H * W);
    for (int c = 0; c < C; ++c) {
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                img[(c * H + y) * W + x] =
                    static_cast<float>(x + y) /
                    static_cast<float>(2 * (W - 1));
            }
        }
    }

    PreprocessConfig cfg;
    // 64x64 = 4096 < min_pixels (65536) → smart_resize upscales.
    // factor=32; round_to_multiple(64,32)=64, 64*64=4096<65536.
    //   beta = sqrt(65536 / 4096) = sqrt(16) = 4
    //   h = ceil(64*4/32)*32 = ceil(8)*32 = 256
    //   w = same = 256
    int rH = 0, rW = 0;
    smart_resize(H, W, cfg.factor(), cfg.min_pixels, cfg.max_pixels, rH, rW);
    assert(rH == 256);
    assert(rW == 256);

    PreprocessedImage out = preprocess_image(img.data(), H, W, cfg);
    assert(out.grid_t == 1);
    assert(out.grid_h == rH / cfg.patch_size);  // 256/16 = 16
    assert(out.grid_w == rW / cfg.patch_size);  // 16
    assert(out.merge_size == cfg.merge_size);

    // Patch tensor shape: (gH*gW, C*tps*P*P) = (256, 3*2*16*16) = (256, 1536).
    const int expected_patches = out.grid_h * out.grid_w;
    const int per_patch =
        C * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size;
    assert(out.patches.rows == expected_patches);
    assert(out.patches.cols == per_patch);

    // num_image_tokens = 1 * (16/2) * (16/2) = 64.
    assert(out.num_image_tokens() == 64);

    // After (x - 0.5) / 0.5, a [0,1] gradient becomes a [-1,1] gradient. Its
    // mean should be close to zero (the gradient is symmetric about 0.5 over
    // the full image; small numerical drift from bilinear resize is fine).
    const float* p = out.patches.host_f32();
    const std::size_t n =
        static_cast<std::size_t>(expected_patches) * per_patch;
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) sum += p[i];
    const double mean = sum / static_cast<double>(n);
    assert(std::fabs(mean) < 0.02);

    // Spot-check: the duplicated temporal-patch frames are identical. For
    // patch 0, frame-0 and frame-1 of channel 0 should match byte-for-byte.
    const int P = cfg.patch_size;
    {
        const float* patch0 = p;                       // first patch
        const float* frame0 = patch0 + 0 * P * P;      // c=0, t=0
        const float* frame1 = patch0 + 1 * P * P;      // c=0, t=1
        for (int k = 0; k < P * P; ++k) {
            if (frame0[k] != frame1[k]) {
                std::fprintf(stderr,
                    "temporal-patch duplicate mismatch at k=%d: %.7f != %.7f\n",
                    k, frame0[k], frame1[k]);
                std::abort();
            }
        }
    }

    std::printf("[ok] preprocess_image gradient: grid=(1,%d,%d), tokens=%d, mean=%.5f\n",
                out.grid_h, out.grid_w, out.num_image_tokens(), mean);
}

void test_mrope_positions_simple() {
    // Synthetic prompt:  [10, 20,  VS, IP, IP, IP, IP,  VE, 30]
    //                     ^text^   ^---image span---^   ^text^
    // For a 1×4×4 grid with merge=2 → llm_grid = (1, 2, 2) → 4 image tokens.
    // We don't include VS/VE in the position-id logic specially (HF treats
    // them as plain text tokens too — they fall in the "text" run).
    const int VS = 200;
    const int VE = 201;
    const int IP = 100;
    const std::vector<int> tokens = {10, 20, VS, IP, IP, IP, IP, VE, 30};

    PreprocessedImage im;
    im.grid_t = 1;
    im.grid_h = 4;       // post-merger 4/2 = 2
    im.grid_w = 4;       // post-merger 4/2 = 2
    im.merge_size = 2;
    // num_image_tokens = 1 * 2 * 2 = 4
    assert(im.num_image_tokens() == 4);

    std::vector<PreprocessedImage> images;
    images.push_back(std::move(im));

    MRopePositions p = build_mrope_position_ids(tokens, images, IP, VS);

    assert(p.t.size() == tokens.size());
    assert(p.h.size() == tokens.size());
    assert(p.w.size() == tokens.size());

    // tokens[0..2] are text → (0,0,0), (1,1,1), (2,2,2).
    // current_pos starts 0, advances to 3 after VS.
    for (int i = 0; i < 3; ++i) {
        assert(p.t[i] == i);
        assert(p.h[i] == i);
        assert(p.w[i] == i);
    }

    // Image span: 4 tokens at current_pos=3.
    //   llm_grid_h = llm_grid_w = 2, llm_grid_t = 1.
    //   For (t=0, h=0, w=0..1), (t=0, h=1, w=0..1).
    //   t-axis: all 3+0 = 3.
    //   h-axis: 3+0, 3+0, 3+1, 3+1.
    //   w-axis: 3+0, 3+1, 3+0, 3+1.
    const int64_t expected_t[4] = {3, 3, 3, 3};
    const int64_t expected_h[4] = {3, 3, 4, 4};
    const int64_t expected_w[4] = {3, 4, 3, 4};
    for (int k = 0; k < 4; ++k) {
        assert(p.t[3 + k] == expected_t[k]);
        assert(p.h[3 + k] == expected_h[k]);
        assert(p.w[3 + k] == expected_w[k]);
    }

    // Post-image text: current_pos += max(2, 2) = 5. So VE token and 30 token
    // get (5,5,5) and (6,6,6).
    assert(p.t[7] == 5); assert(p.h[7] == 5); assert(p.w[7] == 5);
    assert(p.t[8] == 6); assert(p.h[8] == 6); assert(p.w[8] == 6);

    // delta = max(positions)+1 - seq_len. max=6, len=9, delta = 7-9 = -2.
    assert(p.delta == -2);

    // The post-image text token's coords are strictly greater than the max
    // image coord — i.e., the "max+1 on all three axes" property the spec
    // calls out. Max over image span = 4 on each of h,w; t-max within span =
    // 3. After image, VE token = 5 on every axis > 4. Confirm explicitly.
    int64_t img_max_t = 0, img_max_h = 0, img_max_w = 0;
    for (int k = 0; k < 4; ++k) {
        img_max_t = std::max(img_max_t, p.t[3 + k]);
        img_max_h = std::max(img_max_h, p.h[3 + k]);
        img_max_w = std::max(img_max_w, p.w[3 + k]);
    }
    assert(p.t[7] == std::max({img_max_t, img_max_h, img_max_w}) + 1);
    assert(p.h[7] == std::max({img_max_t, img_max_h, img_max_w}) + 1);
    assert(p.w[7] == std::max({img_max_t, img_max_h, img_max_w}) + 1);

    std::printf("[ok] build_mrope_position_ids: image span (1,2,2), delta=%lld\n",
                static_cast<long long>(p.delta));
}

void test_mrope_mismatch_throws() {
    // Run length 3 but image expects 4 → throw.
    const std::vector<int> tokens = {100, 100, 100};  // image_token_id=100
    PreprocessedImage im;
    im.grid_t = 1; im.grid_h = 4; im.grid_w = 4; im.merge_size = 2;
    std::vector<PreprocessedImage> images;
    images.push_back(std::move(im));

    bool threw = false;
    try {
        (void)build_mrope_position_ids(tokens, images, 100, 200);
    } catch (const std::runtime_error&) { threw = true; }
    assert(threw);
    std::printf("[ok] mrope rejects mismatched run length\n");
}

void test_real_checkpoint_if_present() {
    const char* dir = std::getenv("BROLM_QWEN35_DIR");
    if (!dir) {
        std::printf("[skip] BROLM_QWEN35_DIR not set (real preprocessor_config check)\n");
        return;
    }
    // We don't load the JSON here (no JSON dep wired into this TU); the
    // qwen35_config test already validates the file. This stub mirrors the
    // env-gated pattern of test_qwen35_config.cpp so future expansion (e.g.
    // loading a real test image) lands in a uniform place.
    std::printf("[ok] BROLM_QWEN35_DIR set; deferring real-image validation\n");
}

}  // namespace

int main() {
    test_smart_resize_basic();
    test_preprocess_image_gradient();
    test_mrope_positions_simple();
    test_mrope_mismatch_throws();
    test_real_checkpoint_if_present();
    return 0;
}
