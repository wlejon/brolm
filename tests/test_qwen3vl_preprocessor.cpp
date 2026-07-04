// Tests for brolm::qwen3vl preprocessor re-exports.
//
// qwen3vl_preprocessor.h is a deliberate re-export of brolm::qwen35's
// preprocessor (see that header's comment for why) — this test just confirms
// the re-exported names resolve to the same behavior under the qwen3vl
// namespace, not a duplicate of qwen35's own (more exhaustive) preprocessor
// test suite.

#include "brolm/qwen3vl_preprocessor.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

using brolm::qwen3vl::MRopePositions;
using brolm::qwen3vl::PreprocessConfig;
using brolm::qwen3vl::PreprocessedImage;
using brolm::qwen3vl::preprocess_image;
using brolm::qwen3vl::smart_resize;
using brolm::qwen3vl::build_mrope_position_ids;

namespace {

void test_smart_resize() {
    const PreprocessConfig cfg;   // factor=32, min=65536, max=16777216
    int rH = 0, rW = 0;
    smart_resize(1024, 1024, cfg.factor(), cfg.min_pixels, cfg.max_pixels, rH, rW);
    assert(rH == 1024);
    assert(rW == 1024);
    std::printf("[ok] smart_resize passthrough\n");
}

void test_preprocess_image_shape() {
    const PreprocessConfig cfg;
    const int H = 64, W = 64, C = 3;
    std::vector<float> img(static_cast<std::size_t>(C) * H * W);
    for (int c = 0; c < C; ++c)
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x)
                img[(static_cast<std::size_t>(c) * H + y) * W + x] =
                    static_cast<float>(x) / static_cast<float>(W - 1);

    PreprocessedImage pre = preprocess_image(img.data(), H, W, cfg);
    assert(pre.grid_t == 1);
    assert(pre.grid_h > 0 && pre.grid_w > 0);
    assert(pre.patches.rows == pre.grid_t * pre.grid_h * pre.grid_w);
    assert(pre.num_image_tokens() ==
           (pre.grid_h / cfg.merge_size) * (pre.grid_w / cfg.merge_size));
    std::printf("[ok] preprocess_image: grid=(1,%d,%d) tokens=%d\n",
                pre.grid_h, pre.grid_w, pre.num_image_tokens());
}

void test_mrope_position_ids() {
    const PreprocessConfig cfg;
    const int H = 64, W = 64, C = 3;
    std::vector<float> img(static_cast<std::size_t>(C) * H * W, 0.5f);
    PreprocessedImage pre = preprocess_image(img.data(), H, W, cfg);

    const int image_pad_id = 151655, vision_start_id = 151652;
    std::vector<int> tokens;
    tokens.push_back(1);              // leading text token
    tokens.push_back(vision_start_id);
    for (int i = 0; i < pre.num_image_tokens(); ++i) tokens.push_back(image_pad_id);
    tokens.push_back(2);              // trailing text token

    std::vector<PreprocessedImage> images;
    images.push_back(std::move(pre));

    MRopePositions mp = build_mrope_position_ids(tokens, images, image_pad_id,
                                                 vision_start_id);
    assert(mp.t.size() == tokens.size());
    assert(mp.h.size() == tokens.size());
    assert(mp.w.size() == tokens.size());
    std::printf("[ok] build_mrope_position_ids: delta=%lld\n",
                static_cast<long long>(mp.delta));
}

}  // namespace

int main() {
    test_smart_resize();
    test_preprocess_image_shape();
    test_mrope_position_ids();
    return 0;
}
