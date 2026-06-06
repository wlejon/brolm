// Mistral 3.1 (Pixtral) image preprocessor test.
//
// Covers the host-side geometry + patchify + token-span logic (no model
// weights). Checks:
//   1. compute_resized_size — rounds up to whole patch*merge units, scales
//      large images down to fit longest_edge, and always yields a grid
//      divisible by the spatial merge;
//   2. patch layout — at an identity resize size, a constant-per-channel image
//      lands the right normalized channel constant in each channel block
//      (validates the channel-major c*P²+ph*P+pw within-patch layout), and a
//      horizontal ramp confirms pw is the inner (x) index and column order;
//   3. shape — patches is (grid_h*grid_w, C*P²) and num_image_tokens matches
//      the merged grid;
//   4. build_image_token_span — [IMG] count equals num_image_tokens, with one
//      [IMG_BREAK] between rows and a single trailing [IMG_END].

#include "brolm/mistral3_preprocessor.h"
#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace m3 = brolm::mistral3;
namespace bt = brotensor;

static int g_failures = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        ++g_failures; \
    } \
} while (0)

int main() {
    try {
        bt::init();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "init failed: %s\n", e.what());
        return 1;
    }

    m3::PreprocessConfig cfg;  // patch 14, merge 2, longest_edge 1540, CLIP stats
    const int P = cfg.patch_size;
    const int C = cfg.num_channels;
    const int factor = cfg.factor();  // 28

    // ── 1. resize geometry ────────────────────────────────────────────────
    {
        int rh = 0, rw = 0;
        // small image rounds up to whole factor units.
        m3::compute_resized_size(30, 40, cfg.longest_edge, factor, rh, rw);
        CHECK(rh == 56);  // ceil(30/28)*28
        CHECK(rw == 56);  // ceil(40/28)*28
        CHECK(rh % factor == 0 && rw % factor == 0);
        CHECK((rh / P) % cfg.spatial_merge_size == 0);
        CHECK((rw / P) % cfg.spatial_merge_size == 0);

        // exact-multiple image is unchanged.
        m3::compute_resized_size(56, 28, cfg.longest_edge, factor, rh, rw);
        CHECK(rh == 56 && rw == 28);

        // large image scales down to fit longest_edge, still merge-aligned.
        m3::compute_resized_size(3000, 1500, cfg.longest_edge, factor, rh, rw);
        CHECK(rh <= cfg.longest_edge + factor);
        CHECK(rw <= cfg.longest_edge + factor);
        CHECK(rh % factor == 0 && rw % factor == 0);
        CHECK((rh / P) % cfg.spatial_merge_size == 0);
        CHECK((rw / P) % cfg.spatial_merge_size == 0);
        // longest side should be at the cap (1540 = 55*28).
        CHECK(std::max(rh, rw) == 1540);
    }

    // ── 2 + 3. patch layout + shape at an identity resize size ────────────
    // 28×28 → grid 2×2 (gW=2 so pc ∈ {0,1}); resize is identity at this size.
    {
        const int H = 28, W = 28;
        const int gH = 2, gW = 2;

        // Constant-per-channel image: channel c is filled with value vc.
        const float vc[3] = {0.20f, 0.50f, 0.80f};
        std::vector<float> img(static_cast<std::size_t>(C) * H * W);
        for (int c = 0; c < C; ++c)
            for (int i = 0; i < H * W; ++i)
                img[static_cast<std::size_t>(c) * H * W + i] = vc[c];

        m3::PreprocessedImage out = m3::preprocess_image(img.data(), H, W, cfg);
        CHECK(out.grid_h == gH);
        CHECK(out.grid_w == gW);
        CHECK(out.patches.rows == gH * gW);
        CHECK(out.patches.cols == C * P * P);
        CHECK(out.num_image_tokens() == (gH / cfg.spatial_merge_size) * (gW / cfg.spatial_merge_size));

        const float* pat = out.patches.host_f32();
        const int per_patch = C * P * P;
        // Every feature in channel block c of every patch is the normalized vc.
        float max_err = 0.0f;
        for (int n = 0; n < gH * gW; ++n) {
            for (int c = 0; c < C; ++c) {
                const float expect = (vc[c] - cfg.mean[c]) / cfg.std_[c];
                for (int j = 0; j < P * P; ++j) {
                    const float got = pat[static_cast<std::size_t>(n) * per_patch
                                          + static_cast<std::size_t>(c) * P * P + j];
                    max_err = std::max(max_err, std::fabs(got - expect));
                }
            }
        }
        std::printf("mistral3_preprocessor: constant-channel max_err = %.3e\n",
                    static_cast<double>(max_err));
        CHECK(max_err < 1e-4f);
    }

    // ── 2b. horizontal ramp confirms pw = inner x index + column ordering ─
    {
        const int H = 28, W = 28;
        // image[c][y][x] = x for all channels.
        std::vector<float> img(static_cast<std::size_t>(C) * H * W);
        for (int c = 0; c < C; ++c)
            for (int y = 0; y < H; ++y)
                for (int x = 0; x < W; ++x)
                    img[(static_cast<std::size_t>(c) * H + y) * W + x] = static_cast<float>(x) / 28.0f;

        m3::PreprocessedImage out = m3::preprocess_image(img.data(), H, W, cfg);
        const float* pat = out.patches.host_f32();
        const int per_patch = C * P * P;

        auto feat = [&](int patch, int c, int ph, int pw) {
            return pat[static_cast<std::size_t>(patch) * per_patch
                       + static_cast<std::size_t>(c) * P * P
                       + static_cast<std::size_t>(ph) * P + pw];
        };
        // Within patch 0 (pc=0), channel 0: feature should increase with pw
        // (x = pw) and be independent of ph (constant down columns).
        bool monotonic = true, row_indep = true;
        for (int pw = 1; pw < P; ++pw)
            if (!(feat(0, 0, 0, pw) > feat(0, 0, 0, pw - 1))) monotonic = false;
        for (int ph = 1; ph < P; ++ph)
            if (std::fabs(feat(0, 0, ph, 3) - feat(0, 0, 0, 3)) > 1e-4f) row_indep = false;
        CHECK(monotonic);
        CHECK(row_indep);

        // Patch 1 is grid column pc=1 (x starts at 14): its pw=0 feature must
        // exceed patch 0's pw=0 feature — columns advance left→right.
        CHECK(feat(1, 0, 0, 0) > feat(0, 0, 0, 0));
    }

    // ── 4. token span ─────────────────────────────────────────────────────
    {
        m3::PreprocessedImage img;
        img.grid_h = 8;  // tokens_h = 4
        img.grid_w = 8;  // tokens_w = 4
        img.merge  = 2;
        const int IMG = 10, BRK = 12, END = 13;
        std::vector<int> span = m3::build_image_token_span(img, IMG, BRK, END);

        int n_img = 0, n_brk = 0, n_end = 0;
        for (int t : span) {
            if (t == IMG) ++n_img;
            else if (t == BRK) ++n_brk;
            else if (t == END) ++n_end;
        }
        CHECK(n_img == img.num_image_tokens());  // 16
        CHECK(n_brk == img.tokens_h() - 1);       // 3 breaks between 4 rows
        CHECK(n_end == 1);                         // single trailing end
        CHECK(span.back() == END);
        // total = 16 img + 3 brk + 1 end.
        CHECK(static_cast<int>(span.size()) == 16 + 3 + 1);
    }

    if (g_failures == 0) std::printf("mistral3_preprocessor: OK\n");
    else std::fprintf(stderr, "mistral3_preprocessor: %d failure(s)\n", g_failures);
    return g_failures ? 1 : 0;
}
