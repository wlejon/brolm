// Ad-hoc driver: load a real image file, run Qwen3.5-VL on it, print the reply.
//
// usage: run_qwen35_image <weights_dir> <image_path> [user_prompt]
//
// Image decode + area-downscale come from broimage (any format stb supports —
// png/jpg/bmp/...). The decoded RGBA is dropped to RGB, optionally box-averaged
// to fit a per-tool pixel cap, and shuffled to CHW float in [0, 1].

#include "broimage/buffer.h"
#include "broimage/color.h"
#include "broimage/decode.h"
#include "broimage/geometric.h"

#include "brolm/qwen35_vl.h"
#include "brotensor/runtime.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

// Hard cap on input image pixels before it reaches the VLM. Bigger inputs are
// area-resampled down to fit. The cap exists because vision token count grows
// linearly with pixels and the ViT's attention cost grows quadratically with
// token count — a multi-megapixel screenshot will pin the CPU matmul for many
// minutes. ~262 K pixels (~512×512) keeps a single forward pass in the
// tens-of-seconds range on the scalar CPU backend while still being legible
// for screenshot-style content.
static constexpr int kMaxInputPixels = 512 * 512;

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
            "usage: %s <weights_dir> <image_path> [user_prompt]\n", argv[0]);
        return 1;
    }
    const std::string weights    = argv[1];
    const std::string image_path = argv[2];
    const std::string user_prompt =
        argc >= 4 ? argv[3] : "Describe the image.";

    broimage::Image rgba;
    std::string err;
    if (!broimage::decode_file_oriented(image_path, rgba, &err)) {
        std::fprintf(stderr, "broimage: failed to load %s: %s\n",
                     image_path.c_str(), err.c_str());
        return 1;
    }
    const int W = rgba.width;
    const int H = rgba.height;
    std::fprintf(stderr, "image: %dx%d\n", W, H);

    // RGBA8 -> RGB8.
    std::vector<uint8_t> rgb(static_cast<std::size_t>(W) * H * 3);
    broimage::rgba_to_rgb_u8(rgba.pixels.data(), rgb.data(), W * H);

    int effW = W, effH = H;
    const long long pix = static_cast<long long>(W) * H;
    if (pix > kMaxInputPixels) {
        const double scale = std::sqrt(static_cast<double>(kMaxInputPixels) / pix);
        effW = std::max(1, static_cast<int>(std::round(W * scale)));
        effH = std::max(1, static_cast<int>(std::round(H * scale)));
        std::vector<uint8_t> down(static_cast<std::size_t>(effW) * effH * 3);
        broimage::resize_hwc_u8(rgb.data(), W, H, /*channels=*/3,
                                down.data(), effW, effH,
                                broimage::Filter::Area);
        rgb = std::move(down);
        std::fprintf(stderr,
                     "image: downscaled to %dx%d (%.2f MP -> %.2f MP cap)\n",
                     effW, effH, pix / 1.0e6,
                     static_cast<double>(effW) * effH / 1.0e6);
    }

    // HWC u8 -> CHW float in [0, 1].
    std::vector<float> chw(static_cast<std::size_t>(3) * effH * effW);
    const std::size_t plane = static_cast<std::size_t>(effH) * effW;
    for (int y = 0; y < effH; ++y) {
        for (int x = 0; x < effW; ++x) {
            const uint8_t* p = rgb.data() + (static_cast<std::size_t>(y) * effW + x) * 3;
            for (int c = 0; c < 3; ++c) {
                chw[static_cast<std::size_t>(c) * plane +
                    static_cast<std::size_t>(y) * effW + x] = p[c] / 255.0f;
            }
        }
    }

    brotensor::init();

    namespace q35 = brolm::qwen35;
    q35::VLMConfig vcfg;
    vcfg.max_seq_len    = 4096;
    vcfg.max_new_tokens = 64;
    vcfg.temperature    = 0.0f;
    vcfg.seed           = 0;

    q35::VLM vlm(vcfg);
    vlm.load_from_directory(weights);

    q35::ImageInput inp{ chw.data(), effH, effW };
    std::vector<q35::ImageInput> images = { inp };

    const std::string prompt =
        "<|im_start|>user\n"
        "<|vision_start|><|image_pad|><|vision_end|>"
        + user_prompt +
        "<|im_end|>\n"
        "<|im_start|>assistant\n";

    try {
        std::string out = vlm.generate(prompt, images);
        std::printf("%s\n", out.c_str());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "generate failed: %s\n", e.what());
        return 1;
    }
    return 0;
}
