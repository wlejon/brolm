// Ad-hoc driver: load a real image file, run Qwen3.5-VL on it, print the reply.
//
// usage: run_qwen35_image <weights_dir> <image_path> [user_prompt]
//
// Decodes the image with stb_image (any format stb supports — png/jpg/bmp/...),
// converts to CHW float in [0,1], wraps the user prompt in the standard ChatML
// vision template, and prints the generated text to stdout.

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_STATIC
#include "stb_image.h"

#include "brolm/qwen35_vl.h"
#include "brotensor/runtime.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

// Hard cap on input image pixels before it reaches the VLM. Bigger inputs are
// box-averaged down to fit. The cap exists because vision token count grows
// linearly with pixels and the ViT's attention cost grows quadratically with
// token count — a multi-megapixel screenshot will pin the CPU matmul for many
// minutes. ~262 K pixels (~512×512) keeps a single forward pass in the
// tens-of-seconds range on the scalar CPU backend while still being legible
// for screenshot-style content.
static constexpr int kMaxInputPixels = 512 * 512;

// Box-average downscale uint8 HWC RGB image (W,H) to (outW,outH). Allocates
// and returns a new buffer; caller owns it (delete[]). For pure downscaling
// this is a higher-quality choice than nearest/bilinear: every output pixel is
// the unweighted mean of all source pixels whose center falls in its cell.
static unsigned char* box_downscale_rgb(const unsigned char* src,
                                        int W, int H, int outW, int outH) {
    auto* out = new unsigned char[static_cast<std::size_t>(outW) * outH * 3];
    const double sx = static_cast<double>(W) / outW;
    const double sy = static_cast<double>(H) / outH;
    for (int oy = 0; oy < outH; ++oy) {
        const int y0 = static_cast<int>(std::floor(oy * sy));
        const int y1 = std::min(H, static_cast<int>(std::ceil((oy + 1) * sy)));
        for (int ox = 0; ox < outW; ++ox) {
            const int x0 = static_cast<int>(std::floor(ox * sx));
            const int x1 = std::min(W, static_cast<int>(std::ceil((ox + 1) * sx)));
            double rsum = 0, gsum = 0, bsum = 0;
            int n = 0;
            for (int y = y0; y < y1; ++y) {
                const unsigned char* row = src + (static_cast<std::size_t>(y) * W + x0) * 3;
                for (int x = x0; x < x1; ++x, row += 3) {
                    rsum += row[0]; gsum += row[1]; bsum += row[2];
                    ++n;
                }
            }
            const double inv = n > 0 ? 1.0 / n : 0.0;
            unsigned char* d = out + (static_cast<std::size_t>(oy) * outW + ox) * 3;
            d[0] = static_cast<unsigned char>(rsum * inv + 0.5);
            d[1] = static_cast<unsigned char>(gsum * inv + 0.5);
            d[2] = static_cast<unsigned char>(bsum * inv + 0.5);
        }
    }
    return out;
}

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

    int W = 0, H = 0, ch_in = 0;
    unsigned char* pixels = stbi_load(image_path.c_str(), &W, &H, &ch_in, 3);
    if (!pixels) {
        std::fprintf(stderr, "stb_image: failed to load %s: %s\n",
                     image_path.c_str(), stbi_failure_reason());
        return 1;
    }
    std::fprintf(stderr, "image: %dx%d (file channels=%d, forced to 3)\n",
                 W, H, ch_in);

    // Cap input size. Anything bigger than kMaxInputPixels is box-averaged
    // down preserving aspect ratio; the post-cap dims still get snapped to
    // multiples of 32 inside the HF preprocessor, so we don't bother here.
    const long long pix = static_cast<long long>(W) * H;
    int effW = W, effH = H;
    unsigned char* owned = nullptr;
    if (pix > kMaxInputPixels) {
        const double scale = std::sqrt(static_cast<double>(kMaxInputPixels) / pix);
        effW = std::max(1, static_cast<int>(std::round(W * scale)));
        effH = std::max(1, static_cast<int>(std::round(H * scale)));
        owned = box_downscale_rgb(pixels, W, H, effW, effH);
        std::fprintf(stderr,
                     "image: downscaled to %dx%d (%.2f MP -> %.2f MP cap)\n",
                     effW, effH, pix / 1.0e6,
                     static_cast<double>(effW) * effH / 1.0e6);
        stbi_image_free(pixels);
        pixels = owned;
    }

    std::vector<float> chw(static_cast<std::size_t>(3) * effH * effW);
    for (int y = 0; y < effH; ++y) {
        for (int x = 0; x < effW; ++x) {
            for (int c = 0; c < 3; ++c) {
                chw[(static_cast<std::size_t>(c) * effH + y) * effW + x] =
                    pixels[(static_cast<std::size_t>(y) * effW + x) * 3 + c]
                    / 255.0f;
            }
        }
    }
    if (owned) delete[] owned;
    else stbi_image_free(pixels);

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
