// BioCLIP ViT-B/16 integration test on the default GPU (CPU when there is none).
#include "brolm/clip.h"
#include "brolm/clip_image.h"
#include "brolm/detail/compute.h"
#include "test_compute.h"
#include "brotensor/runtime.h"
#include "brotensor/safetensors.h"
#include "brotensor/tensor.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <vector>

namespace bt = brotensor;
namespace st = brotensor::safetensors;

int main() {
    bt::init();
    const bt::Device dev = bt::default_device();
    std::printf("BioCLIP test device: %s\n", bt::to_string(dev).c_str());

    // BROLM_BIOCLIP_DIR names the checkpoint directory (holding
    // model.safetensors); unset, it falls back to the local default. A missing
    // checkpoint is a skip, not a failure, so CPU-only CI without the weights
    // stays green.
    const char* dir_env = std::getenv("BROLM_BIOCLIP_DIR");
    const std::filesystem::path dir = dir_env ? dir_env : "/home/j/models/bioclip";
    const std::string model_path = (dir / "model.safetensors").string();
    if (!std::filesystem::exists(model_path)) {
        std::printf("[skip] BioCLIP checkpoint not found at %s (set BROLM_BIOCLIP_DIR)\n",
                    model_path.c_str());
        return 0;
    }

    auto f = st::File::open(model_path);
    std::printf("Opened %s with %zu tensors\n", model_path.c_str(), f.tensors().size());

    // 1. Vision Encoder (ViT-B/16)
    brolm::clip_image::ImageEncoderConfig vcfg;
    vcfg.image_size       = 224;
    vcfg.patch_size       = 16;
    vcfg.in_channels      = 3;
    vcfg.hidden_dim       = 768;
    vcfg.num_heads        = 12;
    vcfg.num_layers       = 12;
    vcfg.intermediate_dim = 3072;
    vcfg.layer_norm_eps   = 1e-5f;

    brolm::clip_image::ImageEncoder vision_enc(vcfg);
    vision_enc.load_weights(f, "vision_model.");
    std::printf("Vision weights loaded successfully.\n");

    // Synthetic normalized pixel input (1, 3 * 224 * 224)
    const int pixel_count = 3 * 224 * 224;
    std::vector<float> host_pixels(static_cast<std::size_t>(pixel_count), 0.5f);
    bt::Tensor pixels = brolm::detail::upload_host(host_pixels.data(), 1, pixel_count);

    bt::Tensor cls_out;
    vision_enc.forward(pixels, cls_out);
    bt::sync_all();

    std::printf("Vision forward completed: cls_out shape (%d, %d), dev=%s\n",
                cls_out.rows, cls_out.cols, bt::to_string(cls_out.device).c_str());

    std::vector<float> cls_host = bdtest::bd_download(cls_out);
    bool all_finite = true;
    for (float v : cls_host) {
        if (!std::isfinite(v)) {
            all_finite = false;
            break;
        }
    }
    if (!all_finite || cls_out.rows != 1 || cls_out.cols != 768) {
        std::fprintf(stderr, "FAIL: cls_out has non-finite values or incorrect shape\n");
        return 1;
    }
    std::printf("Vision forward output verified finite (mean=%.4f).\n", cls_host[0]);

    // 2. Text Encoder (ViT-B/16 text branch)
    brolm::clip::TextEncoderConfig tcfg;
    tcfg.vocab_size       = 49408;
    tcfg.max_position     = 77;
    tcfg.hidden_dim       = 512;
    tcfg.num_heads        = 8;
    tcfg.num_layers       = 12;
    tcfg.intermediate_dim = 2048;
    tcfg.layer_norm_eps   = 1e-5f;
    tcfg.eos_token_id     = 49407;

    brolm::clip::TextEncoder text_enc(tcfg);
    text_enc.load_weights(f, "text_model.");
    std::printf("Text weights loaded successfully.\n");

    // Test token sequence: BOS, some token, EOS, EOS, ...
    std::vector<int32_t> token_ids(77, 49407);
    token_ids[0] = 49406; // BOS
    token_ids[1] = 320;   // "a"
    token_ids[2] = 2368;  // "photo"

    bt::Tensor text_out, text_pooled;
    text_enc.forward(token_ids.data(), text_out, &text_pooled);
    bt::sync_all();

    std::printf("Text forward completed: text_out (%d, %d), pooled (%d, %d), dev=%s\n",
                text_out.rows, text_out.cols, text_pooled.rows, text_pooled.cols,
                bt::to_string(text_out.device).c_str());

    std::vector<float> pooled_host = bdtest::bd_download(text_pooled);
    all_finite = true;
    for (float v : pooled_host) {
        if (!std::isfinite(v)) {
            all_finite = false;
            break;
        }
    }
    if (!all_finite || text_pooled.rows != 1 || text_pooled.cols != 512) {
        std::fprintf(stderr, "FAIL: text_pooled has non-finite values or incorrect shape\n");
        return 1;
    }
    std::printf("Text forward output verified finite (mean=%.4f).\n", pooled_host[0]);

    std::printf("ALL BIOCLIP TESTS PASSED ON %s!\n", bt::to_string(dev).c_str());
    return 0;
}
