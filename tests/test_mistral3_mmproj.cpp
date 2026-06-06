// Real-weights smoke test for the Mistral 3.1 vision path via the llama.cpp
// mmproj/clip gguf. Gated on the file being present; skips cleanly otherwise.
//
// The mmproj ships the Pixtral vision tower + multimodal projector in F16, so —
// unlike the quantized text gguf — it loads and runs on the CPU backend
// (GgufSource converts F16 → FP32; the tower's attention runs host-FP32). This
// validates the ggml name maps (mistral3_vision_hf_to_ggml /
// mistral3_projector_hf_to_ggml) and Mistral3Config::merge_vision_from_mmproj_gguf
// against the actual checkpoint.
//
// Discovery: $BROLM_MISTRAL_MMPROJ, else
// <repo>/weights/Mistral-Small-3.1-24B-Instruct-2503-GGUF/*mmproj*.gguf.
//
// Checks: config dims are sane; tower.forward on a synthetic patch grid yields
// finite (N, vision_hidden) features; projector.forward yields finite
// (num_image_tokens, text_hidden) embeddings.

#include "brolm/mistral3_config.h"
#include "brolm/mistral3_vision.h"
#include "brolm/mistral3_projector.h"
#include "brolm/detail/compute.h"

#include "brotensor/gguf.h"
#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

#include "test_compute.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

namespace m3 = brolm::mistral3;
namespace bt = brotensor;

static int g_failures = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_failures; } } while (0)

namespace {

std::string find_mmproj() {
    if (const char* env = std::getenv("BROLM_MISTRAL_MMPROJ")) {
        if (*env && std::filesystem::exists(env)) return env;
    }
    namespace fs = std::filesystem;
    const fs::path dir = fs::path("weights") / "Mistral-Small-3.1-24B-Instruct-2503-GGUF";
    if (!fs::is_directory(dir)) return {};
    for (const auto& e : fs::directory_iterator(dir)) {
        if (e.path().extension() != ".gguf") continue;
        if (e.path().filename().string().find("mmproj") != std::string::npos)
            return e.path().string();
    }
    return {};
}

bt::Tensor upload(const std::vector<float>& v, int rows, int cols) {
    if (brolm::compute_dtype() == bt::Dtype::FP16) {
        std::vector<std::uint16_t> bits(v.size());
        for (std::size_t i = 0; i < v.size(); ++i) bits[i] = bt::fp32_to_fp16_bits(v[i]);
        return bt::Tensor::from_host_fp16(bits.data(), rows, cols);
    }
    return bt::Tensor::from_host(v.data(), rows, cols);
}

bool all_finite(const std::vector<float>& v) {
    for (float x : v) if (!std::isfinite(x)) return false;
    return true;
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const std::string path = find_mmproj();
    if (path.empty()) {
        std::printf("[skip] no Mistral mmproj .gguf found (set BROLM_MISTRAL_MMPROJ "
                    "or run scripts/download_mistral_gguf.sh MMPROJ=1)\n");
        return 0;
    }
    std::printf("[info] using %s\n", path.c_str());
    bt::init();

    bt::gguf::File f = bt::gguf::File::open(path);
    if (const auto* a = f.find_meta("general.architecture");
        a && a->type == bt::gguf::ValueType::String) {
        std::printf("[info] general.architecture = %s\n", a->str.c_str());
    }

    m3::Mistral3Config cfg;  // text stays default (hidden 5120 etc.)
    try {
        cfg.merge_vision_from_mmproj_gguf(f);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "merge_vision_from_mmproj_gguf threw: %s\n", e.what());
        return 1;
    }
    const auto& v = cfg.vision;
    std::printf("[info] vision: hidden=%d layers=%d heads=%d hd=%d ff=%d patch=%d eps=%g merge=%d\n",
                v.hidden_size, v.num_hidden_layers, v.num_attention_heads, v.head_dim,
                v.intermediate_size, v.patch_size, static_cast<double>(v.rms_norm_eps),
                cfg.spatial_merge_size);
    CHECK(v.hidden_size > 0);
    CHECK(v.num_hidden_layers > 0);
    CHECK(v.head_dim * v.num_attention_heads == v.hidden_size);
    CHECK(v.head_dim % 4 == 0);
    CHECK(cfg.spatial_merge_size >= 1);

    try {
        // ── load tower + projector from the mmproj ────────────────────────
        m3::VisionTower tower(v);
        tower.load_weights(f);
        m3::MultiModalProjector proj(cfg);
        proj.load_weights(f);
        std::printf("[info] tower + projector loaded\n");

        // ── synthetic patch grid (4×4 patches → 2×2 = 4 image tokens) ─────
        const int gh = 4, gw = 4, N = gh * gw;
        const int CP = v.num_channels * v.patch_size * v.patch_size;
        std::vector<float> patches(static_cast<std::size_t>(N) * CP);
        uint32_t s = 12345;
        for (float& x : patches) { s = s * 1664525u + 1013904223u; x = (static_cast<float>(s >> 8) / 16777216.0f - 0.5f) * 0.5f; }

        bt::Tensor pt = upload(patches, N, CP);
        bt::Tensor feat;
        tower.forward(pt, gh, gw, feat);
        bt::sync_all();
        CHECK(feat.rows == N);
        CHECK(feat.cols == v.hidden_size);
        std::vector<float> fv = bdtest::bd_download(feat);
        CHECK(all_finite(fv));
        std::printf("[info] tower forward OK: (%d, %d) finite\n", feat.rows, feat.cols);

        bt::Tensor out;
        proj.forward(feat, gh, gw, out);
        bt::sync_all();
        const int Ltok = (gh / cfg.spatial_merge_size) * (gw / cfg.spatial_merge_size);
        CHECK(out.rows == Ltok);
        CHECK(out.cols == cfg.text.hidden_size);
        std::vector<float> ov = bdtest::bd_download(out);
        CHECK(all_finite(ov));
        std::printf("[info] projector forward OK: (%d, %d) finite\n", out.rows, out.cols);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "vision path threw: %s\n", e.what());
        ++g_failures;
    }

    if (g_failures) { std::fprintf(stderr, "FAILED: %d\n", g_failures); return 1; }
    std::printf("OK\n");
    return 0;
}
