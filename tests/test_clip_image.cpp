// CLIP ViT image encoder smoke test.
//
// Builds a tiny image encoder (image_size=8, patch_size=4 -> 2x2 patch grid
// + 1 CLS = 5 tokens; hidden_dim=16, num_heads=2, num_layers=2,
// intermediate_dim=32), synthesizes a safetensors fixture with the HF
// "vision_model." tensor names, loads it, and runs forward on a synthetic
// pixel tensor. Verifies output shape/dtype, that all outputs are finite
// (no NaN/Inf), and that two consecutive forwards produce identical results.
//
// Numerical accuracy against a reference is intentionally not checked here —
// that needs real CLIP weights and lives in a future integration test.

#include "brolm/clip_image.h"
#include "brolm/detail/compute.h"
#include "brotensor/safetensors.h"

#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

#include "test_compute.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

namespace clip_image = brolm::clip_image;
namespace st         = brotensor::safetensors;
namespace bt         = brotensor;

static int g_failures = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        ++g_failures; \
    } \
} while (0)

// ─── safetensors fixture builder ───────────────────────────────────────────
//
// Same pattern as test_clip.cpp / test_t5.cpp: an F16 safetensors file built
// in memory and flushed to disk.

namespace {

struct Builder {
    std::string entries;
    std::vector<uint8_t> payload;
    bool first = true;

    void add(const std::string& name, std::vector<int> shape,
             const std::vector<uint16_t>& fp16_bits) {
        std::size_t expected = 1;
        for (int d : shape) expected *= static_cast<std::size_t>(d);
        if (expected != fp16_bits.size()) {
            std::fprintf(stderr, "fixture: shape/data mismatch for %s\n",
                         name.c_str());
            std::abort();
        }
        std::uint64_t start = payload.size();
        const std::uint8_t* bytes =
            reinterpret_cast<const std::uint8_t*>(fp16_bits.data());
        payload.insert(payload.end(), bytes, bytes + fp16_bits.size() * 2);
        std::uint64_t end = payload.size();

        if (!first) entries += ",";
        first = false;
        entries += "\"" + name + "\":{\"dtype\":\"F16\",\"shape\":[";
        for (std::size_t i = 0; i < shape.size(); ++i) {
            if (i) entries += ",";
            entries += std::to_string(shape[i]);
        }
        entries += "],\"data_offsets\":[" + std::to_string(start) + "," +
                   std::to_string(end) + "]}";
    }

    void write(const std::filesystem::path& path) const {
        std::string header = "{" + entries + "}";
        std::uint64_t hdr_size = header.size();
        std::ofstream f(path, std::ios::binary | std::ios::trunc);
        if (!f) std::abort();
        f.write(reinterpret_cast<const char*>(&hdr_size), 8);
        f.write(header.data(), header.size());
        f.write(reinterpret_cast<const char*>(payload.data()),
                static_cast<std::streamsize>(payload.size()));
    }
};

std::vector<uint16_t> fp16_zeros(std::size_t n) {
    return std::vector<uint16_t>(n, 0);
}
std::vector<uint16_t> fp16_ones(std::size_t n) {
    return std::vector<uint16_t>(n, bt::fp32_to_fp16_bits(1.0f));
}
// Small deterministic values: scale, 2*scale, 3*scale, ... so rows/cols differ.
std::vector<uint16_t> fp16_seq(std::size_t n, float scale) {
    std::vector<uint16_t> out(n);
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = bt::fp32_to_fp16_bits((static_cast<float>(i) + 1.0f) * scale);
    }
    return out;
}

}  // namespace

// ─── test ──────────────────────────────────────────────────────────────────

int main() {
    try {
        bt::init();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "init failed: %s\n", e.what());
        return 1;
    }

    clip_image::ImageEncoderConfig cfg;
    cfg.image_size       = 8;
    cfg.patch_size       = 4;     // 2x2 patch grid + 1 CLS = 5 tokens
    cfg.in_channels      = 3;
    cfg.hidden_dim       = 16;
    cfg.num_heads        = 2;
    cfg.num_layers       = 2;
    cfg.intermediate_dim = 32;
    cfg.layer_norm_eps   = 1e-5f;

    const int D = cfg.hidden_dim;
    const int F = cfg.intermediate_dim;
    const int C = cfg.in_channels;
    const int P = cfg.patch_size;
    const int S = cfg.image_size;
    const int T = clip_image::num_tokens(cfg);   // 1 + 2*2 = 5
    CHECK(T == 5);

    Builder b;
    const std::string p = "vision_model.";

    // Embeddings.
    b.add(p + "embeddings.patch_embedding.weight", {D, C, P, P},
          fp16_seq(static_cast<std::size_t>(D) * C * P * P, 0.01f));
    b.add(p + "embeddings.class_embedding", {D}, fp16_seq(D, 0.05f));
    b.add(p + "embeddings.position_embedding.weight", {T, D},
          fp16_seq(static_cast<std::size_t>(T) * D, 0.02f));

    // pre_layrnorm (HF typo spelling).
    b.add(p + "pre_layrnorm.weight", {D}, fp16_ones(D));
    b.add(p + "pre_layrnorm.bias",   {D}, fp16_zeros(D));

    auto W = [&](float s) { return fp16_seq(static_cast<std::size_t>(D) * D, s); };
    for (int i = 0; i < cfg.num_layers; ++i) {
        const std::string lp =
            p + "encoder.layers." + std::to_string(i) + ".";

        b.add(lp + "layer_norm1.weight", {D}, fp16_ones(D));
        b.add(lp + "layer_norm1.bias",   {D}, fp16_zeros(D));

        b.add(lp + "self_attn.q_proj.weight",   {D, D}, W(0.02f));
        b.add(lp + "self_attn.q_proj.bias",     {D},    fp16_zeros(D));
        b.add(lp + "self_attn.k_proj.weight",   {D, D}, W(0.03f));
        b.add(lp + "self_attn.k_proj.bias",     {D},    fp16_zeros(D));
        b.add(lp + "self_attn.v_proj.weight",   {D, D}, W(0.04f));
        b.add(lp + "self_attn.v_proj.bias",     {D},    fp16_zeros(D));
        b.add(lp + "self_attn.out_proj.weight", {D, D}, W(0.05f));
        b.add(lp + "self_attn.out_proj.bias",   {D},    fp16_zeros(D));

        b.add(lp + "layer_norm2.weight", {D}, fp16_ones(D));
        b.add(lp + "layer_norm2.bias",   {D}, fp16_zeros(D));

        b.add(lp + "mlp.fc1.weight", {F, D},
              fp16_seq(static_cast<std::size_t>(F) * D, 0.01f));
        b.add(lp + "mlp.fc1.bias",   {F}, fp16_zeros(F));
        b.add(lp + "mlp.fc2.weight", {D, F},
              fp16_seq(static_cast<std::size_t>(D) * F, 0.01f));
        b.add(lp + "mlp.fc2.bias",   {D}, fp16_zeros(D));
    }

    b.add(p + "post_layernorm.weight", {D}, fp16_ones(D));
    b.add(p + "post_layernorm.bias",   {D}, fp16_zeros(D));

    auto path = std::filesystem::temp_directory_path() /
                "brolm_clip_image_test.safetensors";
    b.write(path);

    std::vector<float> out_vals1, out_vals2;
    {
        auto file = st::File::open(path.string());
        clip_image::ImageEncoder enc(cfg);
        enc.load_weights(file, "vision_model.");

        // Synthetic pixel tensor: (1, C*S*S) at the compute dtype.
        std::vector<float> pixels(static_cast<std::size_t>(C) * S * S);
        for (std::size_t i = 0; i < pixels.size(); ++i) {
            pixels[i] = 0.01f * (static_cast<float>(i % 17) - 8.0f);
        }
        bt::Tensor pixels_dev =
            brolm::detail::upload_host(pixels.data(), 1, C * S * S);

        bt::Tensor cls_out;
        enc.forward(pixels_dev, cls_out);
        bt::sync_all();

        CHECK(cls_out.rows == 1);
        CHECK(cls_out.cols == D);
        CHECK(cls_out.dtype == brolm::compute_dtype());

        out_vals1 = bdtest::bd_download(cls_out);
        CHECK(out_vals1.size() == static_cast<std::size_t>(D));

        int nonfinite = 0;
        for (float v : out_vals1) if (!bdtest::bd_finite(v)) ++nonfinite;
        CHECK(nonfinite == 0);

        // Second forward — must be identical for a deterministic graph.
        enc.forward(pixels_dev, cls_out);
        bt::sync_all();
        out_vals2 = bdtest::bd_download(cls_out);
        CHECK(out_vals1 == out_vals2);
    }

    std::error_code ec;
    std::filesystem::remove(path, ec);

    if (g_failures == 0) std::printf("clip_image: OK\n");
    else std::fprintf(stderr, "clip_image: %d failure(s)\n", g_failures);
    return g_failures ? 1 : 0;
}
