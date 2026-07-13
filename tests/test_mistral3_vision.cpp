// Mistral 3.1 Pixtral vision tower smoke + structural test.
//
// Builds a tiny tower (hidden=32, heads=4, head_dim=8, layers=2,
// intermediate=64, channels=3, patch=4 → C*P²=48), synthesizes a safetensors
// fixture with every HF `vision_tower.*` tensor name (all bias-free; RMSNorms
// carry only a weight), loads it, and runs forward on a 4×4 patch grid. Checks:
//   1. shape (N, hidden) + finiteness of the per-patch hidden states;
//   2. determinism — two identical runs are bitwise-identical;
//   3. 2-D RoPE is active — feeding N *identical* patch rows must NOT yield
//      identical output rows: with full attention and no positional signal the
//      rows would be identical, so distinct rows prove the rotary embedding
//      breaks the permutation symmetry by patch position;
//   4. a missing weight raises.
//
// Numerical parity against HF needs the real Pixtral weights (not downloaded)
// and lives in a future integration test.

#include "brolm/mistral3_config.h"
#include "brolm/mistral3_vision.h"
#include "brolm/detail/compute.h"

#include "brotensor/runtime.h"
#include "brotensor/safetensors.h"
#include "brotensor/tensor.h"

#include "test_compute.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

namespace m3 = brolm::mistral3;
namespace st = brotensor::safetensors;
namespace bt = brotensor;

static int g_failures = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        ++g_failures; \
    } \
} while (0)

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
            std::fprintf(stderr, "fixture: shape/data mismatch for %s\n", name.c_str());
            std::abort();
        }
        std::uint64_t start = payload.size();
        const std::uint8_t* bytes = reinterpret_cast<const std::uint8_t*>(fp16_bits.data());
        payload.insert(payload.end(), bytes, bytes + fp16_bits.size() * 2);
        std::uint64_t end = payload.size();

        if (!first) entries += ",";
        first = false;
        entries += "\"" + name + "\":{\"dtype\":\"F16\",\"shape\":[";
        for (std::size_t i = 0; i < shape.size(); ++i) {
            if (i) entries += ",";
            entries += std::to_string(shape[i]);
        }
        entries += "],\"data_offsets\":[" + std::to_string(start) + "," + std::to_string(end) + "]}";
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

std::vector<uint16_t> fp16_rand(std::size_t n, uint32_t seed) {
    std::vector<uint16_t> out(n);
    uint32_t s = seed * 2654435761u + 1u;
    for (std::size_t i = 0; i < n; ++i) {
        s = s * 1664525u + 1013904223u;
        float v = (static_cast<float>(s >> 8) / 16777216.0f - 0.5f) * 0.2f;
        out[i] = bt::fp32_to_fp16_bits(v);
    }
    return out;
}
std::vector<uint16_t> fp16_ones(std::size_t n) {
    return std::vector<uint16_t>(n, bt::fp32_to_fp16_bits(1.0f));
}

// Build the tiny Pixtral fixture (all bias-free; RMSNorm weights = ones).
std::filesystem::path build_fixture(const m3::Mistral3Config::Vision& cfg) {
    const int D = cfg.hidden_size;
    const int F = cfg.intermediate_size;
    const int C = cfg.num_channels;
    const int P = cfg.patch_size;
    const int CP = C * P * P;

    Builder b;
    uint32_t seed = 1;
    auto R = [&](std::size_t n) { return fp16_rand(n, seed++); };

    b.add("vision_tower.patch_conv.weight", {D, CP}, R(static_cast<std::size_t>(D) * CP));
    b.add("vision_tower.ln_pre.weight", {D}, fp16_ones(D));
    for (int i = 0; i < cfg.num_hidden_layers; ++i) {
        const std::string p = "vision_tower.transformer.layers." + std::to_string(i) + ".";
        b.add(p + "attention_norm.weight", {D}, fp16_ones(D));
        b.add(p + "attention.q_proj.weight", {D, D}, R(static_cast<std::size_t>(D) * D));
        b.add(p + "attention.k_proj.weight", {D, D}, R(static_cast<std::size_t>(D) * D));
        b.add(p + "attention.v_proj.weight", {D, D}, R(static_cast<std::size_t>(D) * D));
        b.add(p + "attention.o_proj.weight", {D, D}, R(static_cast<std::size_t>(D) * D));
        b.add(p + "ffn_norm.weight", {D}, fp16_ones(D));
        b.add(p + "feed_forward.gate_proj.weight", {F, D}, R(static_cast<std::size_t>(F) * D));
        b.add(p + "feed_forward.up_proj.weight",   {F, D}, R(static_cast<std::size_t>(F) * D));
        b.add(p + "feed_forward.down_proj.weight", {D, F}, R(static_cast<std::size_t>(D) * F));
    }

    auto path = std::filesystem::temp_directory_path() / "brolm_mistral3_vision.safetensors";
    b.write(path);
    return path;
}

// Upload an (rows, cols) FP32 host matrix at the pipeline compute dtype.
bt::Tensor upload(const std::vector<float>& v, int rows, int cols) {
    if (brolm::compute_dtype() == bt::Dtype::FP16) {
        std::vector<std::uint16_t> bits(v.size());
        for (std::size_t i = 0; i < v.size(); ++i) bits[i] = bt::fp32_to_fp16_bits(v[i]);
        return bt::Tensor::from_host_fp16(bits.data(), rows, cols);
    }
    return bt::Tensor::from_host(v.data(), rows, cols);
}

}  // namespace

int main() {
    try {
        bt::init();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "init failed: %s\n", e.what());
        return 1;
    }

    m3::Mistral3Config::Vision cfg;
    cfg.hidden_size         = 32;
    cfg.num_attention_heads = 4;
    cfg.head_dim            = 8;
    cfg.intermediate_size   = 64;
    cfg.num_hidden_layers   = 2;
    cfg.num_channels        = 3;
    cfg.patch_size          = 4;
    cfg.rope_theta          = 10000.0f;

    const int D  = cfg.hidden_size;
    const int CP = cfg.num_channels * cfg.patch_size * cfg.patch_size;  // 48
    const int grid_h = 4, grid_w = 4;
    const int N = grid_h * grid_w;  // 16

    auto path = build_fixture(cfg);

    try {
        auto file = st::File::open(path.string());

        // Distinct random patch rows.
        std::vector<float> patches(static_cast<std::size_t>(N) * CP);
        {
            uint32_t s = 777;
            for (float& x : patches) {
                s = s * 1664525u + 1013904223u;
                x = (static_cast<float>(s >> 8) / 16777216.0f - 0.5f) * 0.5f;
            }
        }

        // ── 1. shape + finiteness ─────────────────────────────────────────
        std::vector<float> out_a;
        {
            m3::VisionTower tower(cfg);
            tower.load_weights(file);
            bt::Tensor pt = upload(patches, N, CP);
            bt::Tensor out;
            tower.forward(pt, grid_h, grid_w, out);
            bt::sync_all();
            CHECK(out.rows == N);
            CHECK(out.cols == D);
            CHECK(out.dtype == brolm::compute_dtype());
            out_a = bdtest::bd_download(out);
            int nonfinite = 0;
            for (float v : out_a) if (!bdtest::bd_finite(v)) ++nonfinite;
            CHECK(nonfinite == 0);
            // Exact fingerprint on fixed synthetic weights. The 2-D RoPE and the
            // attention are what mix patch positions; a wrong rotary pairing
            // moves these numbers while every shape/finiteness check above stays
            // perfectly happy. Printed so a refactor can be diffed against the
            // implementation it replaces.
            double sum_abs = 0.0;
            for (float v : out_a) sum_abs += std::fabs(static_cast<double>(v));
            std::printf("FINGERPRINT mistral3 sum=%.6f v0=%.6f v1=%.6f v2=%.6f v3=%.6f\n",
                        sum_abs, out_a[0], out_a[1], out_a[2], out_a[3]);
        }

        // ── 2. determinism ────────────────────────────────────────────────
        {
            m3::VisionTower tower(cfg);
            tower.load_weights(file);
            bt::Tensor pt = upload(patches, N, CP);
            bt::Tensor out;
            tower.forward(pt, grid_h, grid_w, out);
            bt::sync_all();
            std::vector<float> out_b = bdtest::bd_download(out);
            CHECK(out_a == out_b);
        }

        // ── 3. 2-D RoPE is active (permutation sensitivity) ───────────────
        // Swap the content of patches 0 and 1 (grid positions (0,0) and (0,1)),
        // leaving every other patch untouched, and re-run. Attention is over the
        // *set* of keys/values, so WITHOUT a positional signal an unswapped
        // query row (e.g. patch 2) would be unchanged — the multiset of inputs
        // is identical. With 2-D RoPE the content now sits at different
        // positions, changing the relative-position structure patch 2 sees, so
        // its output MUST change. A non-trivial diff therefore proves RoPE is
        // wired and position-dependent.
        {
            std::vector<float> swapped = patches;
            for (int c = 0; c < CP; ++c) {
                std::swap(swapped[static_cast<std::size_t>(c)],
                          swapped[static_cast<std::size_t>(CP + c)]);
            }

            m3::VisionTower tower(cfg);
            tower.load_weights(file);
            bt::Tensor pt = upload(swapped, N, CP);
            bt::Tensor out;
            tower.forward(pt, grid_h, grid_w, out);
            bt::sync_all();
            std::vector<float> o = bdtest::bd_download(out);

            // Row 2 (patch 2) is unswapped; compare its output to the baseline.
            const std::size_t base = static_cast<std::size_t>(2) * D;
            float max_diff = 0.0f;
            for (int c = 0; c < D; ++c) {
                const float d = std::fabs(o[base + static_cast<std::size_t>(c)] -
                                          out_a[base + static_cast<std::size_t>(c)]);
                if (d > max_diff) max_diff = d;
            }
            // Signal (~1e-4 here) sits far above the FP32/FP16 determinism
            // floor (check 2 is bitwise-exact; observed noise ~6e-8); any clear
            // non-zero delta proves RoPE is position-dependent. Near-uniform
            // attention (tiny random weights) keeps the magnitude modest.
            std::printf("mistral3_vision: swap0-1 effect on unswapped row2 max_diff = %.3e\n",
                        static_cast<double>(max_diff));
            CHECK(max_diff > 1e-5f);
        }

        // ── 4. missing weight raises ──────────────────────────────────────
        {
            Builder nb;
            uint32_t s2 = 50;
            auto R2 = [&](std::size_t n) { return fp16_rand(n, s2++); };
            // patch_conv present, ln_pre deliberately omitted.
            nb.add("vision_tower.patch_conv.weight", {D, CP}, R2(static_cast<std::size_t>(D) * CP));
            auto npath = std::filesystem::temp_directory_path() / "brolm_mistral3_vision_missing.safetensors";
            nb.write(npath);
            auto nfile = st::File::open(npath.string());
            bool threw = false;
            try {
                m3::VisionTower tower(cfg);
                tower.load_weights(nfile);
            } catch (const std::exception&) {
                threw = true;
            }
            CHECK(threw);
            std::error_code ec2;
            std::filesystem::remove(npath, ec2);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "mistral3_vision test threw: %s\n", e.what());
        ++g_failures;
    }

    std::error_code ec;
    std::filesystem::remove(path, ec);

    if (g_failures == 0) std::printf("mistral3_vision: OK\n");
    else std::fprintf(stderr, "mistral3_vision: %d failure(s)\n", g_failures);
    return g_failures ? 1 : 0;
}
