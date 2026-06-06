// Mistral 3.1 multimodal projector smoke + structural test.
//
// Builds a tiny projector (vision_hidden=8, text_hidden=16, merge=2),
// synthesizes a safetensors fixture with every HF `multi_modal_projector.*`
// tensor (bias-free), and projects a 4×4 patch grid (N=16) down to L=4 image
// tokens. Checks:
//   1. shape (L, text_hidden) + finiteness, with L = (h/m)*(w/m);
//   2. determinism — two identical runs are bitwise-identical;
//   3. block locality — each output token depends ONLY on its own merge×merge
//      window: perturbing one patch in block 1's window changes output row 1
//      and leaves rows 0/2/3 byte-identical. This pins the unfold gather's
//      (oh, ow) → patch-window mapping;
//   4. a grid not divisible by merge raises;
//   5. a missing weight raises.
//
// Numerical parity against HF needs the real projector weights (not downloaded)
// and lives in a future integration test.

#include "brolm/mistral3_config.h"
#include "brolm/mistral3_projector.h"
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

std::filesystem::path build_fixture(const m3::Mistral3Config& cfg) {
    const int d  = cfg.vision.hidden_size;
    const int T  = cfg.text.hidden_size;
    const int dm = d * cfg.spatial_merge_size * cfg.spatial_merge_size;

    Builder b;
    uint32_t seed = 1;
    auto R = [&](std::size_t n) { return fp16_rand(n, seed++); };

    b.add("multi_modal_projector.norm.weight", {d}, R(static_cast<std::size_t>(d)));
    b.add("multi_modal_projector.patch_merger.merging_layer.weight", {d, dm},
          R(static_cast<std::size_t>(d) * dm));
    b.add("multi_modal_projector.linear_1.weight", {T, d}, R(static_cast<std::size_t>(T) * d));
    b.add("multi_modal_projector.linear_2.weight", {T, T}, R(static_cast<std::size_t>(T) * T));

    auto path = std::filesystem::temp_directory_path() / "brolm_mistral3_projector.safetensors";
    b.write(path);
    return path;
}

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

    m3::Mistral3Config cfg;
    cfg.vision.hidden_size       = 8;
    cfg.text.hidden_size         = 16;
    cfg.text.rms_norm_eps        = 1e-5f;
    cfg.spatial_merge_size       = 2;
    cfg.multimodal_projector_bias = false;

    const int d = cfg.vision.hidden_size;
    const int T = cfg.text.hidden_size;
    const int m = cfg.spatial_merge_size;
    const int grid_h = 4, grid_w = 4;
    const int N = grid_h * grid_w;             // 16
    const int L = (grid_h / m) * (grid_w / m);  // 4

    auto path = build_fixture(cfg);

    try {
        auto file = st::File::open(path.string());

        std::vector<float> features(static_cast<std::size_t>(N) * d);
        {
            uint32_t s = 909;
            for (float& x : features) {
                s = s * 1664525u + 1013904223u;
                x = (static_cast<float>(s >> 8) / 16777216.0f - 0.5f) * 0.5f;
            }
        }

        // ── 1. shape + finiteness ─────────────────────────────────────────
        std::vector<float> base;
        {
            m3::MultiModalProjector proj(cfg);
            proj.load_weights(file);
            bt::Tensor ft = upload(features, N, d);
            bt::Tensor out;
            proj.forward(ft, grid_h, grid_w, out);
            bt::sync_all();
            CHECK(out.rows == L);
            CHECK(out.cols == T);
            CHECK(out.dtype == brolm::compute_dtype());
            base = bdtest::bd_download(out);
            int nonfinite = 0;
            for (float v : base) if (!bdtest::bd_finite(v)) ++nonfinite;
            CHECK(nonfinite == 0);
        }

        // ── 2. determinism ────────────────────────────────────────────────
        {
            m3::MultiModalProjector proj(cfg);
            proj.load_weights(file);
            bt::Tensor ft = upload(features, N, d);
            bt::Tensor out;
            proj.forward(ft, grid_h, grid_w, out);
            bt::sync_all();
            std::vector<float> again = bdtest::bd_download(out);
            CHECK(base == again);
        }

        // ── 3. block locality ─────────────────────────────────────────────
        // Patch index 2 = grid (row 0, col 2) lives in block (0,1) = output
        // token l=1. Perturbing it must change row 1 only; rows 0/2/3 stay
        // byte-identical (each output token reads only its own 2×2 window).
        {
            std::vector<float> perturbed = features;
            for (int c = 0; c < d; ++c)
                perturbed[static_cast<std::size_t>(2) * d + static_cast<std::size_t>(c)] += 0.37f;

            m3::MultiModalProjector proj(cfg);
            proj.load_weights(file);
            bt::Tensor ft = upload(perturbed, N, d);
            bt::Tensor out;
            proj.forward(ft, grid_h, grid_w, out);
            bt::sync_all();
            std::vector<float> p = bdtest::bd_download(out);

            auto row_eq = [&](int r) {
                for (int c = 0; c < T; ++c) {
                    const std::size_t i = static_cast<std::size_t>(r) * T + static_cast<std::size_t>(c);
                    if (p[i] != base[i]) return false;
                }
                return true;
            };
            float row1_diff = 0.0f;
            for (int c = 0; c < T; ++c) {
                const std::size_t i = static_cast<std::size_t>(1) * T + static_cast<std::size_t>(c);
                row1_diff = std::max(row1_diff, std::fabs(p[i] - base[i]));
            }
            std::printf("mistral3_projector: block-1 perturb → row1 max_diff = %.3e, "
                        "rows 0/2/3 unchanged = %d/%d/%d\n",
                        static_cast<double>(row1_diff), row_eq(0), row_eq(2), row_eq(3));
            CHECK(row1_diff > 1e-4f);  // the perturbed block's token must move
            CHECK(row_eq(0));          // other blocks are untouched
            CHECK(row_eq(2));
            CHECK(row_eq(3));
        }

        // ── 4. grid not divisible by merge raises ─────────────────────────
        {
            m3::MultiModalProjector proj(cfg);
            proj.load_weights(file);
            std::vector<float> odd(static_cast<std::size_t>(3 * 3) * d, 0.1f);
            bt::Tensor ft = upload(odd, 3 * 3, d);
            bt::Tensor out;
            bool threw = false;
            try { proj.forward(ft, 3, 3, out); }
            catch (const std::exception&) { threw = true; }
            CHECK(threw);
        }

        // ── 5. missing weight raises ──────────────────────────────────────
        {
            Builder nb;
            uint32_t s2 = 70;
            auto R2 = [&](std::size_t n) { return fp16_rand(n, s2++); };
            const int dm = d * m * m;
            // norm present, merging_layer omitted.
            nb.add("multi_modal_projector.norm.weight", {d}, R2(static_cast<std::size_t>(d)));
            nb.add("multi_modal_projector.linear_1.weight", {T, d}, R2(static_cast<std::size_t>(T) * d));
            nb.add("multi_modal_projector.linear_2.weight", {T, T}, R2(static_cast<std::size_t>(T) * T));
            (void)dm;
            auto npath = std::filesystem::temp_directory_path() / "brolm_mistral3_projector_missing.safetensors";
            nb.write(npath);
            auto nfile = st::File::open(npath.string());
            bool threw = false;
            try {
                m3::MultiModalProjector proj(cfg);
                proj.load_weights(nfile);
            } catch (const std::exception&) { threw = true; }
            CHECK(threw);
            std::error_code ec2;
            std::filesystem::remove(npath, ec2);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "mistral3_projector test threw: %s\n", e.what());
        ++g_failures;
    }

    std::error_code ec;
    std::filesystem::remove(path, ec);

    if (g_failures == 0) std::printf("mistral3_projector: OK\n");
    else std::fprintf(stderr, "mistral3_projector: %d failure(s)\n", g_failures);
    return g_failures ? 1 : 0;
}
