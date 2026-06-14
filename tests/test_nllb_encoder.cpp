// NLLB-200 (M2M-100) encoder smoke test.
//
// Builds a tiny encoder (d_model=8, heads=2, ffn=16, 2 layers, vocab=20),
// synthesizes a safetensors fixture with the HF M2M-100 tensor names, loads it,
// and runs forward. Verifies output shape, that all outputs are finite, and
// that two consecutive forwards are identical. Numerical parity vs HF needs the
// real checkpoint and lives in a gated test.

#include "brolm/nllb.h"
#include "brolm/nllb_config.h"
#include "brolm/detail/compute.h"

#include "brotensor/safetensors.h"
#include "brotensor/runtime.h"
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

namespace nllb = brolm::nllb;
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
        if (expected != fp16_bits.size()) std::abort();
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

std::vector<uint16_t> f16_ones(std::size_t n) {
    return std::vector<uint16_t>(n, bt::fp32_to_fp16_bits(1.0f));
}
std::vector<uint16_t> f16_zeros(std::size_t n) {
    return std::vector<uint16_t>(n, bt::fp32_to_fp16_bits(0.0f));
}
// Small distinct values so rows/cols differ but stay well within FP16 range.
std::vector<uint16_t> f16_seq(std::size_t n, float scale) {
    std::vector<uint16_t> out(n);
    for (std::size_t i = 0; i < n; ++i) {
        const float v = std::sin((static_cast<float>(i) + 1.0f) * scale) * 0.1f;
        out[i] = bt::fp32_to_fp16_bits(v);
    }
    return out;
}

}  // namespace

int main() {
    try { bt::init(); }
    catch (const std::exception& e) {
        std::fprintf(stderr, "init failed: %s\n", e.what());
        return 1;
    }

    nllb::NllbConfig cfg;
    cfg.vocab_size = 20;
    cfg.d_model = 8;
    cfg.encoder_attention_heads = 2;
    cfg.decoder_attention_heads = 2;
    cfg.encoder_ffn_dim = 16;
    cfg.decoder_ffn_dim = 16;
    cfg.encoder_layers = 2;
    cfg.decoder_layers = 2;
    cfg.max_position_embeddings = 32;
    cfg.layer_norm_eps = 1e-5f;

    const int V = cfg.vocab_size, D = cfg.d_model, FF = cfg.encoder_ffn_dim;

    Builder b;
    b.add("model.shared.weight", {V, D},
          f16_seq(static_cast<std::size_t>(V) * D, 0.05f));

    auto WDD = [&](float s) { return f16_seq(static_cast<std::size_t>(D) * D, s); };
    for (int i = 0; i < cfg.encoder_layers; ++i) {
        const std::string p = "model.encoder.layers." + std::to_string(i) + ".";
        const std::string sa = p + "self_attn.";
        b.add(sa + "q_proj.weight", {D, D}, WDD(0.11f + i));
        b.add(sa + "k_proj.weight", {D, D}, WDD(0.13f + i));
        b.add(sa + "v_proj.weight", {D, D}, WDD(0.17f + i));
        b.add(sa + "out_proj.weight", {D, D}, WDD(0.19f + i));
        b.add(sa + "q_proj.bias", {D}, f16_zeros(D));
        b.add(sa + "k_proj.bias", {D}, f16_zeros(D));
        b.add(sa + "v_proj.bias", {D}, f16_zeros(D));
        b.add(sa + "out_proj.bias", {D}, f16_zeros(D));
        b.add(p + "self_attn_layer_norm.weight", {D}, f16_ones(D));
        b.add(p + "self_attn_layer_norm.bias", {D}, f16_zeros(D));
        b.add(p + "fc1.weight", {FF, D}, f16_seq(static_cast<std::size_t>(FF) * D, 0.07f));
        b.add(p + "fc1.bias", {FF}, f16_zeros(FF));
        b.add(p + "fc2.weight", {D, FF}, f16_seq(static_cast<std::size_t>(D) * FF, 0.09f));
        b.add(p + "fc2.bias", {D}, f16_zeros(D));
        b.add(p + "final_layer_norm.weight", {D}, f16_ones(D));
        b.add(p + "final_layer_norm.bias", {D}, f16_zeros(D));
    }
    b.add("model.encoder.layer_norm.weight", {D}, f16_ones(D));
    b.add("model.encoder.layer_norm.bias", {D}, f16_zeros(D));

    auto tmp = std::filesystem::temp_directory_path();
    auto path = tmp / "brolm_nllb_encoder.safetensors";
    b.write(path);

    st::File f = st::File::open(path.string());
    nllb::Encoder enc(cfg);
    enc.load_weights(f);

    const std::vector<int32_t> ids = {3, 4, 5, 6};
    const int L = static_cast<int>(ids.size());

    bt::Tensor out1, out2;
    enc.forward(ids.data(), L, out1);
    enc.forward(ids.data(), L, out2);
    bt::sync_all();

    CHECK(out1.rows == L);
    CHECK(out1.cols == D);
    CHECK(out1.dtype == brolm::compute_dtype());

    std::vector<float> h1 = bdtest::bd_download(out1);
    std::vector<float> h2 = bdtest::bd_download(out2);
    CHECK(h1.size() == static_cast<std::size_t>(L) * D);

    int nonfinite = 0;
    for (float v : h1) if (!bdtest::bd_finite(v)) ++nonfinite;
    CHECK(nonfinite == 0);
    CHECK(h1 == h2);

    std::error_code ec;
    std::filesystem::remove(path, ec);

    if (g_failures == 0) std::printf("nllb_encoder: OK\n");
    else std::fprintf(stderr, "nllb_encoder: %d failure(s)\n", g_failures);
    return g_failures ? 1 : 0;
}
