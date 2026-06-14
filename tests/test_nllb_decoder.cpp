// NLLB-200 (M2M-100) decoder smoke test.
//
// Builds a tiny full model (shared embedding + 1 encoder layer + 1 decoder
// layer with cross-attention), runs the encoder to produce the memory, then
// exercises the decoder: last-token logits shape, finiteness, determinism, that
// the cross-attention memory actually influences the logits, and that the
// causal self-attention makes a longer prefix change the prediction.

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
             const std::vector<uint16_t>& bits) {
        std::size_t expected = 1;
        for (int d : shape) expected *= static_cast<std::size_t>(d);
        if (expected != bits.size()) std::abort();
        std::uint64_t start = payload.size();
        const std::uint8_t* b = reinterpret_cast<const std::uint8_t*>(bits.data());
        payload.insert(payload.end(), b, b + bits.size() * 2);
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
std::vector<uint16_t> f16_seq(std::size_t n, float scale) {
    std::vector<uint16_t> out(n);
    for (std::size_t i = 0; i < n; ++i)
        out[i] = bt::fp32_to_fp16_bits(
            std::sin((static_cast<float>(i) + 1.0f) * scale) * 0.1f);
    return out;
}

int g_D = 8, g_FF = 16;

// Add a q/k/v/out attention block under `base` (e.g. "...self_attn").
void add_attn(Builder& b, const std::string& base, float s) {
    const std::size_t dd = static_cast<std::size_t>(g_D) * g_D;
    b.add(base + ".q_proj.weight", {g_D, g_D}, f16_seq(dd, s + 0.01f));
    b.add(base + ".k_proj.weight", {g_D, g_D}, f16_seq(dd, s + 0.02f));
    b.add(base + ".v_proj.weight", {g_D, g_D}, f16_seq(dd, s + 0.03f));
    b.add(base + ".out_proj.weight", {g_D, g_D}, f16_seq(dd, s + 0.04f));
    b.add(base + ".q_proj.bias", {g_D}, f16_zeros(g_D));
    b.add(base + ".k_proj.bias", {g_D}, f16_zeros(g_D));
    b.add(base + ".v_proj.bias", {g_D}, f16_zeros(g_D));
    b.add(base + ".out_proj.bias", {g_D}, f16_zeros(g_D));
}
void add_ln(Builder& b, const std::string& name) {
    b.add(name + ".weight", {g_D}, f16_ones(g_D));
    b.add(name + ".bias", {g_D}, f16_zeros(g_D));
}
void add_ffn(Builder& b, const std::string& p) {
    b.add(p + "fc1.weight", {g_FF, g_D},
          f16_seq(static_cast<std::size_t>(g_FF) * g_D, 0.07f));
    b.add(p + "fc1.bias", {g_FF}, f16_zeros(g_FF));
    b.add(p + "fc2.weight", {g_D, g_FF},
          f16_seq(static_cast<std::size_t>(g_D) * g_FF, 0.09f));
    b.add(p + "fc2.bias", {g_D}, f16_zeros(g_D));
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
    cfg.encoder_layers = 1;
    cfg.decoder_layers = 1;
    cfg.max_position_embeddings = 32;
    cfg.layer_norm_eps = 1e-5f;
    const int V = cfg.vocab_size, D = cfg.d_model;
    g_D = D; g_FF = cfg.decoder_ffn_dim;

    Builder b;
    b.add("model.shared.weight", {V, D},
          f16_seq(static_cast<std::size_t>(V) * D, 0.05f));

    // encoder layer 0
    add_attn(b, "model.encoder.layers.0.self_attn", 0.11f);
    add_ln(b, "model.encoder.layers.0.self_attn_layer_norm");
    add_ffn(b, "model.encoder.layers.0.");
    add_ln(b, "model.encoder.layers.0.final_layer_norm");
    add_ln(b, "model.encoder.layer_norm");

    // decoder layer 0 (self + cross + ffn)
    add_attn(b, "model.decoder.layers.0.self_attn", 0.21f);
    add_ln(b, "model.decoder.layers.0.self_attn_layer_norm");
    add_attn(b, "model.decoder.layers.0.encoder_attn", 0.31f);
    add_ln(b, "model.decoder.layers.0.encoder_attn_layer_norm");
    add_ffn(b, "model.decoder.layers.0.");
    add_ln(b, "model.decoder.layers.0.final_layer_norm");
    add_ln(b, "model.decoder.layer_norm");

    auto path = std::filesystem::temp_directory_path() /
                "brolm_nllb_decoder.safetensors";
    b.write(path);

    st::File f = st::File::open(path.string());
    nllb::Encoder enc(cfg);
    nllb::Decoder dec(cfg);
    enc.load_weights(f);
    dec.load_weights(f);

    const std::vector<int32_t> src = {3, 7, 5, 9, 2};
    bt::Tensor enc_out;
    enc.forward(src.data(), static_cast<int>(src.size()), enc_out);
    dec.set_encoder_memory(enc_out);

    // Decoder prefix: </s>, target-language BOS, one generated token.
    const std::vector<int32_t> dec_ids = {2, 11, 8};
    bt::Tensor lg1, lg2;
    dec.forward_logits(dec_ids.data(), static_cast<int>(dec_ids.size()), lg1);
    dec.forward_logits(dec_ids.data(), static_cast<int>(dec_ids.size()), lg2);
    bt::sync_all();

    CHECK(lg1.rows == 1);
    CHECK(lg1.cols == V);
    CHECK(lg1.dtype == brolm::compute_dtype());

    std::vector<float> h1 = bdtest::bd_download(lg1);
    std::vector<float> h2 = bdtest::bd_download(lg2);
    int nonfinite = 0;
    for (float v : h1) if (!bdtest::bd_finite(v)) ++nonfinite;
    CHECK(nonfinite == 0);
    CHECK(h1 == h2);   // deterministic

    // Cross-attention must matter: a different source changes the logits.
    {
        const std::vector<int32_t> src2 = {12, 4, 6, 1, 2};
        bt::Tensor eo2, lg3;
        enc.forward(src2.data(), static_cast<int>(src2.size()), eo2);
        dec.set_encoder_memory(eo2);
        dec.forward_logits(dec_ids.data(), static_cast<int>(dec_ids.size()), lg3);
        bt::sync_all();
        std::vector<float> h3 = bdtest::bd_download(lg3);
        // The encoder's final LayerNorm normalizes away most of the token
        // magnitude under these tiny synthetic weights, so the signal is small
        // — but a decoder that ignored the cross-attention memory would give an
        // EXACTLY identical result. Any clear nonzero shift proves it is wired.
        float maxdiff = 0.0f;
        for (std::size_t i = 0; i < h1.size(); ++i)
            maxdiff = std::max(maxdiff, std::fabs(h1[i] - h3[i]));
        CHECK(maxdiff > 1.0e-6f);
        dec.set_encoder_memory(enc_out);   // restore
    }

    // The last token drives the prediction: changing it shifts the logits
    // (it feeds the residual directly, then the tied lm_head).
    {
        bt::Tensor lg_alt;
        const std::vector<int32_t> alt = {2, 11, 9};   // last token 8 -> 9
        dec.forward_logits(alt.data(), static_cast<int>(alt.size()), lg_alt);
        bt::sync_all();
        std::vector<float> ha = bdtest::bd_download(lg_alt);
        float maxdiff = 0.0f;
        for (std::size_t i = 0; i < h1.size(); ++i)
            maxdiff = std::max(maxdiff, std::fabs(h1[i] - ha[i]));
        CHECK(maxdiff > 1.0e-4f);
    }

    // Causal self-attention is still wired (a length-2 prefix runs and the last
    // position attends only to the past): just require a finite, nonzero
    // response — magnitudes are tiny under the synthetic weights, so true
    // causal parity is left to the gated real-checkpoint test.
    {
        bt::Tensor lg_short;
        const std::vector<int32_t> shorter = {2, 11};
        dec.forward_logits(shorter.data(), static_cast<int>(shorter.size()),
                           lg_short);
        bt::sync_all();
        std::vector<float> hs = bdtest::bd_download(lg_short);
        int nf = 0;
        float maxdiff = 0.0f;
        for (std::size_t i = 0; i < h1.size(); ++i) {
            if (!bdtest::bd_finite(hs[i])) ++nf;
            maxdiff = std::max(maxdiff, std::fabs(h1[i] - hs[i]));
        }
        CHECK(nf == 0);
        CHECK(maxdiff > 1.0e-6f);
    }

    std::error_code ec;
    std::filesystem::remove(path, ec);

    if (g_failures == 0) std::printf("nllb_decoder: OK\n");
    else std::fprintf(stderr, "nllb_decoder: %d failure(s)\n", g_failures);
    return g_failures ? 1 : 0;
}
