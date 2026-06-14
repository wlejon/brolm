// NLLB-200 beam-search smoke test.
//
// Builds a tiny full model (shared embedding + 1 encoder + 1 decoder layer),
// runs the encoder, and exercises beam_search: greedy (num_beams=1) must take
// the argmax first token, results are deterministic, hypotheses start with the
// forced prefix, and generation stops at eos or the max-token cap.

#include "brolm/nllb.h"
#include "brolm/nllb_config.h"
#include "brolm/detail/generate.h"

#include "brotensor/safetensors.h"
#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

#include <algorithm>
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

int g_D = 8, g_FF = 16;
std::vector<uint16_t> ones(std::size_t n) {
    return std::vector<uint16_t>(n, bt::fp32_to_fp16_bits(1.0f));
}
std::vector<uint16_t> zeros(std::size_t n) {
    return std::vector<uint16_t>(n, bt::fp32_to_fp16_bits(0.0f));
}
std::vector<uint16_t> seq(std::size_t n, float s) {
    std::vector<uint16_t> out(n);
    for (std::size_t i = 0; i < n; ++i)
        out[i] = bt::fp32_to_fp16_bits(
            std::sin((static_cast<float>(i) + 1.0f) * s) * 0.1f);
    return out;
}
void add_attn(Builder& b, const std::string& base, float s) {
    const std::size_t dd = static_cast<std::size_t>(g_D) * g_D;
    b.add(base + ".q_proj.weight", {g_D, g_D}, seq(dd, s + 0.01f));
    b.add(base + ".k_proj.weight", {g_D, g_D}, seq(dd, s + 0.02f));
    b.add(base + ".v_proj.weight", {g_D, g_D}, seq(dd, s + 0.03f));
    b.add(base + ".out_proj.weight", {g_D, g_D}, seq(dd, s + 0.04f));
    b.add(base + ".q_proj.bias", {g_D}, zeros(g_D));
    b.add(base + ".k_proj.bias", {g_D}, zeros(g_D));
    b.add(base + ".v_proj.bias", {g_D}, zeros(g_D));
    b.add(base + ".out_proj.bias", {g_D}, zeros(g_D));
}
void add_ln(Builder& b, const std::string& name) {
    b.add(name + ".weight", {g_D}, ones(g_D));
    b.add(name + ".bias", {g_D}, zeros(g_D));
}
void add_ffn(Builder& b, const std::string& p) {
    b.add(p + "fc1.weight", {g_FF, g_D},
          seq(static_cast<std::size_t>(g_FF) * g_D, 0.07f));
    b.add(p + "fc1.bias", {g_FF}, zeros(g_FF));
    b.add(p + "fc2.weight", {g_D, g_FF},
          seq(static_cast<std::size_t>(g_D) * g_FF, 0.09f));
    b.add(p + "fc2.bias", {g_D}, zeros(g_D));
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
    cfg.max_position_embeddings = 64;
    cfg.layer_norm_eps = 1e-5f;
    const int V = cfg.vocab_size, D = cfg.d_model;
    g_D = D; g_FF = cfg.decoder_ffn_dim;

    Builder b;
    b.add("model.shared.weight", {V, D},
          seq(static_cast<std::size_t>(V) * D, 0.05f));
    add_attn(b, "model.encoder.layers.0.self_attn", 0.11f);
    add_ln(b, "model.encoder.layers.0.self_attn_layer_norm");
    add_ffn(b, "model.encoder.layers.0.");
    add_ln(b, "model.encoder.layers.0.final_layer_norm");
    add_ln(b, "model.encoder.layer_norm");
    add_attn(b, "model.decoder.layers.0.self_attn", 0.21f);
    add_ln(b, "model.decoder.layers.0.self_attn_layer_norm");
    add_attn(b, "model.decoder.layers.0.encoder_attn", 0.31f);
    add_ln(b, "model.decoder.layers.0.encoder_attn_layer_norm");
    add_ffn(b, "model.decoder.layers.0.");
    add_ln(b, "model.decoder.layers.0.final_layer_norm");
    add_ln(b, "model.decoder.layer_norm");

    auto path = std::filesystem::temp_directory_path() /
                "brolm_nllb_beam.safetensors";
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

    // Forced prefix: </s> (decoder start) then a target-language token.
    const std::vector<int32_t> start = {2, 11};
    const int eos_id = 2;

    // Greedy (num_beams=1): the first generated token must be the argmax of the
    // start prefix's logits.
    {
        bt::Tensor lg;
        dec.forward_logits(start.data(), static_cast<int>(start.size()), lg);
        bt::sync_all();
        std::vector<float> h = brolm::detail::last_row_fp32(lg);
        int argmax = static_cast<int>(
            std::max_element(h.begin(), h.end()) - h.begin());

        nllb::BeamOptions g;
        g.num_beams = 1;
        g.max_new_tokens = 6;
        std::vector<int32_t> out = nllb::beam_search(dec, start, eos_id, g);

        CHECK(out.size() >= start.size());
        CHECK(std::equal(start.begin(), start.end(), out.begin()));
        if (out.size() > start.size()) CHECK(out[start.size()] == argmax);

        // Deterministic.
        std::vector<int32_t> out2 = nllb::beam_search(dec, start, eos_id, g);
        CHECK(out == out2);
    }

    // Beam search (num_beams=3): valid structured hypothesis, bounded length,
    // ends in eos when it stops before the cap.
    {
        nllb::BeamOptions o;
        o.num_beams = 3;
        o.max_new_tokens = 8;
        std::vector<int32_t> out = nllb::beam_search(dec, start, eos_id, o);
        CHECK(std::equal(start.begin(), start.end(), out.begin()));
        CHECK(out.size() <= start.size() + static_cast<std::size_t>(o.max_new_tokens));
        const bool hit_cap =
            out.size() == start.size() + static_cast<std::size_t>(o.max_new_tokens);
        CHECK(out.back() == eos_id || hit_cap);

        std::vector<int32_t> out2 = nllb::beam_search(dec, start, eos_id, o);
        CHECK(out == out2);
    }

    std::error_code ec;
    std::filesystem::remove(path, ec);

    if (g_failures == 0) std::printf("nllb_beam: OK\n");
    else std::fprintf(stderr, "nllb_beam: %d failure(s)\n", g_failures);
    return g_failures ? 1 : 0;
}
