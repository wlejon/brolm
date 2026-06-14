// NLLB-200 end-to-end Translator test.
//
#define _CRT_SECURE_NO_WARNINGS   // std::getenv for the gated checkpoint path
//
// Always-on: writes a tiny but consistent config.json + tokenizer.json +
// model.safetensors into a temp dir, loads a Translator, and runs the full
// tokenize -> encode -> beam-search -> decode path. The output is gibberish
// (random weights) but the whole pipeline must run and stay structurally sound.
//
// Gated: if NLLB_MODEL_DIR (or weights/nllb-200-distilled-600M) holds a real
// converted checkpoint, it translates a known sentence and prints the result.

#include "brolm/nllb.h"

#include "brotensor/safetensors.h"
#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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
    b.add(p + "fc1.weight", {g_FF, g_D}, seq(static_cast<std::size_t>(g_FF) * g_D, 0.07f));
    b.add(p + "fc1.bias", {g_FF}, zeros(g_FF));
    b.add(p + "fc2.weight", {g_D, g_FF}, seq(static_cast<std::size_t>(g_D) * g_FF, 0.09f));
    b.add(p + "fc2.bias", {g_D}, zeros(g_D));
}

void write_text(const std::filesystem::path& p, const std::string& s) {
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    f << s;
}

// Tokenizer + model share vocab_size 20: specials 0-3, metaspace BPE pieces
// 4-13, language codes eng_Latn=18 / fra_Latn=19 (kept in range so their
// embeddings exist). "a cat" -> metaspace ["▁a","▁cat"] -> [eng_Latn, 5, 8, </s>].
void write_tokenizer(const std::filesystem::path& p) {
    const std::string M = "\xE2\x96\x81";
    std::string s = "{";
    s += "\"added_tokens\":[";
    s += "{\"id\":0,\"content\":\"<s>\",\"special\":true},";
    s += "{\"id\":1,\"content\":\"<pad>\",\"special\":true},";
    s += "{\"id\":2,\"content\":\"</s>\",\"special\":true},";
    s += "{\"id\":3,\"content\":\"<unk>\",\"special\":true},";
    s += "{\"id\":18,\"content\":\"eng_Latn\",\"special\":true},";
    s += "{\"id\":19,\"content\":\"fra_Latn\",\"special\":true}";
    s += "],";
    s += "\"model\":{\"type\":\"BPE\",\"unk_token\":\"<unk>\",\"vocab\":{";
    s += "\"<s>\":0,\"<pad>\":1,\"</s>\":2,\"<unk>\":3,";
    s += "\"" + M + "\":4,\"" + M + "a\":5,\"a\":6,\"cat\":7,";
    s += "\"" + M + "cat\":8,\"c\":9,\"at\":10,\"t\":11,\"ca\":12";
    s += "},\"merges\":[";
    s += "\"" + M + " a\",\"c a\",\"ca t\",\"" + M + " cat\"";
    s += "]}}";
    write_text(p, s);
}

void write_config(const std::filesystem::path& p) {
    write_text(p,
        "{\"model_type\":\"m2m_100\",\"d_model\":8,\"encoder_layers\":1,"
        "\"decoder_layers\":1,\"encoder_attention_heads\":2,"
        "\"decoder_attention_heads\":2,\"encoder_ffn_dim\":16,"
        "\"decoder_ffn_dim\":16,\"vocab_size\":20,"
        "\"max_position_embeddings\":64,\"scale_embedding\":true,"
        "\"bos_token_id\":0,\"pad_token_id\":1,\"eos_token_id\":2,"
        "\"decoder_start_token_id\":2}");
}

void write_model(const std::filesystem::path& p) {
    const int V = 20;
    g_D = 8; g_FF = 16;
    Builder b;
    b.add("model.shared.weight", {V, g_D},
          seq(static_cast<std::size_t>(V) * g_D, 0.05f));
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
    b.write(p);
}

std::string gated_model_dir() {
    if (const char* env = std::getenv("NLLB_MODEL_DIR")) {
        if (env[0]) return env;
    }
    std::error_code ec;
    const std::filesystem::path d = "weights/nllb-200-distilled-600M";
    if (std::filesystem::exists(d / "model.safetensors", ec)) return d.string();
    return {};
}

}  // namespace

int main() {
    try { bt::init(); }
    catch (const std::exception& e) {
        std::fprintf(stderr, "init failed: %s\n", e.what());
        return 1;
    }

    // ── always-on synthetic end-to-end ──────────────────────────────────────
    {
        const std::filesystem::path dir =
            std::filesystem::temp_directory_path() / "brolm_nllb_e2e";
        std::filesystem::create_directories(dir);
        write_config(dir / "config.json");
        write_tokenizer(dir / "tokenizer.json");
        write_model(dir / "model.safetensors");

        nllb::Translator tr = nllb::Translator::load(dir.string());

        nllb::BeamOptions opts;
        opts.num_beams = 3;
        opts.max_new_tokens = 8;

        // translate_ids: hypothesis must start with the forced [</s>, fra_Latn].
        const std::vector<int32_t> src =
            tr.tokenizer().encode_source("a cat", "eng_Latn");
        CHECK(src.front() == tr.tokenizer().lang_id("eng_Latn"));
        CHECK(src.back() == tr.tokenizer().eos_id());

        std::vector<int32_t> hyp = tr.translate_ids(src, "fra_Latn", opts);
        CHECK(hyp.size() >= 2);
        CHECK(hyp[0] == tr.tokenizer().eos_id());
        CHECK(hyp[1] == tr.tokenizer().lang_id("fra_Latn"));

        // Full string path runs and is deterministic.
        std::string out1 = tr.translate("a cat", "eng_Latn", "fra_Latn", opts);
        std::string out2 = tr.translate("a cat", "eng_Latn", "fra_Latn", opts);
        CHECK(out1 == out2);

        std::error_code ec;
        std::filesystem::remove_all(dir, ec);
    }

    // ── gated real-checkpoint translation ───────────────────────────────────
    const std::string real = gated_model_dir();
    if (real.empty()) {
        std::printf("nllb: OK (gated real-checkpoint test skipped — set "
                    "NLLB_MODEL_DIR)\n");
    } else {
        try {
            nllb::Translator tr = nllb::Translator::load(real);
            const std::string out = tr.translate(
                "Hello, world.", "eng_Latn", "fra_Latn");
            std::printf("nllb: eng->fra \"Hello, world.\" => \"%s\"\n",
                        out.c_str());
            CHECK(!out.empty());
        } catch (const std::exception& e) {
            std::fprintf(stderr, "gated translation failed: %s\n", e.what());
            ++g_failures;
        }
    }

    if (g_failures == 0) std::printf("nllb: OK\n");
    else std::fprintf(stderr, "nllb: %d failure(s)\n", g_failures);
    return g_failures ? 1 : 0;
}
