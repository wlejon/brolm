// Mistral 3.1 text decoder smoke + correctness test.
//
// Builds a tiny model (vocab=48, hidden=32, intermediate=64, layers=2,
// q_heads=4, kv_heads=2, head_dim=8), synthesizes a safetensors fixture with
// every HF tensor name, loads it, and runs forward. Verifies:
//   1. shape + finiteness of the (L, vocab) logits;
//   2. KV-cache equivalence — prefilling a 4-token sequence in one call
//      produces the same final-token logits as four 1-token decode calls;
//   3. determinism — two identical runs are bitwise-identical.
//
// Mistral differs from the Qwen3 fixture in two ways exercised here: there are
// NO self_attn.q_norm / k_norm tensors (use_qk_norm=false), and the lm_head is
// UNTIED — a distinct lm_head.weight is present (tie_word_embeddings=false).
//
// Numerical accuracy against the HF reference needs real weights and lives in
// a future integration test.

#include "brolm/mistral3_text.h"
#include "brolm/detail/compute.h"
#include "brotensor/safetensors.h"

#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

#include "test_compute.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
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

// ─── safetensors fixture builder ───────────────────────────────────────────

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

// Deterministic small pseudo-random values in roughly [-0.1, 0.1], distinct
// per (name, index) so no two weights are identical.
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

}  // namespace

// ─── test ──────────────────────────────────────────────────────────────────

int main() {
    try {
        bt::init();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "init failed: %s\n", e.what());
        return 1;
    }

    m3::Mistral3Config::Text cfg;
    cfg.vocab_size          = 48;
    cfg.hidden_size         = 32;
    cfg.intermediate_size   = 64;
    cfg.num_hidden_layers   = 2;
    cfg.num_attention_heads = 4;
    cfg.num_key_value_heads = 2;
    cfg.head_dim            = 8;       // q_dim = 32, kv_dim = 16
    cfg.rms_norm_eps        = 1e-5f;
    cfg.rope_theta          = 1000000000.0f;
    cfg.tie_word_embeddings = false;   // Mistral: untied lm_head

    const int V    = cfg.vocab_size;
    const int H    = cfg.hidden_size;
    const int F    = cfg.intermediate_size;
    const int HD   = cfg.head_dim;
    const int q_d  = cfg.num_attention_heads  * HD;   // 32
    const int kv_d = cfg.num_key_value_heads  * HD;   // 16

    Builder b;
    uint32_t seed = 1;
    auto R = [&](std::size_t n) { return fp16_rand(n, seed++); };

    b.add("model.embed_tokens.weight", {V, H}, R(static_cast<std::size_t>(V) * H));

    for (int i = 0; i < cfg.num_hidden_layers; ++i) {
        const std::string p = "model.layers." + std::to_string(i) + ".";
        b.add(p + "input_layernorm.weight", {H}, fp16_ones(H));
        b.add(p + "self_attn.q_proj.weight", {q_d, H},
              R(static_cast<std::size_t>(q_d) * H));
        b.add(p + "self_attn.k_proj.weight", {kv_d, H},
              R(static_cast<std::size_t>(kv_d) * H));
        b.add(p + "self_attn.v_proj.weight", {kv_d, H},
              R(static_cast<std::size_t>(kv_d) * H));
        b.add(p + "self_attn.o_proj.weight", {H, q_d},
              R(static_cast<std::size_t>(H) * q_d));
        // NOTE: no q_norm / k_norm — Mistral has no QK-norm.
        b.add(p + "post_attention_layernorm.weight", {H}, fp16_ones(H));
        b.add(p + "mlp.gate_proj.weight", {F, H},
              R(static_cast<std::size_t>(F) * H));
        b.add(p + "mlp.up_proj.weight", {F, H},
              R(static_cast<std::size_t>(F) * H));
        b.add(p + "mlp.down_proj.weight", {H, F},
              R(static_cast<std::size_t>(H) * F));
    }
    b.add("model.norm.weight", {H}, fp16_ones(H));
    // Untied: a distinct lm_head.weight is present.
    b.add("lm_head.weight", {V, H}, R(static_cast<std::size_t>(V) * H));

    auto path =
        std::filesystem::temp_directory_path() / "brolm_mistral3_text.safetensors";
    b.write(path);

    try {
        auto file = st::File::open(path.string());

        const std::vector<int32_t> seq = {5, 17, 2, 31};
        const int Lseq = static_cast<int>(seq.size());

        // ── 1. shape + finiteness (prefill of the whole sequence) ─────────
        std::vector<float> prefill_logits;
        {
            m3::TextModel model(cfg);
            model.load_weights(file, "");
            model.allocate_cache(16);
            CHECK(model.cache_len() == 0);

            bt::Tensor logits;
            model.forward(seq.data(), Lseq, logits);
            bt::sync_all();

            CHECK(logits.rows == Lseq);
            CHECK(logits.cols == V);
            CHECK(logits.dtype == brolm::compute_dtype());
            CHECK(model.cache_len() == Lseq);

            prefill_logits = bdtest::bd_download(logits);
            int nonfinite = 0;
            for (float v : prefill_logits) {
                if (!bdtest::bd_finite(v)) ++nonfinite;
            }
            CHECK(nonfinite == 0);

            // ── 3. determinism — repeat the exact same run. ───────────────
            m3::TextModel model2(cfg);
            model2.load_weights(file, "");
            model2.allocate_cache(16);
            bt::Tensor logits2;
            model2.forward(seq.data(), Lseq, logits2);
            bt::sync_all();
            std::vector<float> rerun = bdtest::bd_download(logits2);
            CHECK(prefill_logits == rerun);
        }

        // ── 2. KV-cache equivalence ───────────────────────────────────────
        // Decode the same 4 tokens one at a time; the final token's logits
        // must match the prefill's final-token logits.
        {
            m3::TextModel model(cfg);
            model.load_weights(file, "");
            model.allocate_cache(16);
            model.reset_cache();

            std::vector<float> last_decode_logits;
            for (int t = 0; t < Lseq; ++t) {
                bt::Tensor step;
                model.forward(&seq[static_cast<std::size_t>(t)], 1, step);
                bt::sync_all();
                CHECK(step.rows == 1);
                CHECK(step.cols == V);
                CHECK(model.cache_len() == t + 1);
                last_decode_logits = bdtest::bd_download(step);
            }

            const std::size_t base =
                static_cast<std::size_t>(Lseq - 1) * static_cast<std::size_t>(V);
            float max_rel = 0.0f;
            for (int j = 0; j < V; ++j) {
                const float a = prefill_logits[base + static_cast<std::size_t>(j)];
                const float c = last_decode_logits[static_cast<std::size_t>(j)];
                const float denom = std::max(1e-4f, std::fabs(a));
                const float rel = std::fabs(a - c) / denom;
                if (rel > max_rel) max_rel = rel;
            }
            std::printf("mistral3_text: KV-cache max relative error = %.3e\n",
                        static_cast<double>(max_rel));
            CHECK(max_rel < 1e-4f);
        }

        // ── 3b. embed_tokens + forward_embeds == forward(ids) ─────────────
        // The multimodal entry point must reproduce the token-id path exactly
        // when the embeddings are the unmodified token embeddings.
        {
            m3::TextModel model(cfg);
            model.load_weights(file, "");
            model.allocate_cache(16);
            bt::Tensor emb;
            model.embed_tokens(seq.data(), Lseq, emb);
            CHECK(emb.rows == Lseq);
            CHECK(emb.cols == cfg.hidden_size);
            bt::Tensor logits_e;
            model.forward_embeds(emb, Lseq, logits_e);
            bt::sync_all();
            CHECK(model.cache_len() == Lseq);
            std::vector<float> via_embeds = bdtest::bd_download(logits_e);
            CHECK(via_embeds == prefill_logits);
        }

        // ── 4. an untied checkpoint that is missing lm_head must error ─────
        {
            Builder nb;
            uint32_t s2 = 100;
            auto R2 = [&](std::size_t n) { return fp16_rand(n, s2++); };
            nb.add("model.embed_tokens.weight", {V, H},
                   R2(static_cast<std::size_t>(V) * H));
            for (int i = 0; i < cfg.num_hidden_layers; ++i) {
                const std::string p = "model.layers." + std::to_string(i) + ".";
                nb.add(p + "input_layernorm.weight", {H}, fp16_ones(H));
                nb.add(p + "self_attn.q_proj.weight", {q_d, H},
                       R2(static_cast<std::size_t>(q_d) * H));
                nb.add(p + "self_attn.k_proj.weight", {kv_d, H},
                       R2(static_cast<std::size_t>(kv_d) * H));
                nb.add(p + "self_attn.v_proj.weight", {kv_d, H},
                       R2(static_cast<std::size_t>(kv_d) * H));
                nb.add(p + "self_attn.o_proj.weight", {H, q_d},
                       R2(static_cast<std::size_t>(H) * q_d));
                nb.add(p + "post_attention_layernorm.weight", {H}, fp16_ones(H));
                nb.add(p + "mlp.gate_proj.weight", {F, H},
                       R2(static_cast<std::size_t>(F) * H));
                nb.add(p + "mlp.up_proj.weight", {F, H},
                       R2(static_cast<std::size_t>(F) * H));
                nb.add(p + "mlp.down_proj.weight", {H, F},
                       R2(static_cast<std::size_t>(H) * F));
            }
            nb.add("model.norm.weight", {H}, fp16_ones(H));
            // Deliberately omit lm_head.weight.
            auto npath = std::filesystem::temp_directory_path() /
                         "brolm_mistral3_text_notie.safetensors";
            nb.write(npath);
            auto nfile = st::File::open(npath.string());
            bool threw = false;
            try {
                m3::TextModel model(cfg);   // cfg.tie_word_embeddings == false
                model.load_weights(nfile, "");
            } catch (const std::exception&) {
                threw = true;
            }
            CHECK(threw);
            std::error_code ec2;
            std::filesystem::remove(npath, ec2);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "mistral3_text test threw: %s\n", e.what());
        ++g_failures;
    }

    std::error_code ec;
    std::filesystem::remove(path, ec);

    if (g_failures == 0) std::printf("mistral3_text: OK\n");
    else std::fprintf(stderr, "mistral3_text: %d failure(s)\n", g_failures);
    return g_failures ? 1 : 0;
}
