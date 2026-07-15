// LLM2Vec encoder smoke + correctness test.
//
// Builds a tiny LLaMA-shaped model (vocab=48, hidden=32, intermediate=64,
// layers=2, q_heads=4, kv_heads=2, head_dim=8), synthesizes a safetensors
// fixture with every HF tensor name (NO lm_head — the encoder never runs one),
// loads it, and runs the bidirectional encode. Verifies:
//   1. shape + dtype + finiteness of the (L, hidden) per-token embeddings;
//   2. determinism — two identical runs are bitwise-identical;
//   3. BIDIRECTIONALITY — changing only the LAST token perturbs the FIRST
//      token's embedding. A causal model computes row 0 without ever seeing
//      later tokens, so this difference is only possible with the causal mask
//      dropped — the defining LLM2Vec property;
//   4. masked mean pooling — encode_pooled over all tokens equals the manual
//      row-mean of the per-token states, and a partial mask pools only the
//      marked rows.
//
// Numerical accuracy against the HF LLM2Vec reference needs the real merged
// checkpoint (scripts/convert-llm2vec.py) and lives in a future gated test.

#include "brolm/llm2vec.h"
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

namespace l2v = brolm::llm2vec;
namespace st  = brotensor::safetensors;
namespace bt  = brotensor;

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

int main() {
    try {
        bt::init();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "init failed: %s\n", e.what());
        return 1;
    }

    l2v::Config cfg;
    cfg.vocab_size          = 48;
    cfg.hidden_size         = 32;
    cfg.intermediate_size   = 64;
    cfg.num_hidden_layers   = 2;
    cfg.num_attention_heads = 4;
    cfg.num_key_value_heads = 2;
    cfg.head_dim            = 8;       // q_dim = 32, kv_dim = 16
    cfg.rms_norm_eps        = 1e-5f;
    cfg.rope_theta          = 500000.0f;

    const int V    = cfg.vocab_size;
    const int H    = cfg.hidden_size;
    const int F    = cfg.intermediate_size;
    const int HD   = cfg.head_dim;
    const int q_d  = cfg.num_attention_heads * HD;   // 32
    const int kv_d = cfg.num_key_value_heads * HD;   // 16

    Builder b;
    uint32_t seed = 1;
    auto R = [&](std::size_t n) { return fp16_rand(n, seed++); };

    b.add("model.embed_tokens.weight", {V, H}, R(static_cast<std::size_t>(V) * H));
    for (int i = 0; i < cfg.num_hidden_layers; ++i) {
        const std::string p = "model.layers." + std::to_string(i) + ".";
        b.add(p + "input_layernorm.weight", {H}, fp16_ones(H));
        b.add(p + "self_attn.q_proj.weight", {q_d, H}, R(static_cast<std::size_t>(q_d) * H));
        b.add(p + "self_attn.k_proj.weight", {kv_d, H}, R(static_cast<std::size_t>(kv_d) * H));
        b.add(p + "self_attn.v_proj.weight", {kv_d, H}, R(static_cast<std::size_t>(kv_d) * H));
        b.add(p + "self_attn.o_proj.weight", {H, q_d}, R(static_cast<std::size_t>(H) * q_d));
        b.add(p + "post_attention_layernorm.weight", {H}, fp16_ones(H));
        b.add(p + "mlp.gate_proj.weight", {F, H}, R(static_cast<std::size_t>(F) * H));
        b.add(p + "mlp.up_proj.weight", {F, H}, R(static_cast<std::size_t>(F) * H));
        b.add(p + "mlp.down_proj.weight", {H, F}, R(static_cast<std::size_t>(H) * F));
    }
    b.add("model.norm.weight", {H}, fp16_ones(H));
    // No lm_head.weight — the encoder ties the (unused) head to the embeddings.

    auto path =
        std::filesystem::temp_directory_path() / "brolm_llm2vec.safetensors";
    b.write(path);

    try {
        auto file = st::File::open(path.string());

        const std::vector<int32_t> seq = {5, 17, 2, 31};
        const int L = static_cast<int>(seq.size());

        // ── 1. shape + dtype + finiteness + 2. determinism ────────────────
        std::vector<float> hid;
        {
            l2v::Encoder enc(cfg);
            enc.load_weights(file, "");
            bt::Tensor h;
            enc.encode(seq.data(), L, h);
            bt::sync_all();
            CHECK(h.rows == L);
            CHECK(h.cols == H);
            CHECK(h.dtype == brolm::compute_dtype());
            hid = bdtest::bd_download(h);
            int nonfinite = 0;
            for (float v : hid) if (!bdtest::bd_finite(v)) ++nonfinite;
            CHECK(nonfinite == 0);

            l2v::Encoder enc2(cfg);
            enc2.load_weights(file, "");
            bt::Tensor h2;
            enc2.encode(seq.data(), L, h2);
            bt::sync_all();
            CHECK(bdtest::bd_download(h2) == hid);
        }

        // ── 3. bidirectionality: changing the LAST token must perturb the
        //       FIRST token's embedding (impossible under a causal mask) ────
        {
            std::vector<int32_t> seq_b = {5, 17, 2, 9};   // last token differs
            l2v::Encoder enc(cfg);
            enc.load_weights(file, "");
            bt::Tensor hb;
            enc.encode(seq_b.data(), L, hb);
            bt::sync_all();
            std::vector<float> hidb = bdtest::bd_download(hb);

            // Row 0 (first token) — max abs diff between the two runs.
            float row0_diff = 0.0f;
            for (int c = 0; c < H; ++c) {
                const float d = std::fabs(hid[static_cast<std::size_t>(c)] -
                                          hidb[static_cast<std::size_t>(c)]);
                if (d > row0_diff) row0_diff = d;
            }
            std::printf("llm2vec: row0 delta on last-token change = %.4e\n",
                        static_cast<double>(row0_diff));
            CHECK(row0_diff > 1e-3f);   // bidirectional: row 0 sees token 3
        }

        // ── 4. masked mean pooling ────────────────────────────────────────
        {
            l2v::Encoder enc(cfg);
            enc.load_weights(file, "");

            // Unmasked pool == manual row-mean of the per-token states.
            bt::Tensor emb;
            enc.encode_pooled(seq.data(), L, emb, /*pool_mask=*/nullptr);
            bt::sync_all();
            CHECK(emb.rows == H);
            CHECK(emb.cols == 1);
            std::vector<float> pooled = bdtest::bd_download(emb);

            float max_err = 0.0f;
            for (int c = 0; c < H; ++c) {
                float mean = 0.0f;
                for (int r = 0; r < L; ++r)
                    mean += hid[static_cast<std::size_t>(r) * H + c];
                mean /= static_cast<float>(L);
                const float err = std::fabs(mean - pooled[static_cast<std::size_t>(c)]);
                if (err > max_err) max_err = err;
            }
            std::printf("llm2vec: mean-pool max err = %.4e\n",
                        static_cast<double>(max_err));
            CHECK(max_err < 5e-3f);

            // Partial mask: pool only the first two tokens.
            std::vector<float> mask = {1.0f, 1.0f, 0.0f, 0.0f};
            bt::Tensor emb2;
            enc.encode_pooled(seq.data(), L, emb2, mask.data());
            bt::sync_all();
            std::vector<float> pooled2 = bdtest::bd_download(emb2);
            float max_err2 = 0.0f;
            for (int c = 0; c < H; ++c) {
                const float mean2 = 0.5f * (hid[static_cast<std::size_t>(c)] +
                                            hid[static_cast<std::size_t>(H) + c]);
                const float err = std::fabs(mean2 - pooled2[static_cast<std::size_t>(c)]);
                if (err > max_err2) max_err2 = err;
            }
            std::printf("llm2vec: masked mean-pool max err = %.4e\n",
                        static_cast<double>(max_err2));
            CHECK(max_err2 < 5e-3f);
        }

        // ── 5. Config parsing from a flat LLaMA config.json ───────────────
        {
            // head_dim omitted → derived from hidden/heads (4096/32 = 128).
            const std::string txt =
                "{\"vocab_size\":128256,\"hidden_size\":4096,"
                "\"intermediate_size\":14336,\"num_hidden_layers\":32,"
                "\"num_attention_heads\":32,\"num_key_value_heads\":8,"
                "\"rms_norm_eps\":1e-05,\"rope_theta\":500000.0}";
            l2v::Config c = l2v::Config::from_json_text(txt);
            CHECK(c.vocab_size == 128256);
            CHECK(c.hidden_size == 4096);
            CHECK(c.num_key_value_heads == 8);
            CHECK(c.head_dim == 128);   // derived
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "llm2vec test threw: %s\n", e.what());
        ++g_failures;
    }

    std::error_code ec;
    std::filesystem::remove(path, ec);

    if (g_failures == 0) std::printf("llm2vec: OK\n");
    else std::fprintf(stderr, "llm2vec: %d failure(s)\n", g_failures);
    return g_failures ? 1 : 0;
}
