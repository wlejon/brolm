// Mistral 3.1 text-generation layer test — token sampling + autoregressive
// generate, mirroring test_qwen_generate.cpp on the Mistral decoder.
//
// Builds a tiny synthetic Mistral model (vocab=48, hidden=32, layers=2, ...)
// the same way test_mistral3_text.cpp does — NO q_norm/k_norm, an UNTIED
// lm_head — then verifies:
//   1. sampling unit tests — greedy / temperature<=0 / top_k==1 all return the
//      argmax; a fixed seed makes two sample_token calls deterministic; a tiny
//      top_p collapses to the argmax;
//   2. generate length — stop_on_eos=false returns exactly max_new_tokens ids,
//      each in [0, vocab);
//   3. greedy generate is deterministic across two runs;
//   4. KV-cache correctness — greedy generate of N tokens equals a from-scratch
//      teacher-forced argmax sequence (prompt + generated[:-1] fed once);
//   5. EOS stops generation and the eos token is excluded.
//
// The sampler and generate loop are the shared brolm::detail core; this test
// pins that the Mistral binding (untied head, mistral3::generate) drives them
// correctly. Numerical accuracy against HF needs real weights (future test).

#include "brolm/mistral3_text.h"
#include "brolm/mistral3_generate.h"
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
#include <random>
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

// ─── safetensors fixture builder (mirrors test_mistral3_text.cpp) ───────────

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

// Build the tiny Mistral safetensors fixture (untied, no QK-norm) and return
// its path.
std::filesystem::path build_fixture(const m3::Mistral3Config::Text& cfg) {
    const int V    = cfg.vocab_size;
    const int H    = cfg.hidden_size;
    const int F    = cfg.intermediate_size;
    const int HD   = cfg.head_dim;
    const int q_d  = cfg.num_attention_heads * HD;
    const int kv_d = cfg.num_key_value_heads * HD;

    Builder b;
    uint32_t seed = 1;
    auto R = [&](std::size_t n) { return fp16_rand(n, seed++); };

    b.add("model.embed_tokens.weight", {V, H},
          R(static_cast<std::size_t>(V) * H));
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

    auto path = std::filesystem::temp_directory_path() /
                "brolm_mistral3_generate_test.safetensors";
    b.write(path);
    return path;
}

int row_argmax(const std::vector<float>& v) {
    int best = 0;
    for (int i = 1; i < static_cast<int>(v.size()); ++i) {
        if (v[static_cast<std::size_t>(i)] > v[static_cast<std::size_t>(best)]) {
            best = i;
        }
    }
    return best;
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
    cfg.head_dim            = 8;
    cfg.rms_norm_eps        = 1e-5f;
    cfg.rope_theta          = 1000000000.0f;
    cfg.tie_word_embeddings = false;   // Mistral: untied lm_head

    const int V = cfg.vocab_size;
    auto path = build_fixture(cfg);

    try {
        auto file = st::File::open(path.string());

        // ── 1. sampling unit tests ────────────────────────────────────────
        {
            std::vector<float> logits(static_cast<std::size_t>(V), 0.1f);
            logits[5]  = 9.0f;   // argmax
            logits[12] = 3.0f;   // runner-up
            logits[30] = 1.5f;

            {
                m3::SamplingParams sp;
                sp.temperature = 0.0f;
                std::mt19937_64 rng(123);
                CHECK(m3::sample_token(logits.data(), V, sp, rng) == 5);
            }
            {
                m3::SamplingParams sp;
                sp.temperature = -1.0f;
                std::mt19937_64 rng(7);
                CHECK(m3::sample_token(logits.data(), V, sp, rng) == 5);
            }
            {
                m3::SamplingParams sp;
                sp.temperature = 1.0f;
                sp.top_k       = 1;
                std::mt19937_64 rng(99);
                CHECK(m3::sample_token(logits.data(), V, sp, rng) == 5);
            }
            {
                m3::SamplingParams sp;
                sp.temperature = 1.0f;
                sp.top_k       = 10;
                std::mt19937_64 rng_a(20260605ull);
                std::mt19937_64 rng_b(20260605ull);
                int a = m3::sample_token(logits.data(), V, sp, rng_a);
                int b = m3::sample_token(logits.data(), V, sp, rng_b);
                CHECK(a == b);
                CHECK(a >= 0 && a < V);
            }
            {
                m3::SamplingParams sp;
                sp.temperature = 1.0f;
                sp.top_p       = 1e-4f;
                std::mt19937_64 rng(555);
                CHECK(m3::sample_token(logits.data(), V, sp, rng) == 5);
            }
        }

        // ── 2. generate length + id range (stop_on_eos = false) ───────────
        const std::vector<int32_t> prompt = {5, 17, 2, 31};
        {
            m3::TextModel model(cfg);
            model.load_weights(file, "");

            m3::GenerateOptions opts;
            opts.max_new_tokens       = 12;
            opts.stop_on_eos          = false;
            opts.sampling.temperature = 0.0f;  // greedy

            std::vector<int32_t> out =
                m3::generate(model, prompt, /*eos_id=*/-1, opts);
            CHECK(static_cast<int>(out.size()) == opts.max_new_tokens);
            for (int32_t id : out) CHECK(id >= 0 && id < V);
        }

        // ── 3. greedy generate is deterministic ───────────────────────────
        std::vector<int32_t> greedy_out;
        {
            m3::GenerateOptions opts;
            opts.max_new_tokens       = 10;
            opts.stop_on_eos          = false;
            opts.sampling.temperature = 0.0f;

            m3::TextModel model_a(cfg);
            model_a.load_weights(file, "");
            std::vector<int32_t> a = m3::generate(model_a, prompt, -1, opts);

            m3::TextModel model_b(cfg);
            model_b.load_weights(file, "");
            std::vector<int32_t> b = m3::generate(model_b, prompt, -1, opts);

            CHECK(a == b);
            greedy_out = a;
        }

        // ── 4. KV-cache correctness — teacher-forced argmax must match ────
        {
            CHECK(greedy_out.size() >= 2);

            std::vector<int32_t> tf(prompt);
            for (std::size_t i = 0; i + 1 < greedy_out.size(); ++i) {
                tf.push_back(greedy_out[i]);
            }

            m3::TextModel model(cfg);
            model.load_weights(file, "");
            model.allocate_cache(static_cast<int>(tf.size()) + 4);
            model.reset_cache();

            bt::Tensor logits;
            model.forward(tf.data(), static_cast<int>(tf.size()), logits);
            bt::sync_all();
            CHECK(logits.rows == static_cast<int>(tf.size()));
            CHECK(logits.cols == V);

            std::vector<float> all = bdtest::bd_download(logits);

            int mismatches = 0;
            for (std::size_t k = 0; k < greedy_out.size(); ++k) {
                std::size_t r = prompt.size() - 1 + k;
                std::size_t base = r * static_cast<std::size_t>(V);
                std::vector<float> rowv(
                    all.begin() + static_cast<std::ptrdiff_t>(base),
                    all.begin() + static_cast<std::ptrdiff_t>(base + V));
                int am = row_argmax(rowv);
                if (am != greedy_out[k]) ++mismatches;
            }
            std::printf("mistral3-generate: teacher-forced mismatches = %d / %d\n",
                        mismatches, static_cast<int>(greedy_out.size()));
            CHECK(mismatches == 0);
        }

        // ── 5. EOS stops generation and is excluded from the output ───────
        // The greedy sequence is deterministic; pick its 3rd token as the eos
        // id and assert generation halts before emitting it (returning exactly
        // the 2 tokens that preceded it).
        {
            CHECK(greedy_out.size() >= 3);
            const int eos = greedy_out[2];

            m3::GenerateOptions opts;
            opts.max_new_tokens       = 10;
            opts.stop_on_eos          = true;
            opts.sampling.temperature = 0.0f;

            m3::TextModel model(cfg);
            model.load_weights(file, "");
            std::vector<int32_t> out = m3::generate(model, prompt, eos, opts);

            CHECK(out.size() == 2);
            CHECK(out.empty() || out.back() != eos);
            CHECK(out[0] == greedy_out[0]);
            CHECK(out[1] == greedy_out[1]);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "mistral3-generate test threw: %s\n", e.what());
        ++g_failures;
    }

    std::error_code ec;
    std::filesystem::remove(path, ec);

    if (g_failures == 0) std::printf("mistral3-generate: OK\n");
    else std::fprintf(stderr, "mistral3-generate: %d failure(s)\n", g_failures);
    return g_failures ? 1 : 0;
}
