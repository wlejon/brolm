// Gemma-2 decoder smoke + correctness test.
//
// (a) Synthetic — always runs. Tiny config with the four-norm Gemma-2 layer
//     wiring, alternating sliding/global attention, GeGLU MLP, the embedding
//     scale, the (1+weight) norm fold and both soft-caps. Random weights,
//     tied embeddings. Length-5 prefill + 3 greedy decode steps. Asserts:
//     logits shape (·, vocab), finite logits, the final-softcap bound
//     |logit| <= final_logit_softcapping, and a stable decode loop.
// (b) Real checkpoint — gated on BROLM_GEMMA_DIR (an HF gemma-2-2b dir with
//     config.json + tokenizer.json + *.safetensors). Loads config + tokenizer
//     + model, prefills a short prompt and greedy-decodes a few tokens;
//     asserts shape/finiteness, the softcap bound, and a non-degenerate token
//     sequence. No HF python reference — structural + no-NaN + bounded, the
//     way the qwen35 checkpoint test is gated.

#include "brolm/gemma2.h"
#include "brolm/gemma2_config.h"
#include "brolm/gemma_tokenizer.h"
#include "brolm/detail/compute.h"

#include "brotensor/gguf.h"
#include "brotensor/runtime.h"
#include "brotensor/safetensors.h"
#include "brotensor/tensor.h"

#include "test_compute.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

namespace gm = brolm::gemma;
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

// ─── safetensors fixture builder ───────────────────────────────────────────

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

int argmax_row(const std::vector<float>& logits, std::size_t base, int V) {
    int best = 0;
    float bv = logits[base];
    for (int j = 1; j < V; ++j) {
        const float v = logits[base + static_cast<std::size_t>(j)];
        if (v > bv) { bv = v; best = j; }
    }
    return best;
}

void run_synthetic() {
    gm::Gemma2Config cfg;
    cfg.vocab_size            = 64;
    cfg.hidden_size           = 32;
    cfg.intermediate_size     = 64;
    cfg.num_hidden_layers     = 4;
    cfg.num_attention_heads   = 4;
    cfg.num_key_value_heads   = 2;
    cfg.head_dim              = 8;     // q_dim=32, kv_dim=16
    cfg.rms_norm_eps          = 1e-6f;
    cfg.rope_theta            = 10000.0f;
    cfg.tie_word_embeddings   = true;
    cfg.query_pre_attn_scalar = 8.0f; // == head_dim
    cfg.sliding_window        = 3;    // exercise windowing on even layers
    cfg.attn_logit_softcapping  = 50.0f;
    cfg.final_logit_softcapping = 30.0f;
    cfg.max_position_embeddings = 64;

    const int V    = cfg.vocab_size;
    const int H    = cfg.hidden_size;
    const int Fm   = cfg.intermediate_size;
    const int HD   = cfg.head_dim;
    const int n_q  = cfg.num_attention_heads;
    const int n_kv = cfg.num_key_value_heads;
    const int q_d  = n_q  * HD;
    const int kv_d = n_kv * HD;

    Builder b;
    uint32_t seed = 1;
    auto R = [&](std::size_t n) { return fp16_rand(n, seed++); };

    b.add("model.embed_tokens.weight", {V, H},
          R(static_cast<std::size_t>(V) * H));
    for (int i = 0; i < cfg.num_hidden_layers; ++i) {
        const std::string p = "model.layers." + std::to_string(i) + ".";
        b.add(p + "input_layernorm.weight",            {H}, R(H));
        b.add(p + "post_attention_layernorm.weight",   {H}, R(H));
        b.add(p + "pre_feedforward_layernorm.weight",  {H}, R(H));
        b.add(p + "post_feedforward_layernorm.weight", {H}, R(H));
        b.add(p + "self_attn.q_proj.weight", {q_d, H},
              R(static_cast<std::size_t>(q_d) * H));
        b.add(p + "self_attn.k_proj.weight", {kv_d, H},
              R(static_cast<std::size_t>(kv_d) * H));
        b.add(p + "self_attn.v_proj.weight", {kv_d, H},
              R(static_cast<std::size_t>(kv_d) * H));
        b.add(p + "self_attn.o_proj.weight", {H, q_d},
              R(static_cast<std::size_t>(H) * q_d));
        b.add(p + "mlp.gate_proj.weight", {Fm, H},
              R(static_cast<std::size_t>(Fm) * H));
        b.add(p + "mlp.up_proj.weight", {Fm, H},
              R(static_cast<std::size_t>(Fm) * H));
        b.add(p + "mlp.down_proj.weight", {H, Fm},
              R(static_cast<std::size_t>(H) * Fm));
    }
    b.add("model.norm.weight", {H}, R(H));

    auto path = std::filesystem::temp_directory_path() /
                "brolm_gemma2_test.safetensors";
    b.write(path);

    try {
        auto file = st::File::open(path.string());
        gm::Gemma2Model model(cfg);
        model.load_weights(file);  // prefix "" — names are "model.<...>"
        model.allocate_cache(32);

        const std::vector<int32_t> prompt = {5, 17, 2, 31, 9};
        const int L = static_cast<int>(prompt.size());

        bt::Tensor logits;
        model.forward(prompt.data(), L, logits);
        bt::sync_all();

        CHECK(logits.rows == L);
        CHECK(logits.cols == V);
        CHECK(logits.dtype == brolm::compute_dtype());

        std::vector<float> host = bdtest::bd_download(logits);
        int nonfinite = 0;
        float max_abs = 0.0f;
        for (float v : host) {
            if (!bdtest::bd_finite(v)) ++nonfinite;
            if (std::fabs(v) > max_abs) max_abs = std::fabs(v);
        }
        CHECK(nonfinite == 0);
        // Final tanh soft-cap bounds every logit to (-c, c).
        CHECK(max_abs <= cfg.final_logit_softcapping);

        // Greedy decode three steps; cache continues from the prefill.
        std::vector<int32_t> generated;
        int next = argmax_row(host, static_cast<std::size_t>(L - 1) * V, V);
        for (int step = 0; step < 3; ++step) {
            CHECK(next >= 0 && next < V);
            generated.push_back(next);
            bt::Tensor step_logits;
            int32_t tok = next;
            model.forward(&tok, 1, step_logits);
            bt::sync_all();
            CHECK(step_logits.rows == 1);
            CHECK(step_logits.cols == V);
            std::vector<float> sh = bdtest::bd_download(step_logits);
            int snf = 0; float smax = 0.0f;
            for (float v : sh) {
                if (!bdtest::bd_finite(v)) ++snf;
                if (std::fabs(v) > smax) smax = std::fabs(v);
            }
            CHECK(snf == 0);
            CHECK(smax <= cfg.final_logit_softcapping);
            next = argmax_row(sh, 0, V);
        }
        CHECK(model.cache_len() == L + 3);

        // forward_last on a fresh model+cache must reproduce the prefill's last
        // row (same KV ingest; only the L-1 intermediate lm_head rows skipped).
        gm::Gemma2Model model2(cfg);
        model2.load_weights(file);
        model2.allocate_cache(32);
        bt::Tensor last_logits;
        model2.forward_last(prompt.data(), L, last_logits);
        bt::sync_all();
        CHECK(last_logits.rows == 1);
        CHECK(last_logits.cols == V);
        std::vector<float> lh = bdtest::bd_download(last_logits);
        const std::size_t base = static_cast<std::size_t>(L - 1) * V;
        float worst = 0.0f;
        for (int j = 0; j < V; ++j) {
            worst = std::max(worst,
                std::fabs(lh[static_cast<std::size_t>(j)] - host[base + j]));
        }
        const bool is_fp32 = (brolm::compute_dtype() == bt::Dtype::FP32);
        CHECK(worst <= (is_fp32 ? 1e-4f : 5e-2f));
        std::printf("gemma2 synthetic: forward_last vs prefill max abs diff = "
                    "%.3e (max|logit| = %.3f)\n",
                    static_cast<double>(worst), static_cast<double>(max_abs));
    } catch (const std::exception& e) {
        std::fprintf(stderr, "gemma2 synthetic threw: %s\n", e.what());
        ++g_failures;
    }
    std::error_code ec;
    std::filesystem::remove(path, ec);
}

void run_real_checkpoint() {
    const char* dir_env = std::getenv("BROLM_GEMMA_DIR");
    if (!dir_env) {
        std::printf("[skip] BROLM_GEMMA_DIR not set\n");
        return;
    }
    const std::filesystem::path dir = dir_env;
    const std::filesystem::path cfg_path = dir / "config.json";
    const std::filesystem::path tok_path = dir / "tokenizer.json";
    if (!std::filesystem::exists(cfg_path)) {
        std::printf("[skip] config.json not found at %s\n", dir_env);
        return;
    }

    try {
        auto cfg = gm::Gemma2Config::from_safetensors_dir(dir.string());

        // Collect *.safetensors shards in the directory.
        std::vector<std::string> shard_paths;
        for (const auto& entry : std::filesystem::directory_iterator(dir)) {
            const auto& pth = entry.path();
            if (pth.extension() == ".safetensors") {
                shard_paths.push_back(pth.string());
            }
        }
        if (shard_paths.empty()) {
            std::printf("[skip] no *.safetensors in %s\n", dir_env);
            return;
        }
        std::vector<st::File> files;
        files.reserve(shard_paths.size());
        for (const auto& sp : shard_paths) files.push_back(st::File::open(sp));
        std::vector<const st::File*> shard_ptrs;
        for (const auto& f : files) shard_ptrs.push_back(&f);

        gm::Gemma2Model model(cfg);
        model.load_weights(shard_ptrs);
        std::printf("gemma2: loaded real-checkpoint weights (%zu shard(s))\n",
                    shard_paths.size());

        std::vector<int32_t> ids;
        if (std::filesystem::exists(tok_path)) {
            auto tok = gm::Tokenizer::load(tok_path.string());
            ids = tok.encode("The capital of France is");
        } else {
            ids = {2, 651, 6037, 576, 6081, 603};  // plausible fallback
        }
        if (ids.size() > 16) ids.resize(16);
        const int L = static_cast<int>(ids.size());

        model.allocate_cache(L + 8);
        bt::Tensor logits;
        model.forward_last(ids.data(), L, logits);
        bt::sync_all();

        CHECK(logits.rows == 1);
        CHECK(logits.cols == cfg.vocab_size);

        std::vector<float> host = bdtest::bd_download(logits);
        int nonfinite = 0; float max_abs = 0.0f;
        for (float v : host) {
            if (!bdtest::bd_finite(v)) ++nonfinite;
            if (std::fabs(v) > max_abs) max_abs = std::fabs(v);
        }
        CHECK(nonfinite == 0);
        CHECK(max_abs <= cfg.final_logit_softcapping);

        // Greedy-decode 8 tokens; collect the sequence.
        std::vector<int32_t> gen;
        int next = argmax_row(host, 0, cfg.vocab_size);
        for (int step = 0; step < 8; ++step) {
            gen.push_back(next);
            int32_t tok = next;
            bt::Tensor step_logits;
            model.forward(&tok, 1, step_logits);
            bt::sync_all();
            std::vector<float> sh = bdtest::bd_download(step_logits);
            next = argmax_row(sh, 0, cfg.vocab_size);
        }

        std::printf("gemma2: real-checkpoint greedy ids:");
        for (int32_t t : gen) std::printf(" %d", t);
        std::printf("\n");
        if (std::filesystem::exists(tok_path)) {
            auto tok = gm::Tokenizer::load(tok_path.string());
            std::printf("gemma2: decoded = %s\n", tok.decode(gen).c_str());
        }

        // Non-degenerate: not every generated token is identical.
        bool all_same = true;
        for (std::size_t i = 1; i < gen.size(); ++i) {
            if (gen[i] != gen[0]) { all_same = false; break; }
        }
        CHECK(!all_same);
        // EXPECTED-TOKEN-IDS: once you have a reference HF generation for this
        // prompt, paste the expected greedy ids here and assert equality.
        //   const std::vector<int32_t> expected = { ... };
        //   CHECK(gen == expected);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "gemma2 real-checkpoint threw: %s\n", e.what());
        ++g_failures;
    }
}

// ─── gguf checkpoint (gated) ───────────────────────────────────────────────
//
// Gated on BROLM_GEMMA_GGUF (path to a gemma-2-2b .gguf). Loads config + model
// from the gguf container, runs a short prefill + a few greedy decode steps,
// and asserts the same structural / no-NaN / softcap-bound invariants as the
// safetensors checkpoint section. No gguf tokenizer yet (separate follow-up):
// the model is driven with raw token ids.
void run_gguf_checkpoint() {
    const char* gguf_env = std::getenv("BROLM_GEMMA_GGUF");
    if (!gguf_env) {
        std::printf("[skip] BROLM_GEMMA_GGUF not set\n");
        return;
    }
    if (!std::filesystem::exists(gguf_env)) {
        std::printf("[skip] gguf not found at %s\n", gguf_env);
        return;
    }

    try {
        auto file = bt::gguf::File::open(gguf_env);
        auto cfg = gm::Gemma2Config::from_gguf(file);
        std::printf("gemma2 gguf: config hidden=%d layers=%d heads=%d/%d "
                    "head_dim=%d vocab=%d tied=%d\n",
                    cfg.hidden_size, cfg.num_hidden_layers,
                    cfg.num_attention_heads, cfg.num_key_value_heads,
                    cfg.head_dim, cfg.vocab_size,
                    cfg.tie_word_embeddings ? 1 : 0);

        gm::Gemma2Model model(cfg);
        model.load_weights(file);
        std::printf("gemma2 gguf: loaded weights\n");

        // Drive with raw token ids (no gguf tokenizer yet). Keep them in range.
        std::vector<int32_t> ids = {2, 651, 6037, 576, 6081, 603};
        for (int32_t& t : ids) {
            if (t < 0 || t >= cfg.vocab_size) t = t % cfg.vocab_size;
        }
        const int L = static_cast<int>(ids.size());

        model.allocate_cache(L + 8);
        bt::Tensor logits;
        model.forward_last(ids.data(), L, logits);
        bt::sync_all();

        CHECK(logits.rows == 1);
        CHECK(logits.cols == cfg.vocab_size);

        std::vector<float> host = bdtest::bd_download(logits);
        int nonfinite = 0; float max_abs = 0.0f;
        for (float v : host) {
            if (!bdtest::bd_finite(v)) ++nonfinite;
            if (std::fabs(v) > max_abs) max_abs = std::fabs(v);
        }
        CHECK(nonfinite == 0);
        CHECK(max_abs <= cfg.final_logit_softcapping);

        std::vector<int32_t> gen;
        int next = argmax_row(host, 0, cfg.vocab_size);
        for (int step = 0; step < 8; ++step) {
            gen.push_back(next);
            int32_t tok = next;
            bt::Tensor step_logits;
            model.forward(&tok, 1, step_logits);
            bt::sync_all();
            std::vector<float> sh = bdtest::bd_download(step_logits);
            int snf = 0; float smax = 0.0f;
            for (float v : sh) {
                if (!bdtest::bd_finite(v)) ++snf;
                if (std::fabs(v) > smax) smax = std::fabs(v);
            }
            CHECK(snf == 0);
            CHECK(smax <= cfg.final_logit_softcapping);
            next = argmax_row(sh, 0, cfg.vocab_size);
        }
        std::printf("gemma2 gguf: greedy ids:");
        for (int32_t t : gen) std::printf(" %d", t);
        std::printf("\n");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "gemma2 gguf threw: %s\n", e.what());
        ++g_failures;
    }
}

}  // namespace

int main() {
    try {
        bt::init();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "init failed: %s\n", e.what());
        return 1;
    }
    run_synthetic();
    run_real_checkpoint();
    run_gguf_checkpoint();
    if (g_failures == 0) std::printf("gemma2: OK\n");
    else std::fprintf(stderr, "gemma2: %d failure(s)\n", g_failures);
    return g_failures ? 1 : 0;
}
