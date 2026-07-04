// Qwen3-VL text decoder smoke + correctness test.
//
// (a) Synthetic — always runs. Tiny dense config (every layer full
//     attention, full-rotary M-RoPE). Random weights. Length-5 prefill.
//     Asserts shape + finite logits.
// (b) Prefill+decode equivalence — appends one token via a fresh cache slot,
//     compares against a length-6 fresh prefill's final row.
// (c) DeepStack injection — forwarding with a non-trivial DeepstackSplice at
//     a subset of image-token rows changes the output at those rows only
//     changes hidden states (finite, no crash); a zero-feature splice must be
//     a no-op vs no splice at all.
// (d) Real checkpoint — gated on BROLM_QWEN3VL_DIR.

#include "brolm/qwen3vl_config.h"
#include "brolm/qwen3vl_text.h"
#include "brolm/qwen3vl_tokenizer.h"
#include "brolm/detail/compute.h"

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

namespace q3vl = brolm::qwen3vl;
namespace st   = brotensor::safetensors;
namespace bt   = brotensor;

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

q3vl::Qwen3VLConfig::Text make_tiny_cfg() {
    q3vl::Qwen3VLConfig::Text cfg;
    cfg.vocab_size          = 64;
    cfg.hidden_size         = 64;
    cfg.intermediate_size   = 128;
    cfg.num_hidden_layers   = 4;
    cfg.num_attention_heads = 2;
    cfg.num_key_value_heads = 1;
    cfg.head_dim            = 32;     // rotary_dim == 32 (full rotation)
    cfg.attention_bias      = false;
    cfg.rms_norm_eps        = 1e-6f;
    cfg.tie_word_embeddings = true;
    cfg.rope.rope_theta     = 1.0e7f;
    cfg.rope.mrope_section  = {6, 5, 5};   // 2*(6+5+5)=32 == head_dim
    return cfg;
}

std::filesystem::path build_fixture(const q3vl::Qwen3VLConfig::Text& cfg) {
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

    b.add("model.embed_tokens.weight", {V, H}, R(static_cast<std::size_t>(V) * H));

    for (int i = 0; i < cfg.num_hidden_layers; ++i) {
        const std::string p = "model.layers." + std::to_string(i) + ".";
        b.add(p + "input_layernorm.weight", {H}, fp16_ones(H));
        b.add(p + "post_attention_layernorm.weight", {H}, fp16_ones(H));
        b.add(p + "mlp.gate_proj.weight", {Fm, H},
              R(static_cast<std::size_t>(Fm) * H));
        b.add(p + "mlp.up_proj.weight", {Fm, H},
              R(static_cast<std::size_t>(Fm) * H));
        b.add(p + "mlp.down_proj.weight", {H, Fm},
              R(static_cast<std::size_t>(H) * Fm));

        b.add(p + "self_attn.q_proj.weight", {q_d, H},
              R(static_cast<std::size_t>(q_d) * H));
        b.add(p + "self_attn.k_proj.weight", {kv_d, H},
              R(static_cast<std::size_t>(kv_d) * H));
        b.add(p + "self_attn.v_proj.weight", {kv_d, H},
              R(static_cast<std::size_t>(kv_d) * H));
        b.add(p + "self_attn.o_proj.weight", {H, q_d},
              R(static_cast<std::size_t>(H) * q_d));
        b.add(p + "self_attn.q_norm.weight", {HD}, fp16_ones(HD));
        b.add(p + "self_attn.k_norm.weight", {HD}, fp16_ones(HD));
    }
    b.add("model.norm.weight", {H}, fp16_ones(H));

    auto path =
        std::filesystem::temp_directory_path() / "brolm_qwen3vl_text_test.safetensors";
    b.write(path);
    return path;
}

void run_synthetic() {
    const auto cfg = make_tiny_cfg();
    const auto path = build_fixture(cfg);
    const int V = cfg.vocab_size;

    try {
        auto file = st::File::open(path.string());
        q3vl::TextModel model(cfg);
        // Fixture keys are "model.<...>" (no language_model. sub-prefix).
        model.load_weights(file, "model.");

        const std::vector<int> seq = {5, 17, 2, 31, 9};
        const int Lseq = static_cast<int>(seq.size());
        std::vector<int64_t> mt(Lseq), mh(Lseq), mw(Lseq);
        for (int i = 0; i < Lseq; ++i) { mt[i] = mh[i] = mw[i] = i; }

        auto cache = model.make_cache(16);
        bt::Tensor logits;
        model.forward(seq, mt, mh, mw, cache, logits);
        bt::sync_all();

        CHECK(logits.rows == Lseq);
        CHECK(logits.cols == V);
        CHECK(logits.dtype == brolm::compute_dtype());

        std::vector<float> host = bdtest::bd_download(logits);
        int nonfinite = 0;
        for (float v : host) if (!bdtest::bd_finite(v)) ++nonfinite;
        CHECK(nonfinite == 0);

        // Decode one more token with the same cache.
        std::vector<int> next = {21};
        std::vector<int64_t> nt = {Lseq}, nh = {Lseq}, nw = {Lseq};
        bt::Tensor step;
        model.forward(next, nt, nh, nw, cache, step);
        bt::sync_all();
        CHECK(step.rows == 1);
        CHECK(step.cols == V);

        // Cross-check via fresh prefill of length 6 — must match exactly
        // (full attention, no lossy recurrent state to round differently).
        q3vl::TextModel model2(cfg);
        model2.load_weights(file, "model.");
        std::vector<int> full = seq; full.push_back(21);
        std::vector<int64_t> ft(6), fh(6), fw(6);
        for (int i = 0; i < 6; ++i) { ft[i] = fh[i] = fw[i] = i; }
        auto cache2 = model2.make_cache(16);
        bt::Tensor logits6;
        model2.forward(full, ft, fh, fw, cache2, logits6);
        bt::sync_all();
        CHECK(logits6.rows == 6);

        const bool is_fp32 = (brolm::compute_dtype() == bt::Dtype::FP32);
        const float atol = is_fp32 ? 1e-5f : 2e-3f;
        const float rtol = is_fp32 ? 1e-4f : 2e-2f;
        std::vector<float> all = bdtest::bd_download(logits6);
        std::vector<float> dec_row = bdtest::bd_download(step);
        const std::size_t base = static_cast<std::size_t>(5) * V;
        float worst_ratio = 0.0f;
        for (int j = 0; j < V; ++j) {
            const float a = all[base + static_cast<std::size_t>(j)];
            const float c = dec_row[static_cast<std::size_t>(j)];
            const float ratio = std::fabs(a - c) / (atol + rtol * std::fabs(a));
            if (ratio > worst_ratio) worst_ratio = ratio;
        }
        std::printf("qwen3vl_text: prefill-vs-decode worst diff/bound = %.2f\n",
                    static_cast<double>(worst_ratio));
        CHECK(worst_ratio <= 1.0f);

        // truncate_cache: rolling back within range succeeds, out-of-range throws.
        try {
            model.truncate_cache(cache, 3);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "truncate_cache(3) threw: %s\n", e.what());
            ++g_failures;
        }
        bool threw = false;
        try {
            model.truncate_cache(cache, 100);
        } catch (const std::exception&) {
            threw = true;
        }
        CHECK(threw);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "qwen3vl_text synthetic threw: %s\n", e.what());
        ++g_failures;
    }
    std::error_code ec;
    std::filesystem::remove(path, ec);
}

void run_deepstack_injection() {
    const auto cfg = make_tiny_cfg();
    const auto path = build_fixture(cfg);
    const int H = cfg.hidden_size;

    try {
        auto file = st::File::open(path.string());
        q3vl::TextModel model(cfg);
        model.load_weights(file, "model.");

        const std::vector<int> seq = {1, 2, 3, 4};   // 2 leading text + 2 "image" tokens
        const int L = static_cast<int>(seq.size());
        std::vector<int64_t> mt = {0, 1, 2, 2}, mh = {0, 1, 0, 1}, mw = {0, 1, 0, 0};

        bt::Tensor embeds = model.embed_tokens(seq);

        // Baseline: no DeepStack splice.
        auto cache_a = model.make_cache(16);
        bt::Tensor logits_a;
        model.forward_embeds(embeds, mt, mh, mw, cache_a, logits_a);
        bt::sync_all();

        // A DeepstackSplice with all-zero features at rows [2,4) must be a
        // no-op — same logits as the baseline.
        q3vl::DeepstackSplice zero_splice;
        zero_splice.row_start = 2;
        for (int li = 0; li < 2; ++li) {
            zero_splice.per_layer.push_back(bt::Tensor::zeros_on(
                embeds.device, 2, H, embeds.dtype));
        }
        auto cache_b = model.make_cache(16);
        bt::Tensor logits_b;
        model.forward_embeds(embeds, mt, mh, mw, cache_b, logits_b,
                             {zero_splice});
        bt::sync_all();

        std::vector<float> a = bdtest::bd_download(logits_a);
        std::vector<float> bvec = bdtest::bd_download(logits_b);
        CHECK(a.size() == bvec.size());
        float max_diff = 0.0f;
        for (std::size_t i = 0; i < a.size(); ++i) {
            max_diff = std::max(max_diff, std::fabs(a[i] - bvec[i]));
        }
        std::printf("qwen3vl_text: zero-splice max diff vs baseline = %.3e\n",
                    static_cast<double>(max_diff));
        CHECK(max_diff < 1e-3f);

        // A non-zero DeepstackSplice must change the output (still finite).
        q3vl::DeepstackSplice nz_splice;
        nz_splice.row_start = 2;
        for (int li = 0; li < 2; ++li) {
            std::vector<float> feat(static_cast<std::size_t>(2) * H, 5.0f);
            bt::Tensor t;
            if (embeds.dtype == bt::Dtype::FP16) {
                std::vector<uint16_t> bits(feat.size());
                for (std::size_t i = 0; i < feat.size(); ++i)
                    bits[i] = bt::fp32_to_fp16_bits(feat[i]);
                t = bt::Tensor::from_host_fp16(bits.data(), 2, H);
            } else {
                t = bt::Tensor::from_host(feat.data(), 2, H);
            }
            nz_splice.per_layer.push_back(std::move(t));
        }
        auto cache_c = model.make_cache(16);
        bt::Tensor logits_c;
        model.forward_embeds(embeds, mt, mh, mw, cache_c, logits_c, {nz_splice});
        bt::sync_all();
        std::vector<float> cvec = bdtest::bd_download(logits_c);
        int nonfinite = 0;
        float max_diff_nz = 0.0f;
        for (std::size_t i = 0; i < cvec.size(); ++i) {
            if (!bdtest::bd_finite(cvec[i])) ++nonfinite;
            max_diff_nz = std::max(max_diff_nz, std::fabs(cvec[i] - a[i]));
        }
        CHECK(nonfinite == 0);
        std::printf("qwen3vl_text: nonzero-splice max diff vs baseline = %.3e\n",
                    static_cast<double>(max_diff_nz));
        CHECK(max_diff_nz > 1e-3f);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "qwen3vl_text deepstack injection threw: %s\n", e.what());
        ++g_failures;
    }
    std::error_code ec;
    std::filesystem::remove(path, ec);
}

void run_real_checkpoint() {
    const char* dir_env = std::getenv("BROLM_QWEN3VL_DIR");
    if (!dir_env) {
        std::printf("[skip] BROLM_QWEN3VL_DIR not set\n");
        return;
    }
    const std::filesystem::path dir = dir_env;
    const std::filesystem::path cfg_path = dir / "config.json";
    const std::filesystem::path st_path  = dir / "model.safetensors";
    const std::filesystem::path vocab_path  = dir / "vocab.json";
    const std::filesystem::path merges_path = dir / "merges.txt";
    if (!std::filesystem::exists(cfg_path) ||
        !std::filesystem::exists(st_path)) {
        std::printf("[skip] real checkpoint not found at %s\n", dir_env);
        return;
    }

    try {
        auto cfg_full = q3vl::Qwen3VLConfig::load(cfg_path.string());
        const auto& cfg = cfg_full.text;

        q3vl::TextModel model(cfg);
        auto file = st::File::open(st_path.string());
        model.load_weights(file);  // prefix defaults to "model.language_model."
        std::printf("qwen3vl_text: loaded real-checkpoint weights\n");

        std::vector<int> ids;
        if (std::filesystem::exists(vocab_path) &&
            std::filesystem::exists(merges_path)) {
            auto tok = q3vl::Tokenizer::load(vocab_path.string(),
                                             merges_path.string());
            auto enc = tok.encode("The capital of France is");
            ids.assign(enc.begin(), enc.end());
        } else {
            ids = {1, 2, 3, 4, 5};
        }
        if (ids.size() > 16) ids.resize(16);
        const int L = static_cast<int>(ids.size());

        std::vector<int64_t> mt(L), mh(L), mw(L);
        for (int i = 0; i < L; ++i) { mt[i] = mh[i] = mw[i] = i; }

        auto cache = model.make_cache(std::max(L + 4, 32));
        bt::Tensor logits;
        model.forward(ids, mt, mh, mw, cache, logits);
        bt::sync_all();

        CHECK(logits.rows == L);
        CHECK(logits.cols == cfg.vocab_size);

        std::vector<float> host = bdtest::bd_download(logits);
        int nonfinite = 0;
        for (float v : host) if (!bdtest::bd_finite(v)) ++nonfinite;
        CHECK(nonfinite == 0);
        std::printf("qwen3vl_text: real-checkpoint prefill ok (L=%d, nonfinite=%d)\n",
                    L, nonfinite);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "qwen3vl_text real-checkpoint threw: %s\n", e.what());
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
    run_deepstack_injection();
    run_real_checkpoint();
    if (g_failures == 0) std::printf("qwen3vl_text: OK\n");
    else std::fprintf(stderr, "qwen3vl_text: %d failure(s)\n", g_failures);
    return g_failures ? 1 : 0;
}
