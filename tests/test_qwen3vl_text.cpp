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
// (d) forward_capture_hidden_states — capturing every layer must let the
//     final norm + tied lm_head be reproduced by hand from the last captured
//     hidden state, matching forward_embeds' logits; a non-contiguous subset
//     must match the corresponding entries of the all-layers capture; empty/
//     out-of-range/non-ascending capture_layers must throw.
// (e) Real checkpoint — gated on BROLM_QWEN3VL_DIR.

#include "brolm/qwen3vl_config.h"
#include "brolm/qwen3vl_text.h"
#include "brolm/qwen3vl_tokenizer.h"
#include "brolm/detail/compute.h"

#include "brotensor/runtime.h"
#include "brotensor/safetensors.h"
#include "brotensor/tensor.h"

#include "test_compute.h"

#include <algorithm>
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

void run_capture_hidden_states() {
    const auto cfg = make_tiny_cfg();
    const auto path = build_fixture(cfg);
    const int H = cfg.hidden_size;
    const int N = cfg.num_hidden_layers;   // 4 in make_tiny_cfg()

    try {
        auto file = st::File::open(path.string());
        q3vl::TextModel model(cfg);
        model.load_weights(file, "model.");

        const std::vector<int> seq = {5, 17, 2, 31, 9};
        const int L = static_cast<int>(seq.size());
        std::vector<int64_t> mt(L), mh(L), mw(L);
        for (int i = 0; i < L; ++i) { mt[i] = mh[i] = mw[i] = i; }

        bt::Tensor embeds = model.embed_tokens(seq);

        // Capturing every layer's output must let us reconstruct forward_embeds'
        // logits by hand: apply the final norm + tied lm_head to the LAST
        // captured hidden state ourselves and compare against a fresh
        // forward_embeds call over the identical input.
        std::vector<int> all_layers(static_cast<std::size_t>(N));
        for (int i = 0; i < N; ++i) all_layers[static_cast<std::size_t>(i)] = i + 1;

        std::vector<bt::Tensor> hs;
        model.forward_capture_hidden_states(embeds, mt, mh, mw, all_layers, hs);
        bt::sync_all();

        CHECK(hs.size() == static_cast<std::size_t>(N));
        for (const auto& t : hs) {
            CHECK(t.rows == L);
            CHECK(t.cols == H);
        }

        auto cache_ref = model.make_cache(16);
        bt::Tensor logits_ref;
        model.forward_embeds(embeds, mt, mh, mw, cache_ref, logits_ref);
        bt::sync_all();

        // Reproduce forward_embeds' tail (final norm + tied lm_head) by hand
        // on the last captured hidden state, reading the same two weights
        // straight from the fixture file (no TextModel internals needed).
        const auto* norm_view  = file.find("model.norm.weight");
        const auto* embed_view = file.find("model.embed_tokens.weight");
        CHECK(norm_view != nullptr);
        CHECK(embed_view != nullptr);
        bt::Tensor final_norm_w, embed_w;
        st::upload_compute_checked(*norm_view, H, 1, final_norm_w, "model.norm.weight");
        st::upload_compute_checked(*embed_view, cfg.vocab_size, H, embed_w,
                                   "model.embed_tokens.weight");

        bt::Tensor manual_norm, manual_logits;
        bt::rms_norm_forward(hs.back(), final_norm_w, cfg.rms_norm_eps, manual_norm);
        brolm::detail::linear_batched(embed_w, nullptr, manual_norm, manual_logits);
        bt::sync_all();

        std::vector<float> ref = bdtest::bd_download(logits_ref);
        std::vector<float> man = bdtest::bd_download(manual_logits);
        CHECK(ref.size() == man.size());
        float max_diff = 0.0f;
        for (std::size_t i = 0; i < ref.size() && i < man.size(); ++i) {
            max_diff = std::max(max_diff, std::fabs(ref[i] - man[i]));
        }
        std::printf("qwen3vl_text: capture-all-layers vs forward_embeds max diff = %.3e\n",
                    static_cast<double>(max_diff));
        CHECK(max_diff < 1e-2f);

        // A non-contiguous subset (Krea-2-style tap pattern) must return
        // exactly those layers, in order, with finite values.
        std::vector<int> subset = {2, 4};
        std::vector<bt::Tensor> hs2;
        model.forward_capture_hidden_states(embeds, mt, mh, mw, subset, hs2);
        bt::sync_all();
        CHECK(hs2.size() == 2);
        std::vector<float> layer2 = bdtest::bd_download(hs2[0]);
        std::vector<float> layer4 = bdtest::bd_download(hs2[1]);
        std::vector<float> full4  = bdtest::bd_download(hs.back());
        int nonfinite = 0;
        for (float v : layer2) if (!bdtest::bd_finite(v)) ++nonfinite;
        for (float v : layer4) if (!bdtest::bd_finite(v)) ++nonfinite;
        CHECK(nonfinite == 0);
        CHECK(layer4.size() == full4.size());
        float subset_diff = 0.0f;
        for (std::size_t i = 0; i < layer4.size() && i < full4.size(); ++i) {
            subset_diff = std::max(subset_diff, std::fabs(layer4[i] - full4[i]));
        }
        CHECK(subset_diff < 1e-3f);

        // Validation: empty, out-of-range, and non-ascending capture_layers
        // must all throw.
        auto expect_throw = [&](const std::vector<int>& bad, const char* label) {
            std::vector<bt::Tensor> tmp;
            bool threw = false;
            try {
                model.forward_capture_hidden_states(embeds, mt, mh, mw, bad, tmp);
            } catch (const std::exception&) {
                threw = true;
            }
            if (!threw) {
                std::fprintf(stderr, "expected throw for %s\n", label);
                ++g_failures;
            }
        };
        expect_throw({}, "empty");
        expect_throw({0}, "index 0");
        expect_throw({N + 1}, "index beyond num_hidden_layers");
        expect_throw({3, 2}, "non-ascending");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "qwen3vl_text capture_hidden_states threw: %s\n", e.what());
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
    const std::filesystem::path vocab_path  = dir / "vocab.json";
    const std::filesystem::path merges_path = dir / "merges.txt";
    if (!std::filesystem::exists(cfg_path)) {
        std::printf("[skip] real checkpoint not found at %s\n", dir_env);
        return;
    }

    // Discover every model*.safetensors shard (HF's sharded checkpoints split
    // model.language_model.* across more than one file for larger models).
    std::vector<std::filesystem::path> shard_paths;
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        const std::string name = entry.path().filename().string();
        if (name.rfind("model", 0) == 0 && entry.path().extension() == ".safetensors")
            shard_paths.push_back(entry.path());
    }
    std::sort(shard_paths.begin(), shard_paths.end());
    if (shard_paths.empty()) {
        std::printf("[skip] no model*.safetensors shard found in %s\n", dir_env);
        return;
    }

    try {
        auto cfg_full = q3vl::Qwen3VLConfig::load(cfg_path.string());
        const auto& cfg = cfg_full.text;

        std::vector<st::File> shards;
        shards.reserve(shard_paths.size());
        for (const auto& p : shard_paths) shards.emplace_back(st::File::open(p.string()));
        std::vector<const st::File*> shard_ptrs;
        shard_ptrs.reserve(shards.size());
        for (const auto& f : shards) shard_ptrs.push_back(&f);

        q3vl::TextModel model(cfg);
        model.load_weights(shard_ptrs);  // prefix defaults to "model.language_model."
        std::printf("qwen3vl_text: loaded real-checkpoint weights (%zu shard(s))\n",
                    shard_ptrs.size());

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
    run_capture_hidden_states();
    run_real_checkpoint();
    if (g_failures == 0) std::printf("qwen3vl_text: OK\n");
    else std::fprintf(stderr, "qwen3vl_text: %d failure(s)\n", g_failures);
    return g_failures ? 1 : 0;
}
