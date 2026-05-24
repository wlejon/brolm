// Qwen3.5 hybrid text decoder smoke + correctness test.
//
// (a) Synthetic — always runs. Tiny config with mixed L/F layers, one F at the
//     end. Random weights. Length-5 prefill. Asserts shape + finite logits.
// (b) Prefill+decode equivalence — appends one token, compares the final-row
//     logits against the prefill's last row.
// (c) Real checkpoint — gated on BROLM_QWEN35_DIR. Loads
//     model.language_model.* from the real 0.8B safetensors and runs a short
//     prefill. Linear layers are stubbed-as-identity in this chunk so the
//     output is not numerically meaningful; we only assert shape and finiteness.

#include "brolm/qwen35_config.h"
#include "brolm/qwen35_text.h"
#include "brolm/qwen35_tokenizer.h"
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

namespace q35 = brolm::qwen35;
namespace st  = brotensor::safetensors;
namespace bt  = brotensor;

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

void run_synthetic() {
    q35::Qwen35Config::Text cfg;
    cfg.vocab_size          = 64;
    cfg.hidden_size         = 64;
    cfg.intermediate_size   = 128;
    cfg.num_hidden_layers   = 4;
    cfg.num_attention_heads = 2;
    cfg.num_key_value_heads = 1;
    cfg.head_dim            = 32;     // q_dim = 64, kv_dim = 32
    cfg.attention_bias      = false;
    cfg.attn_output_gate    = true;
    cfg.rms_norm_eps        = 1e-6f;
    cfg.tie_word_embeddings = true;
    cfg.layer_types = {
        q35::LayerType::Linear, q35::LayerType::Linear,
        q35::LayerType::Linear, q35::LayerType::Full,
    };
    cfg.full_attention_interval = 4;
    // Tiny linear-attn dims for the synthetic test. (Real 0.8B uses
    // 16 heads x 128 head_dim with kernel=4; the recurrence is dtype/shape-
    // polymorphic so we shrink it.)
    cfg.linear_num_key_heads   = 2;
    cfg.linear_num_value_heads = 2;
    cfg.linear_key_head_dim    = 16;
    cfg.linear_value_head_dim  = 16;
    cfg.linear_conv_kernel_dim = 4;
    cfg.rope.rope_theta            = 10000000.0f;
    cfg.rope.partial_rotary_factor = 0.5f;             // rotary_dim = 16
    cfg.rope.mrope_section         = {3, 3, 2};        // 2*(3+3+2)=16

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

    // Linear-attn tensor shapes (synthetic) — match the real checkpoint layout:
    //   in_proj_qkv : (3 * H_lin * D_k, hidden)
    //   in_proj_z   : (H_lin * D_v, hidden)
    //   in_proj_a   : (H_lin, hidden)          per-head scalar projection
    //   in_proj_b   : (H_lin, hidden)
    //   A_log       : (H_lin,)                 per-head learnable decay
    //   dt_bias     : (H_lin,)
    //   conv1d.w    : (3*H_lin*D_k, 1, kK)     depthwise
    //   norm.w      : (D_v,)                   per-head RMSNorm gain
    //   out_proj    : (hidden, H_lin*D_v)
    const int H_lin      = cfg.linear_num_value_heads;
    const int D_k        = cfg.linear_key_head_dim;
    const int D_v        = cfg.linear_value_head_dim;
    const int lin_kv     = H_lin * D_k;            // == H_lin * D_v (equal here)
    const int lin_vv     = H_lin * D_v;
    const int lin_qkv    = 3 * lin_kv;
    const int conv_ch    = lin_qkv;
    const int conv_kd    = cfg.linear_conv_kernel_dim;

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

        if (cfg.layer_types[static_cast<std::size_t>(i)] == q35::LayerType::Full) {
            // q_proj output is 2*q_d (q + gate per head).
            b.add(p + "self_attn.q_proj.weight", {2 * q_d, H},
                  R(static_cast<std::size_t>(2 * q_d) * H));
            b.add(p + "self_attn.k_proj.weight", {kv_d, H},
                  R(static_cast<std::size_t>(kv_d) * H));
            b.add(p + "self_attn.v_proj.weight", {kv_d, H},
                  R(static_cast<std::size_t>(kv_d) * H));
            b.add(p + "self_attn.o_proj.weight", {H, q_d},
                  R(static_cast<std::size_t>(H) * q_d));
            b.add(p + "self_attn.q_norm.weight", {HD}, fp16_ones(HD));
            b.add(p + "self_attn.k_norm.weight", {HD}, fp16_ones(HD));
        } else {
            const std::string lp = p + "linear_attn.";
            b.add(lp + "A_log",   {H_lin}, R(static_cast<std::size_t>(H_lin)));
            b.add(lp + "conv1d.weight",
                  {conv_ch, 1, conv_kd},
                  R(static_cast<std::size_t>(conv_ch) * conv_kd));
            b.add(lp + "dt_bias", {H_lin}, R(static_cast<std::size_t>(H_lin)));
            b.add(lp + "in_proj_a.weight",   {H_lin, H},
                  R(static_cast<std::size_t>(H_lin) * H));
            b.add(lp + "in_proj_b.weight",   {H_lin, H},
                  R(static_cast<std::size_t>(H_lin) * H));
            b.add(lp + "in_proj_qkv.weight", {lin_qkv, H},
                  R(static_cast<std::size_t>(lin_qkv) * H));
            b.add(lp + "in_proj_z.weight",   {lin_vv, H},
                  R(static_cast<std::size_t>(lin_vv) * H));
            b.add(lp + "norm.weight",        {D_v}, fp16_ones(D_v));
            b.add(lp + "out_proj.weight",    {H, lin_vv},
                  R(static_cast<std::size_t>(H) * lin_vv));
        }
    }
    b.add("model.norm.weight", {H}, fp16_ones(H));

    auto path =
        std::filesystem::temp_directory_path() / "brolm_qwen35_text_test.safetensors";
    b.write(path);

    try {
        auto file = st::File::open(path.string());
        q35::TextModel model(cfg);
        // prefix: synthesized fixture keys are "model.<...>" (no language_model).
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

        // Decode one more token with the same cache; expect (1, V) output.
        // Then a fresh-prefill of the full (5+1) sequence's final row should
        // match the decode's row within tolerance.
        std::vector<int> next = {21};
        std::vector<int64_t> nt = {Lseq}, nh = {Lseq}, nw = {Lseq};
        bt::Tensor step;
        model.forward(next, nt, nh, nw, cache, step);
        bt::sync_all();
        CHECK(step.rows == 1);
        CHECK(step.cols == V);

        // Cross-check via fresh prefill of length 6.
        q35::TextModel model2(cfg);
        model2.load_weights(file, "model.");
        std::vector<int> full = seq; full.push_back(21);
        std::vector<int64_t> ft(6), fh(6), fw(6);
        for (int i = 0; i < 6; ++i) { ft[i] = fh[i] = fw[i] = i; }
        auto cache2 = model2.make_cache(16);
        bt::Tensor logits6;
        model2.forward(full, ft, fh, fw, cache2, logits6);
        bt::sync_all();
        CHECK(logits6.rows == 6);

        std::vector<float> all = bdtest::bd_download(logits6);
        std::vector<float> dec_row = bdtest::bd_download(step);
        const std::size_t base = static_cast<std::size_t>(5) * V;
        float max_rel = 0.0f;
        for (int j = 0; j < V; ++j) {
            const float a = all[base + static_cast<std::size_t>(j)];
            const float c = dec_row[static_cast<std::size_t>(j)];
            const float denom = std::max(1e-4f, std::fabs(a));
            const float rel = std::fabs(a - c) / denom;
            if (rel > max_rel) max_rel = rel;
        }
        std::printf("qwen35_text: prefill-vs-decode max relative error = %.3e\n",
                    static_cast<double>(max_rel));
        CHECK(max_rel < 5e-3f);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "qwen35_text synthetic threw: %s\n", e.what());
        ++g_failures;
    }
    std::error_code ec;
    std::filesystem::remove(path, ec);
}

void run_real_checkpoint() {
    const char* dir_env = std::getenv("BROLM_QWEN35_DIR");
    if (!dir_env) {
        std::printf("[skip] BROLM_QWEN35_DIR not set\n");
        return;
    }
    const std::filesystem::path dir = dir_env;
    const std::filesystem::path cfg_path = dir / "config.json";
    const std::filesystem::path st_path =
        dir / "model.safetensors-00001-of-00001.safetensors";
    const std::filesystem::path vocab_path  = dir / "vocab.json";
    const std::filesystem::path merges_path = dir / "merges.txt";
    if (!std::filesystem::exists(cfg_path) ||
        !std::filesystem::exists(st_path)) {
        std::printf("[skip] real checkpoint not found at %s\n", dir_env);
        return;
    }

    try {
        auto cfg_full = q35::Qwen35Config::load(cfg_path.string());
        const auto& cfg = cfg_full.text;

        q35::TextModel model(cfg);
        auto file = st::File::open(st_path.string());
        model.load_weights(file);  // prefix defaults to "model.language_model."
        std::printf("qwen35_text: loaded real-checkpoint weights\n");

        std::vector<int> ids;
        if (std::filesystem::exists(vocab_path) &&
            std::filesystem::exists(merges_path)) {
            auto tok = q35::Tokenizer::load(vocab_path.string(),
                                            merges_path.string());
            auto enc = tok.encode("The capital of France is");
            ids.assign(enc.begin(), enc.end());
        } else {
            // Fall back to a plausible 5-token prompt.
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

        // Inspect the *last* row's top-5 (next-token distribution).
        const int V = cfg.vocab_size;
        const std::size_t base = static_cast<std::size_t>(L - 1) * V;
        std::vector<int> top_ids(5, -1);
        std::vector<float> top_v(5, -1e30f);
        for (int j = 0; j < V; ++j) {
            const float val = host[base + static_cast<std::size_t>(j)];
            // insertion-sort into the top-5.
            for (int s = 0; s < 5; ++s) {
                if (val > top_v[s]) {
                    for (int t = 4; t > s; --t) {
                        top_v[t] = top_v[t-1];
                        top_ids[t] = top_ids[t-1];
                    }
                    top_v[s] = val;
                    top_ids[s] = j;
                    break;
                }
            }
        }

        std::printf("qwen35_text: real-checkpoint prefill ok "
                    "(L=%d, vocab=%d, nonfinite=%d)\n",
                    L, cfg.vocab_size, nonfinite);
        std::printf("qwen35_text: top-5 next-token candidates:\n");
        if (std::filesystem::exists(vocab_path) &&
            std::filesystem::exists(merges_path)) {
            auto tok = q35::Tokenizer::load(vocab_path.string(),
                                            merges_path.string());
            for (int s = 0; s < 5; ++s) {
                std::string decoded;
                if (top_ids[s] >= 0) {
                    decoded = tok.decode({static_cast<int32_t>(top_ids[s])});
                }
                std::printf("  #%d id=%d logit=%.4f text=%s\n",
                            s, top_ids[s], static_cast<double>(top_v[s]),
                            decoded.c_str());
            }
        } else {
            for (int s = 0; s < 5; ++s) {
                std::printf("  #%d id=%d logit=%.4f\n",
                            s, top_ids[s], static_cast<double>(top_v[s]));
            }
        }
        // Sanity: max isn't pinned to a single duplicated id, and the top-5
        // are distinct.
        bool distinct = true;
        for (int i = 0; i < 5; ++i) {
            for (int j = i+1; j < 5; ++j) {
                if (top_ids[i] == top_ids[j]) distinct = false;
            }
        }
        CHECK(distinct);
        // Some entropy: gap between #1 and #5 logits should be finite and
        // bounded (a sane model has a soft top — not winner-take-all).
        const float spread = top_v[0] - top_v[4];
        CHECK(spread > 0.0f);
        CHECK(spread < 100.0f);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "qwen35_text real-checkpoint threw: %s\n", e.what());
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
    if (g_failures == 0) std::printf("qwen35_text: OK\n");
    else std::fprintf(stderr, "qwen35_text: %d failure(s)\n", g_failures);
    return g_failures ? 1 : 0;
}
