// CLIP score smoke test.
//
// Exercises brolm::clip_score::CLIPScorer end to end with tiny synthetic
// encoders: a CLIP BPE tokenizer (small vocab.json + merges.txt fixture),
// a tiny clip::TextEncoder, a tiny clip_image::ImageEncoder, and a tiny pair
// of cross-modal projections. set_prompt() is called, then score() on a
// synthetic image; the result must be finite and within [-1, 1] (it is a
// cosine similarity between two L2-normalised vectors).
//
// Constraints driving the tiny configs:
//   - clip::Tokenizer::encode always emits exactly clip::max_length (77) ids,
//     and CLIPScorer::set_prompt asserts the tokenizer length equals the text
//     encoder's max_position. So the text encoder MUST use max_position=77.
//   - clip::max_length and the BOS/EOS ids (49406/49407) are fixed constants,
//     so the text encoder's vocab_size must cover the EOS id (49408).
// Every other dimension is shrunk to keep the test fast.
//
// Numerical accuracy is not checked (needs real CLIP weights) — this is a
// shape / finiteness / range smoke test only.

#include "brolm/clip.h"
#include "brolm/clip_image.h"
#include "brolm/clip_score.h"
#include "brolm/tokenizer.h"
#include "brolm/detail/compute.h"
#include "brotensor/safetensors.h"

#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

namespace clip       = brolm::clip;
namespace clip_image = brolm::clip_image;
namespace clip_score = brolm::clip_score;
namespace st         = brotensor::safetensors;
namespace bt         = brotensor;

static int g_failures = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        ++g_failures; \
    } \
} while (0)

// ─── safetensors fixture builder ───────────────────────────────────────────
//
// Same F16 in-memory builder used by test_clip.cpp / test_clip_image.cpp.

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

std::vector<uint16_t> fp16_zeros(std::size_t n) {
    return std::vector<uint16_t>(n, 0);
}
std::vector<uint16_t> fp16_ones(std::size_t n) {
    return std::vector<uint16_t>(n, bt::fp32_to_fp16_bits(1.0f));
}
std::vector<uint16_t> fp16_seq(std::size_t n, float scale) {
    std::vector<uint16_t> out(n);
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = bt::fp32_to_fp16_bits((static_cast<float>(i) + 1.0f) * scale);
    }
    return out;
}

// Build a tiny CLIP BPE vocab.json + merges.txt. Same minimal fixture shape
// as test_tokenizer_clip.cpp: per-character byte tokens plus a couple of
// </w>-suffixed and merged forms. IDs are arbitrary and well under the
// text encoder's vocab_size.
void write_tokenizer_fixture(const std::filesystem::path& vocab_path,
                             const std::filesystem::path& merges_path) {
    {
        std::ofstream f(vocab_path, std::ios::binary | std::ios::trunc);
        f << "{";
        bool first = true;
        auto add = [&](const std::string& tok, int id) {
            if (!first) f << ",";
            first = false;
            f << "\"";
            for (char c : tok) {
                if (c == '"' || c == '\\') f << '\\' << c;
                else f << c;
            }
            f << "\":" << id;
        };
        add("h",       1);
        add("i",       2);
        add("c",       3);
        add("a",       4);
        add("t",       5);
        add("h</w>",   6);
        add("i</w>",   7);
        add("c</w>",   8);
        add("a</w>",   9);
        add("t</w>",  10);
        add("hi</w>", 11);
        add("cat</w>",12);
        add("ca",     13);
        f << "}";
    }
    {
        std::ofstream f(merges_path);
        f << "#version: test\n";
        f << "h i</w>\n";
        f << "c a\n";
        f << "ca t</w>\n";
    }
}

// Build the text-encoder safetensors fixture under the "text_model." prefix.
void add_text_encoder_weights(Builder& b, const clip::TextEncoderConfig& cfg) {
    const int V = cfg.vocab_size;
    const int P = cfg.max_position;
    const int D = cfg.hidden_dim;
    const int F = cfg.intermediate_dim;
    const std::string p = "text_model.";

    b.add(p + "embeddings.token_embedding.weight", {V, D},
          fp16_seq(static_cast<std::size_t>(V) * D, 0.001f));
    b.add(p + "embeddings.position_embedding.weight", {P, D},
          fp16_seq(static_cast<std::size_t>(P) * D, 0.001f));

    auto W = [&](float s) { return fp16_seq(static_cast<std::size_t>(D) * D, s); };
    for (int i = 0; i < cfg.num_layers; ++i) {
        const std::string lp =
            p + "encoder.layers." + std::to_string(i) + ".";
        b.add(lp + "layer_norm1.weight", {D}, fp16_ones(D));
        b.add(lp + "layer_norm1.bias",   {D}, fp16_zeros(D));
        b.add(lp + "self_attn.q_proj.weight",   {D, D}, W(0.02f));
        b.add(lp + "self_attn.q_proj.bias",     {D},    fp16_zeros(D));
        b.add(lp + "self_attn.k_proj.weight",   {D, D}, W(0.03f));
        b.add(lp + "self_attn.k_proj.bias",     {D},    fp16_zeros(D));
        b.add(lp + "self_attn.v_proj.weight",   {D, D}, W(0.04f));
        b.add(lp + "self_attn.v_proj.bias",     {D},    fp16_zeros(D));
        b.add(lp + "self_attn.out_proj.weight", {D, D}, W(0.05f));
        b.add(lp + "self_attn.out_proj.bias",   {D},    fp16_zeros(D));
        b.add(lp + "layer_norm2.weight", {D}, fp16_ones(D));
        b.add(lp + "layer_norm2.bias",   {D}, fp16_zeros(D));
        b.add(lp + "mlp.fc1.weight", {F, D},
              fp16_seq(static_cast<std::size_t>(F) * D, 0.001f));
        b.add(lp + "mlp.fc1.bias",   {F}, fp16_zeros(F));
        b.add(lp + "mlp.fc2.weight", {D, F},
              fp16_seq(static_cast<std::size_t>(D) * F, 0.001f));
        b.add(lp + "mlp.fc2.bias",   {D}, fp16_zeros(D));
    }
    b.add(p + "final_layer_norm.weight", {D}, fp16_ones(D));
    b.add(p + "final_layer_norm.bias",   {D}, fp16_zeros(D));
}

// Build the image-encoder safetensors fixture under the "vision_model." prefix.
void add_image_encoder_weights(Builder& b,
                               const clip_image::ImageEncoderConfig& cfg) {
    const int D = cfg.hidden_dim;
    const int F = cfg.intermediate_dim;
    const int C = cfg.in_channels;
    const int P = cfg.patch_size;
    const int T = clip_image::num_tokens(cfg);
    const std::string p = "vision_model.";

    b.add(p + "embeddings.patch_embedding.weight", {D, C, P, P},
          fp16_seq(static_cast<std::size_t>(D) * C * P * P, 0.01f));
    b.add(p + "embeddings.class_embedding", {D}, fp16_seq(D, 0.05f));
    b.add(p + "embeddings.position_embedding.weight", {T, D},
          fp16_seq(static_cast<std::size_t>(T) * D, 0.02f));
    b.add(p + "pre_layrnorm.weight", {D}, fp16_ones(D));
    b.add(p + "pre_layrnorm.bias",   {D}, fp16_zeros(D));

    auto W = [&](float s) { return fp16_seq(static_cast<std::size_t>(D) * D, s); };
    for (int i = 0; i < cfg.num_layers; ++i) {
        const std::string lp =
            p + "encoder.layers." + std::to_string(i) + ".";
        b.add(lp + "layer_norm1.weight", {D}, fp16_ones(D));
        b.add(lp + "layer_norm1.bias",   {D}, fp16_zeros(D));
        b.add(lp + "self_attn.q_proj.weight",   {D, D}, W(0.02f));
        b.add(lp + "self_attn.q_proj.bias",     {D},    fp16_zeros(D));
        b.add(lp + "self_attn.k_proj.weight",   {D, D}, W(0.03f));
        b.add(lp + "self_attn.k_proj.bias",     {D},    fp16_zeros(D));
        b.add(lp + "self_attn.v_proj.weight",   {D, D}, W(0.04f));
        b.add(lp + "self_attn.v_proj.bias",     {D},    fp16_zeros(D));
        b.add(lp + "self_attn.out_proj.weight", {D, D}, W(0.05f));
        b.add(lp + "self_attn.out_proj.bias",   {D},    fp16_zeros(D));
        b.add(lp + "layer_norm2.weight", {D}, fp16_ones(D));
        b.add(lp + "layer_norm2.bias",   {D}, fp16_zeros(D));
        b.add(lp + "mlp.fc1.weight", {F, D},
              fp16_seq(static_cast<std::size_t>(F) * D, 0.01f));
        b.add(lp + "mlp.fc1.bias",   {F}, fp16_zeros(F));
        b.add(lp + "mlp.fc2.weight", {D, F},
              fp16_seq(static_cast<std::size_t>(D) * F, 0.01f));
        b.add(lp + "mlp.fc2.bias",   {D}, fp16_zeros(D));
    }
    b.add(p + "post_layernorm.weight", {D}, fp16_ones(D));
    b.add(p + "post_layernorm.bias",   {D}, fp16_zeros(D));
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

    // Text encoder: max_position MUST be clip::max_length (77) — the tokenizer
    // always emits exactly that many ids and CLIPScorer::set_prompt checks it.
    // vocab_size must cover the fixed EOS id (49407), so use clip::vocab_size.
    clip::TextEncoderConfig tcfg;
    tcfg.vocab_size       = clip::vocab_size;     // 49408
    tcfg.max_position     = clip::max_length;     // 77
    tcfg.hidden_dim       = 8;
    tcfg.num_heads        = 2;
    tcfg.num_layers       = 1;
    tcfg.intermediate_dim = 16;
    tcfg.layer_norm_eps   = 1e-5f;
    tcfg.eos_token_id     = clip::eos_id;

    // Image encoder: tiny ViT, image_size=8 / patch_size=4 -> 5 tokens.
    clip_image::ImageEncoderConfig icfg;
    icfg.image_size       = 8;
    icfg.patch_size       = 4;
    icfg.in_channels      = 3;
    icfg.hidden_dim       = 8;
    icfg.num_heads        = 2;
    icfg.num_layers       = 1;
    icfg.intermediate_dim = 16;
    icfg.layer_norm_eps   = 1e-5f;

    // Shared cross-modal projection dim.
    clip_score::Config scfg;
    scfg.projection_dim = 8;

    // ── build the weight fixtures ──────────────────────────────────────────
    Builder text_b;
    add_text_encoder_weights(text_b, tcfg);
    auto text_path = std::filesystem::temp_directory_path() /
                     "brolm_clip_score_text.safetensors";
    text_b.write(text_path);

    Builder image_b;
    add_image_encoder_weights(image_b, icfg);
    auto image_path = std::filesystem::temp_directory_path() /
                      "brolm_clip_score_image.safetensors";
    image_b.write(image_path);

    // Projections: visual_projection (P, vision_D), text_projection (P, text_D).
    Builder proj_b;
    proj_b.add("visual_projection.weight",
               {scfg.projection_dim, icfg.hidden_dim},
               fp16_seq(static_cast<std::size_t>(scfg.projection_dim) *
                            icfg.hidden_dim,
                        0.05f));
    proj_b.add("text_projection.weight",
               {scfg.projection_dim, tcfg.hidden_dim},
               fp16_seq(static_cast<std::size_t>(scfg.projection_dim) *
                            tcfg.hidden_dim,
                        0.05f));
    auto proj_path = std::filesystem::temp_directory_path() /
                     "brolm_clip_score_proj.safetensors";
    proj_b.write(proj_path);

    // Tokenizer fixture.
    auto vocab_path = std::filesystem::temp_directory_path() /
                      "brolm_clip_score_vocab.json";
    auto merges_path = std::filesystem::temp_directory_path() /
                       "brolm_clip_score_merges.txt";
    write_tokenizer_fixture(vocab_path, merges_path);

    {
        auto tok = clip::Tokenizer::load(vocab_path.string(),
                                         merges_path.string());

        clip::TextEncoder text_enc(tcfg);
        {
            auto f = st::File::open(text_path.string());
            text_enc.load_weights(f, "text_model.");
        }

        clip_image::ImageEncoder image_enc(icfg);
        {
            auto f = st::File::open(image_path.string());
            image_enc.load_weights(f, "vision_model.");
        }

        clip_score::CLIPScorer scorer(tok, text_enc, image_enc, scfg);
        {
            auto f = st::File::open(proj_path.string());
            scorer.load_projections(f, "");
        }

        // set_prompt: tokenize + encode + pool + project + cache.
        scorer.set_prompt("hi cat");
        const std::vector<float>& tf = scorer.text_feature();
        CHECK(tf.size() == static_cast<std::size_t>(scfg.projection_dim));
        int tf_nonfinite = 0;
        for (float v : tf) if (!std::isfinite(v)) ++tf_nonfinite;
        CHECK(tf_nonfinite == 0);

        // Synthetic image: (3 * H * W) FP32, NCHW planar, values in [-1, 1].
        const int H = 12;
        const int W = 12;
        std::vector<float> image(static_cast<std::size_t>(3) * H * W);
        for (std::size_t i = 0; i < image.size(); ++i) {
            // Deterministic ramp wrapped into [-1, 1].
            float t = static_cast<float>(i % 41) / 40.0f;   // [0, 1]
            image[i] = 2.0f * t - 1.0f;
        }

        const float s = scorer.score(image, H, W);
        std::printf("clip_score: score=%.6f\n", s);
        CHECK(std::isfinite(s));
        // Cosine similarity of two L2-normalised vectors is in [-1, 1].
        // A small epsilon absorbs FP16 rounding drift.
        CHECK(s >= -1.0001f && s <= 1.0001f);

        // score() is callable repeatedly after one set_prompt; a second call
        // on the same image must reproduce the same value (deterministic).
        const float s2 = scorer.score(image, H, W);
        CHECK(s == s2);
    }

    std::error_code ec;
    std::filesystem::remove(text_path, ec);
    std::filesystem::remove(image_path, ec);
    std::filesystem::remove(proj_path, ec);
    std::filesystem::remove(vocab_path, ec);
    std::filesystem::remove(merges_path, ec);

    if (g_failures == 0) std::printf("clip_score: OK\n");
    else std::fprintf(stderr, "clip_score: %d failure(s)\n", g_failures);
    return g_failures ? 1 : 0;
}
