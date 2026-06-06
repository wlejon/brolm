// Mistral 3.1 config parser test.
//
// Parses the real Mistral-Small-3.1-24B-Instruct-2503 config.json (embedded
// verbatim below) and asserts every field lands in the typed struct, then
// checks a few schema edge cases (untied-lm_head default, sliding_window:null
// handling, missing-section errors). No weights or checkpoint needed.

#include "brolm/mistral3_config.h"

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace m3 = brolm::mistral3;

static int g_failures = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        ++g_failures; \
    } \
} while (0)

// Verbatim Mistral-Small-3.1-24B-Instruct-2503 config.json.
static const char* kRealConfig = R"JSON({
  "architectures": ["Mistral3ForConditionalGeneration"],
  "image_token_index": 10,
  "model_type": "mistral3",
  "multimodal_projector_bias": false,
  "projector_hidden_act": "gelu",
  "spatial_merge_size": 2,
  "text_config": {
    "attention_dropout": 0.0,
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 5120,
    "initializer_range": 0.02,
    "intermediate_size": 32768,
    "max_position_embeddings": 131072,
    "model_type": "mistral",
    "num_attention_heads": 32,
    "num_hidden_layers": 40,
    "num_key_value_heads": 8,
    "rms_norm_eps": 1e-05,
    "rope_theta": 1000000000.0,
    "sliding_window": null,
    "use_cache": true,
    "vocab_size": 131072
  },
  "torch_dtype": "bfloat16",
  "transformers_version": "4.50.0.dev0",
  "vision_config": {
    "attention_dropout": 0.0,
    "head_dim": 64,
    "hidden_act": "silu",
    "hidden_size": 1024,
    "image_size": 1540,
    "initializer_range": 0.02,
    "intermediate_size": 4096,
    "model_type": "pixtral",
    "num_attention_heads": 16,
    "num_channels": 3,
    "num_hidden_layers": 24,
    "patch_size": 14,
    "rope_theta": 10000.0
  },
  "vision_feature_layer": -1
})JSON";

static bool approx(float a, float b) {
    return std::fabs(a - b) <= 1e-6f * std::fmax(1.0f, std::fabs(b));
}

int main() {
    try {
        m3::Mistral3Config cfg = m3::Mistral3Config::from_json_text(kRealConfig);

        // ── top-level glue ────────────────────────────────────────────────
        CHECK(cfg.image_token_index == 10);
        CHECK(cfg.spatial_merge_size == 2);
        CHECK(cfg.multimodal_projector_bias == false);

        // ── text_config ───────────────────────────────────────────────────
        const auto& t = cfg.text;
        CHECK(t.vocab_size == 131072);
        CHECK(t.hidden_size == 5120);
        CHECK(t.intermediate_size == 32768);
        CHECK(t.num_hidden_layers == 40);
        CHECK(t.num_attention_heads == 32);
        CHECK(t.num_key_value_heads == 8);
        CHECK(t.head_dim == 128);
        CHECK(approx(t.rms_norm_eps, 1e-5f));
        CHECK(approx(t.rope_theta, 1.0e9f));
        CHECK(t.max_position_embeddings == 131072);
        // tie_word_embeddings is absent from the config → Mistral default false.
        CHECK(t.tie_word_embeddings == false);
        // sliding_window: null → window disabled.
        CHECK(t.has_sliding_window == false);

        // ── vision_config ─────────────────────────────────────────────────
        const auto& v = cfg.vision;
        CHECK(v.hidden_size == 1024);
        CHECK(v.num_attention_heads == 16);
        CHECK(v.head_dim == 64);
        CHECK(v.intermediate_size == 4096);
        CHECK(v.num_hidden_layers == 24);
        CHECK(v.num_channels == 3);
        CHECK(v.patch_size == 14);
        CHECK(v.image_size == 1540);
        CHECK(approx(v.rope_theta, 1.0e4f));
    } catch (const std::exception& e) {
        std::fprintf(stderr, "real-config parse threw: %s\n", e.what());
        ++g_failures;
    }

    // ── sliding_window as a real integer is captured ─────────────────────────
    try {
        const char* windowed = R"JSON({
          "text_config": {"sliding_window": 4096},
          "vision_config": {}
        })JSON";
        m3::Mistral3Config cfg = m3::Mistral3Config::from_json_text(windowed);
        CHECK(cfg.text.has_sliding_window == true);
        CHECK(cfg.text.sliding_window == 4096);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "windowed-config parse threw: %s\n", e.what());
        ++g_failures;
    }

    // ── a missing text_config is a hard error ────────────────────────────────
    {
        bool threw = false;
        try {
            m3::Mistral3Config::from_json_text(R"JSON({"vision_config": {}})JSON");
        } catch (const std::exception&) {
            threw = true;
        }
        CHECK(threw);
    }

    // ── GQA divisibility is validated ────────────────────────────────────────
    {
        bool threw = false;
        try {
            m3::Mistral3Config::from_json_text(R"JSON({
              "text_config": {"num_attention_heads": 32, "num_key_value_heads": 7},
              "vision_config": {}
            })JSON");
        } catch (const std::exception&) {
            threw = true;
        }
        CHECK(threw);
    }

    if (g_failures == 0) std::printf("mistral3_config: OK\n");
    else std::fprintf(stderr, "mistral3_config: %d failure(s)\n", g_failures);
    return g_failures ? 1 : 0;
}
