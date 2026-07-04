// Tests for brolm::qwen3vl::Qwen3VLConfig.
//
// Covers two paths:
//   1. An inline JSON document that matches the real Qwen3-VL-4B-Instruct
//      config.json shape, asserting every field of interest parses to the
//      expected value and that validate() accepts it.
//   2. Negative tests for the validate() invariants (bad mrope_section sum,
//      out_hidden_size mismatch, non-increasing deepstack indexes).
//
// If the env var BROLM_QWEN3VL_DIR is set, the same checks also run against
// the real `config.json` inside that directory. Without weights present the
// inline cases provide full coverage.

#include "brolm/qwen3vl_config.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

using brolm::qwen3vl::Qwen3VLConfig;

namespace {

// Mirror of Qwen/Qwen3-VL-4B-Instruct's config.json.
const char* kQwen3VL_4B_Config = R"({
    "architectures": ["Qwen3VLForConditionalGeneration"],
    "image_token_id": 151655,
    "video_token_id": 151656,
    "vision_start_token_id": 151652,
    "vision_end_token_id": 151653,
    "tie_word_embeddings": true,
    "model_type": "qwen3_vl",
    "text_config": {
        "attention_bias": false,
        "attention_dropout": 0.0,
        "bos_token_id": 151643,
        "eos_token_id": 151645,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 2560,
        "initializer_range": 0.02,
        "intermediate_size": 9728,
        "max_position_embeddings": 262144,
        "model_type": "qwen3_vl_text",
        "num_attention_heads": 32,
        "num_hidden_layers": 36,
        "num_key_value_heads": 8,
        "rms_norm_eps": 1e-06,
        "rope_scaling": {
            "mrope_interleaved": true,
            "mrope_section": [24, 20, 20],
            "rope_type": "default"
        },
        "rope_theta": 5000000,
        "tie_word_embeddings": true,
        "use_cache": true,
        "vocab_size": 151936
    },
    "vision_config": {
        "deepstack_visual_indexes": [5, 11, 17],
        "depth": 24,
        "hidden_act": "gelu_pytorch_tanh",
        "hidden_size": 1024,
        "in_channels": 3,
        "initializer_range": 0.02,
        "intermediate_size": 4096,
        "model_type": "qwen3_vl",
        "num_heads": 16,
        "num_position_embeddings": 2304,
        "out_hidden_size": 2560,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2
    }
})";

void check_4B_parsed(const Qwen3VLConfig& cfg) {
    // Top-level multimodal IDs.
    assert(cfg.image_token_id        == 151655);
    assert(cfg.video_token_id        == 151656);
    assert(cfg.vision_start_token_id == 151652);
    assert(cfg.vision_end_token_id   == 151653);

    // Text backbone.
    assert(cfg.text.vocab_size            == 151936);
    assert(cfg.text.hidden_size           == 2560);
    assert(cfg.text.intermediate_size     == 9728);
    assert(cfg.text.num_hidden_layers     == 36);
    assert(cfg.text.num_attention_heads   == 32);
    assert(cfg.text.num_key_value_heads   == 8);
    assert(cfg.text.head_dim              == 128);
    assert(cfg.text.attention_bias        == false);
    assert(cfg.text.max_position_embeddings == 262144);
    assert(cfg.text.tie_word_embeddings == true);

    // M-RoPE — full rotation (no partial_rotary_factor concept here).
    assert(cfg.text.rope.mrope_interleaved == true);
    assert(cfg.text.rope.mrope_section.size() == 3);
    assert(cfg.text.rope.mrope_section[0] == 24);
    assert(cfg.text.rope.mrope_section[1] == 20);
    assert(cfg.text.rope.mrope_section[2] == 20);
    assert(cfg.text.rope.rope_theta == 5.0e6f);
    assert(cfg.text.rotary_dim() == 128);

    // Vision tower.
    assert(cfg.vision.depth               == 24);
    assert(cfg.vision.hidden_size         == 1024);
    assert(cfg.vision.num_heads           == 16);
    assert(cfg.vision.intermediate_size   == 4096);
    assert(cfg.vision.in_channels         == 3);
    assert(cfg.vision.patch_size          == 16);
    assert(cfg.vision.temporal_patch_size == 2);
    assert(cfg.vision.spatial_merge_size  == 2);
    assert(cfg.vision.out_hidden_size     == 2560);
    assert(cfg.vision.num_position_embeddings == 2304);
    assert(cfg.vision.deepstack_visual_indexes.size() == 3);
    assert(cfg.vision.deepstack_visual_indexes[0] == 5);
    assert(cfg.vision.deepstack_visual_indexes[1] == 11);
    assert(cfg.vision.deepstack_visual_indexes[2] == 17);
}

void test_parse_inline() {
    Qwen3VLConfig cfg = Qwen3VLConfig::from_json_text(kQwen3VL_4B_Config);
    check_4B_parsed(cfg);
    std::printf("[ok] parse inline Qwen3-VL-4B-Instruct config\n");
}

void expect_throws(const char* label, const std::string& json) {
    bool threw = false;
    try {
        (void)Qwen3VLConfig::from_json_text(json);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    if (!threw) {
        std::fprintf(stderr, "expected throw for %s but did not\n", label);
        std::abort();
    }
    std::printf("[ok] rejects %s\n", label);
}

std::string replace_first(std::string s, const std::string& from,
                          const std::string& to) {
    const std::size_t pos = s.find(from);
    if (pos == std::string::npos) {
        std::fprintf(stderr, "test bug: '%s' not found in fixture\n", from.c_str());
        std::abort();
    }
    s.replace(pos, from.size(), to);
    return s;
}

void test_validate_rejects_bad_inputs() {
    // Break the mrope_section sum (20+20+20 -> 60*2=120 != head_dim 128).
    expect_throws("mrope_section sum mismatch",
                  replace_first(kQwen3VL_4B_Config,
                                "[24, 20, 20]", "[20, 20, 20]"));

    // Make vision.out_hidden_size disagree with text.hidden_size.
    expect_throws("vision out_hidden_size mismatch",
                  replace_first(kQwen3VL_4B_Config,
                                "\"out_hidden_size\": 2560",
                                "\"out_hidden_size\": 1024"));

    // Non-increasing deepstack_visual_indexes.
    expect_throws("deepstack indexes not strictly increasing",
                  replace_first(kQwen3VL_4B_Config,
                                "[5, 11, 17]", "[11, 5, 17]"));

    // Out-of-range deepstack index (>= depth).
    expect_throws("deepstack index out of range",
                  replace_first(kQwen3VL_4B_Config,
                                "[5, 11, 17]", "[5, 11, 99]"));
}

void test_real_checkpoint_if_present() {
    const char* dir = std::getenv("BROLM_QWEN3VL_DIR");
    if (!dir) {
        std::printf("[skip] BROLM_QWEN3VL_DIR not set (real config check)\n");
        return;
    }
    const std::string path = std::string(dir) + "/config.json";
    Qwen3VLConfig cfg = Qwen3VLConfig::load(path);
    // Real checkpoint may differ in exact dims from the 4B fixture above; just
    // assert it loads and passes validate() (already run inside load()).
    assert(cfg.text.hidden_size > 0);
    assert(cfg.vision.depth > 0);
    std::printf("[ok] parsed real config at %s\n", path.c_str());
}

}  // namespace

int main() {
    test_parse_inline();
    test_validate_rejects_bad_inputs();
    test_real_checkpoint_if_present();
    return 0;
}
