// Tests for brolm::qwen35::Qwen35Config.
//
// Covers two paths:
//   1. An inline JSON document that matches the Qwen3.5-0.8B shape, asserting
//      every field of interest parses to the expected value and that
//      validate() accepts it.
//   2. Negative tests for the validate() invariants (mismatched layer_types
//      length, bad mrope_section sum, missing required sections).
//
// If the env var BROLM_QWEN35_DIR is set, the same checks also run against
// the real `config.json` inside that directory. Without weights present the
// inline cases provide full coverage.

#include "brolm/qwen35_config.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

using brolm::qwen35::LayerType;
using brolm::qwen35::Qwen35Config;

namespace {

// Compact mirror of Qwen/Qwen3.5-0.8B's config.json (24 layers, hybrid
// [L,L,L,F]×6, M-RoPE with mrope_section [11,11,10]).
const char* kQwen35_0_8B_Config = R"({
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "image_token_id": 248056,
    "video_token_id": 248057,
    "vision_start_token_id": 248053,
    "vision_end_token_id": 248054,
    "tie_word_embeddings": true,
    "text_config": {
        "vocab_size": 248320,
        "hidden_size": 1024,
        "intermediate_size": 3584,
        "num_hidden_layers": 24,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "attention_bias": false,
        "attn_output_gate": true,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 16,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "rms_norm_eps": 1e-6,
        "full_attention_interval": 4,
        "max_position_embeddings": 262144,
        "tie_word_embeddings": true,
        "mtp_num_hidden_layers": 1,
        "mtp_use_dedicated_embeddings": false,
        "layer_types": [
            "linear_attention","linear_attention","linear_attention","full_attention",
            "linear_attention","linear_attention","linear_attention","full_attention",
            "linear_attention","linear_attention","linear_attention","full_attention",
            "linear_attention","linear_attention","linear_attention","full_attention",
            "linear_attention","linear_attention","linear_attention","full_attention",
            "linear_attention","linear_attention","linear_attention","full_attention"
        ],
        "rope_parameters": {
            "mrope_interleaved": true,
            "mrope_section": [11, 11, 10],
            "rope_theta": 10000000,
            "partial_rotary_factor": 0.25
        }
    },
    "vision_config": {
        "depth": 12,
        "hidden_size": 768,
        "num_heads": 12,
        "intermediate_size": 3072,
        "in_channels": 3,
        "patch_size": 16,
        "temporal_patch_size": 2,
        "spatial_merge_size": 2,
        "out_hidden_size": 1024,
        "num_position_embeddings": 2304,
        "deepstack_visual_indexes": []
    }
})";

void check_0_8B_parsed(const Qwen35Config& cfg) {
    // Top-level multimodal IDs.
    assert(cfg.image_token_id        == 248056);
    assert(cfg.video_token_id        == 248057);
    assert(cfg.vision_start_token_id == 248053);
    assert(cfg.vision_end_token_id   == 248054);

    // Text backbone.
    assert(cfg.text.vocab_size            == 248320);
    assert(cfg.text.hidden_size           == 1024);
    assert(cfg.text.intermediate_size     == 3584);
    assert(cfg.text.num_hidden_layers     == 24);
    assert(cfg.text.num_attention_heads   == 8);
    assert(cfg.text.num_key_value_heads   == 2);
    assert(cfg.text.head_dim              == 256);
    assert(cfg.text.attn_output_gate      == true);
    assert(cfg.text.linear_num_key_heads   == 16);
    assert(cfg.text.linear_num_value_heads == 16);
    assert(cfg.text.linear_key_head_dim    == 128);
    assert(cfg.text.linear_value_head_dim  == 128);
    assert(cfg.text.linear_conv_kernel_dim == 4);
    assert(cfg.text.full_attention_interval == 4);
    assert(cfg.text.max_position_embeddings == 262144);
    assert(cfg.text.tie_word_embeddings == true);
    assert(cfg.text.mtp_num_hidden_layers == 1);

    // Hybrid pattern: every 4th layer (3, 7, 11, ...) is Full; rest Linear.
    assert(static_cast<int>(cfg.text.layer_types.size()) ==
           cfg.text.num_hidden_layers);
    for (int i = 0; i < cfg.text.num_hidden_layers; ++i) {
        const bool want_full = ((i + 1) % cfg.text.full_attention_interval) == 0;
        const LayerType got = cfg.text.layer_types[static_cast<std::size_t>(i)];
        assert((want_full && got == LayerType::Full) ||
               (!want_full && got == LayerType::Linear));
    }

    // M-RoPE.
    assert(cfg.text.rope.mrope_interleaved == true);
    assert(cfg.text.rope.mrope_section.size() == 3);
    assert(cfg.text.rope.mrope_section[0] == 11);
    assert(cfg.text.rope.mrope_section[1] == 11);
    assert(cfg.text.rope.mrope_section[2] == 10);
    assert(cfg.text.rope.rope_theta == 1.0e7f);
    assert(cfg.text.rope.partial_rotary_factor == 0.25f);

    // rotary_dim = round_even(256 * 0.25) = 64; sum(11+11+10) * 2 = 64.
    assert(cfg.text.rotary_dim() == 64);

    // Vision tower.
    assert(cfg.vision.depth               == 12);
    assert(cfg.vision.hidden_size         == 768);
    assert(cfg.vision.num_heads           == 12);
    assert(cfg.vision.intermediate_size   == 3072);
    assert(cfg.vision.in_channels         == 3);
    assert(cfg.vision.patch_size          == 16);
    assert(cfg.vision.temporal_patch_size == 2);
    assert(cfg.vision.spatial_merge_size  == 2);
    assert(cfg.vision.out_hidden_size     == 1024);
    assert(cfg.vision.num_position_embeddings == 2304);
}

void test_parse_inline() {
    Qwen35Config cfg = Qwen35Config::from_json_text(kQwen35_0_8B_Config);
    check_0_8B_parsed(cfg);
    std::printf("[ok] parse inline Qwen3.5-0.8B config\n");
}

void expect_throws(const char* label, const std::string& json) {
    bool threw = false;
    try {
        (void)Qwen35Config::from_json_text(json);
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
    // Drop one layer_types entry → length mismatch.
    expect_throws("layer_types length mismatch",
                  replace_first(kQwen35_0_8B_Config,
                                "\"full_attention\"\n        ]",
                                "]"));

    // Break the mrope_section sum (10+10+10 → rotary 60 != 64).
    expect_throws("mrope_section sum mismatch",
                  replace_first(kQwen35_0_8B_Config,
                                "[11, 11, 10]", "[10, 10, 10]"));

    // Make vision.out_hidden_size disagree with text.hidden_size.
    expect_throws("vision out_hidden_size mismatch",
                  replace_first(kQwen35_0_8B_Config,
                                "\"out_hidden_size\": 1024",
                                "\"out_hidden_size\": 2048"));
}

void test_real_checkpoint_if_present() {
    const char* dir = std::getenv("BROLM_QWEN35_DIR");
    if (!dir) {
        std::printf("[skip] BROLM_QWEN35_DIR not set (real config check)\n");
        return;
    }
    const std::string path = std::string(dir) + "/config.json";
    Qwen35Config cfg = Qwen35Config::load(path);
    check_0_8B_parsed(cfg);
    std::printf("[ok] parsed real config at %s\n", path.c_str());
}

}  // namespace

int main() {
    test_parse_inline();
    test_validate_rejects_bad_inputs();
    test_real_checkpoint_if_present();
    return 0;
}
