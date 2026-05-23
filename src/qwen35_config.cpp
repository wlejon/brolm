#include "brolm/qwen35_config.h"

#include "brolm/detail/json.h"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace brolm::qwen35 {

namespace j = brolm::detail::json;

namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("qwen35::Qwen35Config: " + msg);
}

std::string slurp(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) fail("cannot open '" + path + "'");
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

int get_int(const j::Value& obj, const std::string& key, int dflt) {
    return obj.get_int(key, dflt);
}

float get_float(const j::Value& obj, const std::string& key, float dflt) {
    return obj.get_float(key, dflt);
}

bool get_bool(const j::Value& obj, const std::string& key, bool dflt) {
    return obj.get_bool(key, dflt);
}

LayerType parse_layer_type(const std::string& s) {
    if (s == "linear_attention") return LayerType::Linear;
    if (s == "full_attention")   return LayerType::Full;
    fail("unknown layer_type '" + s + "'");
}

// Round `n` UP to the nearest even integer. Qwen3.5's rotary_dim is computed
// by the HF reference as `int(head_dim * partial_rotary_factor)` rounded to
// an even value so it can be split into rotation pairs.
int round_even(int n) {
    return (n + 1) & ~1;
}

}  // namespace

int Qwen35Config::Text::rotary_dim() const {
    return round_even(static_cast<int>(
        static_cast<float>(head_dim) * rope.partial_rotary_factor));
}

Qwen35Config Qwen35Config::load(const std::string& config_json_path) {
    return from_json_text(slurp(config_json_path));
}

Qwen35Config Qwen35Config::from_json_text(const std::string& json_text) {
    j::Value root;
    try {
        root = j::parse(json_text);
    } catch (const std::exception& e) {
        fail(std::string("json parse: ") + e.what());
    }
    if (!root.is_object()) fail("root is not a JSON object");

    Qwen35Config cfg;

    // ── top-level multimodal token IDs ────────────────────────────────────
    cfg.image_token_id          = get_int(root, "image_token_id",
                                          cfg.image_token_id);
    cfg.video_token_id          = get_int(root, "video_token_id",
                                          cfg.video_token_id);
    cfg.vision_start_token_id   = get_int(root, "vision_start_token_id",
                                          cfg.vision_start_token_id);
    cfg.vision_end_token_id     = get_int(root, "vision_end_token_id",
                                          cfg.vision_end_token_id);
    cfg.tie_word_embeddings_top = get_bool(root, "tie_word_embeddings",
                                           cfg.tie_word_embeddings_top);

    // ── text_config ───────────────────────────────────────────────────────
    if (!root.contains("text_config")) fail("missing text_config");
    const j::Value& tc = root.at("text_config");
    Qwen35Config::Text& t = cfg.text;
    t.vocab_size            = get_int(tc, "vocab_size",            t.vocab_size);
    t.hidden_size           = get_int(tc, "hidden_size",           t.hidden_size);
    t.intermediate_size     = get_int(tc, "intermediate_size",     t.intermediate_size);
    t.num_hidden_layers     = get_int(tc, "num_hidden_layers",     t.num_hidden_layers);
    t.num_attention_heads   = get_int(tc, "num_attention_heads",   t.num_attention_heads);
    t.num_key_value_heads   = get_int(tc, "num_key_value_heads",   t.num_key_value_heads);
    t.head_dim              = get_int(tc, "head_dim",              t.head_dim);
    t.attention_bias        = get_bool(tc, "attention_bias",       t.attention_bias);
    t.attn_output_gate      = get_bool(tc, "attn_output_gate",     t.attn_output_gate);
    t.linear_num_key_heads   = get_int(tc, "linear_num_key_heads",   t.linear_num_key_heads);
    t.linear_num_value_heads = get_int(tc, "linear_num_value_heads", t.linear_num_value_heads);
    t.linear_key_head_dim    = get_int(tc, "linear_key_head_dim",    t.linear_key_head_dim);
    t.linear_value_head_dim  = get_int(tc, "linear_value_head_dim",  t.linear_value_head_dim);
    t.linear_conv_kernel_dim = get_int(tc, "linear_conv_kernel_dim", t.linear_conv_kernel_dim);
    t.rms_norm_eps          = get_float(tc, "rms_norm_eps",        t.rms_norm_eps);
    t.full_attention_interval = get_int(tc, "full_attention_interval",
                                        t.full_attention_interval);
    t.max_position_embeddings = get_int(tc, "max_position_embeddings",
                                        t.max_position_embeddings);
    t.tie_word_embeddings     = get_bool(tc, "tie_word_embeddings",
                                         t.tie_word_embeddings);
    t.mtp_num_hidden_layers   = get_int(tc, "mtp_num_hidden_layers",
                                        t.mtp_num_hidden_layers);
    t.mtp_use_dedicated_embeddings = get_bool(tc, "mtp_use_dedicated_embeddings",
                                              t.mtp_use_dedicated_embeddings);

    // layer_types: required, drives the hybrid dispatch.
    if (!tc.contains("layer_types")) fail("text_config.layer_types missing");
    const j::Value& lt = tc.at("layer_types");
    if (!lt.is_array()) fail("text_config.layer_types is not an array");
    t.layer_types.clear();
    t.layer_types.reserve(lt.as_array().size());
    for (const j::Value& v : lt.as_array()) {
        if (!v.is_string()) fail("layer_types element is not a string");
        t.layer_types.push_back(parse_layer_type(v.as_string()));
    }

    // rope_parameters: optional sub-object; if absent the struct defaults stand.
    if (tc.contains("rope_parameters")) {
        const j::Value& rp = tc.at("rope_parameters");
        t.rope.mrope_interleaved     = get_bool(rp,  "mrope_interleaved",
                                                t.rope.mrope_interleaved);
        t.rope.rope_theta            = get_float(rp, "rope_theta",
                                                 t.rope.rope_theta);
        t.rope.partial_rotary_factor = get_float(rp, "partial_rotary_factor",
                                                 t.rope.partial_rotary_factor);
        if (rp.contains("mrope_section")) {
            const j::Value& ms = rp.at("mrope_section");
            if (!ms.is_array()) fail("rope_parameters.mrope_section not array");
            t.rope.mrope_section.clear();
            for (const j::Value& v : ms.as_array()) {
                t.rope.mrope_section.push_back(static_cast<int>(v.as_number()));
            }
        }
    }

    // ── vision_config ─────────────────────────────────────────────────────
    if (!root.contains("vision_config")) fail("missing vision_config");
    const j::Value& vc = root.at("vision_config");
    Qwen35Config::Vision& v = cfg.vision;
    v.depth               = get_int(vc, "depth",               v.depth);
    v.hidden_size         = get_int(vc, "hidden_size",         v.hidden_size);
    v.num_heads           = get_int(vc, "num_heads",           v.num_heads);
    v.intermediate_size   = get_int(vc, "intermediate_size",   v.intermediate_size);
    v.in_channels         = get_int(vc, "in_channels",         v.in_channels);
    v.patch_size          = get_int(vc, "patch_size",          v.patch_size);
    v.temporal_patch_size = get_int(vc, "temporal_patch_size", v.temporal_patch_size);
    v.spatial_merge_size  = get_int(vc, "spatial_merge_size",  v.spatial_merge_size);
    v.out_hidden_size     = get_int(vc, "out_hidden_size",     v.out_hidden_size);
    v.num_position_embeddings = get_int(vc, "num_position_embeddings",
                                        v.num_position_embeddings);
    if (vc.contains("deepstack_visual_indexes")) {
        const j::Value& dv = vc.at("deepstack_visual_indexes");
        if (!dv.is_array()) fail("vision_config.deepstack_visual_indexes not array");
        v.deepstack_visual_indexes.clear();
        for (const j::Value& x : dv.as_array()) {
            v.deepstack_visual_indexes.push_back(static_cast<int>(x.as_number()));
        }
    }

    cfg.validate();
    return cfg;
}

void Qwen35Config::validate() const {
    if (text.hidden_size <= 0 || text.intermediate_size <= 0 ||
        text.num_hidden_layers <= 0 || text.vocab_size <= 0 ||
        text.head_dim <= 0) {
        fail("text_config has non-positive dimension");
    }
    if (text.num_attention_heads <= 0 || text.num_key_value_heads <= 0) {
        fail("num_attention_heads / num_key_value_heads must be positive");
    }
    if (text.num_attention_heads % text.num_key_value_heads != 0) {
        fail("num_attention_heads must be a multiple of num_key_value_heads");
    }
    if (text.head_dim % 2 != 0) {
        fail("head_dim must be even (RoPE rotates pairs)");
    }
    if (static_cast<int>(text.layer_types.size()) != text.num_hidden_layers) {
        fail("layer_types length != num_hidden_layers");
    }
    // mrope_section is given in PAIRS. 2 * sum(mrope_section) must equal the
    // (even, rounded) rotary dimension.
    int pair_sum = 0;
    for (int p : text.rope.mrope_section) pair_sum += p;
    if (2 * pair_sum != text.rotary_dim()) {
        fail("2*sum(mrope_section) != rotary_dim");
    }
    if (vision.patch_size <= 0 || vision.temporal_patch_size <= 0 ||
        vision.spatial_merge_size <= 0) {
        fail("vision patch/merge sizes must be positive");
    }
    if (vision.hidden_size % vision.num_heads != 0) {
        fail("vision.hidden_size % vision.num_heads != 0");
    }
    if (vision.out_hidden_size != text.hidden_size) {
        // The merger projects to text.hidden_size; the config should reflect that.
        fail("vision.out_hidden_size must equal text.hidden_size");
    }
}

}  // namespace brolm::qwen35
