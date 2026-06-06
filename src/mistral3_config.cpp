#include "brolm/mistral3_config.h"

#include "brolm/detail/json.h"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace brolm::mistral3 {

namespace j = brolm::detail::json;

namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("mistral3::Mistral3Config: " + msg);
}

std::string slurp(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) fail("cannot open '" + path + "'");
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

}  // namespace

Mistral3Config Mistral3Config::load(const std::string& config_json_path) {
    return from_json_text(slurp(config_json_path));
}

Mistral3Config Mistral3Config::from_json_text(const std::string& json_text) {
    j::Value root;
    try {
        root = j::parse(json_text);
    } catch (const std::exception& e) {
        fail(std::string("json parse: ") + e.what());
    }
    if (!root.is_object()) fail("root is not a JSON object");

    Mistral3Config cfg;

    // ── top-level multimodal glue ───────────────────────────────────────────
    cfg.image_token_index         = root.get_int("image_token_index",
                                                  cfg.image_token_index);
    cfg.spatial_merge_size        = root.get_int("spatial_merge_size",
                                                  cfg.spatial_merge_size);
    cfg.multimodal_projector_bias = root.get_bool("multimodal_projector_bias",
                                                   cfg.multimodal_projector_bias);

    // ── text_config ─────────────────────────────────────────────────────────
    if (!root.contains("text_config")) fail("missing text_config");
    const j::Value& tc = root.at("text_config");
    Mistral3Config::Text& t = cfg.text;
    t.vocab_size            = tc.get_int("vocab_size",            t.vocab_size);
    t.hidden_size           = tc.get_int("hidden_size",           t.hidden_size);
    t.intermediate_size     = tc.get_int("intermediate_size",     t.intermediate_size);
    t.num_hidden_layers     = tc.get_int("num_hidden_layers",     t.num_hidden_layers);
    t.num_attention_heads   = tc.get_int("num_attention_heads",   t.num_attention_heads);
    t.num_key_value_heads   = tc.get_int("num_key_value_heads",   t.num_key_value_heads);
    t.head_dim              = tc.get_int("head_dim",              t.head_dim);
    t.rms_norm_eps          = tc.get_float("rms_norm_eps",        t.rms_norm_eps);
    t.rope_theta            = tc.get_float("rope_theta",          t.rope_theta);
    t.tie_word_embeddings   = tc.get_bool("tie_word_embeddings",  t.tie_word_embeddings);
    t.max_position_embeddings = tc.get_int("max_position_embeddings",
                                           t.max_position_embeddings);
    // sliding_window is JSON-null when disabled. find() distinguishes "absent
    // or null" (window off) from a real integer (window on).
    if (const j::Value* sw = tc.find("sliding_window"); sw && sw->is_number()) {
        t.has_sliding_window = true;
        t.sliding_window     = static_cast<int>(sw->as_number());
    }

    // ── vision_config ───────────────────────────────────────────────────────
    if (!root.contains("vision_config")) fail("missing vision_config");
    const j::Value& vc = root.at("vision_config");
    Mistral3Config::Vision& v = cfg.vision;
    v.hidden_size         = vc.get_int("hidden_size",         v.hidden_size);
    v.num_attention_heads = vc.get_int("num_attention_heads", v.num_attention_heads);
    v.head_dim            = vc.get_int("head_dim",            v.head_dim);
    v.intermediate_size   = vc.get_int("intermediate_size",   v.intermediate_size);
    v.num_hidden_layers   = vc.get_int("num_hidden_layers",   v.num_hidden_layers);
    v.num_channels        = vc.get_int("num_channels",        v.num_channels);
    v.patch_size          = vc.get_int("patch_size",          v.patch_size);
    v.image_size          = vc.get_int("image_size",          v.image_size);
    v.rope_theta          = vc.get_float("rope_theta",        v.rope_theta);

    cfg.validate();
    return cfg;
}

void Mistral3Config::validate() const {
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
    if (vision.hidden_size <= 0 || vision.num_attention_heads <= 0 ||
        vision.patch_size <= 0) {
        fail("vision_config has non-positive dimension");
    }
    if (vision.hidden_size % vision.num_attention_heads != 0) {
        fail("vision.hidden_size % vision.num_attention_heads != 0");
    }
    if (spatial_merge_size <= 0) {
        fail("spatial_merge_size must be positive");
    }
}

}  // namespace brolm::mistral3
