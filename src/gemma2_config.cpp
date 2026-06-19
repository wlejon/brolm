#include "brolm/gemma2_config.h"

#include "brolm/detail/json.h"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace brolm::gemma {

namespace j = brolm::detail::json;

namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("gemma::Gemma2Config: " + msg);
}

std::string slurp(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) fail("cannot open '" + path + "'");
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

}  // namespace

Gemma2Config Gemma2Config::from_json(const std::string& config_json_path) {
    return from_json_text(slurp(config_json_path));
}

Gemma2Config Gemma2Config::from_safetensors_dir(const std::string& dir) {
    const std::filesystem::path cfg =
        std::filesystem::path(dir) / "config.json";
    return from_json(cfg.string());
}

Gemma2Config Gemma2Config::from_json_text(const std::string& json_text) {
    j::Value root;
    try {
        root = j::parse(json_text);
    } catch (const std::exception& e) {
        fail(std::string("json parse: ") + e.what());
    }
    if (!root.is_object()) fail("root is not a JSON object");

    Gemma2Config cfg;
    cfg.vocab_size          = root.get_int("vocab_size",          cfg.vocab_size);
    cfg.hidden_size         = root.get_int("hidden_size",         cfg.hidden_size);
    cfg.intermediate_size   = root.get_int("intermediate_size",   cfg.intermediate_size);
    cfg.num_hidden_layers   = root.get_int("num_hidden_layers",   cfg.num_hidden_layers);
    cfg.num_attention_heads = root.get_int("num_attention_heads", cfg.num_attention_heads);
    cfg.num_key_value_heads = root.get_int("num_key_value_heads", cfg.num_key_value_heads);
    cfg.head_dim            = root.get_int("head_dim",            cfg.head_dim);
    cfg.rms_norm_eps        = root.get_float("rms_norm_eps",      cfg.rms_norm_eps);
    cfg.rope_theta          = root.get_float("rope_theta",        cfg.rope_theta);
    cfg.tie_word_embeddings = root.get_bool("tie_word_embeddings", cfg.tie_word_embeddings);

    cfg.query_pre_attn_scalar = root.get_float("query_pre_attn_scalar",
                                               cfg.query_pre_attn_scalar);
    cfg.sliding_window        = root.get_int("sliding_window",
                                             cfg.sliding_window);
    cfg.attn_logit_softcapping = root.get_float("attn_logit_softcapping",
                                                cfg.attn_logit_softcapping);
    cfg.final_logit_softcapping = root.get_float("final_logit_softcapping",
                                                 cfg.final_logit_softcapping);
    cfg.max_position_embeddings = root.get_int("max_position_embeddings",
                                               cfg.max_position_embeddings);

    cfg.validate();
    return cfg;
}

void Gemma2Config::validate() const {
    if (hidden_size <= 0 || intermediate_size <= 0 || num_hidden_layers <= 0 ||
        vocab_size <= 0 || head_dim <= 0) {
        fail("config has non-positive dimension");
    }
    if (num_attention_heads <= 0 || num_key_value_heads <= 0) {
        fail("num_attention_heads / num_key_value_heads must be positive");
    }
    if (num_attention_heads % num_key_value_heads != 0) {
        fail("num_attention_heads must be a multiple of num_key_value_heads");
    }
    if (head_dim % 2 != 0) {
        fail("head_dim must be even (RoPE rotates dimension pairs)");
    }
    // brotensor::flash_attention_decode applies the 1/sqrt(head_dim) score
    // scale internally; that matches the model only when query_pre_attn_scalar
    // equals head_dim (gemma-2-2b/9b). Reject the 27B's distinct scalar rather
    // than silently mis-scale.
    if (static_cast<int>(query_pre_attn_scalar) != head_dim) {
        fail("query_pre_attn_scalar (" +
             std::to_string(static_cast<int>(query_pre_attn_scalar)) +
             ") != head_dim (" + std::to_string(head_dim) +
             "); attention scale path supports only the head_dim-scalar case");
    }
}

}  // namespace brolm::gemma
