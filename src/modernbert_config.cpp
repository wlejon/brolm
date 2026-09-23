#include "brolm/modernbert_config.h"

#include "brolm/detail/json.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace brolm::modernbert {

namespace j = brolm::detail::json;

namespace {

[[noreturn]] void fail_cfg(const std::string& msg) {
    throw std::runtime_error("modernbert::Config: " + msg);
}

std::string slurp(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) fail_cfg("cannot open '" + path + "'");
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

}  // namespace

Config Config::load(const std::string& config_json_path) {
    return from_json_text(slurp(config_json_path));
}

Config Config::from_json_text(const std::string& json_text) {
    j::Value root;
    try {
        root = j::parse(json_text);
    } catch (const std::exception& e) {
        fail_cfg(std::string("json parse: ") + e.what());
    }
    if (!root.is_object()) fail_cfg("root is not a JSON object");

    Config cfg;
    cfg.vocab_size          = root.get_int("vocab_size",          cfg.vocab_size);
    cfg.hidden_size         = root.get_int("hidden_size",         cfg.hidden_size);
    cfg.intermediate_size   = root.get_int("intermediate_size",   cfg.intermediate_size);
    cfg.num_hidden_layers   = root.get_int("num_hidden_layers",   cfg.num_hidden_layers);
    cfg.num_attention_heads = root.get_int("num_attention_heads", cfg.num_attention_heads);
    cfg.local_attention     = root.get_int("local_attention",     cfg.local_attention);

    if (const j::Value* eps = root.find("norm_eps"); eps && eps->is_number()) {
        cfg.norm_eps = static_cast<float>(eps->as_number());
    } else if (const j::Value* ln_eps = root.find("layer_norm_eps"); ln_eps && ln_eps->is_number()) {
        cfg.norm_eps = static_cast<float>(ln_eps->as_number());
    }

    if (const j::Value* lt = root.find("layer_types"); lt && lt->is_array()) {
        cfg.layer_types.clear();
        for (const auto& item : lt->as_array()) {
            if (item.is_string()) {
                cfg.layer_types.push_back(item.as_string());
            }
        }
    }

    // Older configs (transformers < 5) carry no layer_types: every
    // global_attn_every_n_layers-th layer (from 0) is global, the rest
    // sliding — how HF's ModernBertAttention decides it.
    if (cfg.layer_types.empty()) {
        const int every = root.get_int("global_attn_every_n_layers", 3);
        for (int i = 0; i < cfg.num_hidden_layers; ++i) {
            cfg.layer_types.push_back(every > 0 && i % every == 0 ? "full_attention" : "sliding_attention");
        }
    }
    // ... and name their rotary bases global_rope_theta / local_rope_theta.
    cfg.rope_theta_full = root.get_float("global_rope_theta", cfg.rope_theta_full);
    cfg.rope_theta_sliding = root.get_float("local_rope_theta", cfg.rope_theta_sliding);
    if (const j::Value* rp = root.find("rope_parameters"); rp && rp->is_object()) {
        if (const j::Value* full = rp->find("full_attention"); full && full->is_object()) {
            if (const j::Value* theta = full->find("rope_theta"); theta && theta->is_number()) {
                cfg.rope_theta_full = static_cast<float>(theta->as_number());
            }
        }
        if (const j::Value* slid = rp->find("sliding_attention"); slid && slid->is_object()) {
            if (const j::Value* theta = slid->find("rope_theta"); theta && theta->is_number()) {
                cfg.rope_theta_sliding = static_cast<float>(theta->as_number());
            }
        }
    }

    cfg.validate();
    return cfg;
}

void Config::validate() const {
    if (vocab_size <= 0) fail_cfg("vocab_size must be positive");
    if (hidden_size <= 0) fail_cfg("hidden_size must be positive");
    if (intermediate_size <= 0) fail_cfg("intermediate_size must be positive");
    if (num_hidden_layers <= 0) fail_cfg("num_hidden_layers must be positive");
    if (num_attention_heads <= 0) fail_cfg("num_attention_heads must be positive");
    if (hidden_size % num_attention_heads != 0) {
        fail_cfg("hidden_size must be divisible by num_attention_heads");
    }
    if (!layer_types.empty() && static_cast<int>(layer_types.size()) != num_hidden_layers) {
        fail_cfg("layer_types size must match num_hidden_layers");
    }
}

}  // namespace brolm::modernbert
