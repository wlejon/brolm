#pragma once

#include <string>
#include <vector>

namespace brolm::modernbert {

struct Config {
    int vocab_size = 50368;
    int hidden_size = 1024;
    int intermediate_size = 2624;
    int num_hidden_layers = 28;
    int num_attention_heads = 16;
    int local_attention = 128;
    float norm_eps = 1e-5f;
    std::vector<std::string> layer_types;
    float rope_theta_full = 160000.0f;
    float rope_theta_sliding = 10000.0f;

    int head_dim() const {
        return num_attention_heads > 0 ? hidden_size / num_attention_heads : 64;
    }

    bool is_sliding(int layer_idx) const {
        if (layer_idx >= 0 && layer_idx < static_cast<int>(layer_types.size())) {
            return layer_types[static_cast<std::size_t>(layer_idx)] == "sliding_attention";
        }
        return false;
    }

    float rope_theta(int layer_idx) const {
        return is_sliding(layer_idx) ? rope_theta_sliding : rope_theta_full;
    }

    static Config from_json_text(const std::string& json_text);
    static Config load(const std::string& config_json_path);
    void validate() const;
};

}  // namespace brolm::modernbert
