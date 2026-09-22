#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace brolm::laya {

struct Config {
    int head_layers = 2;
    int max_len = 512;
    int head_max_len = 192;
    std::vector<float> temperature = {1.636903f, 1.251430f, 1.983400f};
    std::unordered_map<std::string, float> temperature_by_options;

    float get_temperature(int qtype, int num_options) const;

    static Config from_json_text(const std::string& json_text);
    static Config load(const std::string& path);
};

}  // namespace brolm::laya
