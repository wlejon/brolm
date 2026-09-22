#include "brolm/laya_config.h"

#include "brolm/detail/json.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace brolm::laya {

namespace j = brolm::detail::json;

namespace {

[[noreturn]] void fail_cfg(const std::string& msg) {
    throw std::runtime_error("laya::Config: " + msg);
}

std::string slurp(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) fail_cfg("cannot open '" + path + "'");
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

}  // namespace

float Config::get_temperature(int qtype, int num_options) const {
    const std::string size_str = (num_options <= 2) ? "2"
                               : (num_options <= 5) ? "3-5"
                               : (num_options <= 10) ? "6-10" : "11+";
    const std::string qname = (qtype == 0) ? "choice" : (qtype == 1) ? "score" : "noul";
    const std::string key = qname + ":" + size_str;

    auto it = temperature_by_options.find(key);
    if (it != temperature_by_options.end()) {
        return it->second;
    }
    if (qtype >= 0 && qtype < static_cast<int>(temperature.size())) {
        return temperature[static_cast<std::size_t>(qtype)];
    }
    return 1.0f;
}

Config Config::load(const std::string& path) {
    return from_json_text(slurp(path));
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
    cfg.head_layers  = root.get_int("head_layers",  cfg.head_layers);
    cfg.max_len      = root.get_int("max_len",      cfg.max_len);
    cfg.head_max_len = root.get_int("head_max_len", cfg.head_max_len);

    if (const j::Value* t = root.find("temperature"); t && t->is_array()) {
        cfg.temperature.clear();
        for (const auto& item : t->as_array()) {
            if (item.is_number()) {
                cfg.temperature.push_back(static_cast<float>(item.as_number()));
            }
        }
    }

    if (const j::Value* tbo = root.find("temperature_by_options"); tbo && tbo->is_object()) {
        cfg.temperature_by_options.clear();
        for (const auto& [k, v] : tbo->as_object()) {
            if (v.is_number()) {
                cfg.temperature_by_options[k] = static_cast<float>(v.as_number());
            }
        }
    }

    return cfg;
}

}  // namespace brolm::laya
