#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace brolm::laya {

struct Config {
    int head_layers = 2;
    int max_len = 512;
    int head_max_len = 192;
    // Identity, from rl_agent_config.json: the encoder it was fine-tuned from
    // ("answerdotai/ModernBERT-large", "jhu-clsp/mmBERT-base") and its
    // model_name ("rl-agent", "laya-typed-decisions").
    std::string encoder = "answerdotai/ModernBERT-large";
    std::string model_name = "rl-agent";
    // Which member of the Laya family this is: "english", "multilingual",
    // "typed-decisions", or the checkpoint directory's name for anything else
    // (DecisionModel::load_model sets it; see checkpoint_variant()).
    std::string variant = "english";
    std::string model_dir;  // the directory load_model read (empty when synthetic)
    // Training-only: the reference slices a multi-turn conversation episode
    // into at most this many prefixes (episode_prefix_lengths) for TD(lambda)
    // targets. Inference (RLAgent.system_one) never reads it; carried for
    // config() and nothing else.
    int max_prefixes = 6;
    std::vector<float> temperature = {1.636903f, 1.251430f, 1.983400f};
    std::unordered_map<std::string, float> temperature_by_options;

    float get_temperature(int qtype, int num_options) const;

    static Config from_json_text(const std::string& json_text);
    static Config load(const std::string& path);
};

// The family member a checkpoint is: model_name "laya-typed-decisions" ->
// "typed-decisions"; an mmBERT encoder -> "multilingual"; a ModernBERT
// encoder -> "english" when the directory is not a known subfolder;
// otherwise the directory's own name.
std::string checkpoint_variant(const Config& cfg, const std::string& model_dir);

}  // namespace brolm::laya
