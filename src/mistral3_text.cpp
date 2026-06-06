#include "brolm/mistral3_text.h"

#include "brolm/detail/dense_decoder.h"
#include "brolm/detail/weights.h"

#include "brotensor/safetensors.h"

#include <stdexcept>
#include <string>
#include <vector>

namespace brolm::mistral3 {

namespace st = ::brotensor::safetensors;

namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("mistral3::TextModel: " + msg);
}

// Translate the typed Mistral text config into the shared dense-decoder config.
// Mistral is the dense GQA/SwiGLU/RoPE decoder with per-head QK-norm DISABLED;
// lm_head tying follows the config (untied for Mistral 3.1).
brolm::detail::DenseDecoderConfig to_core_config(const Mistral3Config::Text& c) {
    brolm::detail::DenseDecoderConfig d;
    d.vocab_size          = c.vocab_size;
    d.hidden_size         = c.hidden_size;
    d.intermediate_size   = c.intermediate_size;
    d.num_hidden_layers   = c.num_hidden_layers;
    d.num_attention_heads = c.num_attention_heads;
    d.num_key_value_heads = c.num_key_value_heads;
    d.head_dim            = c.head_dim;
    d.rms_norm_eps        = c.rms_norm_eps;
    d.rope_theta          = c.rope_theta;
    d.use_qk_norm         = false;   // Mistral has no QK-norm
    d.tie_word_embeddings = c.tie_word_embeddings;
    return d;
}

}  // namespace

TextModel::TextModel(const Mistral3Config::Text& cfg)
    : cfg_(cfg), core_(to_core_config(cfg)) {}

TextModel::~TextModel() = default;

void TextModel::load_weights(const st::File& f, const std::string& prefix) {
    const std::vector<const st::File*> shards = {&f};
    load_weights(shards, prefix);
}

void TextModel::load_weights(const std::vector<const st::File*>& shards,
                             const std::string& prefix) {
    if (shards.empty()) fail("load_weights: no safetensors shards");
    brolm::detail::weights::SafetensorsSource src(shards, prefix);
    core_.load_weights(src);
}

}  // namespace brolm::mistral3
