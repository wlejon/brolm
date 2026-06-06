#include "brolm/mistral3_text.h"

#include "brolm/detail/dense_decoder.h"
#include "brolm/detail/weights.h"

#include "brotensor/gguf.h"
#include "brotensor/safetensors.h"

#include <stdexcept>
#include <string>
#include <string_view>
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

// ─── HF → ggml tensor-name map (Mistral) ─────────────────────────────────────
//
// Identical to the Qwen3 map except: there are no self_attn.q_norm / k_norm
// tensors (Mistral has no QK-norm), and lm_head.weight maps to output.weight
// for the untied head. The ggml tensor names are arch-independent in llama.cpp
// (Mistral text converts under the "llama" arch but uses the same blk.* names).

namespace {

bool starts_with(std::string_view s, std::string_view p) {
    return s.size() >= p.size() && s.compare(0, p.size(), p) == 0;
}

}  // namespace

std::string mistral3_hf_to_ggml(std::string_view hf_name) {
    if (hf_name == "model.embed_tokens.weight") return "token_embd.weight";
    if (hf_name == "model.norm.weight")         return "output_norm.weight";
    if (hf_name == "lm_head.weight")            return "output.weight";

    constexpr std::string_view kLayersPrefix = "model.layers.";
    if (!starts_with(hf_name, kLayersPrefix)) return {};
    const auto dot = hf_name.find('.', kLayersPrefix.size());
    if (dot == std::string_view::npos) return {};
    const std::string_view idx = hf_name.substr(
        kLayersPrefix.size(), dot - kLayersPrefix.size());
    const std::string_view tail = hf_name.substr(dot + 1);

    auto blk = [&](std::string_view suffix) -> std::string {
        std::string out;
        out.reserve(4 + idx.size() + 1 + suffix.size());
        out.append("blk.");
        out.append(idx);
        out.push_back('.');
        out.append(suffix);
        return out;
    };

    if (tail == "self_attn.q_proj.weight")           return blk("attn_q.weight");
    if (tail == "self_attn.k_proj.weight")           return blk("attn_k.weight");
    if (tail == "self_attn.v_proj.weight")           return blk("attn_v.weight");
    if (tail == "self_attn.o_proj.weight")           return blk("attn_output.weight");
    if (tail == "input_layernorm.weight")            return blk("attn_norm.weight");
    if (tail == "post_attention_layernorm.weight")   return blk("ffn_norm.weight");
    if (tail == "mlp.gate_proj.weight")              return blk("ffn_gate.weight");
    if (tail == "mlp.up_proj.weight")                return blk("ffn_up.weight");
    if (tail == "mlp.down_proj.weight")              return blk("ffn_down.weight");
    return {};
}

void TextModel::load_weights(const brotensor::gguf::File& f) {
    const std::vector<const brotensor::gguf::File*> shards = {&f};
    load_weights(shards);
}

void TextModel::load_weights(
    const std::vector<const brotensor::gguf::File*>& shards) {
    if (shards.empty()) fail("load_weights: no gguf shards");
    brolm::detail::weights::GgufSource src(
        shards, [](std::string_view hf) { return mistral3_hf_to_ggml(hf); });
    core_.load_weights(src);
}

}  // namespace brolm::mistral3
