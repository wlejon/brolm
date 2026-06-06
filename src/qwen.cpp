#include "brolm/qwen.h"

#include "brolm/detail/dense_decoder.h"
#include "brolm/detail/weights.h"

#include "brotensor/gguf.h"
#include "brotensor/safetensors.h"

#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace brolm::qwen {

namespace st = ::brotensor::safetensors;

// ─── helpers ───────────────────────────────────────────────────────────────

namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("qwen::Qwen3Model: " + msg);
}

// Translate the typed Qwen3 config into the shared dense-decoder config. Qwen3
// is the dense GQA/SwiGLU/RoPE decoder with per-head QK-norm enabled.
brolm::detail::DenseDecoderConfig to_core_config(const Qwen3Config& c) {
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
    d.use_qk_norm         = true;   // Qwen3 has per-head QK-norm
    d.tie_word_embeddings = c.tie_word_embeddings;
    return d;
}

}  // namespace

// ─── Qwen3Config::from_gguf ────────────────────────────────────────────────

namespace {

namespace gg = ::brotensor::gguf;

const gg::Value& need_meta(const gg::File& f, const char* key) {
    const gg::Value* v = f.find_meta(key);
    if (!v) throw std::runtime_error(
        std::string("qwen::Qwen3Config::from_gguf: missing metadata '") + key + "'");
    return *v;
}

// Pull an integer-valued metadata into int. Accepts the common int-shaped
// gguf scalar types (u32/i32/u64/i64/u16/i16/u8/i8).
int meta_as_int(const gg::Value& v, const char* key) {
    switch (v.type) {
        case gg::ValueType::U8:  return static_cast<int>(v.scalar.u8);
        case gg::ValueType::I8:  return static_cast<int>(v.scalar.i8);
        case gg::ValueType::U16: return static_cast<int>(v.scalar.u16);
        case gg::ValueType::I16: return static_cast<int>(v.scalar.i16);
        case gg::ValueType::U32: return static_cast<int>(v.scalar.u32);
        case gg::ValueType::I32: return static_cast<int>(v.scalar.i32);
        case gg::ValueType::U64: return static_cast<int>(v.scalar.u64);
        case gg::ValueType::I64: return static_cast<int>(v.scalar.i64);
        default:
            throw std::runtime_error(
                std::string("qwen::Qwen3Config::from_gguf: metadata '") + key +
                "' is not an integer scalar");
    }
}

float meta_as_f32(const gg::Value& v, const char* key) {
    if (v.type == gg::ValueType::F32) return v.scalar.f32;
    if (v.type == gg::ValueType::F64) return static_cast<float>(v.scalar.f64);
    throw std::runtime_error(
        std::string("qwen::Qwen3Config::from_gguf: metadata '") + key +
        "' is not a float scalar");
}

}  // namespace

Qwen3Config Qwen3Config::from_gguf(const gg::File& f) {
    // Architecture check — guards against pointing at, say, a LLaMA gguf.
    if (const auto* arch = f.find_meta("general.architecture")) {
        if (arch->type == gg::ValueType::String && arch->str != "qwen3") {
            throw std::runtime_error(
                "qwen::Qwen3Config::from_gguf: general.architecture is '" +
                arch->str + "', expected 'qwen3'");
        }
    }

    Qwen3Config cfg;
    cfg.hidden_size = meta_as_int(
        need_meta(f, "qwen3.embedding_length"), "qwen3.embedding_length");
    cfg.intermediate_size = meta_as_int(
        need_meta(f, "qwen3.feed_forward_length"), "qwen3.feed_forward_length");
    cfg.num_hidden_layers = meta_as_int(
        need_meta(f, "qwen3.block_count"), "qwen3.block_count");
    cfg.num_attention_heads = meta_as_int(
        need_meta(f, "qwen3.attention.head_count"), "qwen3.attention.head_count");

    if (const auto* v = f.find_meta("qwen3.attention.head_count_kv")) {
        cfg.num_key_value_heads = meta_as_int(*v, "qwen3.attention.head_count_kv");
    } else {
        cfg.num_key_value_heads = cfg.num_attention_heads;
    }

    if (const auto* v = f.find_meta("qwen3.attention.key_length")) {
        cfg.head_dim = meta_as_int(*v, "qwen3.attention.key_length");
    } else {
        cfg.head_dim = cfg.hidden_size / cfg.num_attention_heads;
    }

    if (const auto* v = f.find_meta("qwen3.attention.layer_norm_rms_epsilon")) {
        cfg.rms_norm_eps = meta_as_f32(*v, "qwen3.attention.layer_norm_rms_epsilon");
    }
    if (const auto* v = f.find_meta("qwen3.rope.freq_base")) {
        cfg.rope_theta = meta_as_f32(*v, "qwen3.rope.freq_base");
    }
    if (const auto* v = f.find_meta("qwen3.context_length")) {
        cfg.max_position_embeddings = meta_as_int(*v, "qwen3.context_length");
    }

    // vocab_size: prefer the tokenizer's actual token list length; fall back
    // to a model-level scalar if present.
    if (const auto* toks = f.find_meta("tokenizer.ggml.tokens");
        toks && toks->type == gg::ValueType::Array) {
        cfg.vocab_size = static_cast<int>(toks->array.size());
    } else if (const auto* vs = f.find_meta("qwen3.vocab_size")) {
        cfg.vocab_size = meta_as_int(*vs, "qwen3.vocab_size");
    }

    // Tied embeddings: if there's an explicit metadata flag use it; otherwise
    // infer from the presence of `output.weight`.
    if (const auto* v = f.find_meta("qwen3.tie_word_embeddings");
        v && v->type == gg::ValueType::Bool) {
        cfg.tie_word_embeddings = v->scalar.b;
    } else {
        cfg.tie_word_embeddings = (f.find_tensor("output.weight") == nullptr);
    }

    return cfg;
}

// ─── HF → ggml tensor-name map (Qwen3) ─────────────────────────────────────
//
// HF naming:  model.embed_tokens.weight,
//             model.layers.N.{self_attn.{q,k,v,o}_proj, self_attn.{q,k}_norm,
//                             input_layernorm, post_attention_layernorm,
//                             mlp.{gate,up,down}_proj}.weight,
//             model.norm.weight, lm_head.weight.
// ggml naming (llama.cpp Qwen3 convention):
//             token_embd.weight,
//             blk.N.{attn_q, attn_k, attn_v, attn_output, attn_q_norm,
//                    attn_k_norm, attn_norm, ffn_norm,
//                    ffn_gate, ffn_up, ffn_down}.weight,
//             output_norm.weight, output.weight.

namespace {

bool starts_with(std::string_view s, std::string_view p) {
    return s.size() >= p.size() && s.compare(0, p.size(), p) == 0;
}

}  // namespace

std::string qwen3_hf_to_ggml(std::string_view hf_name) {
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
    if (tail == "self_attn.q_norm.weight")           return blk("attn_q_norm.weight");
    if (tail == "self_attn.k_norm.weight")           return blk("attn_k_norm.weight");
    if (tail == "input_layernorm.weight")            return blk("attn_norm.weight");
    if (tail == "post_attention_layernorm.weight")   return blk("ffn_norm.weight");
    if (tail == "mlp.gate_proj.weight")              return blk("ffn_gate.weight");
    if (tail == "mlp.up_proj.weight")                return blk("ffn_up.weight");
    if (tail == "mlp.down_proj.weight")              return blk("ffn_down.weight");
    return {};
}

// ─── ctor / dtor ───────────────────────────────────────────────────────────

Qwen3Model::Qwen3Model(const Qwen3Config& cfg)
    : cfg_(cfg), core_(to_core_config(cfg)) {}

Qwen3Model::~Qwen3Model() = default;

// ─── load_weights ──────────────────────────────────────────────────────────
//
// Each overload builds the matching container-agnostic Source and hands it to
// the shared DenseDecoder loader. The RoPE row permutation (HF rotate_half →
// brotensor interleaved-pair) is applied inside the Source uniformly across
// safetensors (FP32 host roundtrip) and gguf (byte-level row swap for quant
// carriers, FP32 roundtrip for dense). See dense_decoder.cpp for the layer
// load order.

void Qwen3Model::load_weights(const st::File& f, const std::string& prefix) {
    const std::vector<const st::File*> shards = {&f};
    load_weights(shards, prefix);
}

void Qwen3Model::load_weights(const std::vector<const st::File*>& shards,
                              const std::string& prefix) {
    if (shards.empty()) fail("load_weights: no safetensors shards");
    brolm::detail::weights::SafetensorsSource src(shards, prefix);
    core_.load_weights(src);
}

void Qwen3Model::load_weights(const brotensor::gguf::File& f) {
    const std::vector<const brotensor::gguf::File*> shards = {&f};
    load_weights(shards);
}

void Qwen3Model::load_weights(
    const std::vector<const brotensor::gguf::File*>& shards) {
    if (shards.empty()) fail("load_weights: no gguf shards");
    brolm::detail::weights::GgufSource src(
        shards, [](std::string_view hf) { return qwen3_hf_to_ggml(hf); });
    core_.load_weights(src);
}

}  // namespace brolm::qwen
