#include "brolm/gemma2_config.h"

#include "brolm/detail/json.h"

#include "brotensor/gguf.h"

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

// ─── from_gguf ─────────────────────────────────────────────────────────────

namespace {

namespace gg = ::brotensor::gguf;

const gg::Value& need_meta(const gg::File& f, const char* key) {
    const gg::Value* v = f.find_meta(key);
    if (!v) fail(std::string("from_gguf: missing metadata '") + key + "'");
    return *v;
}

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
            fail(std::string("from_gguf: metadata '") + key +
                 "' is not an integer scalar");
    }
}

float meta_as_f32(const gg::Value& v, const char* key) {
    if (v.type == gg::ValueType::F32) return v.scalar.f32;
    if (v.type == gg::ValueType::F64) return static_cast<float>(v.scalar.f64);
    fail(std::string("from_gguf: metadata '") + key + "' is not a float scalar");
}

}  // namespace

Gemma2Config Gemma2Config::from_gguf(const gg::File& f) {
    if (const auto* arch = f.find_meta("general.architecture")) {
        if (arch->type == gg::ValueType::String && arch->str != "gemma2") {
            fail("from_gguf: general.architecture is '" + arch->str +
                 "', expected 'gemma2'");
        }
    }

    Gemma2Config cfg;
    cfg.hidden_size = meta_as_int(
        need_meta(f, "gemma2.embedding_length"), "gemma2.embedding_length");
    cfg.num_hidden_layers = meta_as_int(
        need_meta(f, "gemma2.block_count"), "gemma2.block_count");
    cfg.intermediate_size = meta_as_int(
        need_meta(f, "gemma2.feed_forward_length"), "gemma2.feed_forward_length");
    cfg.num_attention_heads = meta_as_int(
        need_meta(f, "gemma2.attention.head_count"), "gemma2.attention.head_count");

    if (const auto* v = f.find_meta("gemma2.attention.head_count_kv")) {
        cfg.num_key_value_heads = meta_as_int(*v, "gemma2.attention.head_count_kv");
    } else {
        cfg.num_key_value_heads = cfg.num_attention_heads;
    }

    if (const auto* v = f.find_meta("gemma2.attention.key_length")) {
        cfg.head_dim = meta_as_int(*v, "gemma2.attention.key_length");
    } else {
        cfg.head_dim = cfg.hidden_size / cfg.num_attention_heads;
    }
    // gguf does not carry query_pre_attn_scalar; the 2B/9B invariant is
    // query_pre_attn_scalar == head_dim (validate() enforces it).
    cfg.query_pre_attn_scalar = static_cast<float>(cfg.head_dim);

    if (const auto* v = f.find_meta("gemma2.attention.layer_norm_rms_epsilon")) {
        cfg.rms_norm_eps = meta_as_f32(*v, "gemma2.attention.layer_norm_rms_epsilon");
    }
    if (const auto* v = f.find_meta("gemma2.attention.sliding_window")) {
        cfg.sliding_window = meta_as_int(*v, "gemma2.attention.sliding_window");
    }
    if (const auto* v = f.find_meta("gemma2.attn_logit_softcapping")) {
        cfg.attn_logit_softcapping = meta_as_f32(*v, "gemma2.attn_logit_softcapping");
    }
    if (const auto* v = f.find_meta("gemma2.final_logit_softcapping")) {
        cfg.final_logit_softcapping =
            meta_as_f32(*v, "gemma2.final_logit_softcapping");
    }
    if (const auto* v = f.find_meta("gemma2.context_length")) {
        cfg.max_position_embeddings = meta_as_int(*v, "gemma2.context_length");
    }
    // rope_theta: read freq_base if present, else keep the 10000.0 default.
    if (const auto* v = f.find_meta("gemma2.rope.freq_base")) {
        cfg.rope_theta = meta_as_f32(*v, "gemma2.rope.freq_base");
    }

    // vocab_size: prefer the tokenizer's actual token list length; fall back to
    // a model-level scalar if present.
    if (const auto* toks = f.find_meta("tokenizer.ggml.tokens");
        toks && toks->type == gg::ValueType::Array) {
        cfg.vocab_size = static_cast<int>(toks->array.size());
    } else if (const auto* vs = f.find_meta("gemma2.vocab_size")) {
        cfg.vocab_size = meta_as_int(*vs, "gemma2.vocab_size");
    }

    // Tied embeddings: llama.cpp drops lm_head for Gemma-2, so an absent
    // `output.weight` means the embedding matrix is reused.
    cfg.tie_word_embeddings = (f.find_tensor("output.weight") == nullptr);

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
