#include "brolm/mistral3_config.h"

#include "brolm/detail/json.h"

#include "brotensor/gguf.h"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace brolm::mistral3 {

namespace j = brolm::detail::json;

namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("mistral3::Mistral3Config: " + msg);
}

std::string slurp(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) fail("cannot open '" + path + "'");
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

}  // namespace

Mistral3Config Mistral3Config::load(const std::string& config_json_path) {
    return from_json_text(slurp(config_json_path));
}

Mistral3Config Mistral3Config::from_json_text(const std::string& json_text) {
    j::Value root;
    try {
        root = j::parse(json_text);
    } catch (const std::exception& e) {
        fail(std::string("json parse: ") + e.what());
    }
    if (!root.is_object()) fail("root is not a JSON object");

    Mistral3Config cfg;

    // ── top-level multimodal glue ───────────────────────────────────────────
    cfg.image_token_index         = root.get_int("image_token_index",
                                                  cfg.image_token_index);
    cfg.spatial_merge_size        = root.get_int("spatial_merge_size",
                                                  cfg.spatial_merge_size);
    cfg.multimodal_projector_bias = root.get_bool("multimodal_projector_bias",
                                                   cfg.multimodal_projector_bias);

    // ── text_config ─────────────────────────────────────────────────────────
    if (!root.contains("text_config")) fail("missing text_config");
    const j::Value& tc = root.at("text_config");
    Mistral3Config::Text& t = cfg.text;
    t.vocab_size            = tc.get_int("vocab_size",            t.vocab_size);
    t.hidden_size           = tc.get_int("hidden_size",           t.hidden_size);
    t.intermediate_size     = tc.get_int("intermediate_size",     t.intermediate_size);
    t.num_hidden_layers     = tc.get_int("num_hidden_layers",     t.num_hidden_layers);
    t.num_attention_heads   = tc.get_int("num_attention_heads",   t.num_attention_heads);
    t.num_key_value_heads   = tc.get_int("num_key_value_heads",   t.num_key_value_heads);
    t.head_dim              = tc.get_int("head_dim",              t.head_dim);
    t.rms_norm_eps          = tc.get_float("rms_norm_eps",        t.rms_norm_eps);
    t.rope_theta            = tc.get_float("rope_theta",          t.rope_theta);
    t.tie_word_embeddings   = tc.get_bool("tie_word_embeddings",  t.tie_word_embeddings);
    t.max_position_embeddings = tc.get_int("max_position_embeddings",
                                           t.max_position_embeddings);
    // sliding_window is JSON-null when disabled. find() distinguishes "absent
    // or null" (window off) from a real integer (window on).
    if (const j::Value* sw = tc.find("sliding_window"); sw && sw->is_number()) {
        t.has_sliding_window = true;
        t.sliding_window     = static_cast<int>(sw->as_number());
    }

    // ── vision_config ───────────────────────────────────────────────────────
    if (!root.contains("vision_config")) fail("missing vision_config");
    const j::Value& vc = root.at("vision_config");
    Mistral3Config::Vision& v = cfg.vision;
    v.hidden_size         = vc.get_int("hidden_size",         v.hidden_size);
    v.num_attention_heads = vc.get_int("num_attention_heads", v.num_attention_heads);
    v.head_dim            = vc.get_int("head_dim",            v.head_dim);
    v.intermediate_size   = vc.get_int("intermediate_size",   v.intermediate_size);
    v.num_hidden_layers   = vc.get_int("num_hidden_layers",   v.num_hidden_layers);
    v.num_channels        = vc.get_int("num_channels",        v.num_channels);
    v.patch_size          = vc.get_int("patch_size",          v.patch_size);
    v.image_size          = vc.get_int("image_size",          v.image_size);
    v.rope_theta          = vc.get_float("rope_theta",        v.rope_theta);

    cfg.validate();
    return cfg;
}

// ─── from_gguf ───────────────────────────────────────────────────────────────

namespace {

namespace gg = ::brotensor::gguf;

const gg::Value& need_meta(const gg::File& f, const std::string& key) {
    const gg::Value* v = f.find_meta(key);
    if (!v) fail("from_gguf: missing metadata '" + key + "'");
    return *v;
}

int meta_as_int(const gg::Value& v, const std::string& key) {
    switch (v.type) {
        case gg::ValueType::U8:  return static_cast<int>(v.scalar.u8);
        case gg::ValueType::I8:  return static_cast<int>(v.scalar.i8);
        case gg::ValueType::U16: return static_cast<int>(v.scalar.u16);
        case gg::ValueType::I16: return static_cast<int>(v.scalar.i16);
        case gg::ValueType::U32: return static_cast<int>(v.scalar.u32);
        case gg::ValueType::I32: return static_cast<int>(v.scalar.i32);
        case gg::ValueType::U64: return static_cast<int>(v.scalar.u64);
        case gg::ValueType::I64: return static_cast<int>(v.scalar.i64);
        default: fail("from_gguf: metadata '" + key + "' is not an integer scalar");
    }
}

float meta_as_f32(const gg::Value& v, const std::string& key) {
    if (v.type == gg::ValueType::F32) return v.scalar.f32;
    if (v.type == gg::ValueType::F64) return static_cast<float>(v.scalar.f64);
    fail("from_gguf: metadata '" + key + "' is not a float scalar");
}

}  // namespace

Mistral3Config Mistral3Config::from_gguf(const gg::File& f) {
    // llama.cpp prefixes every model-metadata key with general.architecture.
    // Read it verbatim and use it as the lookup prefix so this works whether
    // the converter tagged the Mistral text model "llama" or "mistral".
    const gg::Value& arch_v = need_meta(f, "general.architecture");
    if (arch_v.type != gg::ValueType::String) {
        fail("from_gguf: general.architecture is not a string");
    }
    const std::string a = arch_v.str + ".";

    Mistral3Config cfg;
    Mistral3Config::Text& t = cfg.text;

    t.hidden_size = meta_as_int(
        need_meta(f, a + "embedding_length"), a + "embedding_length");
    t.intermediate_size = meta_as_int(
        need_meta(f, a + "feed_forward_length"), a + "feed_forward_length");
    t.num_hidden_layers = meta_as_int(
        need_meta(f, a + "block_count"), a + "block_count");
    t.num_attention_heads = meta_as_int(
        need_meta(f, a + "attention.head_count"), a + "attention.head_count");

    if (const auto* v = f.find_meta(a + "attention.head_count_kv")) {
        t.num_key_value_heads = meta_as_int(*v, a + "attention.head_count_kv");
    } else {
        t.num_key_value_heads = t.num_attention_heads;
    }

    // Mistral 3.1's head_dim (128) is not hidden/heads (160), so it must come
    // from key_length. Only fall back to the derived value if the key is
    // genuinely absent.
    if (const auto* v = f.find_meta(a + "attention.key_length")) {
        t.head_dim = meta_as_int(*v, a + "attention.key_length");
    } else {
        t.head_dim = t.hidden_size / t.num_attention_heads;
    }

    if (const auto* v = f.find_meta(a + "attention.layer_norm_rms_epsilon")) {
        t.rms_norm_eps = meta_as_f32(*v, a + "attention.layer_norm_rms_epsilon");
    }
    if (const auto* v = f.find_meta(a + "rope.freq_base")) {
        t.rope_theta = meta_as_f32(*v, a + "rope.freq_base");
    }
    if (const auto* v = f.find_meta(a + "context_length")) {
        t.max_position_embeddings = meta_as_int(*v, a + "context_length");
    }

    // vocab_size: prefer the embedded tokenizer's token list length.
    if (const auto* toks = f.find_meta("tokenizer.ggml.tokens");
        toks && toks->type == gg::ValueType::Array) {
        t.vocab_size = static_cast<int>(toks->array.size());
    } else if (const auto* vs = f.find_meta(a + "vocab_size")) {
        t.vocab_size = meta_as_int(*vs, a + "vocab_size");
    }

    // Mistral is untied: the converter emits a separate output.weight. Absent
    // would mean a tied checkpoint (reuse the embedding matrix).
    t.tie_word_embeddings = (f.find_tensor("output.weight") == nullptr);

    cfg.validate();
    return cfg;
}

void Mistral3Config::validate() const {
    if (text.hidden_size <= 0 || text.intermediate_size <= 0 ||
        text.num_hidden_layers <= 0 || text.vocab_size <= 0 ||
        text.head_dim <= 0) {
        fail("text_config has non-positive dimension");
    }
    if (text.num_attention_heads <= 0 || text.num_key_value_heads <= 0) {
        fail("num_attention_heads / num_key_value_heads must be positive");
    }
    if (text.num_attention_heads % text.num_key_value_heads != 0) {
        fail("num_attention_heads must be a multiple of num_key_value_heads");
    }
    if (text.head_dim % 2 != 0) {
        fail("head_dim must be even (RoPE rotates pairs)");
    }
    if (vision.hidden_size <= 0 || vision.num_attention_heads <= 0 ||
        vision.patch_size <= 0) {
        fail("vision_config has non-positive dimension");
    }
    if (vision.hidden_size % vision.num_attention_heads != 0) {
        fail("vision.hidden_size % vision.num_attention_heads != 0");
    }
    if (spatial_merge_size <= 0) {
        fail("spatial_merge_size must be positive");
    }
}

}  // namespace brolm::mistral3
