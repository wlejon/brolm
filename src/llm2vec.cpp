#include "brolm/llm2vec.h"

#include "brolm/detail/dense_decoder.h"
#include "brolm/detail/json.h"
#include "brolm/detail/weights.h"

#include "brotensor/ops.h"
#include "brotensor/safetensors.h"
#include "brotensor/tensor.h"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace brolm::llm2vec {

namespace bt = ::brotensor;
namespace st = ::brotensor::safetensors;
namespace j  = brolm::detail::json;

namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("llm2vec::Encoder: " + msg);
}

[[noreturn]] void fail_cfg(const std::string& msg) {
    throw std::runtime_error("llm2vec::Config: " + msg);
}

std::string slurp(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) fail_cfg("cannot open '" + path + "'");
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// Translate the typed LLM2Vec config into the shared dense-decoder config.
// LLM2Vec is the dense GQA/SwiGLU/RoPE stack with per-head QK-norm DISABLED
// (LLaMA has none). The encoder never runs an lm_head, so tie the (unused) head
// to the embedding matrix — this lets the checkpoint omit lm_head.weight while
// load_weights still succeeds.
brolm::detail::DenseDecoderConfig to_core_config(const Config& c) {
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
    d.use_qk_norm         = false;   // LLaMA has no QK-norm
    d.tie_word_embeddings = true;    // lm_head unused by the encoder
    return d;
}

}  // namespace

// ─── Config ──────────────────────────────────────────────────────────────────

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
    cfg.num_key_value_heads = root.get_int("num_key_value_heads", cfg.num_key_value_heads);
    cfg.rms_norm_eps        = root.get_float("rms_norm_eps",      cfg.rms_norm_eps);
    cfg.rope_theta          = root.get_float("rope_theta",        cfg.rope_theta);
    // HF LLaMA configs usually omit head_dim → derive it from hidden/heads.
    if (const j::Value* hd = root.find("head_dim"); hd && hd->is_number()) {
        cfg.head_dim = static_cast<int>(hd->as_number());
    } else {
        cfg.head_dim = cfg.hidden_size / cfg.num_attention_heads;
    }

    cfg.validate();
    return cfg;
}

void Config::validate() const {
    if (hidden_size <= 0 || intermediate_size <= 0 || num_hidden_layers <= 0 ||
        vocab_size <= 0 || head_dim <= 0) {
        fail_cfg("non-positive dimension");
    }
    if (num_attention_heads <= 0 || num_key_value_heads <= 0) {
        fail_cfg("num_attention_heads / num_key_value_heads must be positive");
    }
    if (num_attention_heads % num_key_value_heads != 0) {
        fail_cfg("num_attention_heads must be a multiple of num_key_value_heads");
    }
    if (head_dim % 2 != 0) {
        fail_cfg("head_dim must be even (RoPE rotates pairs)");
    }
}

// ─── Encoder ─────────────────────────────────────────────────────────────────

Encoder::Encoder(const Config& cfg)
    : cfg_(cfg), core_(to_core_config(cfg)) {}

Encoder::~Encoder() = default;

void Encoder::load_weights(const st::File& f, const std::string& prefix) {
    const std::vector<const st::File*> shards = {&f};
    load_weights(shards, prefix);
}

void Encoder::load_weights(const std::vector<const st::File*>& shards,
                           const std::string& prefix) {
    if (shards.empty()) fail("load_weights: no safetensors shards");
    brolm::detail::weights::SafetensorsSource src(shards, prefix);
    core_.load_weights(src);
}

void Encoder::encode(const int32_t* ids, int L, bt::Tensor& hidden_out) {
    core_.forward_encode(ids, L, hidden_out);
}

void Encoder::encode_pooled(const int32_t* ids, int L, bt::Tensor& emb_out,
                            const float* pool_mask) {
    bt::Tensor hidden;
    core_.forward_encode(ids, L, hidden);   // (L, hidden_size)

    // masked_mean_pool_forward wants a device-resident length-L mask (1 valid /
    // 0 invalid) or null for "all valid". Upload the host mask to the compute
    // device; keep it alive across the pool call.
    bt::Tensor mask_dev;
    const float* d_mask = nullptr;
    if (pool_mask) {
        mask_dev = bt::Tensor::from_host_on(bt::default_device(), pool_mask, L, 1);
        d_mask   = static_cast<const float*>(mask_dev.data);
    }
    bt::masked_mean_pool_forward(hidden, d_mask, emb_out);   // (hidden_size, 1)
}

}  // namespace brolm::llm2vec
