#include "brolm/nllb_config.h"

#include "brolm/detail/json.h"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace brolm::nllb {

namespace j = brolm::detail::json;

namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("nllb::NllbConfig: " + msg);
}

std::string slurp(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) fail("cannot open '" + path + "'");
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

}  // namespace

NllbConfig NllbConfig::load(const std::string& config_json_path) {
    return from_json_text(slurp(config_json_path));
}

NllbConfig NllbConfig::from_json_text(const std::string& json_text) {
    j::Value root;
    try {
        root = j::parse(json_text);
    } catch (const std::exception& e) {
        fail(std::string("json parse: ") + e.what());
    }
    if (!root.is_object()) fail("root is not a JSON object");

    if (const j::Value* mt = root.find("model_type");
        mt && mt->is_string() && mt->as_string() != "m2m_100") {
        fail("model_type is '" + mt->as_string() + "', expected 'm2m_100'");
    }

    NllbConfig cfg;
    cfg.d_model                 = root.get_int("d_model", cfg.d_model);
    cfg.encoder_layers          = root.get_int("encoder_layers",
                                               cfg.encoder_layers);
    cfg.decoder_layers          = root.get_int("decoder_layers",
                                               cfg.decoder_layers);
    cfg.encoder_attention_heads = root.get_int("encoder_attention_heads",
                                               cfg.encoder_attention_heads);
    cfg.decoder_attention_heads = root.get_int("decoder_attention_heads",
                                               cfg.decoder_attention_heads);
    cfg.encoder_ffn_dim         = root.get_int("encoder_ffn_dim",
                                               cfg.encoder_ffn_dim);
    cfg.decoder_ffn_dim         = root.get_int("decoder_ffn_dim",
                                               cfg.decoder_ffn_dim);
    cfg.vocab_size              = root.get_int("vocab_size", cfg.vocab_size);
    cfg.max_position_embeddings = root.get_int("max_position_embeddings",
                                               cfg.max_position_embeddings);
    cfg.scale_embedding         = root.get_bool("scale_embedding",
                                                cfg.scale_embedding);
    cfg.pad_token_id            = root.get_int("pad_token_id",
                                               cfg.pad_token_id);
    cfg.bos_token_id            = root.get_int("bos_token_id",
                                               cfg.bos_token_id);
    cfg.eos_token_id            = root.get_int("eos_token_id",
                                               cfg.eos_token_id);
    cfg.decoder_start_token_id  = root.get_int("decoder_start_token_id",
                                               cfg.decoder_start_token_id);

    cfg.validate();
    return cfg;
}

void NllbConfig::validate() const {
    if (d_model <= 0 || vocab_size <= 0 || max_position_embeddings <= 0)
        fail("config has a non-positive core dimension");
    if (encoder_layers <= 0 || decoder_layers <= 0)
        fail("encoder_layers and decoder_layers must be positive");
    if (encoder_attention_heads <= 0 || decoder_attention_heads <= 0)
        fail("attention head counts must be positive");
    if (encoder_ffn_dim <= 0 || decoder_ffn_dim <= 0)
        fail("ffn dims must be positive");
    if (d_model % encoder_attention_heads != 0)
        fail("d_model must be a multiple of encoder_attention_heads");
    if (d_model % decoder_attention_heads != 0)
        fail("d_model must be a multiple of decoder_attention_heads");
    // NLLB ties one set of heads across enc/dec; the per-head dim must agree so
    // a single head_dim() is well-defined.
    if (d_model / encoder_attention_heads !=
        d_model / decoder_attention_heads)
        fail("encoder and decoder head_dim disagree");
}

}  // namespace brolm::nllb
