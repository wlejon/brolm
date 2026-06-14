#include "brolm/nllb.h"

#include "brotensor/safetensors.h"

#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace brolm::nllb {

namespace st = ::brotensor::safetensors;

Translator Translator::load(const std::string& model_dir) {
    namespace fs = std::filesystem;
    const fs::path dir(model_dir);
    const fs::path cfg_path = dir / "config.json";
    const fs::path tok_path = dir / "tokenizer.json";
    const fs::path st_path  = dir / "model.safetensors";

    if (!fs::exists(cfg_path))
        throw std::runtime_error("nllb::Translator: missing " + cfg_path.string());
    if (!fs::exists(tok_path))
        throw std::runtime_error("nllb::Translator: missing " + tok_path.string());
    if (!fs::exists(st_path))
        throw std::runtime_error("nllb::Translator: missing " + st_path.string());

    NllbConfig cfg = NllbConfig::load(cfg_path.string());
    Tokenizer  tok = Tokenizer::load(tok_path.string());

    Translator t(std::move(cfg), std::move(tok));
    st::File f = st::File::open(st_path.string());
    t.enc_.load_weights(f);
    t.dec_.load_weights(f);
    return t;
}

std::vector<std::int32_t> Translator::translate_ids(
    const std::vector<std::int32_t>& src_ids, const std::string& tgt_lang,
    const BeamOptions& opts) {
    if (src_ids.empty())
        throw std::runtime_error("nllb::Translator: empty source ids");

    enc_.forward(src_ids.data(), static_cast<int>(src_ids.size()), enc_out_);
    dec_.set_encoder_memory(enc_out_);

    // Decoder is seeded with [</s>, tgt_lang]; beam search decodes to </s>.
    const std::vector<std::int32_t> start = tok_.decoder_start(tgt_lang);
    return beam_search(dec_, start, tok_.eos_id(), opts);
}

std::string Translator::translate(const std::string& text,
                                  const std::string& src_lang,
                                  const std::string& tgt_lang,
                                  const BeamOptions& opts) {
    const std::vector<std::int32_t> src_ids =
        tok_.encode_source(text, src_lang);
    const std::vector<std::int32_t> hyp =
        translate_ids(src_ids, tgt_lang, opts);
    // skip_special drops the </s> prefix, the target-language token, and the
    // trailing </s>, leaving the translated text.
    return tok_.decode(hyp, /*skip_special=*/true);
}

}  // namespace brolm::nllb
