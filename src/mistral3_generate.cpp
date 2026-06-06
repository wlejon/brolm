// Mistral 3.1 text generation: thin binding of the shared generate loop
// (brolm::detail::generate) to mistral3::TextModel + the Mistral tekken
// tokenizer. The sampler and the autoregressive loop live in
// src/detail/generate.cpp and include/brolm/detail/generate.h.

#include "brolm/mistral3_generate.h"

#include <vector>

namespace brolm::mistral3 {

std::vector<int32_t> generate(TextModel& model,
                              const std::vector<int32_t>& prompt_ids,
                              int eos_id,
                              const GenerateOptions& opts) {
    return brolm::detail::generate(model, prompt_ids, eos_id, opts);
}

std::string generate_text(TextModel& model, const mistral::Tokenizer& tok,
                          std::string_view prompt, const GenerateOptions& opts,
                          bool add_special) {
    std::vector<int32_t> prompt_ids = tok.encode(prompt, add_special);
    std::vector<int32_t> out = generate(model, prompt_ids, tok.eos_id(), opts);
    return tok.decode(out);
}

}  // namespace brolm::mistral3
