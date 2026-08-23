// Mistral 3.1 text generation: thin binding of the shared generate loop
// (brolm::detail::generate) to mistral3::TextModel + the Mistral tekken tokenizer.

#include "brolm/mistral3_generate.h"

#include <vector>

namespace brolm::mistral3 {

std::vector<int32_t> generate(TextModel& model,
                              const std::vector<int32_t>& prompt_ids,
                              int eos_id,
                              const GenerateOptions& opts) {
    return brolm::detail::generate(model, prompt_ids, eos_id, opts);
}

std::vector<int32_t> generate(TextModel& model,
                              const mistral::Tokenizer& tok,
                              const std::vector<int32_t>& prompt_ids,
                              int eos_id,
                              const GenerateOptions& opts,
                              TokenCallback callback) {
    return brolm::detail::generate(
        model, prompt_ids, eos_id, opts,
        [&tok](int32_t id) { return tok.decode({id}); },
        std::move(callback));
}

std::string generate_text(TextModel& model, const mistral::Tokenizer& tok,
                          std::string_view prompt, const GenerateOptions& opts,
                          bool add_special,
                          TokenCallback callback) {
    std::vector<int32_t> prompt_ids = tok.encode(prompt, add_special);
    std::vector<int32_t> out = generate(
        model, tok, prompt_ids, tok.eos_id(), opts, std::move(callback));
    return tok.decode(out);
}

}  // namespace brolm::mistral3
