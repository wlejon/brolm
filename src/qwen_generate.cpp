// Qwen3 generation: thin binding of the shared generate loop
// (brolm::detail::generate) to Qwen3Model + the Qwen tokenizer.

#include "brolm/qwen_generate.h"

#include <vector>

namespace brolm::qwen {

std::vector<int32_t> generate(Qwen3Model& model,
                              const std::vector<int32_t>& prompt_ids,
                              int eos_id,
                              const GenerateOptions& opts) {
    return brolm::detail::generate(model, prompt_ids, eos_id, opts);
}

std::vector<int32_t> generate(Qwen3Model& model,
                              const Tokenizer& tok,
                              const std::vector<int32_t>& prompt_ids,
                              int eos_id,
                              const GenerateOptions& opts,
                              TokenCallback callback) {
    return brolm::detail::generate(
        model, prompt_ids, eos_id, opts,
        [&tok](int32_t id) { return tok.decode({id}); },
        std::move(callback));
}

std::string generate_text(Qwen3Model& model, const Tokenizer& tok,
                          std::string_view prompt,
                          const GenerateOptions& opts,
                          TokenCallback callback) {
    std::vector<int32_t> prompt_ids = tok.encode(prompt);
    std::vector<int32_t> out = generate(
        model, tok, prompt_ids, tok.eos_id(), opts, std::move(callback));
    return tok.decode(out);
}

}  // namespace brolm::qwen
