// Qwen3 generation: thin binding of the shared generate loop
// (brolm::detail::generate) to Qwen3Model + the Qwen tokenizer. The sampler and
// the autoregressive loop live in src/detail/generate.cpp and
// include/brolm/detail/generate.h.

#include "brolm/qwen_generate.h"

#include <vector>

namespace brolm::qwen {

std::vector<int32_t> generate(Qwen3Model& model,
                              const std::vector<int32_t>& prompt_ids,
                              int eos_id,
                              const GenerateOptions& opts) {
    return brolm::detail::generate(model, prompt_ids, eos_id, opts);
}

std::string generate_text(Qwen3Model& model, const Tokenizer& tok,
                          std::string_view prompt,
                          const GenerateOptions& opts) {
    std::vector<int32_t> prompt_ids = tok.encode(prompt);
    std::vector<int32_t> out = generate(model, prompt_ids, tok.eos_id(), opts);
    return tok.decode(out);
}

}  // namespace brolm::qwen
