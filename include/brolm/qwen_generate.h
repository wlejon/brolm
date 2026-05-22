#pragma once

// Text-generation layer for the Qwen3 decoder.
//
// Token sampling (greedy / temperature / top-k / top-p) plus the
// autoregressive generate loop on top of brolm::qwen::Qwen3Model. The model is
// inference-only with a KV-cache; this layer wires prefill + decode together,
// samples a token per step, and stops on EOS or a token budget.

#include "brolm/qwen.h"
#include "brolm/qwen_tokenizer.h"

#include <cstdint>
#include <random>
#include <string>
#include <string_view>
#include <vector>

namespace brolm::qwen {

// ─── Sampling ──────────────────────────────────────────────────────────────

struct SamplingParams {
    float    temperature = 1.0f;  // <= 0 => greedy (argmax)
    int      top_k       = 0;     // 0 => disabled
    float    top_p       = 1.0f;  // >= 1 => disabled
    uint64_t seed        = 0;
};

// Draw one token id from `logits` (length `vocab`, raw model logits).
//
// Greedy (argmax of the raw logits) when `temperature <= 0` or `top_k == 1`.
// Otherwise the logits are divided by `temperature` and softmaxed (numerically
// stable — max subtracted), then top-k (keep the `k` highest-probability
// tokens) and/or top-p (sort descending, keep the smallest prefix whose
// cumulative probability reaches `top_p`) prune the distribution. Both filters
// may be active; top-k is applied first. The surviving probabilities are
// renormalised and one token is drawn with `rng`.
int sample_token(const float* logits, int vocab, const SamplingParams& p,
                 std::mt19937_64& rng);

// ─── Generation ────────────────────────────────────────────────────────────

struct GenerateOptions {
    int            max_new_tokens = 64;
    SamplingParams sampling;
    bool           stop_on_eos    = true;  // stop when eos_id is sampled
};

// Autoregressive generation. Sizes + resets the model's KV-cache for
// (prompt + max_new_tokens), prefills `prompt_ids` in one forward, then decodes
// token by token. Returns ONLY the newly generated ids (prompt excluded).
//
// When `stop_on_eos`, generation halts as soon as `eos_id` is sampled, and the
// eos token is NOT included in the returned vector. A negative `eos_id`
// disables EOS stopping regardless of `stop_on_eos`.
//
// Empty prompt: the model requires L >= 1 per forward, so generation cannot be
// primed and an empty vector is returned. `max_new_tokens <= 0` likewise
// returns an empty vector.
std::vector<int32_t> generate(Qwen3Model& model,
                              const std::vector<int32_t>& prompt_ids,
                              int eos_id,
                              const GenerateOptions& opts);

// Convenience: text in, text out. encode(prompt) -> generate -> decode of the
// newly generated ids only. Uses `tok.eos_id()` as the stop token.
std::string generate_text(Qwen3Model& model, const Tokenizer& tok,
                          std::string_view prompt, const GenerateOptions& opts);

}  // namespace brolm::qwen
