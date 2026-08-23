#pragma once

// Text-generation layer for the Qwen3 decoder.
//
// Token sampling (greedy / temperature / top-k / top-p / min-p / DRY / penalties)
// plus the autoregressive generate loop. The sampler and loop are shared across brolm's
// dense decoders and live in brolm/detail/generate.h; this header re-exports them bound to
// brolm::qwen::Qwen3Model + the Qwen tokenizer. The model is inference-only
// with a KV-cache; this layer wires prefill + decode together, samples a token
// per step, and stops on EOS or a token budget.

#include "brolm/qwen.h"
#include "brolm/qwen_tokenizer.h"
#include "brolm/detail/generate.h"

#include <cstdint>
#include <random>
#include <string>
#include <string_view>
#include <vector>

namespace brolm::qwen {

// ─── Sampling ──────────────────────────────────────────────────────────────
//
// Re-exported from brolm::detail so existing call sites (qwen::SamplingParams,
// qwen::sample_token) are unchanged. See detail/generate.h for the semantics.
using SamplingParams  = brolm::detail::SamplingParams;
using GenerateOptions = brolm::detail::GenerateOptions;
using TokenCallback   = brolm::detail::TokenCallback;
using brolm::detail::sample_token;

// ─── Generation ────────────────────────────────────────────────────────────

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

// Streaming autoregressive generation with tokenizer for piece decoding.
std::vector<int32_t> generate(Qwen3Model& model,
                              const Tokenizer& tok,
                              const std::vector<int32_t>& prompt_ids,
                              int eos_id,
                              const GenerateOptions& opts,
                              TokenCallback callback = nullptr);

// Convenience: text in, text out. encode(prompt) -> generate -> decode of the
// newly generated ids only. Uses `tok.eos_id()` as the stop token.
// Supports optional streaming token callback.
std::string generate_text(Qwen3Model& model, const Tokenizer& tok,
                          std::string_view prompt, const GenerateOptions& opts,
                          TokenCallback callback = nullptr);

}  // namespace brolm::qwen
