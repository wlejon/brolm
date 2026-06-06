#pragma once

// Text-generation layer for the Mistral 3.1 text decoder.
//
// Token sampling (greedy / temperature / top-k / top-p) plus the autoregressive
// generate loop, shared with the other dense decoders in brolm/detail/generate.h
// and bound here to brolm::mistral3::TextModel + the Mistral tekken tokenizer.
// The model is inference-only with a KV-cache; this layer wires prefill + decode
// together, samples a token per step, and stops on EOS or a token budget.

#include "brolm/mistral3_text.h"
#include "brolm/mistral_tokenizer.h"
#include "brolm/detail/generate.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace brolm::mistral3 {

// ─── Sampling ──────────────────────────────────────────────────────────────
//
// Re-exported from brolm::detail (shared with qwen); see detail/generate.h.
using SamplingParams  = brolm::detail::SamplingParams;
using GenerateOptions = brolm::detail::GenerateOptions;
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
// Empty prompt or `max_new_tokens <= 0`: returns an empty vector.
std::vector<int32_t> generate(TextModel& model,
                              const std::vector<int32_t>& prompt_ids,
                              int eos_id,
                              const GenerateOptions& opts);

// Convenience: text in, text out. encode(prompt) -> generate -> decode of the
// newly generated ids only. Uses `tok.eos_id()` as the stop token.
//
// `add_special` controls whether the tokenizer prepends BOS (<s>) to the
// prompt. Mistral prefixes BOS at the tokenizer level, so a bare prompt wants
// it (default true); pass false when feeding an already-templated string from
// apply_chat_template (which embeds its own leading <s>).
std::string generate_text(TextModel& model, const mistral::Tokenizer& tok,
                          std::string_view prompt, const GenerateOptions& opts,
                          bool add_special = true);

}  // namespace brolm::mistral3
