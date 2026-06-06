#pragma once

// Shared text-generation core for brolm's dense decoders.
//
// Token sampling (greedy / temperature / top-k / top-p) plus the autoregressive
// generate loop are identical across the LLaMA-family decoders brolm ships
// (Qwen3, Mistral 3.1, ...): the sampler is a pure function of the raw logits,
// and the loop only ever touches a model through `config().vocab_size`,
// `allocate_cache(int)`, and `forward(const int32_t*, int, Tensor&)`. Both
// Qwen3Model and mistral3::TextModel expose exactly that surface, so the loop
// is templated over the model type and lives here once. The per-family headers
// (qwen_generate.h, mistral3_generate.h) are thin re-exports that bind these to
// their concrete model + tokenizer.
//
// This mirrors the dense_decoder.h extraction: the math/wiring is shared in
// detail/, the public per-model classes are thin wrappers.

#include "brotensor/tensor.h"

#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace brolm::detail {

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

// Download the last (vocab,) logits row from a (L, vocab) tensor to host as
// FP32 (FP16 compute tensors are converted). Used by the generate loop to feed
// `sample_token` after each forward.
std::vector<float> last_row_fp32(const brotensor::Tensor& logits);

// ─── Generation ────────────────────────────────────────────────────────────

struct GenerateOptions {
    int            max_new_tokens = 64;
    SamplingParams sampling;
    bool           stop_on_eos    = true;  // stop when eos_id is sampled
};

// Autoregressive generation over any model exposing `config().vocab_size`,
// `allocate_cache(int)`, and `forward(const int32_t*, int, Tensor&)`. Sizes +
// resets the model's KV-cache for (prompt + max_new_tokens), prefills
// `prompt_ids` in one forward, then decodes token by token. Returns ONLY the
// newly generated ids (prompt excluded).
//
// When `stop_on_eos`, generation halts as soon as `eos_id` is sampled, and the
// eos token is NOT included in the returned vector. A negative `eos_id`
// disables EOS stopping regardless of `stop_on_eos`.
//
// Empty prompt: the model requires L >= 1 per forward, so generation cannot be
// primed and an empty vector is returned. `max_new_tokens <= 0` likewise
// returns an empty vector.
template <class Model>
std::vector<int32_t> generate(Model& model,
                              const std::vector<int32_t>& prompt_ids,
                              int eos_id,
                              const GenerateOptions& opts) {
    std::vector<int32_t> generated;

    // The model requires L >= 1 per forward, so an empty prompt cannot prime
    // the decoder. Nothing to generate when the budget is non-positive either.
    if (prompt_ids.empty() || opts.max_new_tokens <= 0) {
        return generated;
    }

    const int vocab = model.config().vocab_size;

    model.allocate_cache(static_cast<int>(prompt_ids.size()) +
                         opts.max_new_tokens);

    std::mt19937_64 rng(opts.sampling.seed);

    // Prefill: one forward of the whole prompt. Sample the first new token from
    // the LAST logits row.
    brotensor::Tensor logits;
    model.forward(prompt_ids.data(), static_cast<int>(prompt_ids.size()),
                  logits);
    std::vector<float> row = last_row_fp32(logits);
    int next = sample_token(row.data(), vocab, opts.sampling, rng);

    const bool stop = opts.stop_on_eos && eos_id >= 0;
    if (stop && next == eos_id) {
        return generated;
    }
    generated.push_back(static_cast<int32_t>(next));

    // Decode loop: feed one token at a time, sample the next.
    while (static_cast<int>(generated.size()) < opts.max_new_tokens) {
        int32_t cur = generated.back();
        model.forward(&cur, 1, logits);
        row = last_row_fp32(logits);
        next = sample_token(row.data(), vocab, opts.sampling, rng);
        if (stop && next == eos_id) break;
        generated.push_back(static_cast<int32_t>(next));
    }

    return generated;
}

}  // namespace brolm::detail
