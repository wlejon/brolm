#pragma once

// Shared text-generation core for brolm's dense decoders.
//
// Token sampling (greedy / temperature / top-k / top-p / min-p / DRY / penalties)
// plus the autoregressive generate loop are identical across the LLaMA-family decoders
// brolm ships (Qwen3, Mistral 3.1, ...): the sampler is a pure function of the raw logits,
// and the loop only ever touches a model through `config().vocab_size`,
// `allocate_cache(int)`, and `forward(const int32_t*, int, Tensor&)`. Both
// Qwen3Model and mistral3::TextModel expose exactly that surface, so the loop
// is templated over the model type and lives here once. The per-family headers
// (qwen_generate.h, mistral3_generate.h) are thin re-exports that bind these to
// their concrete model + tokenizer.
//
// This mirrors the dense_decoder.h extraction: the math/wiring is shared in
// detail/, the public per-model classes are thin wrappers.

#include "brolm/sampler.h"
#include "brolm/grammar.h"
#include "brotensor/tensor.h"

#include <atomic>
#include <cstdint>
#include <functional>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace brolm::detail {

// ─── Sampling ──────────────────────────────────────────────────────────────

using SamplingParams = brolm::SamplingParams;
using brolm::sample_token;

// Streaming token callback: returns false to cancel / early-stop generation.
using TokenCallback = std::function<bool(int32_t token_id, const std::string& token_text)>;

// Download the last (vocab,) logits row from a (L, vocab) tensor to host as
// FP32 (FP16 compute tensors are converted). Only the final row crosses the
// PCIe bus — at Qwen vocab sizes a whole-tensor prefill download would move
// hundreds of MB to use one row. Used by the generate loop to feed
// `sample_token` after each forward.
std::vector<float> last_row_fp32(const brotensor::Tensor& logits);

// ─── Generation ────────────────────────────────────────────────────────────

struct GenerateOptions {
    int            max_new_tokens = 64;
    SamplingParams sampling;
    bool           stop_on_eos    = true;     // stop when eos_id is sampled
    const Grammar* grammar        = nullptr;  // optional grammar constraint for logit masking
    TokenCallback  on_token       = nullptr;  // optional token callback (returns false to stop)
};

// Single-owner gate for a decode call. The loop below drives the model's
// KV-cache in place, so two overlapping calls on one model would interleave
// writes into the same cache. Models that expose a `busy` flag claim it for the
// duration and throw on a second claim; models without one select the variadic
// overload and the guard is inert.
template <class Model>
struct ModelGateGuard {
    std::atomic<bool>* flag_ = nullptr;
    bool claimed_ = false;

    template <class M, class = std::void_t<decltype(std::declval<M&>().busy)>>
    explicit ModelGateGuard(M& m, int) : flag_(&m.busy) {
        bool expected = false;
        if (!flag_->compare_exchange_strong(expected, true)) {
            throw std::runtime_error("brolm: single-owner precondition violated - model operation already in progress");
        }
        claimed_ = true;
    }

    template <class M>
    explicit ModelGateGuard(M&, ...) {}

    ~ModelGateGuard() {
        if (claimed_ && flag_) {
            flag_->store(false, std::memory_order_release);
        }
    }
};

// Autoregressive generation over any model exposing `config().vocab_size`,
// `allocate_cache(int)`, and `forward_last(const int32_t*, int, Tensor&)`.
//
// Supports:
//  - Token decoder mapping `decode_fn(int32_t id) -> std::string`
//  - Streaming token callback `callback(int32_t id, const std::string& text) -> bool`
//  - Constrained grammar decoding and logit masking via `opts.grammar`
//  - Modern sampling: Min-P, DRY repetition penalty, Frequency, Presence penalties
//
// Sizes + resets the model's KV-cache for (prompt + max_new_tokens), prefills
// `prompt_ids` in one forward, then decodes token by token. Returns ONLY the
// newly generated ids (prompt excluded).
template <class Model, class TokenDecoder>
std::vector<int32_t> generate(Model& model,
                              const std::vector<int32_t>& prompt_ids,
                              int eos_id,
                              const GenerateOptions& opts,
                              TokenDecoder&& decode_fn,
                              TokenCallback callback = nullptr) {
    ModelGateGuard<Model> gateGuard(model, 0);
    std::vector<int32_t> generated;

    if (prompt_ids.empty() || opts.max_new_tokens <= 0) {
        return generated;
    }

    const int vocab = model.config().vocab_size;

    model.allocate_cache(static_cast<int>(prompt_ids.size()) +
                         opts.max_new_tokens);

    std::mt19937_64 rng(opts.sampling.seed);

    // Track full context sequence for penalties (DRY, frequency, presence)
    std::vector<int32_t> context = prompt_ids;

    // Clone working grammar state if constraint is present
    Grammar grammar_state;
    bool has_grammar = (opts.grammar != nullptr);
    if (has_grammar) {
        grammar_state = opts.grammar->clone();
    }

    auto get_token_str = [&](int32_t id) -> std::string {
        if constexpr (std::is_invocable_r_v<std::string, TokenDecoder, int32_t>) {
            return decode_fn(id);
        } else {
            return "";
        }
    };

    auto effective_cb = callback ? callback : opts.on_token;

    // Prefill: one forward of the whole prompt. Only the last token's logits
    // are computed — the sampler never reads the intermediate rows.
    brotensor::Tensor logits;
    model.forward_last(prompt_ids.data(), static_cast<int>(prompt_ids.size()),
                       logits);
    std::vector<float> row = last_row_fp32(logits);

    if (has_grammar) {
        grammar_state.mask_logits(row.data(), vocab, [&](int id) -> std::string_view {
            static thread_local std::string s;
            s = get_token_str(static_cast<int32_t>(id));
            return s;
        }, eos_id);
    }

    int next = sample_token(row.data(), vocab, opts.sampling, rng,
                            context.data(), static_cast<int>(context.size()));

    const bool stop = opts.stop_on_eos && eos_id >= 0;
    if (stop && next == eos_id) {
        return generated;
    }

    std::string next_text = get_token_str(static_cast<int32_t>(next));
    if (has_grammar) {
        grammar_state.accept(next_text);
    }

    generated.push_back(static_cast<int32_t>(next));
    context.push_back(static_cast<int32_t>(next));

    if (effective_cb) {
        if (!effective_cb(static_cast<int32_t>(next), next_text)) {
            return generated;
        }
    }

    // Decode loop: feed one token at a time, sample the next.
    while (static_cast<int>(generated.size()) < opts.max_new_tokens) {
        int32_t cur = generated.back();
        model.forward_last(&cur, 1, logits);
        row = last_row_fp32(logits);

        if (has_grammar) {
            grammar_state.mask_logits(row.data(), vocab, [&](int id) -> std::string_view {
                static thread_local std::string s;
                s = get_token_str(static_cast<int32_t>(id));
                return s;
            }, eos_id);
        }

        next = sample_token(row.data(), vocab, opts.sampling, rng,
                            context.data(), static_cast<int>(context.size()));

        if (stop && next == eos_id) break;

        next_text = get_token_str(static_cast<int32_t>(next));
        if (has_grammar) {
            grammar_state.accept(next_text);
        }

        generated.push_back(static_cast<int32_t>(next));
        context.push_back(static_cast<int32_t>(next));

        if (effective_cb) {
            if (!effective_cb(static_cast<int32_t>(next), next_text)) {
                break;
            }
        }
    }

    return generated;
}

// Overload without token decoder
template <class Model>
std::vector<int32_t> generate(Model& model,
                              const std::vector<int32_t>& prompt_ids,
                              int eos_id,
                              const GenerateOptions& opts) {
    return generate(model, prompt_ids, eos_id, opts, [](int32_t) { return std::string(); }, opts.on_token);
}

// Streaming overload taking a token callback directly
template <class Model>
std::vector<int32_t> generate(Model& model,
                              const std::vector<int32_t>& prompt_ids,
                              int eos_id,
                              const GenerateOptions& opts,
                              TokenCallback callback) {
    return generate(model, prompt_ids, eos_id, opts, [](int32_t) { return std::string(); }, std::move(callback));
}

}  // namespace brolm::detail
