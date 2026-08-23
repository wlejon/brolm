#pragma once

// Modern token sampling algorithms and penalties for brolm text generation.
//
// Supports:
//  - Greedy (argmax) & Temperature scaling
//  - Top-K and Top-P (nucleus) sampling
//  - Min-P sampling (filter tokens with prob < min_p * max_prob)
//  - Repetition, Frequency, and Presence penalties
//  - DRY (Don't Repeat Yourself) repetition penalty over recurring n-gram patterns

#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace brolm {

struct SamplingParams {
    float    temperature        = 1.0f;  // <= 0.0f => greedy (argmax)
    int      top_k              = 0;     // 0 => disabled; keep top-k candidates
    float    top_p              = 1.0f;  // >= 1.0f => disabled; nucleus cutoff
    float    min_p              = 0.0f;  // <= 0.0f => disabled; keep tokens with p >= min_p * max_p

    // Repetition, frequency, and presence penalties
    float    repetition_penalty = 1.0f;  // 1.0f => disabled
    float    frequency_penalty  = 0.0f;  // 0.0f => disabled; logit -= freq_pen * count
    float    presence_penalty   = 0.0f;  // 0.0f => disabled; logit -= pres_pen * (count > 0)
    int      penalty_last_n     = 64;    // 0 => entire context window

    // DRY (Don't Repeat Yourself) repetition penalty
    float    dry_multiplier     = 0.0f;  // 0.0f => disabled
    float    dry_base           = 1.75f; // exponential base for penalty scaling
    int      dry_allowed_length = 2;     // maximum n-gram length allowed without penalty
    int      dry_penalty_last_n = 0;     // 0 => entire context window
    std::vector<std::string> dry_sequence_breakers = {"\n", ":", "\"", "*"};
    std::vector<int32_t>     dry_breaker_tokens;

    uint64_t seed               = 0;
};

// Apply frequency, presence, and repetition penalties to raw logits in place.
void apply_repetition_penalties(float* logits, int vocab,
                                const int32_t* context, int context_len,
                                const SamplingParams& params);

// Apply DRY (Don't Repeat Yourself) repetition penalty to raw logits in place.
// Scans preceding context for matching n-gram prefixes and penalizes tokens
// that would repeat an established pattern.
void apply_dry_penalty(float* logits, int vocab,
                       const int32_t* context, int context_len,
                       const SamplingParams& params,
                       const std::vector<std::string>* vocab_tokens = nullptr);

// Apply min-p filter to probability vector in place.
// Tokens with probability < min_p * max_probability have their probability set to 0.
void apply_min_p(std::vector<float>& probs, float min_p);

// Draw one token id from raw `logits` (length `vocab`).
// Applies context penalties (frequency, presence, repetition, DRY) if `context` is non-null.
// Then scales by temperature, softmaxes, and applies Min-P, Top-K, and Top-P filtering.
int sample_token(const float* logits, int vocab, const SamplingParams& p,
                 std::mt19937_64& rng,
                 const int32_t* context = nullptr, int context_len = 0,
                 const std::vector<std::string>* vocab_tokens = nullptr);

// Stateful Sampler that manages its own RNG and context history.
class Sampler {
public:
    explicit Sampler(SamplingParams params = {});

    void set_params(const SamplingParams& params) { params_ = params; }
    const SamplingParams& params() const { return params_; }

    void set_seed(uint64_t seed);
    void reset();

    // Record an existing or sampled token into context history
    void record_token(int32_t token_id);

    // Sample next token given raw logits
    int sample(const float* logits, int vocab,
               const std::vector<std::string>* vocab_tokens = nullptr);

    const std::vector<int32_t>& history() const { return history_; }

private:
    SamplingParams       params_;
    std::mt19937_64      rng_;
    std::vector<int32_t> history_;
};

}  // namespace brolm
