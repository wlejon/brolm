#include "brolm/sampler.h"
#include "brolm/detail/profile.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace brolm {

namespace {

int argmax(const float* logits, int vocab) {
    int best = 0;
    float best_v = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < vocab; ++i) {
        if (logits[i] > best_v) {
            best_v = logits[i];
            best = i;
        }
    }
    return best;
}

}  // namespace

void apply_repetition_penalties(float* logits, int vocab,
                                const int32_t* context, int context_len,
                                const SamplingParams& params) {
    if (!logits || vocab <= 0 || !context || context_len <= 0) return;

    const bool has_rep  = (params.repetition_penalty != 1.0f && params.repetition_penalty > 0.0f);
    const bool has_freq = (params.frequency_penalty != 0.0f);
    const bool has_pres = (params.presence_penalty != 0.0f);

    if (!has_rep && !has_freq && !has_pres) return;

    int start = 0;
    if (params.penalty_last_n > 0 && context_len > params.penalty_last_n) {
        start = context_len - params.penalty_last_n;
    }

    std::unordered_map<int32_t, int> counts;
    for (int i = start; i < context_len; ++i) {
        int32_t id = context[i];
        if (id >= 0 && id < vocab) {
            counts[id]++;
        }
    }

    for (const auto& [id, count] : counts) {
        if (has_rep) {
            if (logits[id] > 0.0f) {
                logits[id] /= params.repetition_penalty;
            } else {
                logits[id] *= params.repetition_penalty;
            }
        }
        if (has_freq) {
            logits[id] -= params.frequency_penalty * static_cast<float>(count);
        }
        if (has_pres) {
            logits[id] -= params.presence_penalty;
        }
    }
}

void apply_dry_penalty(float* logits, int vocab,
                       const int32_t* context, int context_len,
                       const SamplingParams& params,
                       const std::vector<std::string>* vocab_tokens) {
    if (!logits || vocab <= 0 || !context || context_len <= params.dry_allowed_length) return;
    if (params.dry_multiplier <= 0.0f) return;

    int start = 0;
    if (params.dry_penalty_last_n > 0 && context_len > params.dry_penalty_last_n) {
        start = context_len - params.dry_penalty_last_n;
    }

    const int32_t* window = context + start;
    const int window_len = context_len - start;
    if (window_len <= params.dry_allowed_length) return;

    std::unordered_set<int32_t> breakers(params.dry_breaker_tokens.begin(),
                                         params.dry_breaker_tokens.end());

    if (vocab_tokens && !params.dry_sequence_breakers.empty()) {
        for (int id = 0; id < static_cast<int>(vocab_tokens->size()); ++id) {
            const std::string& str = (*vocab_tokens)[id];
            for (const auto& b : params.dry_sequence_breakers) {
                if (!b.empty() && str.find(b) != std::string::npos) {
                    breakers.insert(static_cast<int32_t>(id));
                    break;
                }
            }
        }
    }

    // Match n-grams: find previous occurrences of tokens following matching prefixes
    std::unordered_map<int32_t, int> max_match_lengths;

    for (int j = 0; j < window_len - 1; ++j) {
        int32_t cand = window[j + 1];
        if (cand < 0 || cand >= vocab) continue;
        if (breakers.find(cand) != breakers.end()) continue;

        int match_len = 0;
        while (j - match_len >= 0 && (window_len - 1 - match_len) > (j + 1)) {
            int32_t tj = window[j - match_len];
            int32_t tw = window[window_len - 1 - match_len];
            if (tj != tw) break;
            if (breakers.find(tj) != breakers.end()) break;
            ++match_len;
        }

        if (match_len >= params.dry_allowed_length) {
            auto it = max_match_lengths.find(cand);
            if (it == max_match_lengths.end() || match_len > it->second) {
                max_match_lengths[cand] = match_len;
            }
        }
    }

    for (const auto& [cand, match_len] : max_match_lengths) {
        float exponent = static_cast<float>(match_len - params.dry_allowed_length);
        float penalty = params.dry_multiplier * std::pow(params.dry_base, exponent);
        logits[cand] -= penalty;
    }
}

void apply_min_p(std::vector<float>& probs, float min_p) {
    if (min_p <= 0.0f || probs.empty()) return;

    float max_p = 0.0f;
    for (float p : probs) {
        if (p > max_p) max_p = p;
    }

    const float threshold = min_p * max_p;
    for (float& p : probs) {
        if (p < threshold) {
            p = 0.0f;
        }
    }
}

int sample_token(const float* logits, int vocab, const SamplingParams& p,
                 std::mt19937_64& rng,
                 const int32_t* context, int context_len,
                 const std::vector<std::string>* vocab_tokens) {
    detail::profile::ScopedStage ps(detail::profile::Stage::sample);
    if (vocab <= 0 || !logits) return 0;

    const bool has_penalties = (context && context_len > 0 &&
        (p.repetition_penalty != 1.0f || p.frequency_penalty != 0.0f ||
         p.presence_penalty != 0.0f || p.dry_multiplier > 0.0f));

    std::vector<float> work_logits_buf;
    const float* effective_logits = logits;

    if (has_penalties) {
        work_logits_buf.assign(logits, logits + vocab);
        apply_repetition_penalties(work_logits_buf.data(), vocab, context, context_len, p);
        apply_dry_penalty(work_logits_buf.data(), vocab, context, context_len, p, vocab_tokens);
        effective_logits = work_logits_buf.data();
    }

    // Greedy: argmax of effective logits
    if (p.temperature <= 0.0f || p.top_k == 1) {
        return argmax(effective_logits, vocab);
    }

    // Temperature scaling + numerically-stable softmax
    const float inv_t = 1.0f / p.temperature;
    float max_logit = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < vocab; ++i) {
        if (std::isfinite(effective_logits[i])) {
            max_logit = std::max(max_logit, effective_logits[i] * inv_t);
        }
    }

    if (!std::isfinite(max_logit)) {
        // Fall back if all logits are non-finite (e.g. all masked)
        return argmax(effective_logits, vocab);
    }

    std::vector<float> probs(static_cast<std::size_t>(vocab), 0.0f);
    double sum = 0.0;
    for (int i = 0; i < vocab; ++i) {
        if (effective_logits[i] > -1e30f) {
            float e = std::exp(effective_logits[i] * inv_t - max_logit);
            probs[static_cast<std::size_t>(i)] = e;
            sum += e;
        }
    }

    if (sum <= 0.0) {
        return argmax(effective_logits, vocab);
    }

    const float norm = static_cast<float>(1.0 / sum);
    for (float& v : probs) v *= norm;

    // Apply Min-P cutoff
    if (p.min_p > 0.0f) {
        apply_min_p(probs, p.min_p);
    }

    // Collect candidate indices with non-zero probability
    std::vector<int> order;
    order.reserve(static_cast<std::size_t>(vocab));
    for (int i = 0; i < vocab; ++i) {
        if (probs[static_cast<std::size_t>(i)] > 0.0f) {
            order.push_back(i);
        }
    }

    if (order.empty()) {
        return argmax(effective_logits, vocab);
    }

    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return probs[static_cast<std::size_t>(a)] >
               probs[static_cast<std::size_t>(b)];
    });

    std::size_t keep = order.size();

    // Top-K cutoff
    if (p.top_k > 0 && static_cast<std::size_t>(p.top_k) < keep) {
        keep = static_cast<std::size_t>(p.top_k);
    }

    // Top-P (nucleus) cutoff
    if (p.top_p < 1.0f) {
        double cum = 0.0;
        std::size_t nucleus = 0;
        for (std::size_t i = 0; i < keep; ++i) {
            cum += probs[static_cast<std::size_t>(order[i])];
            ++nucleus;
            if (cum >= static_cast<double>(p.top_p)) break;
        }
        keep = std::max<std::size_t>(nucleus, 1);
    }

    // Renormalize kept probabilities and sample
    double kept_sum = 0.0;
    for (std::size_t i = 0; i < keep; ++i) {
        kept_sum += probs[static_cast<std::size_t>(order[i])];
    }
    if (kept_sum <= 0.0) {
        return order[0];
    }

    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double r = dist(rng) * kept_sum;
    double acc = 0.0;
    for (std::size_t i = 0; i < keep; ++i) {
        acc += probs[static_cast<std::size_t>(order[i])];
        if (r < acc) return order[i];
    }
    return order[keep - 1];
}

Sampler::Sampler(SamplingParams params)
    : params_(std::move(params)), rng_(params_.seed) {}

void Sampler::set_seed(uint64_t seed) {
    params_.seed = seed;
    rng_.seed(seed);
}

void Sampler::reset() {
    rng_.seed(params_.seed);
    history_.clear();
}

void Sampler::record_token(int32_t token_id) {
    history_.push_back(token_id);
}

int Sampler::sample(const float* logits, int vocab,
                    const std::vector<std::string>* vocab_tokens) {
    int tok = sample_token(logits, vocab, params_, rng_,
                           history_.empty() ? nullptr : history_.data(),
                           static_cast<int>(history_.size()),
                           vocab_tokens);
    history_.push_back(static_cast<int32_t>(tok));
    return tok;
}

}  // namespace brolm
