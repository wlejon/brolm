// Token sampling + autoregressive generate loop for the Qwen3 decoder.

#include "brolm/qwen_generate.h"

#include "brolm/detail/compute.h"

#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

namespace brolm::qwen {

namespace {

// Download a tensor to host as FP32 regardless of its compute dtype (FP32
// verbatim, FP16 converted). Mirrors t5.cpp's download_fp32.
std::vector<float> download_fp32(const brotensor::Tensor& t) {
    const std::size_t n = static_cast<std::size_t>(t.size());
    if (t.dtype == brotensor::Dtype::FP16) {
        std::vector<std::uint16_t> bits(n);
        t.copy_to_host_fp16(bits.data());
        std::vector<float> out(n);
        for (std::size_t i = 0; i < n; ++i) {
            out[i] = brotensor::fp16_bits_to_fp32(bits[i]);
        }
        return out;
    }
    return t.to_host_vector();
}

// Argmax of a raw logits array.
int argmax(const float* logits, int vocab) {
    int best = 0;
    float best_v = logits[0];
    for (int i = 1; i < vocab; ++i) {
        if (logits[i] > best_v) {
            best_v = logits[i];
            best = i;
        }
    }
    return best;
}

// Extract the last (vocab,) logits row from a (L, vocab) tensor, on host FP32.
std::vector<float> last_row_fp32(const brotensor::Tensor& logits) {
    std::vector<float> all = download_fp32(logits);
    const std::size_t vocab = static_cast<std::size_t>(logits.cols);
    const std::size_t rows  = static_cast<std::size_t>(logits.rows);
    const std::size_t base  = (rows - 1) * vocab;
    return std::vector<float>(all.begin() + static_cast<std::ptrdiff_t>(base),
                              all.begin() + static_cast<std::ptrdiff_t>(base + vocab));
}

}  // namespace

// ─── Sampling ──────────────────────────────────────────────────────────────

int sample_token(const float* logits, int vocab, const SamplingParams& p,
                 std::mt19937_64& rng) {
    if (vocab <= 0) return 0;

    // Greedy: argmax of the raw logits.
    if (p.temperature <= 0.0f || p.top_k == 1) {
        return argmax(logits, vocab);
    }

    // Temperature scale + numerically-stable softmax.
    const float inv_t = 1.0f / p.temperature;
    float max_logit = logits[0] * inv_t;
    for (int i = 1; i < vocab; ++i) {
        max_logit = std::max(max_logit, logits[i] * inv_t);
    }
    std::vector<float> probs(static_cast<std::size_t>(vocab));
    double sum = 0.0;
    for (int i = 0; i < vocab; ++i) {
        float e = std::exp(logits[i] * inv_t - max_logit);
        probs[static_cast<std::size_t>(i)] = e;
        sum += e;
    }
    const float norm = (sum > 0.0) ? static_cast<float>(1.0 / sum) : 0.0f;
    for (float& v : probs) v *= norm;

    // Indices sorted by descending probability — needed for top-k cutoff and
    // top-p prefix selection.
    std::vector<int> order(static_cast<std::size_t>(vocab));
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return probs[static_cast<std::size_t>(a)] >
               probs[static_cast<std::size_t>(b)];
    });

    // Number of candidates surviving the filters, in `order`.
    std::size_t keep = static_cast<std::size_t>(vocab);

    // top-k: keep the k highest-probability tokens.
    if (p.top_k > 0 && static_cast<std::size_t>(p.top_k) < keep) {
        keep = static_cast<std::size_t>(p.top_k);
    }

    // top-p (nucleus): keep the smallest prefix whose cumulative probability
    // reaches top_p.
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

    // Renormalise the surviving probabilities and draw.
    double kept_sum = 0.0;
    for (std::size_t i = 0; i < keep; ++i) {
        kept_sum += probs[static_cast<std::size_t>(order[i])];
    }
    if (kept_sum <= 0.0) {
        // Degenerate distribution — fall back to the top candidate.
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

// ─── Generation ────────────────────────────────────────────────────────────

std::vector<int32_t> generate(Qwen3Model& model,
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

std::string generate_text(Qwen3Model& model, const Tokenizer& tok,
                          std::string_view prompt,
                          const GenerateOptions& opts) {
    std::vector<int32_t> prompt_ids = tok.encode(prompt);
    std::vector<int32_t> out = generate(model, prompt_ids, tok.eos_id(), opts);
    return tok.decode(out);
}

}  // namespace brolm::qwen
