// Shared sampler + logits-row download for brolm's generate loop. The
// templated generate() itself lives in the header; the non-template pieces
// (the sampler and the FP32 download) are compiled here once.

#include "brolm/detail/generate.h"

#include "brolm/detail/profile.h"

#include "brotensor/tensor.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <vector>

namespace brolm::detail {

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

}  // namespace

std::vector<float> last_row_fp32(const brotensor::Tensor& logits) {
    profile::ScopedStage ps(profile::Stage::logits_download);
    std::vector<float> all = download_fp32(logits);
    const std::size_t vocab = static_cast<std::size_t>(logits.cols);
    const std::size_t rows  = static_cast<std::size_t>(logits.rows);
    const std::size_t base  = (rows - 1) * vocab;
    return std::vector<float>(all.begin() + static_cast<std::ptrdiff_t>(base),
                              all.begin() + static_cast<std::ptrdiff_t>(base + vocab));
}

int sample_token(const float* logits, int vocab, const SamplingParams& p,
                 std::mt19937_64& rng) {
    profile::ScopedStage ps(profile::Stage::sample);
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

}  // namespace brolm::detail
