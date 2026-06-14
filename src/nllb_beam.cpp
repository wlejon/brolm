#include "brolm/nllb.h"

#include "brolm/detail/generate.h"

#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace brolm::nllb {

namespace bt = ::brotensor;

namespace {

struct Hypothesis {
    std::vector<std::int32_t> seq;
    double score = 0.0;   // cumulative log-prob
};

// log(sum exp(logits)) over the whole vocabulary, numerically stable.
double log_sum_exp(const std::vector<float>& logits) {
    float m = -std::numeric_limits<float>::infinity();
    for (float v : logits) m = std::max(m, v);
    double s = 0.0;
    for (float v : logits) s += std::exp(static_cast<double>(v - m));
    return static_cast<double>(m) + std::log(s);
}

// Indices of the `k` largest logits (unordered among themselves is fine — the
// caller re-scores). k is clamped to the vocab size.
std::vector<int> top_k_indices(const std::vector<float>& logits, int k) {
    const int n = static_cast<int>(logits.size());
    k = std::min(k, n);
    std::vector<int> idx(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) idx[static_cast<std::size_t>(i)] = i;
    std::nth_element(idx.begin(), idx.begin() + k, idx.end(),
                     [&](int a, int b) { return logits[a] > logits[b]; });
    idx.resize(static_cast<std::size_t>(k));
    return idx;
}

}  // namespace

std::vector<std::int32_t> beam_search(Decoder& dec,
                                      const std::vector<std::int32_t>& start_ids,
                                      int eos_id, const BeamOptions& opts) {
    if (start_ids.empty())
        throw std::runtime_error("nllb::beam_search: empty start_ids");
    const int beams = std::max(1, opts.num_beams);
    const std::size_t start_len = start_ids.size();

    std::vector<Hypothesis> active = {Hypothesis{start_ids, 0.0}};
    std::vector<Hypothesis> finished;

    bt::Tensor logits;
    for (int step = 0; step < opts.max_new_tokens && !active.empty(); ++step) {
        std::vector<Hypothesis> cand;
        cand.reserve(active.size() * static_cast<std::size_t>(beams));

        for (const Hypothesis& h : active) {
            dec.forward_logits(h.seq.data(), static_cast<int>(h.seq.size()),
                               logits);
            bt::sync_all();
            std::vector<float> lg = brolm::detail::last_row_fp32(logits);
            const double lse = log_sum_exp(lg);
            for (int t : top_k_indices(lg, beams)) {
                Hypothesis nh;
                nh.seq = h.seq;
                nh.seq.push_back(t);
                nh.score = h.score +
                           (static_cast<double>(lg[static_cast<std::size_t>(t)]) - lse);
                cand.push_back(std::move(nh));
            }
        }

        // Keep the best `beams` candidates by raw cumulative log-prob; route
        // those ending in eos to the finished set.
        std::sort(cand.begin(), cand.end(),
                  [](const Hypothesis& a, const Hypothesis& b) {
                      return a.score > b.score;
                  });

        active.clear();
        for (const Hypothesis& h : cand) {
            if (!h.seq.empty() && h.seq.back() == eos_id) {
                finished.push_back(h);
            } else {
                active.push_back(h);
            }
            if (static_cast<int>(active.size()) >= beams) break;
        }

        // Early stop once we have enough completed hypotheses.
        if (static_cast<int>(finished.size()) >= beams) break;
    }

    // Length-penalized final selection over finished hypotheses (or the best
    // unfinished beam if none completed within max_new_tokens).
    const std::vector<Hypothesis>& pool = finished.empty() ? active : finished;
    const Hypothesis* best = nullptr;
    double best_norm = -std::numeric_limits<double>::infinity();
    for (const Hypothesis& h : pool) {
        const double gen_len =
            std::max<double>(1.0, static_cast<double>(h.seq.size() - start_len));
        const double norm =
            h.score / std::pow(gen_len, static_cast<double>(opts.length_penalty));
        if (norm > best_norm) { best_norm = norm; best = &h; }
    }
    if (!best) return start_ids;
    return best->seq;
}

}  // namespace brolm::nllb
