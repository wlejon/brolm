#include "brolm/nllb.h"

#include "brolm/detail/generate.h"

#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace brolm::nllb {

namespace bt = ::brotensor;

namespace {

struct Hypothesis {
    std::vector<std::int32_t> seq;
    double score = 0.0;              // cumulative log-prob
    DecoderState state;             // self-attn cache after consuming `seq`
    std::vector<float> next_logits; // logits predicting the token after `seq`
};

// One expansion candidate: parent beam index + the token to append.
struct Candidate {
    int parent = 0;
    std::int32_t token = 0;
    double score = 0.0;
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
    const int max_new = std::max(1, opts.max_new_tokens);
    const int max_len = static_cast<int>(start_ids.size()) + max_new;

    bt::Tensor logits;

    // Prefill the forced prefix ([</s>, tgt_lang]) into one decode state; the
    // last step's logits predict the first generated token.
    Hypothesis h0;
    h0.seq = start_ids;
    h0.state = dec.make_state(max_len);
    for (std::int32_t t : start_ids) dec.decode_step(h0.state, t, logits);
    bt::sync_all();
    h0.next_logits = brolm::detail::last_row_fp32(logits);

    std::vector<Hypothesis> active;
    active.push_back(std::move(h0));
    std::vector<Hypothesis> finished;

    for (int step = 0; step < max_new && !active.empty(); ++step) {
        // Expand every active beam by its top-`beams` next tokens.
        std::vector<Candidate> cand;
        cand.reserve(active.size() * static_cast<std::size_t>(beams));
        for (int i = 0; i < static_cast<int>(active.size()); ++i) {
            const Hypothesis& h = active[static_cast<std::size_t>(i)];
            const double lse = log_sum_exp(h.next_logits);
            for (int t : top_k_indices(h.next_logits, beams)) {
                cand.push_back({i, t,
                    h.score + (static_cast<double>(
                        h.next_logits[static_cast<std::size_t>(t)]) - lse)});
            }
        }
        std::sort(cand.begin(), cand.end(),
                  [](const Candidate& a, const Candidate& b) {
                      return a.score > b.score;
                  });

        // Keep the best `beams` non-eos candidates as the next active set
        // (cloning + advancing each one's cache); route eos-terminated
        // candidates to the finished pool. Selection order matches the prior
        // recompute beam search, so results are unchanged.
        std::vector<Hypothesis> next_active;
        for (const Candidate& c : cand) {
            std::vector<std::int32_t> seq =
                active[static_cast<std::size_t>(c.parent)].seq;
            seq.push_back(c.token);

            if (c.token == eos_id) {
                Hypothesis fh;
                fh.seq = std::move(seq);
                fh.score = c.score;
                finished.push_back(std::move(fh));
            } else {
                Hypothesis nh;
                nh.score = c.score;
                nh.state = dec.clone_state(
                    active[static_cast<std::size_t>(c.parent)].state);
                dec.decode_step(nh.state, c.token, logits);
                bt::sync_all();
                nh.next_logits = brolm::detail::last_row_fp32(logits);
                nh.seq = std::move(seq);
                next_active.push_back(std::move(nh));
            }
            if (static_cast<int>(next_active.size()) >= beams) break;
        }
        active = std::move(next_active);

        // Early stop once enough hypotheses have completed.
        if (static_cast<int>(finished.size()) >= beams) break;
    }

    // Length-penalized final selection. HF's BeamHypotheses normalizes by the
    // FULL sequence length (hyp.shape[-1], including the forced </s> + tgt_lang
    // prefix) ^ length_penalty — matching it keeps the length bias between
    // competing hypotheses identical to transformers.
    const std::vector<Hypothesis>& pool = finished.empty() ? active : finished;
    const Hypothesis* best = nullptr;
    double best_norm = -std::numeric_limits<double>::infinity();
    for (const Hypothesis& h : pool) {
        const double seq_len =
            std::max<double>(1.0, static_cast<double>(h.seq.size()));
        const double norm =
            h.score / std::pow(seq_len, static_cast<double>(opts.length_penalty));
        if (norm > best_norm) { best_norm = norm; best = &h; }
    }
    if (!best) return start_ids;
    return best->seq;
}

}  // namespace brolm::nllb
