#pragma once

// Gradients of Laya's scorer logits with respect to soft-token inputs.
//
// DecisionModel::forward_items answers questions whose state may be a block of
// soft tokens (embeddings a projector produced from a non-text signal, e.g.
// audio frames). LayaGrad is its training-side twin: the same packed forward,
// run so that dLoss/dlogits can be carried back through the scorer, the two
// head layers and every ModernBERT layer to the soft rows, with Laya itself
// frozen. What trains on it is the thing that produces the soft rows (an
// audio projector), exactly as a LLaVA-style adapter trains through a frozen
// language model.
//
// Memory: the forward keeps one (T, hidden) activation per encoder layer and
// the head / scorer activations; the backward recomputes each encoder layer
// from its input (activation checkpointing), so the extra cost is one more
// encoder forward. All device work is at the compute dtype (FP16 on a GPU);
// the backward scales dLoss/dlogits by `loss_scale` so FP16 gradients stay in
// range, and divides it back out of the FP32 result.
//
// The CPU backend (FP32) runs the same code path.

#include "brolm/laya.h"
#include "brotensor/tensor.h"

#include <memory>
#include <vector>

namespace brolm::laya {

class LayaGrad {
public:
    explicit LayaGrad(DecisionModel& model);
    ~LayaGrad();
    LayaGrad(const LayaGrad&) = delete;
    LayaGrad& operator=(const LayaGrad&) = delete;

    // Training forward over packed items (same packing and soft-row contract
    // as forward_items; several items may read the same soft rows). Returns
    // every marker's raw scorer logit, items in order, markers in order —
    // the values forward_items reports, up to FP16 rounding. Keeps what
    // backward() needs until the next forward().
    std::vector<float> forward(const std::vector<LayaItem>& items, const brotensor::Tensor* soft);

    // Backward of the last forward: dlogits holds dLoss/dlogit for every
    // marker (forward()'s order). Writes dLoss/dsoft into d_soft, an FP32
    // (soft.rows, hidden) tensor on the compute device (rows no item read
    // are zero; rows several items read accumulate). Returns false when the
    // gradient is not finite (an FP16 overflow: skip the step, lower the
    // scale).
    bool backward(const std::vector<float>& dlogits, brotensor::Tensor& d_soft, float loss_scale = 256.0f);

    // Wall-clock of the last forward / backward (ms, device-synchronised).
    double last_forward_ms() const { return fwd_ms_; }
    double last_backward_ms() const { return bwd_ms_; }

private:
    struct State;
    DecisionModel& m_;
    std::unique_ptr<State> s_;
    double fwd_ms_ = 0, bwd_ms_ = 0;

    void encoder_forward_();
    void encoder_backward_(brotensor::Tensor& d_h);  // d_h: dL/d(encoder output) in, dL/d(embeds) out
    void head_forward_();
    void head_backward_(brotensor::Tensor& d_x);
};

}  // namespace brolm::laya
