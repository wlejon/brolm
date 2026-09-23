// Soft-token input and its gradient.
//
//   1. Soft rows that ARE the state's token embeddings (the ids under them
//      replaced by a placeholder) reproduce the text answer, through
//      forward_items (eager, graph capture, graph replay, mixed with a text
//      item) and through LayaGrad::forward.
//   2. LayaGrad::backward matches directional central differences of the
//      forward: for random directions v and the steepest direction,
//      (f(s + eps v) - f(s - eps v)) / 2 eps  ~=  <dL/ds, v>.
//
// Weights from LAYA_MODEL_DIR or the ../laya sibling; skips without them.

#include "brolm/laya.h"
#include "brolm/laya_grad.h"

#include "brotensor/ops.h"
#include "brotensor/runtime.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace bt = ::brotensor;
using brolm::laya::LayaItem;

namespace {

int g_failures = 0;

void check(bool ok, const std::string& msg) {
    if (!ok) {
        std::cerr << "FAILED: " << msg << "\n";
        ++g_failures;
    }
}

std::vector<float> download(const bt::Tensor& t) {
    std::vector<float> out(static_cast<std::size_t>(t.rows) * t.cols);
    if (t.dtype == bt::Dtype::FP32) {
        t.copy_to_host_raw(out.data(), out.size() * sizeof(float));
        return out;
    }
    std::vector<uint16_t> bits(out.size());
    t.copy_to_host_raw(bits.data(), bits.size() * sizeof(uint16_t));
    for (std::size_t i = 0; i < bits.size(); ++i) out[i] = bt::fp16_bits_to_fp32(bits[i]);
    return out;
}

bt::Tensor upload(const std::vector<float>& v, int rows, int cols, bt::Dtype dt) {
    const bt::Device dev = bt::default_device();
    if (dt == bt::Dtype::FP32) return bt::Tensor::from_host_on(dev, v.data(), rows, cols);
    std::vector<uint16_t> bits(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) bits[i] = bt::fp32_to_fp16_bits(v[i]);
    return bt::Tensor::from_raw_bytes_on(dev, bits.data(), rows, cols, dt, bits.size() * sizeof(uint16_t));
}

float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float m = 0.0f;
    for (std::size_t i = 0; i < a.size() && i < b.size(); ++i) m = std::max(m, std::fabs(a[i] - b[i]));
    return a.size() == b.size() ? m : 1e30f;
}

// All checks on the default device. `fp32`: exact enough that random
// directions are judged too (on FP16 only the steepest direction rises above
// the logit rounding noise).
void run_checks(const std::string& dir, bool fp32) {
    std::printf("── %s (%s) ──\n", fp32 ? "CPU" : "GPU", fp32 ? "FP32" : "FP16");
    brolm::laya::DecisionModel model;
    model.load_model(dir);
    const bt::Dtype dt = bt::compute_dtype();
    const int D = model.encoder().config().hidden_size;

    const std::string state = "The customer says the package arrived broken and asks for their money back.";
    brolm::laya::LayaQuestion q;
    q.id = "refund";
    q.type = "noul";
    q.instructions = "The customer wants a refund.";
    const brolm::laya::SequenceResult seq = model.build_sequence(state, q);
    const std::vector<int32_t> st_ids = model.tokenizer().encode_state(state);
    const int n_state = static_cast<int>(st_ids.size());
    const int state_pos = static_cast<int>(seq.input_ids.size()) - 1 - n_state;
    for (int i = 0; i < n_state; ++i) check(seq.input_ids[state_pos + i] == st_ids[i], "state span located");

    const LayaItem text = LayaItem::of(seq, q.qtype_index());
    const std::vector<float> ref = model.forward_items({text})[0].logits;

    // Soft table: the state's token embeddings; the ids under them become
    // placeholders.
    std::vector<int32_t> placeholder = seq.input_ids;
    for (int i = 0; i < n_state; ++i) placeholder[state_pos + i] = model.tokenizer().mask_token_id();
    bt::Tensor idx = bt::Tensor::from_raw_bytes_on(bt::default_device(), st_ids.data(), n_state, 1, bt::Dtype::INT32,
                                                   st_ids.size() * sizeof(int32_t));
    bt::Tensor soft;
    bt::gather_rows(model.encoder().token_embeddings(), idx, soft);
    LayaItem si = text;
    si.input_ids = placeholder.data();
    si.soft_pos = state_pos;
    si.soft_count = n_state;
    si.soft_row = 0;

    // 1. forward_items with soft rows: eager-or-capture, replay, mixed batch.
    for (int rep = 0; rep < 2; ++rep) {
        const std::vector<float> got = model.forward_items({si}, &soft)[0].logits;
        const float d = max_abs_diff(ref, got);
        std::printf("forward_items soft (run %d): logits %.4f %.4f vs text %.4f %.4f, |d| %.4f\n", rep, got[0],
                    got[1], ref[0], ref[1], d);
        check(d < 0.02f, "soft rows equal to the embeddings reproduce the text logits");
    }
    {
        const auto mixed = model.forward_items({text, si, text}, &soft);
        check(max_abs_diff(ref, mixed[1].logits) < 0.02f && max_abs_diff(ref, mixed[0].logits) < 0.02f &&
                  max_abs_diff(ref, mixed[2].logits) < 0.02f,
              "a mixed text + soft batch answers every item as alone");
    }
    // Text-only after a soft batch still matches (graph keys kept apart).
    check(max_abs_diff(ref, model.forward_items({text})[0].logits) < 0.02f, "text path after soft batches");

    // LayaGrad forward, text and soft.
    brolm::laya::LayaGrad grad(model);
    {
        const std::vector<float> g_text = grad.forward({text}, nullptr);
        const std::vector<float> g_soft = grad.forward({si, si}, &soft);  // two items share one soft block
        std::printf("LayaGrad forward: text |d| %.4f, soft |d| %.4f\n", max_abs_diff(ref, g_text),
                    max_abs_diff(ref, {g_soft[0], g_soft[1]}));
        check(max_abs_diff(ref, g_text) < 0.05f, "LayaGrad text forward matches forward_items");
        check(max_abs_diff(ref, {g_soft[0], g_soft[1]}) < 0.05f && max_abs_diff(ref, {g_soft[2], g_soft[3]}) < 0.05f,
              "LayaGrad soft forward matches forward_items");
    }

    // 2. Gradient of L = logit[true] - logit[false] (two items share the soft
    //    block, so dL/ds sums both) against directional differences.
    grad.forward({si, si}, &soft);
    bt::Tensor d_soft;
    const bool finite = grad.backward({-1.0f, 1.0f, -1.0f, 1.0f}, d_soft);
    check(finite, "gradient is finite");
    std::printf("LayaGrad forward %.1f ms, backward %.1f ms (%d rows x 2 items)\n", grad.last_forward_ms(),
                grad.last_backward_ms(), static_cast<int>(seq.input_ids.size()));
    const std::vector<float> g = download(d_soft);
    const std::vector<float> s0 = download(soft);
    double gnorm = 0, snorm = 0;
    for (std::size_t i = 0; i < g.size(); ++i) {
        gnorm += double(g[i]) * g[i];
        snorm += double(s0[i]) * s0[i];
    }
    gnorm = std::sqrt(gnorm);
    snorm = std::sqrt(snorm);
    check(gnorm > 0, "gradient is non-zero");
    // The embedding LayerNorm makes the answer invariant to shifting or
    // scaling a row, so each row's gradient is orthogonal to the ones vector
    // and to the row itself.
    {
        double worst_sum = 0, worst_dot = 0;
        for (int r = 0; r < n_state; ++r) {
            double sum = 0, dotx = 0, gn = 0, xn = 0;
            for (int c = 0; c < D; ++c) {
                const double gv = g[static_cast<std::size_t>(r) * D + c], xv = s0[static_cast<std::size_t>(r) * D + c];
                sum += gv;
                dotx += gv * xv;
                gn += gv * gv;
                xn += xv * xv;
            }
            gn = std::sqrt(gn) + 1e-30;
            worst_sum = std::max(worst_sum, std::fabs(sum) / (gn * std::sqrt(double(D))));
            worst_dot = std::max(worst_dot, std::fabs(dotx) / (gn * std::sqrt(xn)));
        }
        std::printf("row invariance: worst |sum g|/(|g| sqrt D) %.4f, worst |g.x|/(|g||x|) %.4f\n", worst_sum,
                    worst_dot);
        check(worst_sum < (fp32 ? 1e-3 : 2e-2) && worst_dot < (fp32 ? 1e-3 : 2e-2),
              "soft-row gradient respects the embedding LayerNorm's invariances");
    }

    auto f = [&](const std::vector<float>& s) {
        bt::Tensor t = upload(s, n_state, D, dt);
        const auto out = model.forward_items({si, si}, &t);
        return 2.0 * (out[0].logits[1] - out[0].logits[0]);
    };
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (int trial = 0; trial < (fp32 ? 4 : 1); ++trial) {
        std::vector<float> v(g.size());
        if (trial == 0) {
            for (std::size_t i = 0; i < v.size(); ++i) v[i] = static_cast<float>(g[i] / gnorm);
        } else {
            double vn = 0;
            for (float& x : v) {
                x = nd(rng);
                vn += double(x) * x;
            }
            for (float& x : v) x = static_cast<float>(x / std::sqrt(vn));
        }
        double an = 0;
        for (std::size_t i = 0; i < v.size(); ++i) an += double(g[i]) * v[i];
        // Step along the unit direction: 2 % of the soft block's norm on
        // FP16 (above the rounding noise); on FP32 0.2 %, and 0.02 % along
        // the steepest direction, where the answer curves hardest.
        const float eps = static_cast<float>((fp32 ? (trial == 0 ? 0.0002 : 0.002) : 0.02) * snorm);
        std::vector<float> sp(s0), sm(s0);
        for (std::size_t i = 0; i < v.size(); ++i) {
            sp[i] += eps * v[i];
            sm[i] -= eps * v[i];
        }
        const double fd = (f(sp) - f(sm)) / (2.0 * eps);
        // Random unit directions see ~|g|/sqrt(n) of the gradient; judge them
        // against 1 % of |g| rather than their own tiny magnitude.
        const double err = std::fabs(fd - an) / std::max({std::fabs(fd), std::fabs(an), 1e-2 * gnorm});
        std::printf("direction %d: finite difference %.5f, analytic %.5f, rel err %.3f\n", trial, fd, an, err);
        check(err < (fp32 ? 0.02 : 0.1), "directional derivative matches");
    }
}

}  // namespace

int main() {
    const char* env = std::getenv("LAYA_MODEL_DIR");
    const std::string dir = (env && env[0]) ? env : std::string(BROLM_SIBLING_DIR) + "/laya";
    if (!std::filesystem::exists(dir + "/model.safetensors")) {
        std::printf("SKIP: no Laya checkpoint at %s\n", dir.c_str());
        return 0;
    }
    bt::init();
    if (bt::default_device().type != bt::DeviceType::CPU) run_checks(dir, false);
    // The FP32 CPU oracle takes minutes (ModernBERT-large on the CPU backend);
    // LAYA_GRAD_CPU=0 skips it.
    const char* cpu_env = std::getenv("LAYA_GRAD_CPU");
    if (!(cpu_env && cpu_env[0] == '0')) {
        bt::DeviceScope cpu(bt::Device::CPU);
        run_checks(dir, true);
    }

    if (g_failures) {
        std::printf("Laya grad FAILED (%d)\n", g_failures);
        return 1;
    }
    std::printf("Laya grad PASSED\n");
    return 0;
}
