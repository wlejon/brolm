#include "brolm/detail/nllb_ops.h"

#include "brolm/detail/compute.h"

#include "brotensor/ops.h"
#include "brotensor/runtime.h"

#include <cmath>
#include <cstring>
#include <vector>

namespace brolm::nllb::detail {

namespace bt = ::brotensor;

void layer_norm(const bt::Tensor& X, const bt::Tensor& gamma,
                const bt::Tensor& beta, bt::Tensor& Y, float eps) {
    if (brolm::compute_dtype() == bt::Dtype::FP16) {
        bt::layernorm_forward_inference_batched_fp16(X, gamma, beta, Y, eps);
    } else {
        bt::layernorm_forward_inference_batched(X, gamma, beta, Y, eps);
    }
}

void linear(const bt::Tensor& W, const bt::Tensor& bias,
            const bt::Tensor& X, bt::Tensor& Y) {
    if (brolm::compute_dtype() == bt::Dtype::FP16) {
        bt::linear_forward_batched_fp16(W, &bias, X, Y);
    } else {
        bt::linear_forward_batched(W, bias, X, Y);
    }
}

bt::Tensor to_compute(const float* host, int r, int c) {
    bt::Tensor f32 = bt::Tensor::from_host(host, r, c).to(bt::default_device());
    if (brolm::compute_dtype() == bt::Dtype::FP16) {
        bt::Tensor f16;
        bt::cast(f32, f16, bt::Dtype::FP16);
        return f16;
    }
    return f32;
}

bt::Tensor sinusoidal_positions(int L, int d_model, int pad_id, int past_len) {
    const int half = d_model / 2;
    const double log_base = std::log(10000.0);
    const double denom = (half > 1) ? static_cast<double>(half - 1) : 1.0;

    std::vector<float> tab(static_cast<std::size_t>(L) * d_model, 0.0f);
    for (int i = 0; i < L; ++i) {
        const double pos =
            static_cast<double>(past_len + i + pad_id + 1);
        float* row = tab.data() + static_cast<std::size_t>(i) * d_model;
        for (int j = 0; j < half; ++j) {
            const double freq = std::exp(-static_cast<double>(j) * log_base / denom);
            const double a = pos * freq;
            row[j]        = static_cast<float>(std::sin(a));
            row[half + j] = static_cast<float>(std::cos(a));
        }
        // Odd d_model would leave the final column zero (M2M-100 pads it); NLLB
        // d_model is even so this never triggers.
    }
    return to_compute(tab.data(), L, d_model);
}

bt::Tensor upload_ids(const std::int32_t* ids, int n) {
    bt::Tensor cpu = bt::Tensor::empty_on(bt::Device::CPU, n, 1, bt::Dtype::INT32);
    std::memcpy(cpu.host_raw_mut(), ids,
                static_cast<std::size_t>(n) * sizeof(std::int32_t));
    return cpu.to(bt::default_device());
}

}  // namespace brolm::nllb::detail
