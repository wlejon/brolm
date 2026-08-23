// Shared logits-row download for brolm's generate loop.
// The templated generate() lives in detail/generate.h; sampling algorithms live in sampler.cpp.

#include "brolm/detail/generate.h"
#include "brolm/detail/profile.h"
#include "brotensor/tensor.h"

#include <cstddef>
#include <cstdint>
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

}  // namespace

std::vector<float> last_row_fp32(const brotensor::Tensor& logits) {
    profile::ScopedStage ps(profile::Stage::logits_download);
    if (logits.rows == 1) return download_fp32(logits);
    // Download only the final row — a (1, vocab) view into the contiguous
    // row-major buffer. The intermediate rows never cross the bus.
    brotensor::Tensor last = brotensor::Tensor::view(
        logits.device,
        static_cast<char*>(logits.data) +
            static_cast<std::size_t>(logits.rows - 1) *
                static_cast<std::size_t>(logits.cols) *
                static_cast<std::size_t>(
                    brotensor::dtype_size_bytes(logits.dtype)),
        1, logits.cols, logits.dtype);
    return download_fp32(last);
}

}  // namespace brolm::detail
