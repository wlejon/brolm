#pragma once

// Weight-source adapter — uniform API over brotensor::safetensors and
// brotensor::gguf checkpoints.
//
// Model loaders (qwen.cpp, qwen35_text.cpp, etc.) call a single Source
// interface and stay agnostic of the on-disk container. Names passed in are
// the Hugging Face naming convention; the GgufSource maps them to ggml
// (`token_embd.weight`, `blk.N.attn_q.weight`, ...) via a caller-supplied
// translation function.
//
// Two upload entry points:
//   - upload_compute_checked: standard dense-weight load. Source must be
//     F16/F32/BF16 (safetensors) or F16/F32/BF16/quant (gguf). Quant tensors
//     keep their on-disk dtype (Q4_K / Q6_K / Q8_0); brolm::detail::linear_batched
//     dispatches on it. Dense tensors land at brolm::compute_dtype().
//   - upload_compute_rope_permuted: same as above, but applies the HF →
//     interleaved-pair RoPE row permutation within each head's head_dim block.
//     Dense path goes through an FP32 host roundtrip (matches the pre-adapter
//     safetensors loader); quant path does a byte-level row swap (each row of
//     a (rows, cols) quant tensor with cols % block_size == 0 occupies a
//     contiguous (cols/block_size)*block_bytes byte run, so row swapping is
//     bit-faithful).

#include "brolm/detail/compute.h"

#include "brotensor/gguf.h"
#include "brotensor/safetensors.h"
#include "brotensor/tensor.h"
#include "brotensor/runtime.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace brolm::detail::weights {

class Source {
public:
    virtual ~Source() = default;

    // True iff a tensor with HF-style `name` exists in the source.
    virtual bool has(const std::string& name) const = 0;

    // Upload `name` as (rows, cols) into `dst`. For dense source dtypes
    // (F16/F32/BF16) the destination dtype is brolm::compute_dtype(); for
    // quant gguf dtypes the destination keeps the on-disk dtype.
    // `label` is included in error messages.
    virtual void upload_compute_checked(const std::string& name,
                                        int rows, int cols,
                                        brotensor::Tensor& dst,
                                        const std::string& label) const = 0;

    // Like upload_compute_checked, but permute the per-head head_dim row
    // ordering of an (num_heads * head_dim, cols) weight from HF rotate_half
    // order into brotensor's interleaved-pair RoPE order:
    //   dst rows (2i,   2i+1) <- src rows (i, i + head_dim/2)   within each head.
    // `rows` must equal num_heads * head_dim.
    virtual void upload_compute_rope_permuted(const std::string& name,
                                              int rows, int cols,
                                              int num_heads, int head_dim,
                                              brotensor::Tensor& dst,
                                              const std::string& label) const = 0;

    // Read `name` into the caller's host FP16 buffer, resizing it to
    // rows*cols elements. Source dtype must be a dense float type
    // (F16/F32/BF16); F16 is memcpy'd verbatim, F32 / BF16 are converted via
    // brotensor::fp32_to_fp16_bits. Used by callers that do further host-side
    // processing (e.g. T5's INT8 W8A16 quantization) before upload.
    virtual void download_host_fp16(const std::string& name,
                                    int rows, int cols,
                                    std::vector<std::uint16_t>& out,
                                    const std::string& label) const = 0;
};

// ─── permute helpers ───────────────────────────────────────────────────────

namespace detail_ {

// FP32 host buffer row permute (interleaved <- HF) — used by both safetensors
// and the dense gguf paths.
inline std::vector<float> permute_rope_fp32(const std::vector<float>& src,
                                            int num_heads, int head_dim,
                                            int cols) {
    const int half = head_dim / 2;
    std::vector<float> dst(src.size());
    for (int h = 0; h < num_heads; ++h) {
        const std::size_t base =
            static_cast<std::size_t>(h) * head_dim *
            static_cast<std::size_t>(cols);
        for (int i = 0; i < half; ++i) {
            const std::size_t d0 = base + static_cast<std::size_t>(2 * i) * cols;
            const std::size_t d1 = base + static_cast<std::size_t>(2 * i + 1) * cols;
            const std::size_t s0 = base + static_cast<std::size_t>(i) * cols;
            const std::size_t s1 = base + static_cast<std::size_t>(i + half) * cols;
            std::memcpy(&dst[d0], &src[s0],
                        static_cast<std::size_t>(cols) * sizeof(float));
            std::memcpy(&dst[d1], &src[s1],
                        static_cast<std::size_t>(cols) * sizeof(float));
        }
    }
    return dst;
}

// Byte-level row permute (interleaved <- HF). Works for any dtype where a
// "row" of `cols` elements occupies `bytes_per_row` contiguous bytes —
// dense (FP32/FP16/BF16/INT8) and gguf quant carriers alike.
inline void permute_rope_bytes(const uint8_t* src, uint8_t* dst,
                               int num_heads, int head_dim,
                               std::size_t bytes_per_row) {
    const int half = head_dim / 2;
    for (int h = 0; h < num_heads; ++h) {
        const std::size_t base =
            static_cast<std::size_t>(h) * head_dim * bytes_per_row;
        for (int i = 0; i < half; ++i) {
            std::memcpy(dst + base + static_cast<std::size_t>(2 * i) * bytes_per_row,
                        src + base + static_cast<std::size_t>(i) * bytes_per_row,
                        bytes_per_row);
            std::memcpy(dst + base + static_cast<std::size_t>(2 * i + 1) * bytes_per_row,
                        src + base + static_cast<std::size_t>(i + half) * bytes_per_row,
                        bytes_per_row);
        }
    }
}

// FP32 download of an already-loaded tensor (handles both FP16-on-GPU and
// FP32-on-CPU compute dtypes).
inline std::vector<float> download_fp32(const brotensor::Tensor& t) {
    const std::size_t n = static_cast<std::size_t>(t.size());
    if (t.dtype == brotensor::Dtype::FP16) {
        std::vector<uint16_t> bits(n);
        t.copy_to_host_fp16(bits.data());
        std::vector<float> out(n);
        for (std::size_t i = 0; i < n; ++i) {
            out[i] = brotensor::fp16_bits_to_fp32(bits[i]);
        }
        return out;
    }
    return t.to_host_vector();
}

}  // namespace detail_

// ─── SafetensorsSource ─────────────────────────────────────────────────────

class SafetensorsSource final : public Source {
public:
    SafetensorsSource(std::vector<const brotensor::safetensors::File*> shards,
                      std::string prefix = "")
        : shards_(std::move(shards)), prefix_(std::move(prefix)) {}

    bool has(const std::string& name) const override {
        return find_(prefix_ + name) != nullptr;
    }

    void upload_compute_checked(const std::string& name, int rows, int cols,
                                brotensor::Tensor& dst,
                                const std::string& label) const override {
        const auto& view = need_(prefix_ + name);
        brotensor::safetensors::upload_compute_checked(view, rows, cols, dst, label);
    }

    void upload_compute_rope_permuted(const std::string& name,
                                      int rows, int cols,
                                      int num_heads, int head_dim,
                                      brotensor::Tensor& dst,
                                      const std::string& label) const override {
        brotensor::Tensor raw;
        const auto& view = need_(prefix_ + name);
        brotensor::safetensors::upload_compute_checked(view, rows, cols, raw, label);
        std::vector<float> host = detail_::download_fp32(raw);
        std::vector<float> perm = detail_::permute_rope_fp32(host, num_heads,
                                                             head_dim, cols);
        dst = brolm::detail::upload_host(perm.data(), rows, cols);
    }

    void download_host_fp16(const std::string& name, int rows, int cols,
                            std::vector<std::uint16_t>& out,
                            const std::string& label) const override {
        const auto& view = need_(prefix_ + name);
        namespace stns = brotensor::safetensors;
        const std::int64_t expected =
            static_cast<std::int64_t>(rows) * static_cast<std::int64_t>(cols);
        if (view.numel() != expected) {
            throw std::runtime_error(
                label + " ('" + view.name + "'): shape mismatch (expected " +
                std::to_string(rows) + "x" + std::to_string(cols) + ", got " +
                std::to_string(view.numel()) + " elements)");
        }
        const std::size_t n = static_cast<std::size_t>(expected);
        out.resize(n);
        switch (view.dtype) {
            case stns::Dtype::F16:
                std::memcpy(out.data(), view.data, n * sizeof(std::uint16_t));
                return;
            case stns::Dtype::F32: {
                const auto* src = reinterpret_cast<const float*>(view.data);
                for (std::size_t i = 0; i < n; ++i) {
                    out[i] = brotensor::fp32_to_fp16_bits(src[i]);
                }
                return;
            }
            case stns::Dtype::BF16: {
                const auto* src = reinterpret_cast<const std::uint16_t*>(view.data);
                for (std::size_t i = 0; i < n; ++i) {
                    out[i] = brotensor::fp32_to_fp16_bits(
                        brotensor::bf16_bits_to_fp32(src[i]));
                }
                return;
            }
            default:
                throw std::runtime_error(
                    label + " ('" + view.name +
                    "'): unsupported source dtype for download_host_fp16");
        }
    }

private:
    const brotensor::safetensors::TensorView* find_(const std::string& key) const {
        for (const auto* f : shards_) {
            if (const auto* v = f->find(key)) return v;
        }
        return nullptr;
    }
    const brotensor::safetensors::TensorView& need_(const std::string& key) const {
        if (const auto* v = find_(key)) return *v;
        throw std::runtime_error("weights::SafetensorsSource: missing tensor '" +
                                 key + "'");
    }

    std::vector<const brotensor::safetensors::File*> shards_;
    std::string prefix_;
};

// ─── GgufSource ────────────────────────────────────────────────────────────

class GgufSource final : public Source {
public:
    // NameMap translates a HF-style tensor name (e.g.
    // "model.layers.3.self_attn.q_proj.weight") into the ggml name used in
    // the gguf file ("blk.3.attn_q.weight"). Empty return means "no such
    // mapping exists for this name" — used by has() to report absence.
    using NameMap = std::function<std::string(std::string_view)>;

    GgufSource(std::vector<const brotensor::gguf::File*> shards, NameMap mapper)
        : shards_(std::move(shards)), mapper_(std::move(mapper)) {}

    bool has(const std::string& name) const override {
        const std::string ggml = mapper_(name);
        if (ggml.empty()) return false;
        return find_(ggml) != nullptr;
    }

    void upload_compute_checked(const std::string& name, int rows, int cols,
                                brotensor::Tensor& dst,
                                const std::string& label) const override {
        const auto& info = need_(name, label);
        check_shape_(info, rows, cols, label);
        if (brotensor::dtype_is_quant(info.dtype)) {
            // Keep the on-disk quant carrier; the matmul op dispatches on it.
            brotensor::gguf::upload_raw(info, rows, cols, dst);
            return;
        }
        upload_dense_(info, rows, cols, dst, label);
    }

    void upload_compute_rope_permuted(const std::string& name,
                                      int rows, int cols,
                                      int num_heads, int head_dim,
                                      brotensor::Tensor& dst,
                                      const std::string& label) const override {
        const auto& info = need_(name, label);
        check_shape_(info, rows, cols, label);
        if (rows != num_heads * head_dim) {
            throw std::runtime_error(
                label + " ('" + info.name +
                "'): rope-permuted rows (" + std::to_string(rows) +
                ") != num_heads*head_dim (" +
                std::to_string(num_heads * head_dim) + ")");
        }

        if (brotensor::dtype_is_quant(info.dtype)) {
            // Byte-level row permute, then upload as the same quant dtype.
            const int bs = brotensor::dtype_block_size(info.dtype);
            if (cols % bs != 0) {
                throw std::runtime_error(
                    label + " ('" + info.name + "'): cols (" +
                    std::to_string(cols) +
                    ") not a multiple of quant block size " +
                    std::to_string(bs));
            }
            const std::size_t bpr = static_cast<std::size_t>(
                brotensor::dtype_storage_bytes(info.dtype,
                                               static_cast<std::int64_t>(cols)));
            std::vector<uint8_t> permuted(info.nbytes);
            detail_::permute_rope_bytes(info.data, permuted.data(),
                                        num_heads, head_dim, bpr);
            brotensor::Tensor cpu_t = brotensor::Tensor::empty_on(
                brotensor::Device::CPU, rows, cols, info.dtype);
            if (cpu_t.bytes() != info.nbytes) {
                throw std::runtime_error(
                    label + " ('" + info.name +
                    "'): cpu tensor bytes != info.nbytes");
            }
            std::memcpy(cpu_t.host_raw_mut(), permuted.data(), info.nbytes);
            const brotensor::Device target = brotensor::default_device();
            dst = (target == brotensor::Device::CPU) ? std::move(cpu_t)
                                                     : cpu_t.to(target);
            return;
        }

        // Dense path: FP32 host roundtrip, mirroring SafetensorsSource.
        brotensor::Tensor raw;
        upload_dense_(info, rows, cols, raw, label);
        std::vector<float> host = detail_::download_fp32(raw);
        std::vector<float> perm = detail_::permute_rope_fp32(host, num_heads,
                                                             head_dim, cols);
        dst = brolm::detail::upload_host(perm.data(), rows, cols);
    }

    void download_host_fp16(const std::string& name, int rows, int cols,
                            std::vector<std::uint16_t>& out,
                            const std::string& label) const override {
        const auto& info = need_(name, label);
        check_shape_(info, rows, cols, label);
        const brotensor::Dtype dt = info.dtype;
        if (dt != brotensor::Dtype::FP16 && dt != brotensor::Dtype::FP32 &&
            dt != brotensor::Dtype::BF16) {
            throw std::runtime_error(
                label + " ('" + info.name +
                "'): unsupported source dtype for download_host_fp16 "
                "(quant dtypes cannot be host-FP16-projected without dequant)");
        }
        const std::size_t n =
            static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
        out.resize(n);
        if (dt == brotensor::Dtype::FP16) {
            std::memcpy(out.data(), info.data, n * sizeof(std::uint16_t));
            return;
        }
        if (dt == brotensor::Dtype::FP32) {
            const auto* src = reinterpret_cast<const float*>(info.data);
            for (std::size_t i = 0; i < n; ++i) {
                out[i] = brotensor::fp32_to_fp16_bits(src[i]);
            }
            return;
        }
        // BF16
        const auto* src = reinterpret_cast<const std::uint16_t*>(info.data);
        for (std::size_t i = 0; i < n; ++i) {
            out[i] = brotensor::fp32_to_fp16_bits(brotensor::bf16_bits_to_fp32(src[i]));
        }
    }

private:
    const brotensor::gguf::TensorInfo* find_(const std::string& ggml_name) const {
        for (const auto* f : shards_) {
            if (const auto* v = f->find_tensor(ggml_name)) return v;
        }
        return nullptr;
    }

    const brotensor::gguf::TensorInfo& need_(const std::string& hf_name,
                                             const std::string& label) const {
        const std::string ggml = mapper_(hf_name);
        if (ggml.empty()) {
            throw std::runtime_error(
                label + ": no ggml mapping for HF name '" + hf_name + "'");
        }
        if (const auto* v = find_(ggml)) return *v;
        throw std::runtime_error(
            label + ": missing tensor '" + ggml +
            "' (mapped from HF name '" + hf_name + "')");
    }

    static void check_shape_(const brotensor::gguf::TensorInfo& info,
                             int rows, int cols, const std::string& label) {
        const std::int64_t expected =
            static_cast<std::int64_t>(rows) * static_cast<std::int64_t>(cols);
        if (info.numel != expected) {
            throw std::runtime_error(
                label + " ('" + info.name + "'): shape mismatch (expected " +
                std::to_string(rows) + "x" + std::to_string(cols) + " = " +
                std::to_string(expected) + " elements, got " +
                std::to_string(info.numel) + ")");
        }
    }

    // Dense (F16/F32/BF16) upload to brolm::compute_dtype(). Mirrors
    // brotensor::safetensors::upload_compute, but reads from gguf TensorInfo
    // (which already uses brotensor::Dtype directly).
    static void upload_dense_(const brotensor::gguf::TensorInfo& info,
                              int rows, int cols,
                              brotensor::Tensor& dst,
                              const std::string& label) {
        const brotensor::Dtype dt = info.dtype;
        if (dt != brotensor::Dtype::FP32 && dt != brotensor::Dtype::FP16 &&
            dt != brotensor::Dtype::BF16) {
            throw std::runtime_error(
                label + " ('" + info.name +
                "'): unsupported dense gguf dtype (expected FP32/FP16/BF16)");
        }
        const std::size_t n =
            static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);

        if (brolm::compute_dtype() == brotensor::Dtype::FP32) {
            if (dt == brotensor::Dtype::FP32) {
                dst = brotensor::Tensor::from_host(
                    reinterpret_cast<const float*>(info.data), rows, cols);
            } else {
                const uint16_t* src =
                    reinterpret_cast<const uint16_t*>(info.data);
                std::vector<float> tmp(n);
                for (std::size_t i = 0; i < n; ++i) {
                    tmp[i] = (dt == brotensor::Dtype::FP16)
                                 ? brotensor::fp16_bits_to_fp32(src[i])
                                 : brotensor::bf16_bits_to_fp32(src[i]);
                }
                dst = brotensor::Tensor::from_host(tmp.data(), rows, cols);
            }
        } else {  // FP16 compute
            if (dt == brotensor::Dtype::FP16) {
                dst = brotensor::Tensor::from_host_fp16(
                    reinterpret_cast<const uint16_t*>(info.data), rows, cols);
            } else {
                const float* src32 = reinterpret_cast<const float*>(info.data);
                const uint16_t* src16 =
                    reinterpret_cast<const uint16_t*>(info.data);
                std::vector<uint16_t> tmp(n);
                for (std::size_t i = 0; i < n; ++i) {
                    const float f = (dt == brotensor::Dtype::FP32)
                                        ? src32[i]
                                        : brotensor::bf16_bits_to_fp32(src16[i]);
                    tmp[i] = brotensor::fp32_to_fp16_bits(f);
                }
                dst = brotensor::Tensor::from_host_fp16(tmp.data(), rows, cols);
            }
        }
    }

    std::vector<const brotensor::gguf::File*> shards_;
    NameMap mapper_;
};

}  // namespace brolm::detail::weights
