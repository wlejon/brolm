#pragma once

// Internal compute helpers shared by the NLLB-200 encoder and decoder.
//
// These wrap the backend dtype split (FP32 on CPU, FP16 on a GPU) for the
// handful of ops the M2M-100 blocks need beyond the dtype-dispatching attention
// kernels: LayerNorm-with-bias, batched Linear-with-bias, the computed
// sinusoidal position table, and the int32 id upload. Keeping them in one place
// stops the encoder and decoder from re-deriving the same dispatch.

#include "brotensor/tensor.h"

#include <cstdint>

namespace brolm::nllb::detail {

// Y = LayerNorm(X) * gamma + beta over each length-D row of the (R,D) X.
// Dispatches FP32 / FP16 on the active compute dtype.
void layer_norm(const brotensor::Tensor& X,
                const brotensor::Tensor& gamma,
                const brotensor::Tensor& beta,
                brotensor::Tensor& Y, float eps);

// Y[b,:] = W * X[b,:] + bias over the (B,in) X. W is (out,in), bias (out,1).
// Dispatches FP32 / FP16 on the active compute dtype.
void linear(const brotensor::Tensor& W, const brotensor::Tensor& bias,
            const brotensor::Tensor& X, brotensor::Tensor& Y);

// Upload an FP32 host (R,C) buffer to the active device at the compute dtype
// (casting to FP16 when the backend is FP16).
brotensor::Tensor to_compute(const float* host, int r, int c);

// M2M-100 sinusoidal position table for `L` positions starting at decode
// offset `past_len`. Row i holds the sinusoid for absolute position
// (past_len + i + pad_id + 1) — M2M-100 offsets positions past the padding
// index. Returns an (L, d_model) tensor at the compute dtype.
//   emb[:half]  = sin(pos * exp(-j*log(10000)/(half-1)))
//   emb[half:]  = cos(pos * exp(-j*log(10000)/(half-1)))   half = d_model/2
brotensor::Tensor sinusoidal_positions(int L, int d_model, int pad_id,
                                       int past_len);

// Stage `n` int32 ids on the host and migrate them to the active device
// (brotensor has no INT32 from_host path).
brotensor::Tensor upload_ids(const std::int32_t* ids, int n);

}  // namespace brolm::nllb::detail
