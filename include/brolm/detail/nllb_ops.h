#pragma once

// Internal compute helpers shared by the NLLB-200 encoder and decoder.
//
// These cover the NLLB-specific pieces the M2M-100 blocks need on top of the
// shared brolm::detail dtype helpers (linear_batched / layernorm_batched) and
// the dtype-polymorphic attention kernels: the computed sinusoidal position
// table, an FP32-host upload at the compute dtype, and the int32 id upload.

#include "brotensor/tensor.h"

#include <cstdint>

namespace brolm::nllb::detail {

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
