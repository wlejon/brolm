#pragma once

// Mistral 3.1 text decoder — inference-only, KV-cached.
//
// The text backbone of `Mistral3ForConditionalGeneration` (and, used
// standalone, a plain Mistral causal LM). Forward-only. Runs on whichever
// backend brotensor resolves at runtime — FP32 on CPU, FP16 on a GPU backend.
//
// Mistral's text decoder is exactly the shared brolm::detail::DenseDecoder with
// per-head QK-norm DISABLED and an UNTIED lm_head — the same dense GQA / SwiGLU
// / plain-1-D-RoPE causal stack as Qwen3, re-parameterized. This class is a
// thin wrapper that owns the typed Mistral3Config::Text and a DenseDecoder; the
// layer math, KV cache, and weight load all live in dense_decoder.{h,cpp}. See
// that header for the per-layer forward and the HF rotate_half →
// interleaved-pair RoPE weight permute.
//
// Tensor names follow the standalone HF Mistral convention rooted at `model.`
// (`model.embed_tokens.weight`, `model.layers.N.self_attn.q_proj.weight`,
// `model.norm.weight`) with a top-level `lm_head.weight`; pass a `prefix` to
// rebase them. The VLM checkpoint nests the text backbone under
// `language_model.` and carries vision/projector tensors alongside — that
// fusion wiring lands with the Pixtral tower; this loader targets the text
// weights only.
//
// GGUF: deferred. Mistral 3.1 (`mistral3` / Pixtral) needs its own verified
// ggml tensor-name map before a gguf path is wired; HF safetensors is the load
// path today, matching brolm's no-offline-conversion convention.

#include "brolm/detail/dense_decoder.h"
#include "brolm/mistral3_config.h"
#include "brotensor/tensor.h"

#include <cstdint>
#include <string>
#include <vector>

namespace brotensor::safetensors { class File; }

namespace brolm::mistral3 {

class TextModel {
public:
    explicit TextModel(const Mistral3Config::Text& cfg);
    ~TextModel();

    TextModel(const TextModel&)            = delete;
    TextModel& operator=(const TextModel&) = delete;
    TextModel(TextModel&&) noexcept            = default;
    TextModel& operator=(TextModel&&) noexcept = default;

    // Load all weights from a single safetensors file under `prefix`. Tensor
    // names follow the HF convention; source tensors may be F16, F32, or BF16.
    // An untied checkpoint carries `lm_head.weight`; a tied one omits it and
    // the embedding matrix is reused (Mistral 3.1 is untied). Throws
    // std::runtime_error on a missing name or shape mismatch.
    void load_weights(const brotensor::safetensors::File& f,
                      const std::string& prefix = "");

    // Sharded overload: a tensor is resolved by scanning the shards in order,
    // first match wins (Mistral-Small-3.1 ships several safetensors shards).
    void load_weights(
        const std::vector<const brotensor::safetensors::File*>& shards,
        const std::string& prefix = "");

    // Allocate the per-layer K/V cache for sequences up to `max_seq_len`
    // tokens. Sized once; resets cache_len to 0.
    void allocate_cache(int max_seq_len) { core_.allocate_cache(max_seq_len); }

    // Reset the cache length to 0, keeping the allocation.
    void reset_cache() { core_.reset_cache(); }

    int cache_len() const { return core_.cache_len(); }

    // Append L tokens at absolute positions [cache_len, cache_len + L), run the
    // decoder, and write `logits_out` := (L, vocab_size) at the compute dtype.
    // Advances cache_len by L. Prefill = one call with the whole prompt;
    // decode = a call with L == 1.
    //   ids: host pointer to L int32 token IDs in [0, vocab_size).
    // brotensor::init() must have been called once before any forward.
    void forward(const int32_t* ids, int L, brotensor::Tensor& logits_out) {
        core_.forward(ids, L, logits_out);
    }

    const Mistral3Config::Text& config() const { return cfg_; }

private:
    Mistral3Config::Text cfg_;
    brolm::detail::DenseDecoder core_;
};

}  // namespace brolm::mistral3
