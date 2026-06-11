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
#include <string_view>
#include <vector>

namespace brotensor::safetensors { class File; }
namespace brotensor::gguf { class File; }

namespace brolm::mistral3 {

// Translate a HF-style Mistral tensor name (e.g.
// "model.layers.3.self_attn.q_proj.weight") into the ggml/llama.cpp name used
// in a Mistral .gguf checkpoint ("blk.3.attn_q.weight"). This is the Qwen3 map
// minus the q_norm/k_norm entries — Mistral has no QK-norm — and with
// `lm_head.weight` → `output.weight` for the untied head. Returns an empty
// string if the name matches no known Mistral text weight. Exposed so callers
// can build their own gguf-shard pipelines.
std::string mistral3_hf_to_ggml(std::string_view hf_name);

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

    // GGUF overloads. Tensor names follow the ggml/llama.cpp convention
    // (`token_embd.weight`, `blk.N.attn_q.weight`, `output.weight`, ...); see
    // mistral3_hf_to_ggml. Quantized weights (Q4_K / Q6_K / Q8_0) keep their
    // on-disk dtype and dispatch through brotensor's fused-dequant matmuls
    // (GPU-only today; CPU + quant weights throws at first matmul). Mistral is
    // untied, so `output.weight` is expected to be present.
    void load_weights(const brotensor::gguf::File& f);
    void load_weights(const std::vector<const brotensor::gguf::File*>& shards);

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

    // Like forward(), but `logits_out` := (1, vocab_size) — logits for the
    // last appended token only. The KV cache still ingests all L tokens. The
    // generate loop's prefill path: a sampler never reads the L-1
    // intermediate logit rows, and at prompt lengths the skipped lm_head rows
    // dominate the whole forward (vocab >> hidden).
    void forward_last(const int32_t* ids, int L, brotensor::Tensor& logits_out) {
        core_.forward_last(ids, L, logits_out);
    }

    // Embedding lookup only: `out` := (L, hidden_size) input embeddings, no KV
    // cache touched. The VL path uses this to build the input stream before
    // overwriting [IMG] rows with projector embeddings.
    void embed_tokens(const int32_t* ids, int L, brotensor::Tensor& out) {
        core_.embed_tokens(ids, L, out);
    }

    // Forward from precomputed (L, hidden_size) input embeddings (e.g. text
    // embeddings with image rows spliced in). Appends at [cache_len, +L).
    void forward_embeds(const brotensor::Tensor& embeds, int L,
                        brotensor::Tensor& logits_out) {
        core_.forward_embeds(embeds, L, logits_out);
    }

    const Mistral3Config::Text& config() const { return cfg_; }

private:
    Mistral3Config::Text cfg_;
    brolm::detail::DenseDecoder core_;
};

}  // namespace brolm::mistral3
