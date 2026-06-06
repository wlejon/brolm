#pragma once

// Qwen3 decoder transformer — inference-only, KV-cached.
//
// Faithful port of Hugging Face `Qwen3ForCausalLM`. Forward-only. Runs on
// whichever backend brotensor resolves at runtime — CPU by default, CUDA when
// available — at that backend's compute dtype (FP32 on CPU, FP16 on a GPU).
//
// Thin wrapper over the shared brolm::detail::DenseDecoder core: Qwen3 is the
// dense GQA/SwiGLU/RoPE decoder with per-head QK-norm enabled. This class owns
// the typed Qwen3Config (and its gguf parsing / name map); the layer math, KV
// cache, and weight load live in dense_decoder.{h,cpp}. See that header for the
// per-layer forward and the HF rotate_half → interleaved-pair RoPE permute.

#include "brolm/detail/dense_decoder.h"
#include "brotensor/tensor.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace brotensor::safetensors { class File; struct TensorView; }
namespace brotensor::gguf { class File; }

namespace brolm::qwen {

// Translate a HF-style Qwen3 tensor name (e.g.
// "model.layers.3.self_attn.q_proj.weight") into the ggml/llama.cpp name used
// in a Qwen3 .gguf checkpoint ("blk.3.attn_q.weight"). Returns an empty
// string if the name does not match any known Qwen3 weight. Exposed so
// callers can build their own gguf-shard pipelines.
std::string qwen3_hf_to_ggml(std::string_view hf_name);

struct Qwen3Config {
    int   vocab_size            = 151936;
    int   hidden_size           = 1024;
    int   intermediate_size     = 3072;
    int   num_hidden_layers     = 28;
    int   num_attention_heads   = 16;     // query heads
    int   num_key_value_heads   = 8;      // KV heads (GQA)
    int   head_dim              = 128;    // independent of hidden_size/num_heads
    float rms_norm_eps          = 1e-6f;
    float rope_theta            = 1000000.0f;
    bool  tie_word_embeddings   = true;
    int   max_position_embeddings = 40960;

    // Populate a Qwen3Config from the metadata of a Qwen3 .gguf file. Reads
    // the llama.cpp-convention keys (`qwen3.embedding_length`,
    // `qwen3.attention.head_count`, ...) plus the tokenizer vocab length for
    // `vocab_size`. `tie_word_embeddings` is set from whether the file
    // contains an `output.weight` tensor (absent = tied). Throws
    // std::runtime_error on missing required metadata or an architecture
    // mismatch (`general.architecture` != "qwen3").
    static Qwen3Config from_gguf(const brotensor::gguf::File& f);
};

class Qwen3Model {
public:
    explicit Qwen3Model(const Qwen3Config& cfg);
    ~Qwen3Model();

    // Non-copyable; movable.
    Qwen3Model(const Qwen3Model&) = delete;
    Qwen3Model& operator=(const Qwen3Model&) = delete;
    Qwen3Model(Qwen3Model&&) noexcept = default;
    Qwen3Model& operator=(Qwen3Model&&) noexcept = default;

    // Load all weights from a single safetensors file under `prefix`. Tensor
    // names follow the HF convention; see qwen.cpp for the full list. Source
    // tensors may be F16, F32, or BF16. When tie_word_embeddings is true,
    // `lm_head.weight` is expected to be absent and equals embed_tokens.weight.
    // Throws std::runtime_error on a missing name or shape mismatch.
    void load_weights(const brotensor::safetensors::File& f,
                      const std::string& prefix = "");

    // Sharded overload: a tensor is resolved by scanning the shards in order,
    // first match wins.
    void load_weights(
        const std::vector<const brotensor::safetensors::File*>& shards,
        const std::string& prefix = "");

    // GGUF overloads. Tensor names follow the ggml/llama.cpp Qwen3 convention
    // (`token_embd.weight`, `blk.N.attn_q.weight`, ...); see qwen3_hf_to_ggml.
    // Quantized weights (Q4_K / Q6_K / Q8_0) keep their on-disk dtype and
    // dispatch through brotensor's fused-dequant matmuls (GPU-only today;
    // CPU + quant weights throws at first matmul). When tie_word_embeddings
    // is true, the `output.weight` tensor is expected to be absent and the
    // embedding matrix is reused.
    void load_weights(const brotensor::gguf::File& f);
    void load_weights(
        const std::vector<const brotensor::gguf::File*>& shards);

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

    const Qwen3Config& config() const { return cfg_; }

private:
    Qwen3Config cfg_;
    brolm::detail::DenseDecoder core_;
};

}  // namespace brolm::qwen
