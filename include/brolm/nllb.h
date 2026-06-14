#pragma once

// NLLB-200 (M2M-100) encoder-decoder translation model.
//
// Forward-only. Runs on whichever backend brotensor resolves at runtime — CPU
// (FP32) by default, CUDA (FP16) when enabled. Architecture
// (facebook/nllb-200-distilled-600M): a vanilla pre-norm encoder-decoder
// transformer of the fairseq / BART lineage.
//
//   Embedding (shared between encoder, decoder, and lm_head):
//     x = embedding_lookup(shared, ids) * sqrt(d_model) + sinusoidal_positions
//
//   Encoder layer (pre-norm, bidirectional self-attention):
//     r = x;  x = layernorm(x, self_attn_layer_norm)
//     x = r + self_attn(x)                       (MHA with q/k/v/out biases)
//     r = x;  x = layernorm(x, final_layer_norm)
//     x = r + fc2(relu(fc1(x)))
//   ...then a final encoder layernorm.
//
//   Decoder layer adds a cross-attention sub-layer over the encoder output and
//   makes its self-attention causal (see nllb_decoder).
//
// LayerNorm carries weight AND bias; every linear (q/k/v/out, fc1, fc2) carries
// a bias. Positions are COMPUTED sinusoids (nothing to load). This header
// declares the Encoder; the Decoder, KV cache, and beam-search Translator land
// alongside it.

#include "brolm/nllb_config.h"
#include "brolm/tokenizer_nllb.h"

#include "brotensor/tensor.h"

#include <cstdint>
#include <string>
#include <vector>

namespace brotensor::safetensors { class File; }
namespace brolm::detail::weights { class Source; }

namespace brolm::nllb {

class Encoder {
public:
    explicit Encoder(const NllbConfig& cfg);
    ~Encoder();

    Encoder(const Encoder&) = delete;
    Encoder& operator=(const Encoder&) = delete;
    Encoder(Encoder&&) noexcept = default;
    Encoder& operator=(Encoder&&) noexcept = default;

    // Load encoder weights (and the shared token embedding) from HF
    // safetensors. Default prefix "" — the converted nllb checkpoint uses the
    // full HF names ("model.shared.weight", "model.encoder.layers.N....").
    void load_weights(const brotensor::safetensors::File& f,
                      const std::string& prefix = "");
    void load_weights(
        const std::vector<const brotensor::safetensors::File*>& shards,
        const std::string& prefix = "");

    // Run the encoder over a length-L int32 source-token sequence (host
    // pointer; produced by nllb::Tokenizer::encode_source). Writes the
    // (L, d_model) encoder output — the cross-attention memory for the decoder
    // — to `enc_out` at the compute dtype. brotensor::init() must have run.
    void forward(const std::int32_t* ids, int L, brotensor::Tensor& enc_out);

    const NllbConfig& config() const { return cfg_; }

    // The shared (vocab_size, d_model) token embedding, also used by the
    // decoder and the tied lm_head. Valid after load_weights().
    const brotensor::Tensor& token_embedding() const { return token_embed_; }

private:
    struct Layer {
        brotensor::Tensor sa_ln_w, sa_ln_b;       // self_attn_layer_norm
        brotensor::Tensor Wq, Wk, Wv, Wo;         // (D,D)
        brotensor::Tensor bq, bk, bv, bo;         // (D,1) FP32 (mha contract)
        brotensor::Tensor ff_ln_w, ff_ln_b;       // final_layer_norm (FFN pre-norm)
        brotensor::Tensor fc1_w, fc1_b;           // (ffn,D),(ffn,1)
        brotensor::Tensor fc2_w, fc2_b;           // (D,ffn),(D,1)
    };

    void load_weights_impl_(const brolm::detail::weights::Source& src);

    NllbConfig cfg_;
    brotensor::Tensor token_embed_;               // (vocab_size, d_model)
    std::vector<Layer> layers_;
    brotensor::Tensor enc_ln_w_, enc_ln_b_;       // final encoder layer_norm

    // Per-call scratch (kept alive to avoid realloc).
    brotensor::Tensor ids_dev_, x_, n_, attn_, h1_, ff_out_;
    brotensor::Tensor Qh_, Kh_, Vh_, Attnh_, Yconcat_;   // mha caches
};

// Per-hypothesis self-attention KV cache for incremental decoding. Holds one
// pair of (max_len, d_model) K/V caches per decoder layer plus the number of
// tokens already consumed. Created by Decoder::make_state, advanced one row per
// Decoder::decode_step, and deep-copied (valid prefix only) by
// Decoder::clone_state when a beam branches. The cross-attention K/V are NOT
// here — those are source-fixed and live on the Decoder (set_encoder_memory).
struct DecoderState {
    std::vector<brotensor::Tensor> k_self;   // [decoder_layers] (max_len, D)
    std::vector<brotensor::Tensor> v_self;   // [decoder_layers] (max_len, D)
    int len = 0;        // tokens consumed so far (valid cache rows)
    int max_len = 0;    // cache capacity
};

// Decoder: causal self-attention + cross-attention over the encoder memory +
// ReLU FFN, with a tied lm_head. set_encoder_memory() projects and caches the
// cross-attention K/V once per source (shared across all decode steps and beam
// hypotheses).
//
// Two decode paths share the same weights:
//   - forward_logits(): recomputes the whole prefix each call. Stateless and
//     backend-trivial; used by tests and as the reference.
//   - make_state()/decode_step(): a growing self-attention KV cache that
//     consumes one token per call (O(1) projections/FFN per step instead of
//     O(T)). beam_search() uses this; clone_state() forks a beam's cache.
class Decoder {
public:
    explicit Decoder(const NllbConfig& cfg);
    ~Decoder();

    Decoder(const Decoder&) = delete;
    Decoder& operator=(const Decoder&) = delete;
    Decoder(Decoder&&) noexcept = default;
    Decoder& operator=(Decoder&&) noexcept = default;

    void load_weights(const brotensor::safetensors::File& f,
                      const std::string& prefix = "");
    void load_weights(
        const std::vector<const brotensor::safetensors::File*>& shards,
        const std::string& prefix = "");

    // Project and cache the cross-attention K/V from the (Lk, d_model) encoder
    // output. Call once per source before decoding; reused across beams.
    void set_encoder_memory(const brotensor::Tensor& enc_out);

    // Run the decoder over the length-T int32 prefix `dec_ids` (host pointer;
    // starts with the decoder_start </s> then the forced target-language BOS).
    // Writes the last position's (1, vocab_size) logits to `logits` at the
    // compute dtype. set_encoder_memory() must have run.
    void forward_logits(const std::int32_t* dec_ids, int T,
                        brotensor::Tensor& logits);

    // Allocate an empty decode state with self-attention caches sized for up to
    // `max_len` total decoder tokens (start prefix + generated).
    DecoderState make_state(int max_len) const;

    // Deep-copy a decode state's valid cache prefix into a fresh state of the
    // same capacity — used when a beam forks into multiple children.
    DecoderState clone_state(const DecoderState& src) const;

    // Consume one token: append its self-attention K/V to the state's caches,
    // run the decoder over just that row (self-attn over the cache, cross-attn
    // over the encoder memory, ReLU FFN, tied lm_head), and write the
    // (1, vocab_size) logits for the NEXT token. Advances state.len by 1.
    // set_encoder_memory() must have run.
    void decode_step(DecoderState& state, std::int32_t token,
                     brotensor::Tensor& logits);

    const NllbConfig& config() const { return cfg_; }

private:
    struct Layer {
        brotensor::Tensor sa_ln_w, sa_ln_b;       // self_attn_layer_norm
        brotensor::Tensor sWq, sWk, sWv, sWo;
        brotensor::Tensor sbq, sbk, sbv, sbo;
        brotensor::Tensor ca_ln_w, ca_ln_b;       // encoder_attn_layer_norm
        brotensor::Tensor cWq, cWk, cWv, cWo;
        brotensor::Tensor cbq, cbk, cbv, cbo;
        brotensor::Tensor ff_ln_w, ff_ln_b;       // final_layer_norm
        brotensor::Tensor fc1_w, fc1_b, fc2_w, fc2_b;
        brotensor::Tensor K_enc, V_enc;           // cached cross-attn K/V (Lk,D)
    };

    void load_weights_impl_(const brolm::detail::weights::Source& src);

    NllbConfig cfg_;
    brotensor::Tensor token_embed_;               // (vocab_size, d_model), tied
    brotensor::Tensor final_logits_bias_;         // (1, vocab) or empty
    std::vector<Layer> layers_;
    brotensor::Tensor dec_ln_w_, dec_ln_b_;       // final decoder layer_norm
    int enc_len_ = 0;

    // Per-call scratch.
    brotensor::Tensor ids_dev_, x_, xln_, Q_, K_, V_, attn_, proj_, h1_, xn_;
};

// Beam-search decoding options. Defaults follow common NLLB practice
// (num_beams 5, max_length 200, no length penalty).
struct BeamOptions {
    int   num_beams      = 5;
    int   max_new_tokens = 200;   // cap on generated target tokens
    float length_penalty = 1.0f;  // score / length^penalty at final selection
};

// Beam search over a prepared Decoder (set_encoder_memory() already called for
// the source). `start_ids` is the forced decoder prefix
// [decoder_start_token (</s>), tgt_lang]. Returns the best hypothesis as the
// full decoder token sequence [</s>, tgt_lang, t1, ..., eos]; pass it to
// nllb::Tokenizer::decode (skip_special) to recover the translation text.
// Deterministic — no sampling.
std::vector<std::int32_t> beam_search(
    Decoder& dec, const std::vector<std::int32_t>& start_ids, int eos_id,
    const BeamOptions& opts = {});

// End-to-end translator: tokenizer + config + encoder + decoder + beam search.
// The high-level entry point — load a converted NLLB checkpoint directory and
// translate text between FLORES-200 language codes.
class Translator {
public:
    // Load from a directory holding config.json, tokenizer.json, and
    // model.safetensors (the output of scripts/convert-nllb.py). Throws
    // std::runtime_error on any missing/unparseable file.
    static Translator load(const std::string& model_dir);

    // Translate `text` from `src_lang` into `tgt_lang` (FLORES-200 codes such
    // as "eng_Latn", "fra_Latn"). Throws if a code is unknown.
    std::string translate(const std::string& text,
                          const std::string& src_lang,
                          const std::string& tgt_lang,
                          const BeamOptions& opts = {});

    // Lower-level: translate an already-tokenized source (the output of
    // Tokenizer::encode_source) into the raw target hypothesis token ids
    // [</s>, tgt_lang, t1, ..., eos]. Useful for streaming drivers that manage
    // their own segmentation.
    std::vector<std::int32_t> translate_ids(
        const std::vector<std::int32_t>& src_ids, const std::string& tgt_lang,
        const BeamOptions& opts = {});

    const Tokenizer& tokenizer() const { return tok_; }
    const NllbConfig& config() const { return cfg_; }

private:
    Translator(NllbConfig cfg, Tokenizer tok)
        : cfg_(cfg), tok_(std::move(tok)), enc_(cfg_), dec_(cfg_) {}

    NllbConfig cfg_;
    Tokenizer  tok_;
    Encoder    enc_;
    Decoder    dec_;
    brotensor::Tensor enc_out_;   // per-call scratch
};

}  // namespace brolm::nllb
