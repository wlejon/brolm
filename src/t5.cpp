#include "brolm/t5.h"

#include "brotensor/gguf.h"
#include "brotensor/safetensors.h"
#include "brolm/detail/compute.h"
#include "brolm/detail/weights.h"

#include "brotensor/ops.h"
#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace brolm::t5 {

namespace bt = ::brotensor;
namespace st = ::brotensor::safetensors;

// ─── helpers ───────────────────────────────────────────────────────────────

namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("t5::TextEncoder: " + msg);
}

// Build a device-resident INT32 buffer holding `n` token ids. brotensor has
// no from_host path for INT32, so stage on the host then migrate to the
// default device.
bt::Tensor make_idx_device(const int32_t* host, int n) {
    bt::Tensor cpu = bt::Tensor::empty_on(bt::Device::CPU, n, 1, bt::Dtype::INT32);
    std::memcpy(cpu.host_raw_mut(), host,
                static_cast<std::size_t>(n) * sizeof(int32_t));
    return cpu.to(bt::default_device());
}

// Download a loaded weight tensor into a host FP32 vector, handling both the
// FP16 (GPU) and FP32 (CPU) compute-dtype cases — used to pull the
// relative-position-bias table back to the host after upload.
std::vector<float> download_fp32(const bt::Tensor& t) {
    const std::size_t n = static_cast<std::size_t>(t.size());
    if (t.dtype == bt::Dtype::FP16) {
        std::vector<std::uint16_t> bits(n);
        t.copy_to_host_fp16(bits.data());
        std::vector<float> out(n);
        for (std::size_t i = 0; i < n; ++i) {
            out[i] = bt::fp16_bits_to_fp32(bits[i]);
        }
        return out;
    }
    return t.to_host_vector();
}

// T5 bidirectional relative-position bucket (encoder). Mirrors HF's
// _relative_position_bucket with bidirectional=True.
int relative_position_bucket(int relative_position, int num_buckets,
                             int max_distance) {
    int ret = 0;
    num_buckets /= 2;
    if (relative_position > 0) ret += num_buckets;
    int n = std::abs(relative_position);
    int max_exact = num_buckets / 2;
    if (n < max_exact) {
        ret += n;
    } else {
        int large = max_exact +
            static_cast<int>(std::log(static_cast<double>(n) / max_exact) /
                             std::log(static_cast<double>(max_distance) / max_exact) *
                             static_cast<double>(num_buckets - max_exact));
        if (large > num_buckets - 1) large = num_buckets - 1;
        ret += large;
    }
    return ret;
}

}  // namespace

// ─── ctor / dtor ───────────────────────────────────────────────────────────

TextEncoder::TextEncoder(const T5Config& cfg) : cfg_(cfg) {
    if (cfg_.d_model <= 0 || cfg_.num_heads <= 0 ||
        cfg_.d_model % cfg_.num_heads != 0) {
        fail("d_model must be a positive multiple of num_heads");
    }
    if (cfg_.num_layers <= 0 || cfg_.vocab_size <= 0 || cfg_.d_ff <= 0 ||
        cfg_.relative_attention_num_buckets <= 0 ||
        cfg_.relative_attention_max_distance <= 0) {
        fail("config has non-positive dimension");
    }
    blocks_.resize(static_cast<std::size_t>(cfg_.num_layers));
}

TextEncoder::~TextEncoder() = default;

// ─── HF → ggml tensor-name map (T5) ────────────────────────────────────────
//
// HF naming:  shared.weight (or encoder.embed_tokens.weight),
//             encoder.block.N.layer.0.layer_norm.weight,
//             encoder.block.N.layer.0.SelfAttention.{q,k,v,o}.weight,
//             encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight,
//             encoder.block.N.layer.1.layer_norm.weight,
//             encoder.block.N.layer.1.DenseReluDense.{wi_0,wi_1,wo}.weight,
//             encoder.final_layer_norm.weight.
// ggml naming (llama.cpp / city96 T5 GGUFs):
//             token_embd.weight,
//             enc.blk.N.attn_norm.weight,
//             enc.blk.N.attn_{q,k,v,o}.weight,
//             enc.blk.0.attn_rel_b.weight,
//             enc.blk.N.ffn_norm.weight,
//             enc.blk.N.ffn_{gate,up,down}.weight,   (wi_0/wi_1/wo)
//             enc.output_norm.weight.

namespace {

bool starts_with(std::string_view s, std::string_view p) {
    return s.size() >= p.size() && s.compare(0, p.size(), p) == 0;
}

}  // namespace

std::string t5_hf_to_ggml(std::string_view hf_name) {
    if (hf_name == "shared.weight" ||
        hf_name == "encoder.embed_tokens.weight") {
        return "token_embd.weight";
    }
    if (hf_name == "encoder.final_layer_norm.weight") {
        return "enc.output_norm.weight";
    }
    constexpr std::string_view kBlk = "encoder.block.";
    if (!starts_with(hf_name, kBlk)) return {};
    const auto dot = hf_name.find('.', kBlk.size());
    if (dot == std::string_view::npos) return {};
    const std::string_view idx  = hf_name.substr(kBlk.size(),
                                                 dot - kBlk.size());
    const std::string_view tail = hf_name.substr(dot + 1);

    auto blk = [&](std::string_view suffix) -> std::string {
        std::string out;
        out.reserve(8 + idx.size() + 1 + suffix.size());
        out.append("enc.blk.");
        out.append(idx);
        out.push_back('.');
        out.append(suffix);
        return out;
    };

    if (tail == "layer.0.layer_norm.weight")       return blk("attn_norm.weight");
    if (tail == "layer.0.SelfAttention.q.weight")  return blk("attn_q.weight");
    if (tail == "layer.0.SelfAttention.k.weight")  return blk("attn_k.weight");
    if (tail == "layer.0.SelfAttention.v.weight")  return blk("attn_v.weight");
    if (tail == "layer.0.SelfAttention.o.weight")  return blk("attn_o.weight");
    if (tail == "layer.0.SelfAttention.relative_attention_bias.weight")
        return blk("attn_rel_b.weight");
    if (tail == "layer.1.layer_norm.weight")       return blk("ffn_norm.weight");
    if (tail == "layer.1.DenseReluDense.wi_0.weight") return blk("ffn_gate.weight");
    if (tail == "layer.1.DenseReluDense.wi_1.weight") return blk("ffn_up.weight");
    if (tail == "layer.1.DenseReluDense.wo.weight")   return blk("ffn_down.weight");
    return {};
}

// ─── T5Config::from_gguf ───────────────────────────────────────────────────

namespace {

namespace gg = ::brotensor::gguf;

const gg::Value& need_meta(const gg::File& f, const char* key) {
    const gg::Value* v = f.find_meta(key);
    if (!v) throw std::runtime_error(
        std::string("t5::T5Config::from_gguf: missing metadata '") + key + "'");
    return *v;
}

int meta_as_int(const gg::Value& v, const char* key) {
    switch (v.type) {
        case gg::ValueType::U8:  return static_cast<int>(v.scalar.u8);
        case gg::ValueType::I8:  return static_cast<int>(v.scalar.i8);
        case gg::ValueType::U16: return static_cast<int>(v.scalar.u16);
        case gg::ValueType::I16: return static_cast<int>(v.scalar.i16);
        case gg::ValueType::U32: return static_cast<int>(v.scalar.u32);
        case gg::ValueType::I32: return static_cast<int>(v.scalar.i32);
        case gg::ValueType::U64: return static_cast<int>(v.scalar.u64);
        case gg::ValueType::I64: return static_cast<int>(v.scalar.i64);
        default:
            throw std::runtime_error(
                std::string("t5::T5Config::from_gguf: metadata '") + key +
                "' is not an integer scalar");
    }
}

float meta_as_f32(const gg::Value& v, const char* key) {
    if (v.type == gg::ValueType::F32) return v.scalar.f32;
    if (v.type == gg::ValueType::F64) return static_cast<float>(v.scalar.f64);
    throw std::runtime_error(
        std::string("t5::T5Config::from_gguf: metadata '") + key +
        "' is not a float scalar");
}

}  // namespace

T5Config T5Config::from_gguf(const gg::File& f) {
    if (const auto* arch = f.find_meta("general.architecture")) {
        if (arch->type == gg::ValueType::String && arch->str != "t5") {
            throw std::runtime_error(
                "t5::T5Config::from_gguf: general.architecture is '" +
                arch->str + "', expected 't5'");
        }
    }
    T5Config cfg;
    cfg.d_model    = meta_as_int(need_meta(f, "t5.embedding_length"),
                                 "t5.embedding_length");
    cfg.d_ff       = meta_as_int(need_meta(f, "t5.feed_forward_length"),
                                 "t5.feed_forward_length");
    cfg.num_layers = meta_as_int(need_meta(f, "t5.block_count"),
                                 "t5.block_count");
    cfg.num_heads  = meta_as_int(need_meta(f, "t5.attention.head_count"),
                                 "t5.attention.head_count");

    if (const auto* v = f.find_meta("t5.attention.key_length")) {
        cfg.d_kv = meta_as_int(*v, "t5.attention.key_length");
    } else {
        cfg.d_kv = cfg.d_model / cfg.num_heads;
    }
    if (const auto* v = f.find_meta("t5.attention.relative_buckets_count")) {
        cfg.relative_attention_num_buckets =
            meta_as_int(*v, "t5.attention.relative_buckets_count");
    }
    if (const auto* v = f.find_meta("t5.attention.layer_norm_epsilon")) {
        cfg.layer_norm_eps = meta_as_f32(*v, "t5.attention.layer_norm_epsilon");
    }
    if (const auto* toks = f.find_meta("tokenizer.ggml.tokens");
        toks && toks->type == gg::ValueType::Array) {
        cfg.vocab_size = static_cast<int>(toks->array.size());
    } else if (const auto* vs = f.find_meta("t5.vocab_size")) {
        cfg.vocab_size = meta_as_int(*vs, "t5.vocab_size");
    }
    return cfg;
}

// ─── load_weights ──────────────────────────────────────────────────────────

void TextEncoder::load_weights(const st::File& f, const std::string& prefix) {
    const std::vector<const st::File*> shards = {&f};
    load_weights(shards, prefix);
}

void TextEncoder::load_weights(const std::vector<const st::File*>& shards,
                               const std::string& prefix) {
    if (shards.empty()) fail("load_weights: no safetensors shards");
    brolm::detail::weights::SafetensorsSource src(shards, prefix);
    load_weights_impl_(src);
}

void TextEncoder::load_weights(const bt::gguf::File& f) {
    const std::vector<const bt::gguf::File*> shards = {&f};
    load_weights(shards);
}

void TextEncoder::load_weights(
    const std::vector<const bt::gguf::File*>& shards) {
    if (shards.empty()) fail("load_weights: no gguf shards");
    brolm::detail::weights::GgufSource src(
        shards, [](std::string_view hf) { return t5_hf_to_ggml(hf); });
    load_weights_impl_(src);
}

void TextEncoder::load_weights_impl_(
    const brolm::detail::weights::Source& src) {
    const int V  = cfg_.vocab_size;
    const int D  = cfg_.d_model;
    const int FF = cfg_.d_ff;
    const int NB = cfg_.relative_attention_num_buckets;
    const int H  = cfg_.num_heads;

    // Token embedding: HF naming is "shared.weight"; encoder-only exports
    // sometimes use "encoder.embed_tokens.weight" instead. Both map to
    // "token_embd.weight" in gguf, so either probe succeeds there.
    if (src.has("shared.weight")) {
        src.upload_compute_checked("shared.weight", V, D, token_embed_,
                                   "shared.weight");
    } else if (src.has("encoder.embed_tokens.weight")) {
        src.upload_compute_checked("encoder.embed_tokens.weight", V, D,
                                   token_embed_, "shared.weight");
    } else {
        fail("missing token embedding ('shared.weight' or "
             "'encoder.embed_tokens.weight')");
    }

    // INT8 (W8A16) quantisation decision. GPU-only.
    const bool do_quantize =
        cfg_.quantize_weights &&
        (brolm::compute_dtype() == bt::Dtype::FP16);
    if (cfg_.quantize_weights && !do_quantize) {
        std::fprintf(stderr,
            "brolm: INT8 weight quantization is GPU-only; ignoring "
            "T5Config::quantize_weights on the CPU backend.\n");
    }

    auto load_w = [&](const std::string& key, int out, int in,
                      bt::Tensor& W, QWeight& q, const std::string& name) {
        if (do_quantize) {
            quantize_weight_from_source_(src, key, out, in, q, name);
        } else {
            src.upload_compute_checked(key, out, in, W, name);
        }
    };

    for (int i = 0; i < cfg_.num_layers; ++i) {
        const std::string p = "encoder.block." + std::to_string(i) + ".";
        Block& B = blocks_[static_cast<std::size_t>(i)];

        src.upload_compute_checked(p + "layer.0.layer_norm.weight",
                                   D, 1, B.ln0, "block.layer.0.layer_norm");

        const std::string sa = p + "layer.0.SelfAttention.";
        load_w(sa + "q.weight", D, D, B.Wq, B.Wq_q, "SelfAttention.q");
        load_w(sa + "k.weight", D, D, B.Wk, B.Wk_q, "SelfAttention.k");
        load_w(sa + "v.weight", D, D, B.Wv, B.Wv_q, "SelfAttention.v");
        load_w(sa + "o.weight", D, D, B.Wo, B.Wo_q, "SelfAttention.o");

        src.upload_compute_checked(p + "layer.1.layer_norm.weight",
                                   D, 1, B.ln1, "block.layer.1.layer_norm");

        const std::string dr = p + "layer.1.DenseReluDense.";
        load_w(dr + "wi_0.weight", FF, D, B.wi_0, B.wi_0_q, "DenseReluDense.wi_0");
        load_w(dr + "wi_1.weight", FF, D, B.wi_1, B.wi_1_q, "DenseReluDense.wi_1");
        load_w(dr + "wo.weight",   D, FF, B.wo,  B.wo_q,  "DenseReluDense.wo");
    }

    // Relative-position bias table — block 0 only, shared by every layer.
    {
        bt::Tensor rel;
        src.upload_compute_checked(
            "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight",
            NB, H, rel, "relative_attention_bias");
        rel_attn_bias_ = download_fp32(rel);
    }

    src.upload_compute_checked("encoder.final_layer_norm.weight",
                               D, 1, final_ln_, "final_layer_norm");

    pos_bias_L_ = -1;
}

// ─── INT8 (W8A16) quantisation ─────────────────────────────────────────────

void TextEncoder::quantize_weight_from_source_(
    const brolm::detail::weights::Source& src, const std::string& name,
    int out, int in, QWeight& q, const std::string& label) {
    const std::size_t n =
        static_cast<std::size_t>(out) * static_cast<std::size_t>(in);
    std::vector<std::uint16_t> host_fp16;
    src.download_host_fp16(name, out, in, host_fp16, label);

    std::vector<std::int8_t> host_int8(n);
    std::vector<float>       host_scales(static_cast<std::size_t>(out));
    bt::quantize_int8_per_row_host(host_fp16.data(), out, in,
                                   host_int8.data(), host_scales.data());

    bt::Tensor cpu_int8 =
        bt::Tensor::empty_on(bt::Device::CPU, out, in, bt::Dtype::INT8);
    std::memcpy(cpu_int8.host_raw_mut(), host_int8.data(), n);
    q.W_int8 = cpu_int8.to(bt::default_device());
    q.scales = bt::Tensor::from_host_on(bt::default_device(),
                                        host_scales.data(), out, 1);
}

void TextEncoder::ffn_linear_(const bt::Tensor& W, const QWeight& q,
                              const bt::Tensor& X, bt::Tensor& Y) {
    if (q.active()) {
        bt::linear_forward_batched_int8w_fp16(q.W_int8, q.scales,
                                              /*bias=*/nullptr, X, Y);
    } else {
        detail::linear_batched(W, /*bias=*/nullptr, X, Y);
    }
}

// ─── relative-position bias ────────────────────────────────────────────────

void TextEncoder::rebuild_position_bias_(int L) {
    if (pos_bias_L_ == L && !pos_bias_.empty()) return;

    const int H  = cfg_.num_heads;
    const int NB = cfg_.relative_attention_num_buckets;
    const int MD = cfg_.relative_attention_max_distance;

    // bias[h*L + q][k] = rel_attn_bias[bucket(k - q)][h]
    std::vector<float> bias(static_cast<std::size_t>(H) * L * L);
    for (int q = 0; q < L; ++q) {
        for (int k = 0; k < L; ++k) {
            const int bucket = relative_position_bucket(k - q, NB, MD);
            for (int h = 0; h < H; ++h) {
                const std::size_t row =
                    static_cast<std::size_t>(h) * L + q;
                bias[row * L + k] =
                    rel_attn_bias_[static_cast<std::size_t>(bucket) * H + h];
            }
        }
    }

    // attn_bias is FP32 on every backend.
    bt::Tensor host = bt::Tensor::from_host(bias.data(), H * L, L);
    pos_bias_ = host.to(bt::default_device());
    pos_bias_L_ = L;
}

// ─── forward ───────────────────────────────────────────────────────────────

void TextEncoder::forward(const int32_t* ids, int L, bt::Tensor& out,
                          int pad_id) {
    if (!ids) fail("forward: ids pointer is null");
    if (L <= 0) fail("forward: L must be positive");
    if (token_embed_.size() == 0) fail("forward: weights not loaded");
    if (rel_attn_bias_.empty()) fail("forward: position bias table not loaded");

    const int H = cfg_.num_heads;

    // Optional padding mask. When pad_id >= 0, positions holding the pad
    // token are excluded from self-attention — HF passes the equivalent as
    // attention_mask. brotensor's d_mask is a length-L FP32 device buffer
    // (1 = attend, 0 = ignore) that gates both keys and padded query rows.
    const float* d_mask = nullptr;
    if (pad_id >= 0) {
        std::vector<float> mask(static_cast<std::size_t>(L));
        int valid = 0;
        for (int i = 0; i < L; ++i) {
            const bool keep = ids[i] != pad_id;
            mask[static_cast<std::size_t>(i)] = keep ? 1.0f : 0.0f;
            valid += keep ? 1 : 0;
        }
        if (valid == 0) fail("forward: every token equals pad_id");
        attn_mask_ =
            bt::Tensor::from_host(mask.data(), L, 1).to(bt::default_device());
        d_mask = static_cast<const float*>(attn_mask_.data);
    }

    // T5-XXL's residual stream grows monotonically across its 24 pre-norm
    // blocks and overflows FP16 (±65504) after roughly a dozen layers — the
    // well-known "T5 fp16" problem. Clamp the residual back into range after
    // each sub-layer (HuggingFace does exactly this). It is sufficient
    // because every sub-layer input is RMS-normed, hence scale-invariant: a
    // clamped residual still feeds an O(1) normalised input to the next
    // block. The CPU/FP32 path has no overflow and needs no clamp.
    const bool clamp_residual =
        (brolm::compute_dtype() == bt::Dtype::FP16);
    constexpr float kFp16Clamp = 64504.0f;   // finfo(fp16).max - 1000

    rebuild_position_bias_(L);

    // Embedding: x = embedding_lookup(shared, ids). No position embedding,
    // no embedding scaling.
    ids_dev_ = make_idx_device(ids, L);
    bt::embedding_lookup_forward(
        token_embed_, static_cast<const int32_t*>(ids_dev_.data), L, x_);
    // Own the residual stream — embedding output buffer is otherwise reused.
    x_ = x_.clone();

    for (auto& B : blocks_) {
        // ── self-attention sub-layer ──────────────────────────────────────
        bt::rms_norm_forward(x_, B.ln0, cfg_.layer_norm_eps, n_);
        if (B.Wq_q.active()) {
            bt::self_attention_bias_int8w_fp16(
                n_, B.Wq_q.W_int8, B.Wq_q.scales,
                B.Wk_q.W_int8, B.Wk_q.scales,
                B.Wv_q.W_int8, B.Wv_q.scales,
                B.Wo_q.W_int8, B.Wo_q.scales,
                d_mask, &pos_bias_, H, /*scale=*/1.0f, attn_);
        } else {
            bt::self_attention_bias_forward(
                n_, B.Wq, B.Wk, B.Wv, B.Wo,
                d_mask, &pos_bias_, H, /*scale=*/1.0f, attn_);
        }
        bt::add_inplace(x_, attn_);
        if (clamp_residual) bt::clamp(x_, -kFp16Clamp, kFp16Clamp);

        // ── FFN sub-layer (gated-gelu) ────────────────────────────────────
        bt::rms_norm_forward(x_, B.ln1, cfg_.layer_norm_eps, n_);
        ffn_linear_(B.wi_0, B.wi_0_q, n_, g_);
        bt::gelu_forward(g_, g_);                       // tanh-approx GELU
        ffn_linear_(B.wi_1, B.wi_1_q, n_, l_);
        bt::mul_inplace(g_, l_);                        // h = gelu(wi_0 n) * wi_1 n
        ffn_linear_(B.wo, B.wo_q, g_, ffn_out_);
        bt::add_inplace(x_, ffn_out_);
        if (clamp_residual) bt::clamp(x_, -kFp16Clamp, kFp16Clamp);
    }

    bt::rms_norm_forward(x_, final_ln_, cfg_.layer_norm_eps, out);
}

}  // namespace brolm::t5
