#include "brolm/qwen35_text.h"
#include "qwen35_detail.h"

#include "brolm/detail/compute.h"
#include "brolm/detail/device.h"
#include "brolm/detail/weights.h"
#include "brotensor/gguf.h"
#include "brotensor/ops.h"
#include "brotensor/runtime.h"
#include "brotensor/safetensors.h"
#include "brotensor/tensor.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace brolm::qwen35 {

namespace bt = ::brotensor;
namespace st = ::brotensor::safetensors;

using st::upload_compute_checked;

namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("qwen35::TextModel: " + msg);
}

// HF's Qwen3_5RMSNorm applies `(1 + weight)` as the gain (init zeros, centred
// on identity). brotensor's rms_norm_forward expects the raw gain, so we add
// 1.0 to every Qwen3_5RMSNorm weight at load time. Qwen3_5RMSNormGated (the
// linear-attn `linear_attn.norm`) uses plain `weight` (init ones) and is
// EXCLUDED from this transform. See HF transformers
// `Qwen3_5RMSNorm.forward` and the `_init_weights` comment "We initialize
// with 0s to be 1 centered as the RMSNorm here does (1 + weight)".
void add_one_to_norm_weight(bt::Tensor& t) {
    std::vector<float> h(static_cast<std::size_t>(t.size()));
    if (t.dtype == bt::Dtype::FP16) {
        std::vector<std::uint16_t> bits(h.size());
        t.copy_to_host_fp16(bits.data());
        for (std::size_t i = 0; i < h.size(); ++i)
            h[i] = bt::fp16_bits_to_fp32(bits[i]) + 1.0f;
    } else {
        h = t.to_host_vector();
        for (float& v : h) v += 1.0f;
    }
    const int r = t.rows;
    const int c = t.cols;
    const bt::Device dev = t.device;
    if (t.dtype == bt::Dtype::FP16) {
        std::vector<std::uint16_t> bits(h.size());
        for (std::size_t i = 0; i < bits.size(); ++i) {
            bits[i] = bt::fp32_to_fp16_bits(h[i]);
        }
        t = bt::Tensor::from_host_fp16_on(dev, bits.data(), r, c);
    } else {
        t = bt::Tensor::from_host_on(dev, h.data(), r, c);
    }
}

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

std::vector<int> rotary_row_perm(int rotary_dim, int d_t, int d_h, int d_w) {
    const int half = rotary_dim / 2;
    internal::MRopePairing P = internal::mrope_pairing(d_t, d_h, d_w);
    std::vector<int> hf_pair_for_brolm;
    hf_pair_for_brolm.reserve(static_cast<std::size_t>(half));
    hf_pair_for_brolm.insert(hf_pair_for_brolm.end(), P.t_pairs.begin(), P.t_pairs.end());
    hf_pair_for_brolm.insert(hf_pair_for_brolm.end(), P.h_pairs.begin(), P.h_pairs.end());
    hf_pair_for_brolm.insert(hf_pair_for_brolm.end(), P.w_pairs.begin(), P.w_pairs.end());
    std::vector<int> brolm_to_hf(static_cast<std::size_t>(rotary_dim));
    for (int b = 0; b < half; ++b) {
        const int hp = hf_pair_for_brolm[static_cast<std::size_t>(b)];
        brolm_to_hf[static_cast<std::size_t>(2 * b)] = hp;
        brolm_to_hf[static_cast<std::size_t>(2 * b + 1)] = hp + half;
    }
    return brolm_to_hf;
}

std::vector<float> permute_rotary_rows(const std::vector<float>& src,
                                       int num_heads, int head_dim,
                                       int rotary_dim, int cols,
                                       int d_t, int d_h, int d_w) {
    std::vector<int> brolm_to_hf = rotary_row_perm(rotary_dim, d_t, d_h, d_w);
    std::vector<float> dst(src.size());
    const std::size_t row_bytes = static_cast<std::size_t>(cols) * sizeof(float);
    for (int h = 0; h < num_heads; ++h) {
        const std::size_t base =
            static_cast<std::size_t>(h) * head_dim *
            static_cast<std::size_t>(cols);
        for (int b = 0; b < rotary_dim; ++b) {
            const std::size_t doff = base + static_cast<std::size_t>(b) * cols;
            const std::size_t soff = base +
                static_cast<std::size_t>(brolm_to_hf[static_cast<std::size_t>(b)]) * cols;
            std::memcpy(&dst[doff], &src[soff], row_bytes);
        }
        for (int r = rotary_dim; r < head_dim; ++r) {
            const std::size_t off = base + static_cast<std::size_t>(r) * cols;
            std::memcpy(&dst[off], &src[off], row_bytes);
        }
    }
    return dst;
}

void split_q_gate_rows(const std::vector<float>& src,
                       int n_q, int head_dim, int cols,
                       std::vector<float>& q_rows,
                       std::vector<float>& g_rows) {
    const std::size_t per_head_rows = static_cast<std::size_t>(2 * head_dim);
    const std::size_t row_bytes = static_cast<std::size_t>(cols) * sizeof(float);
    const std::size_t single_dim = static_cast<std::size_t>(n_q) *
                                   static_cast<std::size_t>(head_dim) *
                                   static_cast<std::size_t>(cols);
    q_rows.assign(single_dim, 0.0f);
    g_rows.assign(single_dim, 0.0f);
    for (int h = 0; h < n_q; ++h) {
        const std::size_t src_base = static_cast<std::size_t>(h) *
                                     per_head_rows *
                                     static_cast<std::size_t>(cols);
        const std::size_t dst_base = static_cast<std::size_t>(h) *
                                     static_cast<std::size_t>(head_dim) *
                                     static_cast<std::size_t>(cols);
        for (int r = 0; r < head_dim; ++r) {
            const std::size_t soff = src_base + static_cast<std::size_t>(r) * cols;
            const std::size_t goff = src_base + static_cast<std::size_t>(head_dim + r) * cols;
            const std::size_t doff = dst_base + static_cast<std::size_t>(r) * cols;
            std::memcpy(&q_rows[doff], &src[soff], row_bytes);
            std::memcpy(&g_rows[doff], &src[goff], row_bytes);
        }
    }
}

bool starts_with(std::string_view s, std::string_view p) {
    return s.size() >= p.size() && s.compare(0, p.size(), p) == 0;
}

}  // namespace

std::string qwen35_hf_to_ggml(std::string_view hf_name) {
    if (hf_name == "model.embed_tokens.weight" || hf_name == "embed_tokens.weight" ||
        hf_name == "model.language_model.embed_tokens.weight" || hf_name == "language_model.embed_tokens.weight") {
        return "token_embd.weight";
    }
    if (hf_name == "model.norm.weight" || hf_name == "norm.weight" ||
        hf_name == "model.language_model.norm.weight" || hf_name == "language_model.norm.weight") {
        return "output_norm.weight";
    }
    if (hf_name == "lm_head.weight" || hf_name == "model.lm_head.weight" ||
        hf_name == "model.language_model.lm_head.weight" || hf_name == "language_model.lm_head.weight") {
        return "output.weight";
    }

    auto match_layer = [&](std::string_view prefix) -> std::pair<std::string_view, std::string_view> {
        if (!starts_with(hf_name, prefix)) return {{}, {}};
        const auto dot = hf_name.find('.', prefix.size());
        if (dot == std::string_view::npos) return {{}, {}};
        const std::string_view idx = hf_name.substr(prefix.size(), dot - prefix.size());
        const std::string_view tail = hf_name.substr(dot + 1);
        return {idx, tail};
    };

    auto res = match_layer("model.layers.");
    if (res.first.empty()) res = match_layer("layers.");
    if (res.first.empty()) res = match_layer("model.language_model.layers.");
    if (res.first.empty()) res = match_layer("language_model.layers.");
    if (res.first.empty()) return {};

    const std::string_view idx = res.first;
    const std::string_view tail = res.second;

    auto blk = [&](std::string_view suffix) -> std::string {
        std::string out;
        out.reserve(4 + idx.size() + 1 + suffix.size());
        out.append("blk.");
        out.append(idx);
        out.push_back('.');
        out.append(suffix);
        return out;
    };

    if (tail == "input_layernorm.weight")          return blk("attn_norm.weight");
    if (tail == "post_attention_layernorm.weight") return blk("post_attention_norm.weight");
    if (tail == "mlp.gate_proj.weight")            return blk("ffn_gate.weight");
    if (tail == "mlp.up_proj.weight")              return blk("ffn_up.weight");
    if (tail == "mlp.down_proj.weight")            return blk("ffn_down.weight");

    // Full attention
    if (tail == "self_attn.q_proj.weight")         return blk("attn_q.weight");
    if (tail == "self_attn.k_proj.weight")         return blk("attn_k.weight");
    if (tail == "self_attn.v_proj.weight")         return blk("attn_v.weight");
    if (tail == "self_attn.o_proj.weight")         return blk("attn_output.weight");
    if (tail == "self_attn.q_norm.weight")         return blk("attn_q_norm.weight");
    if (tail == "self_attn.k_norm.weight")         return blk("attn_k_norm.weight");

    // Linear attention
    if (tail == "linear_attn.in_proj_qkv.weight")  return blk("attn_qkv.weight");
    if (tail == "linear_attn.in_proj_z.weight")    return blk("attn_gate.weight");
    if (tail == "linear_attn.in_proj_a.weight")    return blk("ssm_alpha.weight");
    if (tail == "linear_attn.in_proj_b.weight")    return blk("ssm_beta.weight");
    if (tail == "linear_attn.A_log")               return blk("ssm_a");
    if (tail == "linear_attn.conv1d.weight")       return blk("ssm_conv1d.weight");
    if (tail == "linear_attn.dt_bias")             return blk("ssm_dt.bias");
    if (tail == "linear_attn.norm.weight")         return blk("ssm_norm.weight");
    if (tail == "linear_attn.out_proj.weight")     return blk("ssm_out.weight");

    return {};
}

// ─── load_weights ──────────────────────────────────────────────────────────

void TextModel::load_weights(const st::File& f, const std::string& prefix) {
    const std::vector<const st::File*> shards = {&f};
    load_weights(shards, prefix);
}

void TextModel::load_weights(const std::vector<const st::File*>& shards,
                             const std::string& prefix) {
    if (shards.empty()) fail("load_weights: no safetensors shards");
    brolm::detail::weights::SafetensorsSource src(shards, prefix);
    load_weights_impl_(src);
}

void TextModel::load_weights(const brotensor::gguf::File& f) {
    const std::vector<const brotensor::gguf::File*> shards = {&f};
    load_weights(shards);
}

void TextModel::load_weights(const std::vector<const brotensor::gguf::File*>& shards) {
    if (shards.empty()) fail("load_weights: no gguf shards");
    brolm::detail::weights::GgufSource src(shards, [](std::string_view hf) {
        return qwen35_hf_to_ggml(hf);
    });
    load_weights_impl_(src);
}

void TextModel::load_weights_impl_(const brolm::detail::weights::Source& src) {
    const int V    = cfg_.vocab_size;
    const int H    = cfg_.hidden_size;
    const int Fm   = cfg_.intermediate_size;
    const int HD   = cfg_.head_dim;
    const int n_q  = cfg_.num_attention_heads;
    const int n_kv = cfg_.num_key_value_heads;
    const int q_dim    = n_q  * HD;
    const int kv_dim   = n_kv * HD;
    const int q_dim2   = 2 * q_dim;        // q_proj output width (q + gate)

    const int lin_h_v = cfg_.linear_num_value_heads;
    const int lin_h_k = cfg_.linear_num_key_heads;
    const int lin_d_v = cfg_.linear_value_head_dim;
    const int lin_d_k = cfg_.linear_key_head_dim;
    const int kdim    = lin_h_k * lin_d_k;
    const int vdim    = lin_h_v * lin_d_v;
    const int qkv_ch  = 2 * kdim + vdim;
    const int conv_kd = cfg_.linear_conv_kernel_dim;

    const bool is_gguf = src.is_gguf();

    {
        bt::DeviceScope scope(embed_device());
        src.upload_compute_dequant("embed_tokens.weight",
                                   V, H, embed_, "embed_tokens.weight");
    }

    for (int i = 0; i < cfg_.num_hidden_layers; ++i) {
        const bt::Device dev = layer_device(i);
        bt::DeviceScope scope(dev);
        const std::string p = "layers." + std::to_string(i) + ".";
        LayerSlot& L = layers_[static_cast<std::size_t>(i)];

        src.upload_compute_dequant(p + "input_layernorm.weight",
                                   H, 1, L.in_norm, "input_layernorm.weight");
        if (!is_gguf) add_one_to_norm_weight(L.in_norm);

        src.upload_compute_dequant(p + "post_attention_layernorm.weight",
                                   H, 1, L.post_attn_norm, "post_attention_layernorm.weight");
        if (!is_gguf) add_one_to_norm_weight(L.post_attn_norm);

        // MLP — same shape on every layer.
        src.upload_compute_checked(p + "mlp.gate_proj.weight",
                                   Fm, H, L.mlp.gate_W, "mlp.gate_proj.weight");
        src.upload_compute_checked(p + "mlp.up_proj.weight",
                                   Fm, H, L.mlp.up_W, "mlp.up_proj.weight");
        src.upload_compute_checked(p + "mlp.down_proj.weight",
                                   H, Fm, L.mlp.down_W, "mlp.down_proj.weight");

        // Fused gate_up_W: vertically stack gate_W and up_W for single-GEMV SwiGLU
        L.mlp.gate_up_W = bt::Tensor::empty_on(dev, 2 * Fm, H, L.mlp.gate_W.dtype);
        bt::copy_d2d(L.mlp.gate_W, 0, L.mlp.gate_up_W, 0, Fm * H);
        bt::copy_d2d(L.mlp.up_W,   0, L.mlp.gate_up_W, Fm * H, Fm * H);

        if (L.type == LayerType::Full) {
            // q_proj: (2*q_dim, hidden), per-head layout [q, gate]. Load raw,
            // split into Wq/Wg, then permute the rotary subrange of Wq's rows
            // (per-head, only the q half) and of Wk's rows for interleaved RoPE.
            bt::Tensor q_raw;
            src.upload_compute_dequant(p + "self_attn.q_proj.weight",
                                       q_dim2, H, q_raw, "self_attn.q_proj.weight");
            std::vector<float> q_host = download_fp32(q_raw);
            std::vector<float> q_rows, g_rows;
            split_q_gate_rows(q_host, n_q, HD, H, q_rows, g_rows);
            std::vector<float> q_perm =
                permute_rotary_rows(q_rows, n_q, HD, rotary_dim_, H,
                                    d_t_, d_h_, d_w_);
            L.full.Wq = brolm::detail::upload_host(q_perm.data(), q_dim, H);
            L.full.Wg = brolm::detail::upload_host(g_rows.data(), q_dim, H);

            bt::Tensor k_raw;
            src.upload_compute_dequant(p + "self_attn.k_proj.weight",
                                       kv_dim, H, k_raw, "self_attn.k_proj.weight");
            std::vector<float> k_host = download_fp32(k_raw);
            std::vector<float> k_perm =
                permute_rotary_rows(k_host, n_kv, HD, rotary_dim_, H,
                                    d_t_, d_h_, d_w_);
            L.full.Wk = brolm::detail::upload_host(k_perm.data(), kv_dim, H);

            src.upload_compute_checked(p + "self_attn.v_proj.weight",
                                       kv_dim, H, L.full.Wv, "self_attn.v_proj.weight");
            src.upload_compute_checked(p + "self_attn.o_proj.weight",
                                       H, q_dim, L.full.Wo, "self_attn.o_proj.weight");

            // Per-head norms: permute the rotary subrange [0, rotary_dim).
            bt::Tensor qn_raw, kn_raw;
            src.upload_compute_dequant(p + "self_attn.q_norm.weight",
                                       HD, 1, qn_raw, "self_attn.q_norm.weight");
            src.upload_compute_dequant(p + "self_attn.k_norm.weight",
                                       HD, 1, kn_raw, "self_attn.k_norm.weight");
            std::vector<float> qn_host = download_fp32(qn_raw);
            std::vector<float> kn_host = download_fp32(kn_raw);
            if (!is_gguf) {
                for (float& v : qn_host) v += 1.0f;
                for (float& v : kn_host) v += 1.0f;
            }
            std::vector<float> qn_perm =
                permute_rotary_rows(qn_host, /*num_heads=*/1, HD, rotary_dim_, 1,
                                    d_t_, d_h_, d_w_);
            std::vector<float> kn_perm =
                permute_rotary_rows(kn_host, /*num_heads=*/1, HD, rotary_dim_, 1,
                                    d_t_, d_h_, d_w_);
            L.full.q_norm = brolm::detail::upload_host(qn_perm.data(), HD, 1);
            L.full.k_norm = brolm::detail::upload_host(kn_perm.data(), HD, 1);
        } else {
            // Linear-attn layer (Gated DeltaNet).
            const std::string lp = p + "linear_attn.";
            auto load_fp32 = [&](const std::string& key, int rows, int cols, bt::Tensor& dst, const std::string& label) {
                bt::Tensor tmp;
                src.upload_compute_dequant(key, rows, cols, tmp, label);
                if (tmp.dtype == bt::Dtype::FP32 && tmp.device == dev) {
                    dst = std::move(tmp);
                } else {
                    std::vector<float> host = download_fp32(tmp);
                    dst = bt::Tensor::from_host_on(dev, host.data(), rows, cols);
                }
            };
            load_fp32(lp + "A_log",             lin_h_v, 1, L.lin.A_log, "A_log");
            load_fp32(lp + "conv1d.weight",     qkv_ch, conv_kd, L.lin.conv1d, "conv1d");
            load_fp32(lp + "dt_bias",           lin_h_v, 1, L.lin.dt_bias, "dt_bias");
            load_fp32(lp + "norm.weight",       lin_d_v, 1, L.lin.norm, "norm");

            src.upload_compute_checked(lp + "in_proj_qkv.weight", qkv_ch, H, L.lin.in_proj_qkv, "in_proj_qkv");
            src.upload_compute_checked(lp + "in_proj_z.weight",   vdim, H,   L.lin.in_proj_z,   "in_proj_z");
            src.upload_compute_checked(lp + "in_proj_a.weight",  lin_h_v, H, L.lin.in_proj_a,  "in_proj_a");
            src.upload_compute_checked(lp + "in_proj_b.weight",  lin_h_v, H, L.lin.in_proj_b,  "in_proj_b");
            src.upload_compute_checked(lp + "out_proj.weight",    H, vdim,   L.lin.out_proj,    "out_proj");

            // Fused in_proj_all: vertically stack in_proj_qkv, in_proj_z, in_proj_a, in_proj_b
            if (L.lin.in_proj_qkv.dtype == L.lin.in_proj_z.dtype &&
                L.lin.in_proj_qkv.dtype == L.lin.in_proj_a.dtype &&
                L.lin.in_proj_qkv.dtype == L.lin.in_proj_b.dtype) {
                const int total_in_rows = qkv_ch + vdim + 2 * lin_h_v;
                L.lin.in_proj_all = bt::Tensor::empty_on(dev, total_in_rows, H, L.lin.in_proj_qkv.dtype);
                bt::copy_d2d(L.lin.in_proj_qkv, 0, L.lin.in_proj_all, 0, qkv_ch * H);
                bt::copy_d2d(L.lin.in_proj_z,   0, L.lin.in_proj_all, qkv_ch * H, vdim * H);
                bt::copy_d2d(L.lin.in_proj_a,   0, L.lin.in_proj_all, (qkv_ch + vdim) * H, lin_h_v * H);
                bt::copy_d2d(L.lin.in_proj_b,   0, L.lin.in_proj_all, (qkv_ch + vdim + lin_h_v) * H, lin_h_v * H);
            }
        }
    }

    {
        bt::DeviceScope scope(final_device());
        src.upload_compute_dequant("norm.weight",
                                   H, 1, final_norm_, "norm.weight");
        if (!is_gguf) add_one_to_norm_weight(final_norm_);

        if (src.has("lm_head.weight")) {
            src.upload_compute_checked("lm_head.weight",
                                       V, H, lm_head_, "lm_head.weight");
        } else {
            if (!cfg_.tie_word_embeddings) {
                fail("tie_word_embeddings=false but lm_head.weight missing");
            }
            if (embed_.device == final_device()) {
                lm_head_ = embed_.clone();
            } else {
                lm_head_ = embed_.to(final_device());
            }
        }
    }
}

}  // namespace brolm::qwen35
