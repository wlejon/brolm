#include "brolm/qwen3vl_text.h"

#include "brolm/detail/compute.h"
#include "brolm/detail/device.h"
#include "brolm/detail/weights.h"
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
#include <vector>

namespace brolm::qwen3vl {

namespace bt = ::brotensor;
namespace st = ::brotensor::safetensors;

using st::upload_compute_checked;

namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("qwen3vl::TextModel: " + msg);
}

const st::TensorView& need(const std::vector<const st::File*>& shards,
                           const std::string& key) {
    for (const st::File* f : shards) {
        if (const auto* v = f->find(key)) return *v;
    }
    fail("missing tensor '" + key + "'");
}

const st::TensorView* find_in(const std::vector<const st::File*>& shards,
                              const std::string& key) {
    for (const st::File* f : shards) {
        if (const auto* v = f->find(key)) return v;
    }
    return nullptr;
}

bt::Tensor make_idx_device(const int32_t* host, int n) {
    bt::Tensor cpu = bt::Tensor::empty_on(bt::Device::CPU, n, 1, bt::Dtype::INT32);
    std::memcpy(cpu.host_raw_mut(), host,
                static_cast<std::size_t>(n) * sizeof(int32_t));
    return cpu.to(bt::default_device());
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

// ── M-RoPE axis pairing / row permutation ──────────────────────────────────
//
// Same scheme as qwen35_text.cpp's mrope_pairing / rotary_row_perm /
// permute_rotary_rows / build_axis_tables — see that file for the derivation
// against HF's apply_interleaved_mrope. The only difference here is that
// rotary_dim == head_dim (full rotation, no partial_rotary_factor), so the
// "pass-through tail" loop in permute_rotary_rows simply has an empty range.

struct MRopePairing {
    std::vector<int> t_pairs, h_pairs, w_pairs;
};

MRopePairing mrope_pairing(int d_t, int d_h, int d_w) {
    const int half = d_t + d_h + d_w;
    std::vector<char> owner(static_cast<std::size_t>(half), 'T');
    for (int k = 0; k < d_h; ++k) owner[static_cast<std::size_t>(1 + 3*k)] = 'H';
    for (int k = 0; k < d_w; ++k) owner[static_cast<std::size_t>(2 + 3*k)] = 'W';
    MRopePairing p;
    for (int j = 0; j < half; ++j) {
        switch (owner[static_cast<std::size_t>(j)]) {
            case 'T': p.t_pairs.push_back(j); break;
            case 'H': p.h_pairs.push_back(j); break;
            case 'W': p.w_pairs.push_back(j); break;
        }
    }
    return p;
}

std::vector<int> rotary_row_perm(int rotary_dim, int d_t, int d_h, int d_w) {
    const int half = rotary_dim / 2;
    MRopePairing P = mrope_pairing(d_t, d_h, d_w);
    std::vector<int> hf_pair_for_brolm;
    hf_pair_for_brolm.reserve(static_cast<std::size_t>(half));
    hf_pair_for_brolm.insert(hf_pair_for_brolm.end(), P.t_pairs.begin(), P.t_pairs.end());
    hf_pair_for_brolm.insert(hf_pair_for_brolm.end(), P.h_pairs.begin(), P.h_pairs.end());
    hf_pair_for_brolm.insert(hf_pair_for_brolm.end(), P.w_pairs.begin(), P.w_pairs.end());
    std::vector<int> brolm_to_hf(static_cast<std::size_t>(rotary_dim));
    for (int b = 0; b < half; ++b) {
        const int hp = hf_pair_for_brolm[static_cast<std::size_t>(b)];
        brolm_to_hf[static_cast<std::size_t>(2*b)]     = hp;
        brolm_to_hf[static_cast<std::size_t>(2*b + 1)] = hp + half;
    }
    return brolm_to_hf;
}

template <typename T>
std::vector<T> permute_rotary_rows(const std::vector<T>& src,
                                   int num_heads, int head_dim,
                                   int rotary_dim, int cols,
                                   int d_t, int d_h, int d_w) {
    std::vector<int> brolm_to_hf = rotary_row_perm(rotary_dim, d_t, d_h, d_w);
    std::vector<T> dst(src.size());
    const std::size_t row_bytes = static_cast<std::size_t>(cols) * sizeof(T);
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
        // Pass-through tail [rotary_dim, head_dim): empty here since
        // rotary_dim == head_dim for Qwen3-VL, but kept general.
        for (int r = rotary_dim; r < head_dim; ++r) {
            const std::size_t off = base + static_cast<std::size_t>(r) * cols;
            std::memcpy(&dst[off], &src[off], row_bytes);
        }
    }
    return dst;
}

void build_axis_tables(int max_pos_inclusive, int d_axis, int rotary_dim,
                       float rope_theta,
                       const std::vector<int>& freq_indices,
                       bt::Tensor& cos_t, bt::Tensor& sin_t) {
    const int rows = std::max(1, max_pos_inclusive + 1);
    if (d_axis <= 0) {
        cos_t = bt::Tensor::zeros_on(bt::default_device(), rows, 1, bt::Dtype::FP32);
        sin_t = bt::Tensor::zeros_on(bt::default_device(), rows, 1, bt::Dtype::FP32);
        return;
    }
    if (static_cast<int>(freq_indices.size()) != d_axis) {
        fail("build_axis_tables: freq_indices.size() != d_axis");
    }
    std::vector<float> cos_h(static_cast<std::size_t>(rows) * d_axis);
    std::vector<float> sin_h(static_cast<std::size_t>(rows) * d_axis);
    std::vector<float> inv_freq(static_cast<std::size_t>(d_axis));
    for (int i = 0; i < d_axis; ++i) {
        const int gj = freq_indices[static_cast<std::size_t>(i)];
        const float exp = 2.0f * static_cast<float>(gj) /
                          static_cast<float>(rotary_dim);
        inv_freq[static_cast<std::size_t>(i)] =
            1.0f / std::pow(rope_theta, exp);
    }
    for (int p = 0; p < rows; ++p) {
        for (int i = 0; i < d_axis; ++i) {
            const float angle = static_cast<float>(p) *
                                inv_freq[static_cast<std::size_t>(i)];
            cos_h[static_cast<std::size_t>(p) * d_axis + i] = std::cos(angle);
            sin_h[static_cast<std::size_t>(p) * d_axis + i] = std::sin(angle);
        }
    }
    cos_t = bt::Tensor::from_host_on(bt::default_device(), cos_h.data(), rows, d_axis);
    sin_t = bt::Tensor::from_host_on(bt::default_device(), sin_h.data(), rows, d_axis);
}

// Additive splice: h[row_start : row_start+feature.rows, :] += feature.
// Aliases the target rows of `h` via a writable view rather than a copy.
void add_inplace_rows(bt::Tensor& h, int row_start, const bt::Tensor& feature) {
    const int n      = feature.rows;
    const int hidden  = feature.cols;
    if (h.cols != hidden) {
        fail("add_inplace_rows: hidden mismatch " +
             std::to_string(h.cols) + " vs " + std::to_string(hidden));
    }
    if (row_start < 0 || row_start + n > h.rows) {
        fail("add_inplace_rows: row range out of bounds");
    }
    if (h.dtype != feature.dtype) {
        fail("add_inplace_rows: dtype mismatch");
    }
    const std::size_t byte_off =
        static_cast<std::size_t>(row_start) * static_cast<std::size_t>(hidden) *
        static_cast<std::size_t>(bt::dtype_size_bytes(h.dtype));
    bt::Tensor view = bt::Tensor::view(
        h.device, static_cast<char*>(h.data) + byte_off, n, hidden, h.dtype);
    bt::add_inplace(view, feature);
}

}  // namespace

// ─── ctor / dtor ───────────────────────────────────────────────────────────

TextModel::TextModel(const Qwen3VLConfig::Text& cfg) : cfg_(cfg) {
    if (cfg_.hidden_size <= 0 || cfg_.intermediate_size <= 0 ||
        cfg_.num_hidden_layers <= 0 || cfg_.vocab_size <= 0 ||
        cfg_.head_dim <= 0) {
        fail("config has non-positive dimension");
    }
    if (cfg_.num_attention_heads <= 0 || cfg_.num_key_value_heads <= 0) {
        fail("num_attention_heads / num_key_value_heads must be positive");
    }
    if (cfg_.num_attention_heads % cfg_.num_key_value_heads != 0) {
        fail("num_attention_heads must be a multiple of num_key_value_heads");
    }
    if (cfg_.head_dim % 2 != 0) {
        fail("head_dim must be even");
    }
    if (cfg_.rope.mrope_section.size() != 3) {
        fail("mrope_section must have 3 entries (t,h,w)");
    }
    d_t_ = cfg_.rope.mrope_section[0];
    d_h_ = cfg_.rope.mrope_section[1];
    d_w_ = cfg_.rope.mrope_section[2];
    if (2 * (d_t_ + d_h_ + d_w_) != cfg_.rotary_dim()) {
        fail("2*sum(mrope_section) != head_dim");
    }
    layers_.resize(static_cast<std::size_t>(cfg_.num_hidden_layers));
}

TextModel::~TextModel() = default;

// ─── load_weights ──────────────────────────────────────────────────────────

void TextModel::load_weights(const st::File& f, const std::string& prefix) {
    const std::vector<const st::File*> shards = {&f};
    load_weights_impl_(shards, prefix);
}

void TextModel::load_weights(const std::vector<const st::File*>& shards,
                             const std::string& prefix) {
    load_weights_impl_(shards, prefix);
}

void TextModel::load_weights_impl_(
    const std::vector<const st::File*>& shards, const std::string& prefix) {
    if (shards.empty()) fail("load_weights: no safetensors shards");

    const int V    = cfg_.vocab_size;
    const int H    = cfg_.hidden_size;
    const int Fm   = cfg_.intermediate_size;
    const int HD   = cfg_.head_dim;
    const int rd   = cfg_.rotary_dim();   // == HD (full rotation)
    const int n_q  = cfg_.num_attention_heads;
    const int n_kv = cfg_.num_key_value_heads;
    const int q_dim  = n_q  * HD;
    const int kv_dim = n_kv * HD;

    upload_compute_checked(need(shards, prefix + "embed_tokens.weight"),
                           V, H, embed_, "embed_tokens.weight");

    upload_compute_checked(need(shards, prefix + "norm.weight"),
                           H, 1, final_norm_, "norm.weight");

    // INT8 (W8A16) quantisation decision. GPU-only.
    const bool do_quantize =
        cfg_.quantize_weights &&
        (brolm::compute_dtype() == bt::Dtype::FP16);
    if (cfg_.quantize_weights && !do_quantize) {
        std::fprintf(stderr,
            "brolm: INT8 weight quantization is GPU-only; ignoring "
            "Qwen3VLConfig::Text::quantize_weights on the CPU backend.\n");
    }
    const detail::weights::SafetensorsSource wsrc(shards, prefix);

    // Quantise a host FP16 weight to INT8 with per-output-row symmetric
    // FP32 scales and upload only the INT8 bytes + scales.
    auto quantize_host = [](const std::vector<std::uint16_t>& host_fp16,
                            int out, int in, QWeight& q) {
        const std::size_t n =
            static_cast<std::size_t>(out) * static_cast<std::size_t>(in);
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
    };

    // Dense-or-quantised loader for the unpermuted per-layer linears
    // (v_proj/o_proj/mlp.*). In the quantised path the dense weight never
    // lands on the device.
    auto load_lin = [&](const std::string& name, int out, int in,
                        bt::Tensor& W, QWeight& q) {
        if (do_quantize) {
            std::vector<std::uint16_t> host_fp16;
            wsrc.download_host_fp16(name, out, in, host_fp16, name);
            quantize_host(host_fp16, out, in, q);
        } else {
            upload_compute_checked(need(shards, prefix + name), out, in, W,
                                   name);
        }
    };

    // q_proj/k_proj loader: permute the rotary subrange of rows for the HF
    // rotate_half -> brotensor interleaved-pair convention (rd == HD here so
    // every row is permuted — no pass-through tail), then upload dense or
    // quantise. The permutation reorders whole rows, so it commutes with the
    // per-row INT8 quantisation.
    auto load_qk = [&](const std::string& name, int out, int num_heads,
                       bt::Tensor& W, QWeight& q) {
        if (do_quantize) {
            std::vector<std::uint16_t> host_fp16;
            wsrc.download_host_fp16(name, out, H, host_fp16, name);
            std::vector<std::uint16_t> perm =
                permute_rotary_rows(host_fp16, num_heads, HD, rd, H,
                                   d_t_, d_h_, d_w_);
            quantize_host(perm, out, H, q);
        } else {
            bt::Tensor raw;
            upload_compute_checked(need(shards, prefix + name), out, H, raw,
                                   name);
            std::vector<float> perm =
                permute_rotary_rows(download_fp32(raw), num_heads, HD, rd, H,
                                   d_t_, d_h_, d_w_);
            W = brolm::detail::upload_host(perm.data(), out, H);
        }
    };

    for (int i = 0; i < cfg_.num_hidden_layers; ++i) {
        const std::string p = prefix + "layers." + std::to_string(i) + ".";
        const std::string pl = "layers." + std::to_string(i) + ".";
        LayerSlot& L = layers_[static_cast<std::size_t>(i)];

        upload_compute_checked(need(shards, p + "input_layernorm.weight"),
                               H, 1, L.in_norm, "input_layernorm.weight");
        upload_compute_checked(
            need(shards, p + "post_attention_layernorm.weight"),
            H, 1, L.post_attn_norm, "post_attention_layernorm.weight");

        load_lin(pl + "mlp.gate_proj.weight", Fm, H, L.gate_W, L.gate_q);
        load_lin(pl + "mlp.up_proj.weight",   Fm, H, L.up_W,   L.up_q);
        load_lin(pl + "mlp.down_proj.weight", H, Fm, L.down_W, L.down_q);

        load_qk(pl + "self_attn.q_proj.weight", q_dim,  n_q,  L.Wq, L.Wq_q);
        load_qk(pl + "self_attn.k_proj.weight", kv_dim, n_kv, L.Wk, L.Wk_q);

        load_lin(pl + "self_attn.v_proj.weight", kv_dim, H, L.Wv, L.Wv_q);
        load_lin(pl + "self_attn.o_proj.weight", H, q_dim, L.Wo, L.Wo_q);

        // Per-head norms: permute the (whole, since rd==HD) rotary range.
        bt::Tensor qn_raw, kn_raw;
        upload_compute_checked(need(shards, p + "self_attn.q_norm.weight"),
                               HD, 1, qn_raw, "self_attn.q_norm.weight");
        upload_compute_checked(need(shards, p + "self_attn.k_norm.weight"),
                               HD, 1, kn_raw, "self_attn.k_norm.weight");
        std::vector<float> qn_perm =
            permute_rotary_rows(download_fp32(qn_raw), /*num_heads=*/1, HD, rd, 1,
                               d_t_, d_h_, d_w_);
        std::vector<float> kn_perm =
            permute_rotary_rows(download_fp32(kn_raw), /*num_heads=*/1, HD, rd, 1,
                               d_t_, d_h_, d_w_);
        L.q_norm = brolm::detail::upload_host(qn_perm.data(), HD, 1);
        L.k_norm = brolm::detail::upload_host(kn_perm.data(), HD, 1);
    }

    if (!cfg_.tie_word_embeddings) {
        // The untied head (Qwen3-VL-8B) lives at the checkpoint ROOT
        // ("lm_head.weight"), not under the language-model prefix.
        const st::TensorView* head = find_in(shards, "lm_head.weight");
        if (head == nullptr) head = find_in(shards, prefix + "lm_head.weight");
        if (head == nullptr) {
            fail("tie_word_embeddings=false but lm_head.weight missing");
        }
        upload_compute_checked(*head, V, H, lm_head_, "lm_head.weight");
    }
}

// ─── cache ─────────────────────────────────────────────────────────────────

std::vector<LayerCache> TextModel::make_cache(int max_seq) const {
    if (max_seq <= 0) fail("make_cache: max_seq must be positive");
    const int n_kv = cfg_.num_key_value_heads;
    const int cache_cols = n_kv * cfg_.head_dim;
    const bt::Dtype dt = brolm::compute_dtype();
    const bt::Device dev = bt::default_device();

    std::vector<LayerCache> out(static_cast<std::size_t>(cfg_.num_hidden_layers));
    for (auto& c : out) {
        brolm::detail::resize_like(c.k, max_seq, cache_cols, dt, dev);
        brolm::detail::resize_like(c.v, max_seq, cache_cols, dt, dev);
        c.len = 0;
    }
    return out;
}

void TextModel::truncate_cache(std::vector<LayerCache>& cache, int len) const {
    if (len < 0) fail("truncate_cache: len must be non-negative");
    if (cache.size() != static_cast<std::size_t>(cfg_.num_hidden_layers)) {
        fail("truncate_cache: cache size mismatch");
    }
    for (const auto& c : cache) {
        if (len > c.len) {
            fail("truncate_cache: len " + std::to_string(len) +
                 " exceeds current cache len " + std::to_string(c.len));
        }
    }
    for (auto& c : cache) c.len = len;
}

// ─── full M-RoPE ────────────────────────────────────────────────────────────

void TextModel::prepare_mrope_(const std::vector<int32_t>& pos_t,
                               const std::vector<int32_t>& pos_h,
                               const std::vector<int32_t>& pos_w, int L) {
    int max_pos = 0;
    auto upd = [&](const std::vector<int32_t>& v) {
        for (int32_t p : v) if (p > max_pos) max_pos = p;
    };
    upd(pos_t); upd(pos_h); upd(pos_w);

    if (max_pos > mrope_tbl_max_pos_) {
        const int cap = std::max({max_pos, 2 * mrope_tbl_max_pos_, 1023});
        const int rd  = cfg_.rotary_dim();
        MRopePairing pairing = mrope_pairing(d_t_, d_h_, d_w_);
        build_axis_tables(cap, d_t_, rd, cfg_.rope.rope_theta,
                          pairing.t_pairs, mrope_cos_t_, mrope_sin_t_);
        build_axis_tables(cap, d_h_, rd, cfg_.rope.rope_theta,
                          pairing.h_pairs, mrope_cos_h_, mrope_sin_h_);
        build_axis_tables(cap, d_w_, rd, cfg_.rope.rope_theta,
                          pairing.w_pairs, mrope_cos_w_, mrope_sin_w_);
        mrope_tbl_max_pos_ = cap;
    }

    pos_t_dev_ = make_idx_device(pos_t.data(), L);
    pos_h_dev_ = make_idx_device(pos_h.data(), L);
    pos_w_dev_ = make_idx_device(pos_w.data(), L);
}

void TextModel::apply_mrope_(bt::Tensor& qk, int num_heads, int L) {
    const int HD = cfg_.head_dim;
    const int rd = cfg_.rotary_dim();   // == HD
    const int rot_cols = num_heads * rd;

    bt::Tensor& scratch = (num_heads == cfg_.num_attention_heads) ? q_rot_ : k_rot_;

    if (rd == HD) {
        // Full rotation: rope_apply_mrope can run directly on qk in place —
        // no subrange copy needed since the "rotary subrange" is everything.
        brolm::detail::resize_like(scratch, L, rot_cols, qk.dtype, qk.device);
        bt::copy_d2d(qk, 0, scratch, 0, L * rot_cols);
        bt::rope_apply_mrope(
            scratch,
            mrope_cos_t_, mrope_sin_t_,
            mrope_cos_h_, mrope_sin_h_,
            mrope_cos_w_, mrope_sin_w_,
            static_cast<const int32_t*>(pos_t_dev_.data),
            static_cast<const int32_t*>(pos_h_dev_.data),
            static_cast<const int32_t*>(pos_w_dev_.data),
            /*head_dim=*/rd,
            /*num_heads=*/num_heads,
            /*d_t=*/d_t_, /*d_h=*/d_h_, /*d_w=*/d_w_,
            scratch);
        bt::copy_d2d(scratch, 0, qk, 0, L * rot_cols);
        return;
    }

    // General (partial-rotary) path — not exercised while rd == HD, kept for
    // completeness in case a future Qwen3-VL release adds partial rotation.
    brolm::detail::resize_like(scratch, L, rot_cols, qk.dtype, qk.device);
    bt::copy_d2d_strided(qk, 0, HD, scratch, 0, rd, rd, L * num_heads);
    bt::rope_apply_mrope(
        scratch,
        mrope_cos_t_, mrope_sin_t_,
        mrope_cos_h_, mrope_sin_h_,
        mrope_cos_w_, mrope_sin_w_,
        static_cast<const int32_t*>(pos_t_dev_.data),
        static_cast<const int32_t*>(pos_h_dev_.data),
        static_cast<const int32_t*>(pos_w_dev_.data),
        /*head_dim=*/rd,
        /*num_heads=*/num_heads,
        /*d_t=*/d_t_, /*d_h=*/d_h_, /*d_w=*/d_w_,
        scratch);
    bt::copy_d2d_strided(scratch, 0, rd, qk, 0, HD, rd, L * num_heads);
}

// ─── linear dispatch ───────────────────────────────────────────────────────

void TextModel::linear_(const bt::Tensor& W, const QWeight& q,
                        const bt::Tensor& X, bt::Tensor& Y) {
    if (q.active()) {
        bt::linear_forward_batched_int8w_fp16(q.W_int8, q.scales,
                                              /*bias=*/nullptr, X, Y);
    } else {
        detail::linear_batched(W, /*bias=*/nullptr, X, Y);
    }
}

// ─── MLP block ─────────────────────────────────────────────────────────────

void TextModel::mlp_block_(const LayerSlot& layer, int L) {
    (void)L;
    linear_(layer.gate_W, layer.gate_q, norm_, mlp_gate_);
    linear_(layer.up_W,   layer.up_q,   norm_, mlp_up_);

    bt::silu_forward(mlp_gate_, mlp_gate_);
    bt::mul_inplace(mlp_gate_, mlp_up_);
    linear_(layer.down_W, layer.down_q, mlp_gate_, proj_);
    bt::add_inplace(h_, proj_);
}

// ─── forward ───────────────────────────────────────────────────────────────

bt::Tensor TextModel::embed_tokens(const std::vector<int>& token_ids) const {
    if (token_ids.empty()) fail("embed_tokens: token_ids empty");
    if (embed_.size() == 0) fail("embed_tokens: weights not loaded");
    const int L = static_cast<int>(token_ids.size());
    std::vector<int32_t> ids32(L);
    for (int i = 0; i < L; ++i) {
        ids32[static_cast<std::size_t>(i)] =
            static_cast<int32_t>(token_ids[static_cast<std::size_t>(i)]);
    }
    bt::Tensor ids_dev = make_idx_device(ids32.data(), L);
    bt::Tensor out;
    bt::embedding_lookup_forward(
        embed_, static_cast<const int32_t*>(ids_dev.data), L, out);
    return out.clone();
}

void TextModel::forward(const std::vector<int>& token_ids,
                        const std::vector<int64_t>& mrope_t,
                        const std::vector<int64_t>& mrope_h,
                        const std::vector<int64_t>& mrope_w,
                        std::vector<LayerCache>& cache,
                        bt::Tensor& logits_out) {
    bt::Tensor embeds = embed_tokens(token_ids);
    forward_embeds(embeds, mrope_t, mrope_h, mrope_w, cache, logits_out);
}

void TextModel::forward_embeds(const bt::Tensor& embeds,
                               const std::vector<int64_t>& mrope_t,
                               const std::vector<int64_t>& mrope_h,
                               const std::vector<int64_t>& mrope_w,
                               std::vector<LayerCache>& cache,
                               bt::Tensor& logits_out,
                               const std::vector<DeepstackSplice>& deepstack) {
    if (embeds.rows <= 0) fail("forward_embeds: empty input");
    const int L = embeds.rows;
    if (embeds.cols != cfg_.hidden_size) {
        fail("forward_embeds: embeds.cols != hidden_size");
    }
    if (static_cast<int>(mrope_t.size()) != L ||
        static_cast<int>(mrope_h.size()) != L ||
        static_cast<int>(mrope_w.size()) != L) {
        fail("forward_embeds: mrope_{t,h,w} length must match embeds.rows");
    }
    if (cache.size() != static_cast<std::size_t>(cfg_.num_hidden_layers)) {
        fail("forward_embeds: cache size mismatch");
    }
    if (embed_.size() == 0) fail("forward_embeds: weights not loaded");

    const float eps = cfg_.rms_norm_eps;

    std::vector<int32_t> pos_t(L), pos_h(L), pos_w(L);
    for (int i = 0; i < L; ++i) {
        pos_t[static_cast<std::size_t>(i)] = static_cast<int32_t>(mrope_t[static_cast<std::size_t>(i)]);
        pos_h[static_cast<std::size_t>(i)] = static_cast<int32_t>(mrope_h[static_cast<std::size_t>(i)]);
        pos_w[static_cast<std::size_t>(i)] = static_cast<int32_t>(mrope_w[static_cast<std::size_t>(i)]);
    }
    prepare_mrope_(pos_t, pos_h, pos_w, L);

    h_ = embeds.clone();
    run_decoder_layers_(L, cache, deepstack, /*capture_layers=*/nullptr,
                       /*hidden_states_out=*/nullptr);

    bt::rms_norm_forward(h_, final_norm_, eps, norm_);
    detail::linear_batched(cfg_.tie_word_embeddings ? embed_ : lm_head_,
                           /*bias=*/nullptr, norm_, logits_out);
}

void TextModel::run_decoder_layers_(
    int L, std::vector<LayerCache>& cache,
    const std::vector<DeepstackSplice>& deepstack,
    const std::vector<int>* capture_layers,
    std::vector<bt::Tensor>* hidden_states_out) {
    const int HD   = cfg_.head_dim;
    const int n_q  = cfg_.num_attention_heads;
    const int n_kv = cfg_.num_key_value_heads;
    const float eps = cfg_.rms_norm_eps;

    std::size_t capture_cursor = 0;

    for (int li = 0; li < cfg_.num_hidden_layers; ++li) {
        LayerSlot& layer = layers_[static_cast<std::size_t>(li)];
        LayerCache& c    = cache[static_cast<std::size_t>(li)];
        if (c.k.size() == 0) fail("run_decoder_layers_: cache not allocated");
        const int max_seq = c.k.rows;
        if (c.len + L > max_seq) {
            fail("run_decoder_layers_: cache_len + L exceeds allocated capacity");
        }

        bt::rms_norm_forward(h_, layer.in_norm, eps, norm_);

        linear_(layer.Wq, layer.Wq_q, norm_, q_);
        linear_(layer.Wk, layer.Wk_q, norm_, k_);
        linear_(layer.Wv, layer.Wv_q, norm_, v_);

        auto headnorm = [&](const bt::Tensor& src, int num_heads,
                            const bt::Tensor& gain, bt::Tensor& dst) {
            const int rows = src.rows;
            bt::Tensor src_v = bt::Tensor::view(
                src.device, src.data, rows * num_heads, HD, src.dtype);
            bt::rms_norm_forward(src_v, gain, eps, dst);
            dst.rows = rows;
            dst.cols = num_heads * HD;
        };
        headnorm(q_, n_q,  layer.q_norm, qn_);
        headnorm(k_, n_kv, layer.k_norm, kn_);

        apply_mrope_(qn_, n_q,  L);
        apply_mrope_(kn_, n_kv, L);

        bt::kv_cache_append(kn_, v_, c.len, c.k, c.v);
        bt::flash_attention_decode(qn_, c.k, c.v, c.len + L, n_q, n_kv, attn_);
        c.len += L;

        linear_(layer.Wo, layer.Wo_q, attn_, proj_);
        bt::add_inplace(h_, proj_);

        // DeepStack injection: this decoder layer receives image i's feature
        // if it's within that image's per-layer list.
        for (const DeepstackSplice& d : deepstack) {
            if (static_cast<std::size_t>(li) < d.per_layer.size()) {
                add_inplace_rows(h_, d.row_start, d.per_layer[static_cast<std::size_t>(li)]);
            }
        }

        bt::rms_norm_forward(h_, layer.post_attn_norm, eps, norm_);
        mlp_block_(layer, L);

        if (capture_layers != nullptr &&
            capture_cursor < capture_layers->size() &&
            (*capture_layers)[capture_cursor] == li + 1) {
            hidden_states_out->push_back(h_.clone());
            ++capture_cursor;
        }
    }
}

void TextModel::forward_capture_hidden_states(
    const bt::Tensor& embeds,
    const std::vector<int64_t>& mrope_t,
    const std::vector<int64_t>& mrope_h,
    const std::vector<int64_t>& mrope_w,
    const std::vector<int>& capture_layers,
    std::vector<bt::Tensor>& hidden_states_out,
    const std::vector<DeepstackSplice>& deepstack) {
    if (embeds.rows <= 0) fail("forward_capture_hidden_states: empty input");
    const int L = embeds.rows;
    if (embeds.cols != cfg_.hidden_size) {
        fail("forward_capture_hidden_states: embeds.cols != hidden_size");
    }
    if (static_cast<int>(mrope_t.size()) != L ||
        static_cast<int>(mrope_h.size()) != L ||
        static_cast<int>(mrope_w.size()) != L) {
        fail("forward_capture_hidden_states: mrope_{t,h,w} length must match "
             "embeds.rows");
    }
    if (embed_.size() == 0) fail("forward_capture_hidden_states: weights not loaded");
    if (capture_layers.empty()) {
        fail("forward_capture_hidden_states: capture_layers must be non-empty");
    }
    for (std::size_t i = 0; i < capture_layers.size(); ++i) {
        const int idx = capture_layers[i];
        if (idx < 1 || idx > cfg_.num_hidden_layers) {
            fail("forward_capture_hidden_states: capture_layers entry " +
                 std::to_string(idx) + " out of range [1, " +
                 std::to_string(cfg_.num_hidden_layers) + "]");
        }
        if (i > 0 && capture_layers[i] <= capture_layers[i - 1]) {
            fail("forward_capture_hidden_states: capture_layers must be "
                 "strictly ascending");
        }
    }

    std::vector<int32_t> pos_t(L), pos_h(L), pos_w(L);
    for (int i = 0; i < L; ++i) {
        pos_t[static_cast<std::size_t>(i)] = static_cast<int32_t>(mrope_t[static_cast<std::size_t>(i)]);
        pos_h[static_cast<std::size_t>(i)] = static_cast<int32_t>(mrope_h[static_cast<std::size_t>(i)]);
        pos_w[static_cast<std::size_t>(i)] = static_cast<int32_t>(mrope_w[static_cast<std::size_t>(i)]);
    }
    prepare_mrope_(pos_t, pos_h, pos_w, L);

    h_ = embeds.clone();
    std::vector<LayerCache> scratch_cache = make_cache(L);

    hidden_states_out.clear();
    hidden_states_out.reserve(capture_layers.size());
    run_decoder_layers_(L, scratch_cache, deepstack, &capture_layers,
                       &hidden_states_out);
}

}  // namespace brolm::qwen3vl
