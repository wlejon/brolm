#include "brolm/laya.h"

#include "brolm/detail/compute.h"
#include "brolm/detail/device.h"
#include "brolm/detail/weights.h"

#include "brotensor/ops.h"
#include "brotensor/runtime.h"
#include "brotensor/safetensors.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace brolm::laya {

namespace bt = ::brotensor;

namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("laya::DecisionModel: " + msg);
}

bt::Tensor make_idx_device(const int32_t* host, int n) {
    bt::Tensor cpu = bt::Tensor::empty_on(bt::Device::CPU, n, 1, bt::Dtype::INT32);
    std::memcpy(cpu.host_raw_mut(), host,
                static_cast<std::size_t>(n) * sizeof(int32_t));
    return cpu.to(bt::default_device());
}

}  // namespace

DecisionModel::DecisionModel() {
    head_layers_.resize(static_cast<std::size_t>(cfg_.head_layers));
}

DecisionModel::~DecisionModel() = default;

void DecisionModel::load_model(const std::string& model_dir) {
    cfg_ = Config::load(model_dir + "/rl_agent_config.json");
    modernbert::Config enc_cfg =
        modernbert::Config::load(model_dir + "/encoder/config.json");
    encoder_ = modernbert::ModernBertModel(std::move(enc_cfg));
    tokenizer_ = LayaTokenizer::load(model_dir + "/tokenizer/tokenizer.json");
    load_safetensors(model_dir + "/model.safetensors");
}

void DecisionModel::load_safetensors(const std::string& safetensors_path) {
    bt::safetensors::File st = bt::safetensors::File::open(safetensors_path);
    brolm::detail::weights::SafetensorsSource src({&st});

    encoder_.load_weights(src, "encoder.");

    const int D = encoder_.config().hidden_size;

    src.upload_compute_checked("type_emb.weight", 3, D, type_emb_, "type_emb.weight");

    head_layers_.resize(static_cast<std::size_t>(cfg_.head_layers));
    for (int i = 0; i < cfg_.head_layers; ++i) {
        const std::string p = "head.layers." + std::to_string(i) + ".";
        TransformerHeadLayer& L = head_layers_[static_cast<std::size_t>(i)];

        src.upload_compute_checked(p + "norm1.weight", D, 1, L.norm1_g, "norm1.weight");
        src.upload_compute_checked(p + "norm1.bias",   D, 1, L.norm1_b, "norm1.bias");

        src.upload_compute_checked(p + "self_attn.in_proj_weight", 3 * D, D,
                                   L.in_proj_W, "in_proj_weight");
        src.upload_compute_checked(p + "self_attn.in_proj_bias",   3 * D, 1,
                                   L.in_proj_b, "in_proj_bias");

        src.upload_compute_checked(p + "self_attn.out_proj.weight", D, D,
                                   L.out_proj_W, "out_proj.weight");
        src.upload_compute_checked(p + "self_attn.out_proj.bias",   D, 1,
                                   L.out_proj_b, "out_proj.bias");

        src.upload_compute_checked(p + "norm2.weight", D, 1, L.norm2_g, "norm2.weight");
        src.upload_compute_checked(p + "norm2.bias",   D, 1, L.norm2_b, "norm2.bias");

        src.upload_compute_checked(p + "linear1.weight", 4 * D, D,
                                   L.linear1_W, "linear1.weight");
        src.upload_compute_checked(p + "linear1.bias",   4 * D, 1,
                                   L.linear1_b, "linear1.bias");

        src.upload_compute_checked(p + "linear2.weight", D, 4 * D,
                                   L.linear2_W, "linear2.weight");
        src.upload_compute_checked(p + "linear2.bias",   D, 1,
                                   L.linear2_b, "linear2.bias");
    }

    src.upload_compute_checked("scorer.0.weight", D, 1, scorer_ln_g_, "scorer.0.weight");
    src.upload_compute_checked("scorer.0.bias",   D, 1, scorer_ln_b_, "scorer.0.bias");
    src.upload_compute_checked("scorer.1.weight", D, D, scorer_l1_W_, "scorer.1.weight");
    src.upload_compute_checked("scorer.1.bias",   D, 1, scorer_l1_b_, "scorer.1.bias");
    src.upload_compute_checked("scorer.3.weight", 1, D, scorer_l2_W_, "scorer.3.weight");
    src.upload_compute_checked("scorer.3.bias",   1, 1, scorer_l2_b_, "scorer.3.bias");

    src.upload_compute_checked("act_head.0.weight", 256, D + 4, act_l1_W_, "act_head.0.weight");
    src.upload_compute_checked("act_head.0.bias",   256, 1,     act_l1_b_, "act_head.0.bias");
    src.upload_compute_checked("act_head.2.weight", 2,   256,   act_l2_W_, "act_head.2.weight");
    src.upload_compute_checked("act_head.2.bias",   2,   1,     act_l2_b_, "act_head.2.bias");

    if (src.has("temperature")) {
        std::vector<std::uint16_t> fp16_buf;
        try {
            src.download_host_fp16("temperature", 3, 1, fp16_buf, "temperature");
            if (fp16_buf.size() >= 3) {
                cfg_.temperature.resize(3);
                for (std::size_t k = 0; k < 3; ++k) {
                    cfg_.temperature[k] = bt::fp16_bits_to_fp32(fp16_buf[k]);
                }
            }
        } catch (...) {
            // Temperature was already read from config.json if present
        }
    }
}

void DecisionModel::init_synthetic(const modernbert::Config& enc_cfg,
                                   const Config& laya_cfg) {
    cfg_ = laya_cfg;
    encoder_ = modernbert::ModernBertModel(enc_cfg);
    encoder_.init_synthetic();

    const int D = enc_cfg.hidden_size;

    std::vector<float> ones_D(D, 1.0f);
    std::vector<float> zeros_D(D, 0.0f);
    std::vector<float> small_3D(static_cast<std::size_t>(3) * D, 0.01f);
    std::vector<float> ones_3D(static_cast<std::size_t>(3) * D, 1.0f);
    std::vector<float> small_3DD(static_cast<std::size_t>(3 * D) * D, 0.01f);
    std::vector<float> small_DD(static_cast<std::size_t>(D) * D, 0.01f);
    std::vector<float> ones_4D(static_cast<std::size_t>(4) * D, 1.0f);
    std::vector<float> zeros_4D(static_cast<std::size_t>(4) * D, 0.0f);
    std::vector<float> small_4DD(static_cast<std::size_t>(4 * D) * D, 0.01f);
    std::vector<float> small_D4D(static_cast<std::size_t>(D) * (4 * D), 0.01f);
    std::vector<float> small_1D(D, 0.01f);

    type_emb_ = brolm::detail::upload_host(small_3D.data(), 3, D);

    head_layers_.resize(static_cast<std::size_t>(cfg_.head_layers));
    for (int i = 0; i < cfg_.head_layers; ++i) {
        TransformerHeadLayer& L = head_layers_[static_cast<std::size_t>(i)];
        L.norm1_g    = brolm::detail::upload_host(ones_D.data(), D, 1);
        L.norm1_b    = brolm::detail::upload_host(zeros_D.data(), D, 1);
        L.in_proj_W  = brolm::detail::upload_host(small_3DD.data(), 3 * D, D);
        L.in_proj_b  = brolm::detail::upload_host(small_3D.data(), 3 * D, 1);
        L.out_proj_W = brolm::detail::upload_host(small_DD.data(), D, D);
        L.out_proj_b = brolm::detail::upload_host(zeros_D.data(), D, 1);
        L.norm2_g    = brolm::detail::upload_host(ones_D.data(), D, 1);
        L.norm2_b    = brolm::detail::upload_host(zeros_D.data(), D, 1);
        L.linear1_W  = brolm::detail::upload_host(small_4DD.data(), 4 * D, D);
        L.linear1_b  = brolm::detail::upload_host(zeros_4D.data(), 4 * D, 1);
        L.linear2_W  = brolm::detail::upload_host(small_D4D.data(), D, 4 * D);
        L.linear2_b  = brolm::detail::upload_host(zeros_D.data(), D, 1);
    }

    scorer_ln_g_ = brolm::detail::upload_host(ones_D.data(), D, 1);
    scorer_ln_b_ = brolm::detail::upload_host(zeros_D.data(), D, 1);
    scorer_l1_W_ = brolm::detail::upload_host(small_DD.data(), D, D);
    scorer_l1_b_ = brolm::detail::upload_host(zeros_D.data(), D, 1);
    scorer_l2_W_ = brolm::detail::upload_host(small_1D.data(), 1, D);
    float zero_scalar = 0.0f;
    scorer_l2_b_ = brolm::detail::upload_host(&zero_scalar, 1, 1);

    std::vector<float> small_act0(static_cast<std::size_t>(256) * (D + 4), 0.01f);
    std::vector<float> zeros_256(256, 0.0f);
    std::vector<float> small_act2(static_cast<std::size_t>(2) * 256, 0.01f);
    std::vector<float> zeros_2(2, 0.0f);
    act_l1_W_ = brolm::detail::upload_host(small_act0.data(), 256, D + 4);
    act_l1_b_ = brolm::detail::upload_host(zeros_256.data(), 256, 1);
    act_l2_W_ = brolm::detail::upload_host(small_act2.data(), 2, 256);
    act_l2_b_ = brolm::detail::upload_host(zeros_2.data(), 2, 1);
}

LayaAnswer DecisionModel::forward_question(const LayaQuestion& q,
                                          const std::vector<int32_t>& input_ids,
                                          const std::vector<int32_t>& marker_pos) {
    if (input_ids.empty()) fail("forward_question: input_ids is empty");
    if (marker_pos.empty()) fail("forward_question: marker_pos is empty");

    const int seq_len = static_cast<int>(input_ids.size());
    const int K = static_cast<int>(marker_pos.size());
    const int D = encoder_.config().hidden_size;
    const int H = encoder_.config().num_attention_heads;
    const bt::Device dev = bt::default_device();
    const bt::Dtype dt = brolm::compute_dtype();

    // 1. ModernBERT encoder
    encoder_.forward(input_ids.data(), seq_len, h_);

    // 2. Add question type embedding across all rows
    const int qtype = q.qtype_index();
    bt::Tensor type_row;
    brolm::detail::resize_like(type_row, 1, D, dt, dev);
    bt::copy_d2d(type_emb_, qtype * D, type_row, 0, D);
    bt::add_row_bias_inplace(h_, type_row);

    // 3. Head layers (Pre-LN, MHA with biases, FFN with biases)
    for (std::size_t i = 0; i < head_layers_.size(); ++i) {
        TransformerHeadLayer& L = head_layers_[i];

        brolm::detail::layernorm_batched(h_, L.norm1_g, L.norm1_b, h_norm_, 1e-5f);

        brolm::detail::linear_batched(L.in_proj_W, &L.in_proj_b, h_norm_, qkv_);

        brolm::detail::resize_like(q_, seq_len, D, dt, dev);
        brolm::detail::resize_like(k_, seq_len, D, dt, dev);
        brolm::detail::resize_like(v_, seq_len, D, dt, dev);
        bt::copy_d2d_strided(qkv_, 0,     3 * D, q_, 0, D, D, seq_len);
        bt::copy_d2d_strided(qkv_, D,     3 * D, k_, 0, D, D, seq_len);
        bt::copy_d2d_strided(qkv_, 2 * D, 3 * D, v_, 0, D, D, seq_len);

        bt::flash_attention_gqa_forward(q_, k_, v_, nullptr, H, H, /*causal=*/false, attn_out_);

        brolm::detail::linear_batched(L.out_proj_W, &L.out_proj_b, attn_out_, proj_out_);
        bt::add_inplace(h_, proj_out_);

        brolm::detail::layernorm_batched(h_, L.norm2_g, L.norm2_b, h_norm_, 1e-5f);
        brolm::detail::linear_batched(L.linear1_W, &L.linear1_b, h_norm_, ffn1_);
        bt::relu_forward(ffn1_, ffn1_);
        brolm::detail::linear_batched(L.linear2_W, &L.linear2_b, ffn1_, ffn2_);
        bt::add_inplace(h_, ffn2_);
    }

    // 4. Gather marker rows
    bt::Tensor idx_dev = make_idx_device(marker_pos.data(), K);
    bt::gather_rows(h_, idx_dev, m_);

    // 5. Scorer: LayerNorm -> Linear -> GELU -> Linear -> raw logits (K, 1)
    brolm::detail::layernorm_batched(m_, scorer_ln_g_, scorer_ln_b_, scorer_out0_, 1e-5f);
    brolm::detail::linear_batched(scorer_l1_W_, &scorer_l1_b_, scorer_out0_, scorer_out1_);
    bt::gelu_exact_forward(scorer_out1_, scorer_out2_);
    brolm::detail::linear_batched(scorer_l2_W_, &scorer_l2_b_, scorer_out2_, scorer_out3_);

    std::vector<float> logits = brolm::detail::weights::detail_::download_fp32(scorer_out3_);

    // 6. Act Head: pooled = h[0], feats from uncalibrated softmax distribution
    bt::Tensor row0;
    brolm::detail::resize_like(row0, 1, D, dt, dev);
    bt::copy_d2d(h_, 0, row0, 0, D);
    std::vector<float> pooled = brolm::detail::weights::detail_::download_fp32(row0);

    float max_l = -1e30f;
    for (float v_val : logits) {
        if (v_val > max_l) max_l = v_val;
    }
    float sum_exp = 0.0f;
    std::vector<float> uncal_p(static_cast<std::size_t>(K));
    for (int i = 0; i < K; ++i) {
        uncal_p[static_cast<std::size_t>(i)] = std::exp(logits[static_cast<std::size_t>(i)] - max_l);
        sum_exp += uncal_p[static_cast<std::size_t>(i)];
    }
    for (int i = 0; i < K; ++i) {
        uncal_p[static_cast<std::size_t>(i)] /= sum_exp;
    }

    const float k_float = std::max(2.0f, static_cast<float>(K));
    float ent = 0.0f;
    for (float pv : uncal_p) {
        ent -= pv * std::log(std::max(pv, 1e-9f));
    }
    ent /= std::log(k_float);

    std::vector<float> sorted_p = uncal_p;
    std::sort(sorted_p.begin(), sorted_p.end(), std::greater<float>());
    const float top1 = sorted_p[0];
    const float top2 = (K >= 2) ? sorted_p[1] : 0.0f;

    std::vector<float> act_in_vec(static_cast<std::size_t>(D + 4));
    std::copy(pooled.begin(), pooled.end(), act_in_vec.begin());
    act_in_vec[static_cast<std::size_t>(D + 0)] = top1;
    act_in_vec[static_cast<std::size_t>(D + 1)] = top1 - top2;
    act_in_vec[static_cast<std::size_t>(D + 2)] = ent;
    act_in_vec[static_cast<std::size_t>(D + 3)] = k_float / 255.0f;

    bt::Tensor act_in = brolm::detail::upload_host(act_in_vec.data(), 1, D + 4);
    brolm::detail::linear_batched(act_l1_W_, &act_l1_b_, act_in, act_h1_);
    bt::gelu_exact_forward(act_h1_, act_h2_);
    brolm::detail::linear_batched(act_l2_W_, &act_l2_b_, act_h2_, act_logits_);

    std::vector<float> act_l = brolm::detail::weights::detail_::download_fp32(act_logits_);
    float max_al = std::max(act_l[0], act_l[1]);
    float ea0 = std::exp(act_l[0] - max_al);
    float ea1 = std::exp(act_l[1] - max_al);
    float act_prob = ea0 / (ea0 + ea1);

    // 7. Temperature calibration
    const float T = cfg_.get_temperature(qtype, K);
    float max_z = -1e30f;
    std::vector<float> z(static_cast<std::size_t>(K));
    for (int i = 0; i < K; ++i) {
        z[static_cast<std::size_t>(i)] = logits[static_cast<std::size_t>(i)] / T;
        if (z[static_cast<std::size_t>(i)] > max_z) max_z = z[static_cast<std::size_t>(i)];
    }
    float sum_z = 0.0f;
    std::vector<float> probs(static_cast<std::size_t>(K));
    for (int i = 0; i < K; ++i) {
        probs[static_cast<std::size_t>(i)] = std::exp(z[static_cast<std::size_t>(i)] - max_z);
        sum_z += probs[static_cast<std::size_t>(i)];
    }
    for (int i = 0; i < K; ++i) {
        probs[static_cast<std::size_t>(i)] /= sum_z;
    }

    float confidence = 1.0f;
    if (K >= 2) {
        float ent_cal = 0.0f;
        for (float pv : probs) {
            ent_cal -= pv * std::log(std::max(pv, 1e-12f));
        }
        confidence = 1.0f - ent_cal / std::log(static_cast<float>(K));
    }

    // 8. Result packaging
    LayaAnswer ans;
    ans.type = q.type;
    ans.confidence = confidence;
    ans.act_probability = act_prob;

    if (q.type == "choice") {
        int best_idx = 0;
        float best_p = -1.0f;
        ans.probabilities.reserve(static_cast<std::size_t>(K));
        for (int i = 0; i < K; ++i) {
            std::string k_str = (i < static_cast<int>(q.criteria_choice.size()))
                                    ? q.criteria_choice[static_cast<std::size_t>(i)].first
                                    : std::to_string(i);
            ans.probabilities.emplace_back(k_str, probs[static_cast<std::size_t>(i)]);
            if (probs[static_cast<std::size_t>(i)] > best_p) {
                best_p = probs[static_cast<std::size_t>(i)];
                best_idx = i;
            }
        }
        ans.choice = (best_idx < static_cast<int>(q.criteria_choice.size()))
                         ? q.criteria_choice[static_cast<std::size_t>(best_idx)].first
                         : "";
    } else if (q.type == "score") {
        float expected_score = 0.0f;
        ans.probabilities.reserve(static_cast<std::size_t>(K));
        for (int i = 0; i < K; ++i) {
            ans.probabilities.emplace_back(std::to_string(i), probs[static_cast<std::size_t>(i)]);
            expected_score += static_cast<float>(i) * probs[static_cast<std::size_t>(i)];
        }
        ans.score = expected_score;
    } else {  // noul
        ans.probabilities.reserve(static_cast<std::size_t>(K));
        ans.probabilities.emplace_back("false", probs[0]);
        if (K > 1) {
            ans.probabilities.emplace_back("true", probs[1]);
            ans.noul = probs[1];
        } else {
            ans.noul = probs[0];
        }
    }

    return ans;
}

LayaResult DecisionModel::predict(const std::string& state_json_or_text,
                                 const std::vector<LayaQuestion>& questions) {
    LayaResult res;
    int total_tokens = 0;

    for (const auto& q : questions) {
        SequenceResult seq = tokenizer_.build_sequence(state_json_or_text, q,
                                                       cfg_.max_len, cfg_.head_max_len);
        total_tokens += static_cast<int>(seq.input_ids.size());
        LayaAnswer ans = forward_question(q, seq.input_ids, seq.marker_pos);
        res.answers[q.id] = std::move(ans);
    }

    res.input_tokens = total_tokens;
    return res;
}

}  // namespace brolm::laya
