#include "brolm/laya.h"

#include "brolm/detail/compute.h"
#include "brolm/detail/device.h"
#include "brolm/detail/weights.h"

#include "brotensor/ops.h"
#include "brotensor/runtime.h"
#include "brotensor/safetensors.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <vector>

namespace brolm::laya {

namespace bt = ::brotensor;
using Clock = std::chrono::steady_clock;

namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("laya::DecisionModel: " + msg);
}

double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// Times one stage into `acc` when profiling; syncs so GPU work lands in the
// stage that issued it.
class StageTimer {
public:
    StageTimer(bool on, double& acc) : on_(on), acc_(acc) {
        if (on_) {
            bt::sync_all();
            t0_ = Clock::now();
        }
    }
    ~StageTimer() {
        if (on_) {
            bt::sync_all();
            acc_ += ms_since(t0_);
        }
    }
    StageTimer(const StageTimer&) = delete;
    StageTimer& operator=(const StageTimer&) = delete;

private:
    bool on_;
    double& acc_;
    Clock::time_point t0_;
};

// Numerically-stable softmax of logits / T.
std::vector<float> softmax_scaled(const std::vector<float>& logits, float T) {
    std::vector<float> p(logits.size());
    float mx = -INFINITY;
    for (float v : logits) mx = std::max(mx, v / T);
    float sum = 0.0f;
    for (std::size_t i = 0; i < logits.size(); ++i) {
        p[i] = std::exp(logits[i] / T - mx);
        sum += p[i];
    }
    for (float& v : p) v /= sum;
    return p;
}

}  // namespace

DecisionModel::DecisionModel() {
    head_layers_.resize(static_cast<std::size_t>(cfg_.head_layers));
}

DecisionModel::~DecisionModel() = default;

void DecisionModel::set_profiling(bool on) {
    profiling_ = on;
    encoder_.set_profiling(on);
}

void DecisionModel::load_model(const std::string& model_dir) {
    cfg_ = Config::load(model_dir + "/rl_agent_config.json");
    modernbert::Config enc_cfg =
        modernbert::Config::load(model_dir + "/encoder/config.json");
    encoder_ = modernbert::ModernBertModel(std::move(enc_cfg));
    encoder_.set_profiling(profiling_);
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

    // The checkpoint's `temperature` buffer is deliberately ignored: the
    // reference (RLAgent) calibrates from rl_agent_config.json's
    // temperature / temperature_by_options only, and the shipped buffer is
    // the untouched torch.ones(3) init.
}

void DecisionModel::init_synthetic(const modernbert::Config& enc_cfg,
                                   const Config& laya_cfg) {
    cfg_ = laya_cfg;
    encoder_ = modernbert::ModernBertModel(enc_cfg);
    encoder_.init_synthetic();
    encoder_.set_profiling(profiling_);

    const int D = enc_cfg.hidden_size;
    auto fill = [](std::size_t n, float v) { return std::vector<float>(n, v); };
    const std::size_t d = static_cast<std::size_t>(D);
    auto up = [](const std::vector<float>& v, int r, int c) {
        return brolm::detail::upload_host(v.data(), r, c);
    };

    type_emb_ = up(fill(3 * d, 0.01f), 3, D);
    head_layers_.resize(static_cast<std::size_t>(cfg_.head_layers));
    for (auto& L : head_layers_) {
        L.norm1_g    = up(fill(d, 1.0f), D, 1);
        L.norm1_b    = up(fill(d, 0.0f), D, 1);
        L.in_proj_W  = up(fill(3 * d * d, 0.01f), 3 * D, D);
        L.in_proj_b  = up(fill(3 * d, 0.01f), 3 * D, 1);
        L.out_proj_W = up(fill(d * d, 0.01f), D, D);
        L.out_proj_b = up(fill(d, 0.0f), D, 1);
        L.norm2_g    = up(fill(d, 1.0f), D, 1);
        L.norm2_b    = up(fill(d, 0.0f), D, 1);
        L.linear1_W  = up(fill(4 * d * d, 0.01f), 4 * D, D);
        L.linear1_b  = up(fill(4 * d, 0.0f), 4 * D, 1);
        L.linear2_W  = up(fill(4 * d * d, 0.01f), D, 4 * D);
        L.linear2_b  = up(fill(d, 0.0f), D, 1);
    }
    scorer_ln_g_ = up(fill(d, 1.0f), D, 1);
    scorer_ln_b_ = up(fill(d, 0.0f), D, 1);
    scorer_l1_W_ = up(fill(d * d, 0.01f), D, D);
    scorer_l1_b_ = up(fill(d, 0.0f), D, 1);
    scorer_l2_W_ = up(fill(d, 0.01f), 1, D);
    scorer_l2_b_ = up(fill(1, 0.0f), 1, 1);
    act_l1_W_ = up(fill(256 * (d + 4), 0.01f), 256, D + 4);
    act_l1_b_ = up(fill(256, 0.0f), 256, 1);
    act_l2_W_ = up(fill(512, 0.01f), 2, 256);
    act_l2_b_ = up(fill(2, 0.0f), 2, 1);
}

SequenceResult DecisionModel::build_sequence(const std::string& state_json_or_text,
                                             const LayaQuestion& q,
                                             const PredictOptions& opts) const {
    if (q.type != "choice" && q.type != "score" && q.type != "noul") {
        fail("question '" + q.id + "': unknown type '" + q.type +
             "' (expected choice, score or noul)");
    }
    const std::size_t n_opts = LayaTokenizer::render_options(q).size();
    if (n_opts == 0) fail("question '" + q.id + "': " + q.type + " question has no criteria");
    const int max_len = opts.max_len > 0 ? opts.max_len : cfg_.max_len;
    const int head_max_len = opts.head_max_len > 0 ? opts.head_max_len : cfg_.head_max_len;
    SequenceResult seq = tokenizer_.build_sequence(state_json_or_text, q, max_len,
                                                   head_max_len, opts.truncate_left);
    if (seq.marker_pos.size() != n_opts) {
        fail("question '" + q.id + "': options do not fit in head_max_len=" +
             std::to_string(head_max_len) + " tokens (" + std::to_string(seq.marker_pos.size()) +
             " of " + std::to_string(n_opts) + " option markers inside max_len=" +
             std::to_string(max_len) + "); raise head_max_len / max_len or split the options");
    }
    return seq;
}

LayaAnswer DecisionModel::forward_question(const LayaQuestion& q,
                                          const std::vector<int32_t>& input_ids,
                                          const std::vector<int32_t>& marker_pos) {
    if (input_ids.empty()) fail("forward_question: input_ids is empty");
    if (marker_pos.empty()) fail("forward_question: marker_pos is empty");

    const int seq_len = static_cast<int>(input_ids.size());
    const int K = static_cast<int>(marker_pos.size());
    const int D = encoder_.config().hidden_size;
    const int H = std::max(1, D / 64);  // nn.TransformerEncoderLayer(d, d // 64 heads)
    const bt::Device dev = bt::default_device();
    const bt::Dtype dt = brolm::compute_dtype();
    const bool prof = profiling_;
    LayaTimings& T = timings_;

    // 1. ModernBERT encoder
    {
        StageTimer t(prof, T.encoder_ms);
        encoder_.forward(input_ids.data(), seq_len, h_);
        ++T.uploads;
    }

    // 2-3. Question-type embedding on every row, then the Pre-LN head layers
    //      (MHA with biases, ReLU FFN — nn.TransformerEncoderLayer defaults).
    const int qtype = q.qtype_index();
    {
        StageTimer t(prof, T.head_ms);
        brolm::detail::resize_like(type_row_, 1, D, dt, dev);
        bt::copy_d2d(type_emb_, qtype * D, type_row_, 0, D);
        bt::add_row_bias_inplace(h_, type_row_);

        for (TransformerHeadLayer& L : head_layers_) {
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
    }

    // 4-5. Gather the [MASK] marker rows, score them:
    //      LayerNorm -> Linear -> GELU -> Linear -> raw logits (K, 1).
    {
        StageTimer t(prof, T.scorer_ms);
        idx_dev_ = bt::Tensor::from_raw_bytes_on(dev, marker_pos.data(), K, 1, bt::Dtype::INT32,
                                                 static_cast<std::size_t>(K) * sizeof(int32_t));
        ++T.uploads;
        bt::gather_rows(h_, idx_dev_, m_);
        brolm::detail::layernorm_batched(m_, scorer_ln_g_, scorer_ln_b_, scorer_out0_, 1e-5f);
        brolm::detail::linear_batched(scorer_l1_W_, &scorer_l1_b_, scorer_out0_, scorer_out1_);
        bt::gelu_exact_forward(scorer_out1_, scorer_out2_);
        brolm::detail::linear_batched(scorer_l2_W_, &scorer_l2_b_, scorer_out2_, scorer_out3_);
    }

    std::vector<float> logits;
    {
        StageTimer t(prof, T.download_ms);
        logits = brolm::detail::weights::detail_::download_fp32(scorer_out3_);
        ++T.downloads;
    }

    // 6. Act head: [h[0] | top1, top1 - top2, normalized entropy, k / 255]
    //    from the uncalibrated distribution. h[0] stays on the device.
    std::vector<float> act_l;
    {
        StageTimer t(prof, T.act_ms);
        const std::vector<float> p = softmax_scaled(logits, 1.0f);
        const float k_float = std::max(2.0f, static_cast<float>(K));
        float ent = 0.0f;
        for (float pv : p) ent -= pv * std::log(std::max(pv, 1e-9f));
        ent /= std::log(k_float);
        std::vector<float> sorted_p = p;
        std::sort(sorted_p.begin(), sorted_p.end(), std::greater<float>());
        const float top1 = sorted_p[0];
        const float top2 = K >= 2 ? sorted_p[1] : 0.0f;
        const float feats[4] = {top1, top1 - top2, ent, k_float / 255.0f};

        brolm::detail::resize_like(act_in_, 1, D + 4, dt, dev);
        bt::copy_d2d(h_, 0, act_in_, 0, D);
        bt::Tensor feats_dev = brolm::detail::upload_host(feats, 1, 4);
        ++T.uploads;
        bt::copy_d2d(feats_dev, 0, act_in_, D, 4);
        brolm::detail::linear_batched(act_l1_W_, &act_l1_b_, act_in_, act_h1_);
        bt::gelu_exact_forward(act_h1_, act_h2_);
        brolm::detail::linear_batched(act_l2_W_, &act_l2_b_, act_h2_, act_logits_);
    }
    {
        StageTimer t(prof, T.download_ms);
        act_l = brolm::detail::weights::detail_::download_fp32(act_logits_);
        ++T.downloads;
    }

    // 7-8. Temperature calibration + packaging (RLAgent.system_one).
    StageTimer tc(prof, T.calibrate_ms);
    const double mx = std::max(act_l[0], act_l[1]);
    const double e0 = std::exp(act_l[0] - mx), e1 = std::exp(act_l[1] - mx);

    LayaAnswer ans;
    ans.type = q.type;
    ans.act_probability = static_cast<float>(e0 / (e0 + e1));
    ans.temperature = cfg_.get_temperature(qtype, K);
    ans.logits = logits;
    ans.act_logits = act_l;

    const std::vector<float> probs = softmax_scaled(logits, ans.temperature);
    ans.confidence = 1.0f;
    if (K >= 2) {
        float ent_cal = 0.0f;
        for (float pv : probs) ent_cal -= pv * std::log(std::max(pv, 1e-12f));
        ans.confidence = 1.0f - ent_cal / std::log(static_cast<float>(K));
    }

    ans.probabilities.reserve(static_cast<std::size_t>(K));
    if (q.type == "choice") {
        const int best = static_cast<int>(std::max_element(probs.begin(), probs.end()) - probs.begin());
        for (int i = 0; i < K; ++i) {
            ans.probabilities.emplace_back(
                i < static_cast<int>(q.criteria_choice.size())
                    ? q.criteria_choice[static_cast<std::size_t>(i)].first
                    : std::to_string(i),
                probs[static_cast<std::size_t>(i)]);
        }
        ans.choice = best < static_cast<int>(q.criteria_choice.size())
                         ? q.criteria_choice[static_cast<std::size_t>(best)].first
                         : "";
    } else if (q.type == "score") {
        float expected = 0.0f;
        for (int i = 0; i < K; ++i) {
            ans.probabilities.emplace_back(std::to_string(i), probs[static_cast<std::size_t>(i)]);
            expected += static_cast<float>(i) * probs[static_cast<std::size_t>(i)];
        }
        ans.score = expected;
    } else {  // noul: options are always [false, true]
        ans.probabilities.emplace_back("false", probs[0]);
        ans.probabilities.emplace_back("true", probs[1]);
        ans.noul = probs[1];
    }
    return ans;
}

LayaResult DecisionModel::predict(const std::string& state_json_or_text,
                                  const std::vector<LayaQuestion>& questions,
                                  const PredictOptions& opts) {
    const Clock::time_point t_start = Clock::now();
    timings_ = LayaTimings{};
    encoder_.timings() = modernbert::EncoderTimings{};

    // Tokenize everything first so an option-fit error throws before any
    // device work, as the reference does.
    std::vector<SequenceResult> seqs;
    seqs.reserve(questions.size());
    {
        StageTimer t(profiling_, timings_.tokenize_ms);
        for (const auto& q : questions) seqs.push_back(build_sequence(state_json_or_text, q, opts));
    }

    LayaResult res;
    for (std::size_t i = 0; i < questions.size(); ++i) {
        res.input_tokens += static_cast<int>(seqs[i].input_ids.size());
        res.answers[questions[i].id] =
            forward_question(questions[i], seqs[i].input_ids, seqs[i].marker_pos);
    }

    timings_.total_ms = ms_since(t_start);
    timings_.questions = static_cast<int>(questions.size());
    timings_.tokens = res.input_tokens;
    timings_.encoder = encoder_.timings();
    return res;
}

LayaResult DecisionModel::predict(const std::string& state_json_or_text,
                                  const std::unordered_map<std::string, LayaQuestion>& questions,
                                  const PredictOptions& opts) {
    std::vector<LayaQuestion> q_vec;
    q_vec.reserve(questions.size());
    for (const auto& [id, q] : questions) {
        q_vec.push_back(q);
        q_vec.back().id = id;
    }
    return predict(state_json_or_text, q_vec, opts);
}

}  // namespace brolm::laya
