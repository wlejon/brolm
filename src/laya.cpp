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

// Zero-pad W's columns up to `cols` (the input dim a fast GEMM path needs a
// multiple of 8 of); the matching activation columns are kept zero too.
void pad_cols(bt::Tensor& W, int cols) {
    if (W.cols >= cols) return;
    bt::Tensor P = bt::Tensor::zeros_on(W.device, W.rows, cols, W.dtype);
    bt::copy_d2d_strided(W, 0, W.cols, P, 0, cols, W.cols, W.rows);
    W = std::move(P);
}

// Zero-pad W's rows (a linear's outputs) up to `rows`, so an N=1 or N=2
// projection runs on the tensor-core path instead of a naive kernel.
void pad_rows(bt::Tensor& W, int rows) {
    if (W.rows >= rows) return;
    bt::Tensor P = bt::Tensor::zeros_on(W.device, rows, W.cols, W.dtype);
    bt::copy_d2d(W, 0, P, 0, W.rows * W.cols);
    W = std::move(P);
}

double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

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

void DecisionModel::set_profiling(bool on) {
    profiling_ = on;
    encoder_.set_profiling(on);
}

void DecisionModel::load_model(const std::string& model_dir) {
    // Everything checkpoint-specific comes from the directory: lengths and
    // temperatures (rl_agent_config.json), the encoder's shape, rotary bases,
    // window and global-layer pattern (encoder/config.json), the tokenizer
    // pipeline and special-token roles (tokenizer/).
    cfg_ = Config::load(model_dir + "/rl_agent_config.json");
    cfg_.variant = checkpoint_variant(cfg_, model_dir);
    cfg_.model_dir = model_dir;
    modernbert::Config enc_cfg =
        modernbert::Config::load(model_dir + "/encoder/config.json");
    tokenizer_ = LayaTokenizer::load(model_dir + "/tokenizer/tokenizer.json");
    for (const int32_t id : {tokenizer_.cls_token_id(), tokenizer_.sep_token_id(), tokenizer_.pad_token_id(),
                             tokenizer_.mask_token_id()}) {
        if (id < 0 || id >= enc_cfg.vocab_size) {
            fail("tokenizer special id " + std::to_string(id) + " is outside the encoder vocabulary (" +
                 std::to_string(enc_cfg.vocab_size) + ")");
        }
    }
    encoder_ = modernbert::ModernBertModel(std::move(enc_cfg));
    encoder_.set_profiling(profiling_);
    load_safetensors(model_dir + "/model.safetensors");
}

void DecisionModel::load_safetensors(const std::string& safetensors_path) {
    bt::safetensors::File st = bt::safetensors::File::open(safetensors_path);
    brolm::detail::weights::SafetensorsSource src({&st});

    reset_batch_();  // cached graphs reference the old weights
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
    pad_small_heads_();

    // The checkpoint's `temperature` buffer is deliberately ignored: the
    // reference (RLAgent) calibrates from rl_agent_config.json's
    // temperature / temperature_by_options only, and the shipped buffer is
    // the untouched torch.ones(3) init.
}

void DecisionModel::init_synthetic(const modernbert::Config& enc_cfg,
                                   const Config& laya_cfg) {
    cfg_ = laya_cfg;
    reset_batch_();
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
    pad_small_heads_();
}

void DecisionModel::pad_small_heads_() {
    pad_cols(act_l1_W_, act_l1_W_.cols - 4 + kActPad);
    pad_rows(scorer_l2_W_, kOutPad);
    pad_rows(scorer_l2_b_, kOutPad);
    pad_rows(act_l2_W_, kOutPad);
    pad_rows(act_l2_b_, kOutPad);
}

SequenceResult DecisionModel::build_sequence(const std::string& state_json_or_text,
                                             const LayaQuestion& q,
                                             const PredictOptions& opts) const {
    return build_sequence_ids_(tokenizer_.encode_state(state_json_or_text), q, opts);
}

std::vector<SequenceResult> DecisionModel::build_sequences(const std::string& state_json_or_text,
                                                           const std::vector<LayaQuestion>& questions,
                                                           const PredictOptions& opts) const {
    const std::vector<int32_t> state_ids = tokenizer_.encode_state(state_json_or_text);
    std::vector<SequenceResult> seqs;
    seqs.reserve(questions.size());
    for (const auto& q : questions) seqs.push_back(build_sequence_ids_(state_ids, q, opts));
    return seqs;
}

SequenceResult DecisionModel::build_sequence_ids_(const std::vector<int32_t>& state_ids,
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
    SequenceResult seq = tokenizer_.build_sequence_ids(state_ids, q, max_len, head_max_len,
                                                       opts.truncate_left);
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
    const LayaItem item{input_ids.data(), static_cast<int>(input_ids.size()), marker_pos.data(),
                        static_cast<int>(marker_pos.size()), q.qtype_index()};
    const std::vector<LayaItemLogits> out = forward_items({item});
    return answer_from_logits(q, out[0]);
}

// Temperature calibration + packaging (RLAgent.system_one).
LayaAnswer DecisionModel::answer_from_logits(const LayaQuestion& q, const LayaItemLogits& out) const {
    const int K = static_cast<int>(out.logits.size());
    if (K == 0) fail("answer_from_logits: no logits");
    const double mx = std::max(out.act_logits[0], out.act_logits[1]);
    const double e0 = std::exp(out.act_logits[0] - mx), e1 = std::exp(out.act_logits[1] - mx);

    LayaAnswer ans;
    ans.type = q.type;
    ans.act_probability = static_cast<float>(e0 / (e0 + e1));
    ans.temperature = cfg_.get_temperature(q.qtype_index(), K);
    ans.logits = out.logits;
    ans.act_logits = {out.act_logits[0], out.act_logits[1]};

    const std::vector<float> probs = softmax_scaled(out.logits, ans.temperature);
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
    {
        const Clock::time_point t0 = Clock::now();
        seqs = build_sequences(state_json_or_text, questions, opts);
        timings_.tokenize_ms = ms_since(t0);
    }

    LayaResult res;
    if (questions.empty()) return res;
    std::vector<LayaItem> items;
    items.reserve(questions.size());
    for (std::size_t i = 0; i < questions.size(); ++i) {
        items.push_back(LayaItem::of(seqs[i], questions[i].qtype_index()));
        res.input_tokens += static_cast<int>(seqs[i].input_ids.size());
    }
    const LayaTimings tok = timings_;
    const std::vector<LayaItemLogits> outs = forward_items(items);  // resets stage timings
    timings_.tokenize_ms = tok.tokenize_ms;

    const Clock::time_point t_cal = Clock::now();
    for (std::size_t i = 0; i < questions.size(); ++i) {
        res.answers[questions[i].id] = answer_from_logits(questions[i], outs[i]);
    }
    timings_.calibrate_ms = ms_since(t_cal);

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
