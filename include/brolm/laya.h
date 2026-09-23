#pragma once

#include "brolm/laya_config.h"
#include "brolm/laya_tokenizer.h"
#include "brolm/modernbert.h"
#include "brotensor/tensor.h"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace brolm::laya {

struct LayaQuestion {
    std::string id;
    std::string type;  // "choice", "score", "noul"
    std::string instructions;

    // For choice: ordered list of (key, description)
    std::vector<std::pair<std::string, std::string>> criteria_choice;

    // For score: ordered list of criteria descriptions
    std::vector<std::string> criteria_score;

    // For noul: optional criteria descriptions for false and true
    std::string criteria_noul_false;
    std::string criteria_noul_true;

    int qtype_index() const {
        if (type == "choice") return 0;
        if (type == "score") return 1;
        return 2;
    }
};

// Parse a Jev-shaped questions object, {"id": {"type", "instructions",
// "criteria"}, ...}, into questions in key order — the same shapes the
// reference RLAgent._to_internal accepts: choice criteria as an object
// (key -> description, null/"" = none) or a list of keys; score criteria as a
// list (an object contributes its keys, as Python's enumerate(dict) does);
// noul criteria as {"false": ..., "true": ...}. Non-string instructions are
// serialised Python-style. Throws std::runtime_error on malformed input.
std::vector<LayaQuestion> parse_questions_json(const std::string& json_text);

struct LayaAnswer {
    std::string type;  // "choice", "score", "noul"
    std::string choice;
    float score = 0.0f;
    float noul = 0.0f;
    float confidence = 0.0f;
    std::vector<std::pair<std::string, float>> probabilities;
    float act_probability = 0.0f;

    // Pre-temperature scorer logits, one per option in option order, and the
    // two raw act-head logits ([act, escalate]) — what a downstream caller
    // needs to refit temperatures on its own data.
    std::vector<float> logits;
    std::vector<float> act_logits;
    // The temperature that was applied (per qtype / option-count bucket).
    float temperature = 1.0f;
};

// One item of a batched forward: a tokenized (state, question) sequence as
// build_sequence(s) produces it, plus the question-type index (qtype_index()).
// The pointed-to ids / markers must outlive the forward_items() call.
struct LayaItem {
    const int32_t* input_ids = nullptr;
    int num_ids = 0;
    const int32_t* marker_pos = nullptr;  // positions inside this item's sequence
    int num_markers = 0;
    int qtype = 0;

    static LayaItem of(const SequenceResult& seq, int qtype) {
        return LayaItem{seq.input_ids.data(), static_cast<int>(seq.input_ids.size()),
                        seq.marker_pos.data(), static_cast<int>(seq.marker_pos.size()), qtype};
    }
};

// Raw outputs of one item: pre-temperature scorer logits (one per marker, in
// marker order) and the two act-head logits [act, escalate].
struct LayaItemLogits {
    std::vector<float> logits;
    float act_logits[2] = {0.0f, 0.0f};
};

struct LayaResult {
    std::string model = "rl-agent";
    std::unordered_map<std::string, LayaAnswer> answers;
    int input_tokens = 0;
};

// Per-call overrides of the checkpoint's rl_agent_config.json values.
struct PredictOptions {
    int max_len = 0;             // <= 0: config max_len
    int head_max_len = 0;        // <= 0: config head_max_len
    bool truncate_left = false;  // keep the newest state tokens (conversations)
};

// Wall-clock stage breakdown of the last predict(), filled only while
// profiling is on (set_profiling). Stage boundaries sync the device, so the
// stages sum to a serialised total that is slower than an unprofiled call.
struct LayaTimings {
    double tokenize_ms = 0;
    double encoder_ms = 0;
    double head_ms = 0;       // type embedding + the 2 head transformer layers
    double scorer_ms = 0;     // marker gather + scorer MLP
    double act_ms = 0;        // act features + act MLP
    double download_ms = 0;   // device->host reads (logits / act logits)
    double calibrate_ms = 0;  // host softmax / result packaging
    double total_ms = 0;
    int uploads = 0;          // host->device transfers issued
    int downloads = 0;        // device->host transfers (each one is a sync)
    int questions = 0;
    int tokens = 0;
    modernbert::EncoderTimings encoder;  // encoder_ms split by op family
};

struct TransformerHeadLayer {
    brotensor::Tensor norm1_g;
    brotensor::Tensor norm1_b;
    brotensor::Tensor in_proj_W;
    brotensor::Tensor in_proj_b;
    brotensor::Tensor out_proj_W;
    brotensor::Tensor out_proj_b;
    brotensor::Tensor norm2_g;
    brotensor::Tensor norm2_b;
    brotensor::Tensor linear1_W;
    brotensor::Tensor linear1_b;
    brotensor::Tensor linear2_W;
    brotensor::Tensor linear2_b;
};

class DecisionModel {
public:
    DecisionModel();
    ~DecisionModel();

    DecisionModel(const DecisionModel&) = delete;
    DecisionModel& operator=(const DecisionModel&) = delete;
    DecisionModel(DecisionModel&&) noexcept;
    DecisionModel& operator=(DecisionModel&&) noexcept;

    void load_model(const std::string& model_dir);
    void load_safetensors(const std::string& safetensors_path);

    void init_synthetic(const modernbert::Config& enc_cfg = modernbert::Config{},
                        const Config& laya_cfg = Config{});

    // Tokenize one question against the state with the effective limits.
    // Throws std::runtime_error (reference wording) when the options do not
    // all fit — the reference raises rather than answering over a truncated
    // answer space.
    SequenceResult build_sequence(const std::string& state_json_or_text,
                                  const LayaQuestion& q,
                                  const PredictOptions& opts = {}) const;

    // Tokenize the state once and build every question's sequence (throws
    // before any device work if a question's options do not fit).
    std::vector<SequenceResult> build_sequences(const std::string& state_json_or_text,
                                                const std::vector<LayaQuestion>& questions,
                                                const PredictOptions& opts = {}) const;

    // ── Batched forward ────────────────────────────────────────────────
    // One packed forward over any number of items — each a tokenized
    // (state, question) sequence, from one request or many. The sequences
    // are packed back to back without padding and run through the encoder,
    // head, scorer and act head together; the result per item equals running
    // it alone. One host->device upload and one device->host readback per
    // call. On CUDA, the device work replays a CUDA graph cached per
    // (token-count, item-count, marker-count) bucket.
    // Not thread-safe: one call at a time per model.
    std::vector<LayaItemLogits> forward_items(const std::vector<LayaItem>& items);

    // Calibrate one item's raw outputs into the reference answer shape
    // (host only: temperature softmax, confidence, option labels).
    LayaAnswer answer_from_logits(const LayaQuestion& q, const LayaItemLogits& out) const;

    // Single-question convenience over forward_items.
    LayaAnswer forward_question(const LayaQuestion& q,
                                const std::vector<int32_t>& input_ids,
                                const std::vector<int32_t>& marker_pos);

    // Enable / disable CUDA-graph replay (default on; env BROLM_LAYA_GRAPHS=0
    // turns it off). Graphs are never used while profiling.
    void set_graphs_enabled(bool on);
    bool graphs_enabled() const;
    std::size_t cached_graphs() const;

    // The padded row count forward_items runs `tokens` packed rows at when
    // replaying graphs (16 steps per power of two, 16-row floor): the graph
    // cache key, and the unit of a forward's cost.
    static int token_bucket(int tokens);

    // Pre-warm: capture the CUDA graph of every token bucket up to
    // bucket(max_tokens) (largest first, so scratch and rotary tables are
    // reserved once), then time one replay of each — host packing, upload,
    // device work and readback, i.e. what a forward_items() of that size
    // costs on this device. Without graphs (CPU, disabled) it only times a
    // few sizes. Returns (bucket rows, ms) points for a cost model.
    struct WarmPoint {
        int tokens = 0;
        double ms = 0;
    };
    std::vector<WarmPoint> prewarm_graphs(int max_tokens);

    LayaResult predict(const std::string& state_json_or_text,
                       const std::vector<LayaQuestion>& questions,
                       const PredictOptions& opts = {});
    LayaResult predict(const std::string& state_json_or_text,
                       const std::unordered_map<std::string, LayaQuestion>& questions,
                       const PredictOptions& opts = {});

    void set_profiling(bool on);
    bool profiling() const { return profiling_; }
    const LayaTimings& last_timings() const { return timings_; }

    const modernbert::ModernBertModel& encoder() const { return encoder_; }
    const Config& config() const { return cfg_; }
    Config& mutable_config() { return cfg_; }
    const LayaTokenizer& tokenizer() const { return tokenizer_; }

private:
    SequenceResult build_sequence_ids_(const std::vector<int32_t>& state_ids,
                                       const LayaQuestion& q,
                                       const PredictOptions& opts) const;

    Config cfg_;
    modernbert::ModernBertModel encoder_;
    LayaTokenizer tokenizer_;

    brotensor::Tensor type_emb_;  // (3, D)
    std::vector<TransformerHeadLayer> head_layers_;

    // Scorer
    brotensor::Tensor scorer_ln_g_;
    brotensor::Tensor scorer_ln_b_;
    brotensor::Tensor scorer_l1_W_;
    brotensor::Tensor scorer_l1_b_;
    brotensor::Tensor scorer_l2_W_;
    brotensor::Tensor scorer_l2_b_;

    // Act Head. act_l1_W_ is (256, D + kActPad): the checkpoint's D + 4
    // input columns then zeros, so its K is a multiple of 8. scorer_l2 and
    // act_l2 (1 and 2 outputs) are zero-padded to kOutPad output rows, so
    // every head linear takes the tensor-core path; the extra outputs are
    // dropped on the device.
    static constexpr int kActPad = 8;
    static constexpr int kOutPad = 8;
    void pad_small_heads_();
    brotensor::Tensor act_l1_W_;
    brotensor::Tensor act_l1_b_;
    brotensor::Tensor act_l2_W_;
    brotensor::Tensor act_l2_b_;

    bool profiling_ = false;
    LayaTimings timings_;

    // Packed-batch scratch, index buffers and the CUDA-graph cache
    // (laya_batch.cpp).
    struct Batch;
    std::unique_ptr<Batch> batch_;
    Batch& batch();
    void reset_batch_();
    void reserve_batch_(int T, int N, int K);
    // Y = epilogue(act(X · Wᵀ + b)) on the encoder's split-K workspace.
    void linear_(const brotensor::Tensor& W, const brotensor::Tensor& b, const brotensor::Tensor& X, int act,
                 int epilogue, brotensor::Tensor& Y);
    void run_device_(Batch& b, int T, int N, int K);
};

}  // namespace brolm::laya

namespace brolm {
using LayaModel = laya::DecisionModel;
using LayaQuestion = laya::LayaQuestion;
using LayaAnswer = laya::LayaAnswer;
using LayaResult = laya::LayaResult;
using LayaPredictOptions = laya::PredictOptions;
}  // namespace brolm
