#include "brolm/laya.h"
#include "brolm/laya_config.h"
#include "brolm/laya_tokenizer.h"
#include "brolm/modernbert.h"
#include "brolm/modernbert_config.h"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

void check(bool condition, const std::string& msg) {
    if (!condition) {
        std::cerr << "FAILED: " << msg << std::endl;
        std::exit(1);
    }
}

void test_synthetic() {
    std::cout << "--- Running Test 1: Synthetic Weights ---" << std::endl;

    brolm::modernbert::Config enc_cfg;
    enc_cfg.vocab_size = 50368;
    enc_cfg.hidden_size = 64;
    enc_cfg.intermediate_size = 128;
    enc_cfg.num_hidden_layers = 2;
    enc_cfg.num_attention_heads = 4;
    enc_cfg.local_attention = 16;
    enc_cfg.layer_types = {"full_attention", "sliding_attention"};

    brolm::laya::Config laya_cfg;
    laya_cfg.head_layers = 2;
    laya_cfg.max_len = 64;
    laya_cfg.head_max_len = 32;

    std::cout << "Initializing synthetic model..." << std::endl;
    brolm::laya::DecisionModel model;
    model.init_synthetic(enc_cfg, laya_cfg);
    std::cout << "Synthetic model initialized." << std::endl;

    // Test choice question
    brolm::laya::LayaQuestion q_choice;
    q_choice.id = "q_choice";
    q_choice.type = "choice";
    q_choice.instructions = "Select an option";
    q_choice.criteria_choice = {{"optA", "Option A"}, {"optB", "Option B"}, {"optC", "Option C"}};

    std::vector<int32_t> ids = {1, 10, 20, 30, 2, 50284, 5, 50284, 6, 50284, 7, 2, 8, 9, 2};
    std::vector<int32_t> markers = {5, 7, 9};

    std::cout << "Calling forward_question..." << std::endl;
    brolm::laya::LayaAnswer ans = model.forward_question(q_choice, ids, markers);
    std::cout << "forward_question returned." << std::endl;
    check(!ans.choice.empty(), "Choice answer should not be empty");
    check(ans.probabilities.size() == 3, "Choice answer should have 3 probabilities");

    float sum_p = 0.0f;
    for (const auto& [k, p] : ans.probabilities) {
        check(std::isfinite(p), "Probability must be finite");
        check(p >= 0.0f && p <= 1.0f, "Probability must be in [0, 1]");
        sum_p += p;
    }
    check(std::abs(sum_p - 1.0f) < 1e-4f, "Choice probabilities must sum to 1");
    check(std::isfinite(ans.confidence), "Confidence must be finite");
    check(ans.confidence >= 0.0f && ans.confidence <= 1.0f, "Confidence must be in [0, 1]");
    check(std::isfinite(ans.act_probability), "Act probability must be finite");
    check(ans.act_probability >= 0.0f && ans.act_probability <= 1.0f, "Act probability must be in [0, 1]");

    // Test score question
    brolm::laya::LayaQuestion q_score;
    q_score.id = "q_score";
    q_score.type = "score";
    q_score.instructions = "Rate severity";
    q_score.criteria_score = {"low", "medium", "high"};

    brolm::laya::LayaAnswer ans_s = model.forward_question(q_score, ids, markers);
    check(ans_s.type == "score", "Answer type must be score");
    check(std::isfinite(ans_s.score), "Score must be finite");
    check(ans_s.score >= 0.0f && ans_s.score <= 2.0f, "Score must be in range [0, 2]");
    float sum_s = 0.0f;
    for (const auto& [k, p] : ans_s.probabilities) {
        sum_s += p;
    }
    check(std::abs(sum_s - 1.0f) < 1e-4f, "Score probabilities must sum to 1");

    // Test noul question
    brolm::laya::LayaQuestion q_noul;
    q_noul.id = "q_noul";
    q_noul.type = "noul";
    q_noul.instructions = "Is this urgent?";

    std::vector<int32_t> ids_noul = {1, 10, 2, 50284, 5, 50284, 6, 2, 8, 2};
    std::vector<int32_t> markers_noul = {3, 5};

    brolm::laya::LayaAnswer ans_n = model.forward_question(q_noul, ids_noul, markers_noul);
    check(ans_n.type == "noul", "Answer type must be noul");
    check(std::isfinite(ans_n.noul), "Noul must be finite");
    check(ans_n.noul >= 0.0f && ans_n.noul <= 1.0f, "Noul must be in [0, 1]");

    std::cout << "Test 1 passed successfully!" << std::endl;
}

void test_real_checkpoint(const std::string& model_dir) {
    std::cout << "--- Running Test 2: Real Checkpoint from " << model_dir << " ---" << std::endl;

    if (!fs::exists(model_dir + "/model.safetensors")) {
        std::cout << "Skipping Test 2: checkpoint not found at " << model_dir << std::endl;
        return;
    }

    brolm::laya::DecisionModel model;
    model.load_model(model_dir);

    const std::string state =
        "{\"from\": \"user@acme.com\", \"subject\": \"Duplicate charge on invoice #4411\", "
        "\"body\": \"Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan.\"}";

    std::vector<brolm::laya::LayaQuestion> questions;

    brolm::laya::LayaQuestion q_dept;
    q_dept.id = "department";
    q_dept.type = "choice";
    q_dept.instructions = "Which department should handle this request?";
    q_dept.criteria_choice = {
        {"billing", "invoices, payments, refunds"},
        {"technical", "bugs, outages, system errors"},
        {"sales", "pricing, new contracts"},
        {"other", "everything else"}
    };
    questions.push_back(q_dept);

    brolm::laya::LayaQuestion q_urg;
    q_urg.id = "urgency";
    q_urg.type = "score";
    q_urg.instructions = "How urgent is this request?";
    q_urg.criteria_score = {"not urgent", "soon", "critical deadline or blocking issue"};
    questions.push_back(q_urg);

    brolm::laya::LayaQuestion q_churn;
    q_churn.id = "churn_risk";
    q_churn.type = "noul";
    q_churn.instructions = "Does the user threaten to cancel or leave?";
    questions.push_back(q_churn);

    brolm::laya::LayaQuestion q_ref;
    q_ref.id = "refund_requested";
    q_ref.type = "noul";
    q_ref.instructions = "Does the user explicitly request a refund?";
    questions.push_back(q_ref);



    brolm::laya::LayaResult res = model.predict(state, questions);

    check(res.input_tokens > 0, "Input tokens must be > 0");
    std::cout << "Input tokens: " << res.input_tokens << std::endl;

    // 1. Department assertion: choice is "billing" with confidence > 0.85
    check(res.answers.count("department") == 1, "Must contain department answer");
    const auto& ans_dept = res.answers.at("department");
    std::cout << "Department choice: " << ans_dept.choice
              << ", confidence: " << ans_dept.confidence
              << ", act_probability: " << ans_dept.act_probability << std::endl;
    for (const auto& [k, p] : ans_dept.probabilities) {
        std::cout << "  p(" << k << ") = " << p << std::endl;
    }
    check(ans_dept.choice == "billing", "Department choice must be 'billing'");
    check(ans_dept.confidence > 0.85f, "Department confidence must be > 0.85");

    // 2. Urgency assertion: probabilities finite and calibrated
    check(res.answers.count("urgency") == 1, "Must contain urgency answer");
    const auto& ans_urg = res.answers.at("urgency");
    std::cout << "Urgency score: " << ans_urg.score
              << ", confidence: " << ans_urg.confidence
              << ", act_probability: " << ans_urg.act_probability << std::endl;
    for (const auto& [k, p] : ans_urg.probabilities) {
        std::cout << "  p(" << k << ") = " << p << std::endl;
    }
    check(std::isfinite(ans_urg.score), "Urgency score must be finite");
    check(ans_urg.score >= 0.0f && ans_urg.score <= 2.0f, "Urgency score in [0, 2]");
    float urg_sum = 0.0f;
    for (const auto& [k, p] : ans_urg.probabilities) urg_sum += p;
    check(std::abs(urg_sum - 1.0f) < 1e-4f, "Urgency probabilities must sum to 1");

    // 3. Churn risk assertion: finite, in [0, 1], calibrated
    check(res.answers.count("churn_risk") == 1, "Must contain churn_risk answer");
    const auto& ans_churn = res.answers.at("churn_risk");
    std::cout << "Churn risk noul: " << ans_churn.noul
              << ", act_probability: " << ans_churn.act_probability << std::endl;
    check(std::isfinite(ans_churn.noul), "Churn risk must be finite");
    check(ans_churn.noul >= 0.0f && ans_churn.noul <= 1.0f, "Churn risk must be in [0, 1]");

    // 4. Refund requested assertion: finite, in [0, 1], calibrated
    check(res.answers.count("refund_requested") == 1, "Must contain refund_requested answer");
    const auto& ans_ref = res.answers.at("refund_requested");
    std::cout << "Refund requested noul: " << ans_ref.noul
              << ", act_probability: " << ans_ref.act_probability << std::endl;
    check(std::isfinite(ans_ref.noul), "Refund requested must be finite");
    check(ans_ref.noul >= 0.0f && ans_ref.noul <= 1.0f, "Refund requested must be in [0, 1]");

    std::cout << "Test 2 passed successfully!" << std::endl;
}

}  // namespace

int main() {
    test_synthetic();

    const char* env_dir = std::getenv("LAYA_MODEL_DIR");
    std::string model_dir = (env_dir && env_dir[0]) ? env_dir : "D:/projects/laya";
    test_real_checkpoint(model_dir);

    std::cout << "All Laya tests PASSED!" << std::endl;
    return 0;
}
