// Numeric parity of brolm's Laya port against the upstream PyTorch reference.
//
// tests/ref/laya_parity.json comes from tests/ref/gen_laya_parity.py (run the
// reference rl_common.py / rl_agent_api.py over a varied set of calls). This
// test checks, against it:
//   - tokenizer ids for a battery of strings, exactly;
//   - python_json_spacing(JSON.stringify form) == json.dumps form;
//   - build_sequence ids + marker positions per call, exactly (default and
//     overridden max_len / head_max_len, both truncation directions);
//   - the options-do-not-fit error;
//   - raw scorer logits, act probability and calibrated probabilities against
//     the fp32 reference, with the tolerance judged against how far the
//     reference's own bf16 autocast run lands from fp32.
// Weights are looked up at LAYA_MODEL_DIR or ../laya; the model half skips
// (tokenizer half still runs) when they are absent.

#include "brolm/laya.h"
#include "brolm/detail/json.h"

#include "brotensor/runtime.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace j = brolm::detail::json;

namespace {

int g_failures = 0;

void check(bool ok, const std::string& msg) {
    if (!ok) {
        std::cerr << "FAILED: " << msg << "\n";
        ++g_failures;
    }
}

std::string slurp(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

std::vector<int32_t> ints(const j::Value& a) {
    std::vector<int32_t> out;
    for (const auto& e : a.as_array()) out.push_back(static_cast<int32_t>(e.as_number()));
    return out;
}

std::vector<double> nums(const j::Value& a) {
    std::vector<double> out;
    for (const auto& e : a.as_array()) out.push_back(e.as_number());
    return out;
}

std::string show(const std::vector<int32_t>& v, std::size_t at) {
    std::ostringstream s;
    const std::size_t lo = at > 4 ? at - 4 : 0;
    for (std::size_t i = lo; i < std::min(v.size(), at + 5); ++i) s << (i == at ? "*" : "") << v[i] << " ";
    return s.str();
}

bool same_ids(const std::vector<int32_t>& got, const std::vector<int32_t>& want, const std::string& what) {
    if (got == want) return true;
    std::size_t i = 0;
    while (i < got.size() && i < want.size() && got[i] == want[i]) ++i;
    check(false, what + ": ids differ at " + std::to_string(i) + " (got len " + std::to_string(got.size()) +
                     ", want " + std::to_string(want.size()) + ")\n  got:  " + show(got, i) +
                     "\n  want: " + show(want, i));
    return false;
}

void test_tokenizer(const brolm::laya::LayaTokenizer& tok, const j::Value& cases) {
    int ok = 0, n = 0;
    for (const auto& c : cases.as_array()) {
        ++n;
        const std::string& text = c.at("text").as_string();
        if (same_ids(tok.encode(text), ints(c.at("ids")), "tokenizer '" + text.substr(0, 40) + "'")) ++ok;
    }
    std::cout << "tokenizer: " << ok << "/" << n << " strings exact\n";
}

// Numeric tolerances. The reference's own bf16 autocast run lands up to
// ~0.23 logits from its fp32 run on these inputs (on a logit of 17); brolm
// computes in FP16 (3 more mantissa bits than bf16) with FP32 accumulation.
// Most questions agree to < 0.02; a saturated logit (|l| ~ 17) moves ~2 %.
constexpr double kLogitAbsTol = 0.15;     // |logit - fp32 ref| ...
constexpr double kLogitRelTol = 0.015;    // ... + this * max |ref logit|
constexpr double kProbAbsTol = 0.02;      // calibrated probability
constexpr double kActProbAbsTol = 1e-3;

struct Worst {
    double logit = 0, logit_bf16 = 0, prob = 0, act = 0;
};

void test_model(brolm::laya::DecisionModel& model, const j::Value& calls) {
    Worst all;
    std::printf("%-20s %4s %9s %9s %9s %9s\n", "call", "q", "dlogit", "bf16ref", "dprob", "dact");
    for (const auto& call : calls.as_array()) {
        const std::string name = call.at("name").as_string();
        const std::string state = call.at("state").as_string();
        brolm::laya::PredictOptions opts;
        opts.max_len = static_cast<int>(call.at("max_len").as_number());
        opts.head_max_len = static_cast<int>(call.at("head_max_len").as_number());
        opts.truncate_left = call.at("truncate_left").as_bool();

        if (const j::Value* compact = call.find("state_compact")) {
            check(brolm::laya::python_json_spacing(compact->as_string()) == state,
                  name + ": python_json_spacing(compact) != json.dumps state");
        }

        const auto questions = brolm::laya::parse_questions_json(call.at("questions_json").as_string());
        const auto& ref_qs = call.at("questions").as_array();
        check(questions.size() == ref_qs.size(), name + ": question count");

        // Sequences (the tokenizer's build_sequence, before the fit check).
        for (std::size_t i = 0; i < questions.size() && i < ref_qs.size(); ++i) {
            const auto seq = model.tokenizer().build_sequence(state, questions[i], opts.max_len,
                                                              opts.head_max_len, opts.truncate_left);
            same_ids(seq.input_ids, ints(ref_qs[i].at("ids")), name + "/" + questions[i].id + " sequence");
            check(seq.marker_pos == ints(ref_qs[i].at("markers")), name + "/" + questions[i].id + " markers");
        }

        if (const j::Value* err = call.find("error")) {
            bool threw = false;
            try {
                model.predict(state, questions, opts);
            } catch (const std::exception& e) {
                threw = std::string(e.what()).find("options do not fit") != std::string::npos;
            }
            check(threw, name + ": expected '" + err->as_string() + "'");
            std::printf("%-20s  raises: options do not fit (ok)\n", name.c_str());
            continue;
        }

        const brolm::laya::LayaResult res = model.predict(state, questions, opts);
        for (std::size_t i = 0; i < questions.size(); ++i) {
            const auto& q = questions[i];
            const auto& r = ref_qs[i];
            const auto& a = res.answers.at(q.id);
            const std::vector<double> l32 = nums(r.at("logits_fp32"));
            const std::vector<double> l16 = nums(r.at("logits_bf16"));
            const std::vector<double> p32 = nums(r.at("probs_fp32"));
            check(a.logits.size() == l32.size(), name + "/" + q.id + ": logit count");
            double dl = 0, dl16 = 0, dp = 0;
            for (std::size_t k = 0; k < l32.size() && k < a.logits.size(); ++k) {
                dl = std::max(dl, std::fabs(a.logits[k] - l32[k]));
                dl16 = std::max(dl16, std::fabs(l16[k] - l32[k]));
                dp = std::max(dp, std::fabs(a.probabilities[k].second - p32[k]));
            }
            const double da = std::fabs(a.act_probability - r.at("act_probability_fp32").as_number());
            check(std::fabs(a.temperature - r.at("temperature").as_number()) < 1e-5,
                  name + "/" + q.id + ": temperature bucket");
            double lmax = 0;
            for (double v : l32) lmax = std::max(lmax, std::fabs(v));
            check(dl <= kLogitAbsTol + kLogitRelTol * lmax, name + "/" + q.id + ": logits off by " + std::to_string(dl));
            check(dp <= kProbAbsTol, name + "/" + q.id + ": probabilities off by " + std::to_string(dp));
            check(da <= kActProbAbsTol, name + "/" + q.id + ": act probability off by " + std::to_string(da));
            // Argmax must agree unless the reference's own top-2 are within tolerance.
            const auto top = [](const std::vector<double>& v) {
                return static_cast<std::size_t>(std::max_element(v.begin(), v.end()) - v.begin());
            };
            std::vector<double> mine(a.logits.begin(), a.logits.end());
            if (top(mine) != top(l32)) {
                std::vector<double> s = l32;
                std::sort(s.rbegin(), s.rend());
                check(s.size() > 1 && s[0] - s[1] <= 2 * kLogitAbsTol, name + "/" + q.id + ": argmax differs");
            }
            std::printf("%-20s %4zu %9.4f %9.4f %9.5f %9.2e\n", (name + "/" + q.id).substr(0, 20).c_str(),
                        l32.size(), dl, dl16, dp, da);
            all.logit = std::max(all.logit, dl);
            all.logit_bf16 = std::max(all.logit_bf16, dl16);
            all.prob = std::max(all.prob, dp);
            all.act = std::max(all.act, da);
        }
    }
    std::printf("worst: |dlogit| %.4f (reference bf16 vs fp32: %.4f), |dprob| %.5f, |dact| %.2e\n",
                all.logit, all.logit_bf16, all.prob, all.act);
}

}  // namespace

int main() {
    const std::string fixture = std::string(BROLM_TEST_REF_DIR) + "/laya_parity.json";
    if (!fs::exists(fixture)) {
        std::cout << "SKIP: fixture not found at " << fixture << "\n";
        return 0;
    }
    const j::Value root = j::parse(slurp(fixture));

    const char* env_dir = std::getenv("LAYA_MODEL_DIR");
    const std::string model_dir = (env_dir && env_dir[0]) ? env_dir : std::string(BROLM_SIBLING_DIR) + "/laya";
    if (!fs::exists(model_dir + "/tokenizer/tokenizer.json")) {
        std::cout << "SKIP: Laya checkpoint not found at " << model_dir << "\n";
        return 0;
    }

    const auto tok = brolm::laya::LayaTokenizer::load(model_dir + "/tokenizer/tokenizer.json");
    test_tokenizer(tok, root.at("tokenizer"));

    if (fs::exists(model_dir + "/model.safetensors")) {
        brotensor::init();
        brolm::laya::DecisionModel model;
        model.load_model(model_dir);
        test_model(model, root.at("calls"));
    } else {
        std::cout << "SKIP model half: no model.safetensors in " << model_dir << "\n";
    }

    if (g_failures) {
        std::cerr << g_failures << " check(s) FAILED\n";
        return 1;
    }
    std::cout << "Laya parity PASSED\n";
    return 0;
}
