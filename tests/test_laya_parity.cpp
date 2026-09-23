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
//     reference's own bf16 autocast run lands from fp32;
//   - the same answers through the request scheduler, on every device, with
//     the calls submitted concurrently from several threads.
// Weights are looked up at LAYA_MODEL_DIR or ../laya; the model half skips
// (tokenizer half still runs) when they are absent.

#include "brolm/laya.h"
#include "brolm/laya_scheduler.h"
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
#include <thread>
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

// One answer against its reference question record `r`.
void check_answer(const std::string& name, const brolm::laya::LayaQuestion& q, const brolm::laya::LayaAnswer& a,
                  const j::Value& r, Worst& all, bool print = true) {
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
    if (print) {
        std::printf("%-20s %4zu %9.4f %9.4f %9.5f %9.2e\n", (name + "/" + q.id).substr(0, 20).c_str(), l32.size(),
                    dl, dl16, dp, da);
    }
    all.logit = std::max(all.logit, dl);
    all.logit_bf16 = std::max(all.logit_bf16, dl16);
    all.prob = std::max(all.prob, dp);
    all.act = std::max(all.act, da);
}

// A fixture call kept for the all-calls packed pass.
struct Call {
    std::string name, state;
    brolm::laya::PredictOptions opts;
    std::vector<brolm::laya::LayaQuestion> questions;
    const j::Value* ref;  // the call's "questions" array
};

void test_model(brolm::laya::DecisionModel& model, const j::Value& calls) {
    Worst all;
    std::vector<Call> packed;
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
        packed.push_back(Call{name, state, opts, questions, &call.at("questions")});
        for (std::size_t i = 0; i < questions.size(); ++i) {
            check_answer(name, questions[i], res.answers.at(questions[i].id), ref_qs[i], all);
        }
    }
    std::printf("worst: |dlogit| %.4f (reference bf16 vs fp32: %.4f), |dprob| %.5f, |dact| %.2e\n",
                all.logit, all.logit_bf16, all.prob, all.act);

    // Every call's items packed into ONE forward_items() — the multi-request
    // batch a scheduler builds. Each item must still match its reference.
    std::vector<std::vector<brolm::laya::SequenceResult>> seqs;
    std::vector<brolm::laya::LayaItem> items;
    for (const Call& c : packed) seqs.push_back(model.build_sequences(c.state, c.questions, c.opts));
    for (std::size_t n = 0; n < packed.size(); ++n)
        for (std::size_t i = 0; i < packed[n].questions.size(); ++i)
            items.push_back(brolm::laya::LayaItem::of(seqs[n][i], packed[n].questions[i].qtype_index()));
    const auto outs = model.forward_items(items);
    check(outs.size() == items.size(), "packed: one output per item");
    Worst pk;
    std::size_t k = 0;
    for (const Call& c : packed) {
        const auto& ref_qs = c.ref->as_array();
        for (std::size_t i = 0; i < c.questions.size() && k < outs.size(); ++i, ++k) {
            const auto a = model.answer_from_logits(c.questions[i], outs[k]);
            check_answer("packed/" + c.name, c.questions[i], a, ref_qs[i], pk, /*print=*/false);
        }
    }
    std::printf("packed: %zu items from %zu calls in one forward_items(); worst |dlogit| %.4f, |dprob| %.5f\n",
                items.size(), packed.size(), pk.logit, pk.prob);
}

// The request scheduler over every device: every fixture call submitted
// several times from several threads at once, so items from different calls
// share forwards, big calls split across forwards and devices, and replicas
// on every GPU answer. Each answer must still match its reference.
void test_scheduler(const std::string& model_dir, const j::Value& calls) {
    brolm::laya::SchedulerOptions so;
    so.devices = brolm::laya::Scheduler::all_devices();
    brolm::laya::Scheduler sched(model_dir, so);
    sched.wait_ready();

    struct Job {
        std::string name, state;
        brolm::laya::RequestOptions opts;
        std::vector<brolm::laya::LayaQuestion> questions;
        const j::Value* ref;
    };
    std::vector<Job> jobs;
    for (const auto& call : calls.as_array()) {
        Job jb;
        jb.name = call.at("name").as_string();
        jb.state = call.at("state").as_string();
        jb.opts.predict.max_len = static_cast<int>(call.at("max_len").as_number());
        jb.opts.predict.head_max_len = static_cast<int>(call.at("head_max_len").as_number());
        jb.opts.predict.truncate_left = call.at("truncate_left").as_bool();
        jb.questions = brolm::laya::parse_questions_json(call.at("questions_json").as_string());
        jb.ref = &call.at("questions");
        if (call.find("error")) {
            bool threw = false;
            try {
                sched.submit(jb.state, jb.questions, jb.opts);
            } catch (const std::exception& e) {
                threw = std::string(e.what()).find("options do not fit") != std::string::npos;
            }
            check(threw, "scheduler/" + jb.name + ": option-fit error must throw at submit");
            continue;
        }
        jobs.push_back(std::move(jb));
    }

    constexpr int kThreads = 4, kRounds = 3;
    std::vector<std::vector<std::pair<const Job*, std::future<brolm::laya::ScheduledResult>>>> futs(kThreads);
    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t] {
            for (int r = 0; r < kRounds; ++r) {
                for (std::size_t k = t; k < jobs.size() * 2; k += kThreads) {
                    const Job& jb = jobs[k % jobs.size()];
                    brolm::laya::RequestOptions o = jb.opts;
                    o.priority = static_cast<int>(k % 3);
                    futs[static_cast<std::size_t>(t)].emplace_back(&jb, sched.submit(jb.state, jb.questions, o));
                }
            }
        });
    }
    for (auto& th : threads) th.join();

    Worst all;
    int results = 0;
    for (auto& per : futs) {
        for (auto& [jb, f] : per) {
            brolm::laya::ScheduledResult r;
            try {
                r = f.get();
            } catch (const std::exception& e) {
                check(false, "scheduler/" + jb->name + ": " + e.what());
                continue;
            }
            ++results;
            check(r.order.size() == jb->questions.size(), "scheduler/" + jb->name + ": answer count");
            check(r.timing.total_ms > 0 && r.timing.forwards >= 1, "scheduler/" + jb->name + ": timing");
            const auto& ref_qs = jb->ref->as_array();
            for (std::size_t i = 0; i < jb->questions.size(); ++i) {
                check_answer("sched/" + jb->name, jb->questions[i], r.result.answers.at(jb->questions[i].id),
                             ref_qs[i], all, /*print=*/false);
            }
        }
    }
    const brolm::laya::SchedulerStats st = sched.stats();
    check(st.completed == static_cast<uint64_t>(results) && st.failed == 0, "scheduler: completion count");
    std::printf("scheduler: %d requests over %zu device(s), %llu forwards (mean %.1f items, %.0f rows, budget %d); "
                "worst |dlogit| %.4f, |dprob| %.5f\n",
                results, st.devices.size(), static_cast<unsigned long long>(st.forwards), st.mean_batch_items,
                st.mean_batch_tokens, st.token_budget, all.logit, all.prob);
    for (const auto& d : st.devices) {
        std::printf("  device %d (%s): %llu forwards, %zu graphs\n", d.device, d.name.c_str(),
                    static_cast<unsigned long long>(d.forwards), d.graphs);
    }
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);  // keep the table if a later phase dies
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
        test_scheduler(model_dir, root.at("calls"));
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
