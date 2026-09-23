// Laya request scheduler behaviour (answers themselves are checked against
// the torch reference by brolm_test_laya_parity): priority admission, a
// request split across idle devices, deadline accounting, empty requests,
// and shutdown with work still queued. Needs the checkpoint (LAYA_MODEL_DIR
// or the ../laya sibling); skips without it.

#include "brolm/laya_scheduler.h"

#include "brotensor/runtime.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <future>
#include <mutex>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using brolm::laya::RequestOptions;
using brolm::laya::ScheduledResult;
using brolm::laya::Scheduler;
using brolm::laya::SchedulerOptions;

namespace {

int g_failures = 0;

void check(bool ok, const std::string& msg) {
    if (!ok) {
        std::fprintf(stderr, "FAILED: %s\n", msg.c_str());
        ++g_failures;
    }
}

std::vector<brolm::LayaQuestion> questions(int n) {
    std::vector<brolm::LayaQuestion> qs;
    for (int i = 0; i < n; ++i) {
        brolm::LayaQuestion q;
        q.id = "q" + std::to_string(i);
        q.type = "choice";
        q.instructions = "Which team should handle this (rule " + std::to_string(i) + ")?";
        q.criteria_choice = {{"billing", "invoices, refunds"}, {"technical", "bugs, outages"}, {"other", ""}};
        qs.push_back(std::move(q));
    }
    return qs;
}

std::string state(int i, int turns = 1) {
    std::string s = "{\"ticket\": " + std::to_string(i) + ", \"body\": \"";
    for (int t = 0; t < turns; ++t) s += "We were billed twice for March and the export button returns a 502. ";
    return s + "\"}";
}

void test_basics(Scheduler& sched) {
    // Empty request: completes at once with no answers.
    const ScheduledResult empty = sched.predict(state(0), {});
    check(empty.result.answers.empty() && empty.order.empty(), "empty request: no answers");

    // A deadline nobody can meet is reported missed, and counted.
    sched.reset_stats();
    RequestOptions o;
    o.deadline_ms = 0.001;
    const ScheduledResult late = sched.predict(state(1), questions(3), o);
    check(late.timing.deadline_missed, "deadline: 1 us deadline reported missed");
    check(sched.stats().deadline_missed >= 1, "deadline: counted in stats");

    // Option-fit errors throw at submit, before anything is queued.
    auto big = questions(1);
    big[0].criteria_choice.clear();
    for (int k = 0; k < 400; ++k) big[0].criteria_choice.emplace_back("option_" + std::to_string(k), "a long option");
    bool threw = false;
    try {
        sched.submit(state(2), big);
    } catch (const std::exception&) {
        threw = true;
    }
    check(threw, "submit: options that do not fit throw at submit");
}

// High priority overtakes queued low-priority work.
void test_priority(const std::string& dir) {
    SchedulerOptions so;
    so.devices = {Scheduler::all_devices().front()};
    Scheduler sched(dir, so);
    sched.wait_ready();

    std::mutex mu;
    std::vector<std::string> order;
    std::vector<std::future<void>> waits;
    auto submit = [&](const std::string& tag, int prio, int turns, int nq) {
        auto p = std::make_shared<std::promise<void>>();
        waits.push_back(p->get_future());
        RequestOptions o;
        o.priority = prio;
        sched.submit(state(static_cast<int>(waits.size()), turns), questions(nq), o,
                     [&, tag, p](ScheduledResult&&, std::exception_ptr) {
                         {
                             std::lock_guard<std::mutex> lk(mu);
                             order.push_back(tag);
                         }
                         p->set_value();
                     });
    };
    submit("blocker", 0, 8, 20);  // occupies the device for a few forwards
    for (int i = 0; i < 12; ++i) submit("low", 0, 1, 5);
    submit("high", 5, 1, 5);
    for (auto& w : waits) w.get();

    std::size_t at = 0;
    while (at < order.size() && order[at] != "high") ++at;
    int low_before = 0;
    for (std::size_t i = 0; i < at; ++i) low_before += order[i] == "low";
    std::printf("priority: high finished after %d of 12 low-priority requests\n", low_before);
    check(at < order.size() && low_before <= 4, "priority: high-priority request overtakes the queue");
}

// With two devices idle, one large request is split across both.
void test_split(const std::string& dir) {
    const auto devs = Scheduler::all_devices();
    if (devs.size() < 2) {
        std::printf("split: SKIP (one device)\n");
        return;
    }
    SchedulerOptions so;
    so.devices = devs;
    Scheduler sched(dir, so);
    sched.wait_ready();
    sched.reset_stats();
    const ScheduledResult r = sched.predict(state(7, 4), questions(12));
    const auto st = sched.stats();
    int used = 0;
    for (const auto& d : st.devices) used += d.forwards > 0;
    std::printf("split: 12 questions over %d forward(s) on %d device(s), %.2f ms\n", r.timing.forwards, used,
                r.timing.total_ms);
    check(r.timing.forwards >= 2 && used >= 2, "split: a large request spreads over the idle devices");
    check(r.result.answers.size() == 12, "split: every answer delivered");
}

// Destroying the scheduler with work queued completes every request exactly once.
void test_shutdown(const std::string& dir) {
    std::atomic<int> calls{0}, failed{0};
    {
        SchedulerOptions so;
        so.prewarm = false;
        Scheduler sched(dir, so);
        sched.wait_ready();
        for (int i = 0; i < 40; ++i) {
            sched.submit(state(i, 2), questions(5), {}, [&](ScheduledResult&&, std::exception_ptr e) {
                calls.fetch_add(1);
                if (e) failed.fetch_add(1);
            });
        }
    }
    std::printf("shutdown: 40 submitted, %d completed (%d failed as shut down)\n", calls.load(), failed.load());
    check(calls.load() == 40, "shutdown: every request completes exactly once");
}

}  // namespace

int main() {
    const char* env = std::getenv("LAYA_MODEL_DIR");
    const std::string dir = (env && env[0]) ? env : std::string(BROLM_SIBLING_DIR) + "/laya";
    if (!fs::exists(dir + "/model.safetensors")) {
        std::printf("SKIP: Laya checkpoint not found at %s\n", dir.c_str());
        return 0;
    }
    brotensor::init();
    {
        Scheduler sched(dir);
        sched.wait_ready();
        test_basics(sched);
    }
    test_priority(dir);
    test_split(dir);
    test_shutdown(dir);
    if (g_failures) {
        std::fprintf(stderr, "%d check(s) FAILED\n", g_failures);
        return 1;
    }
    std::printf("Laya scheduler PASSED\n");
    return 0;
}
