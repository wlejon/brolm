#pragma once

// Open-loop load cells for brolm_bench_laya: clients submitting to the Laya
// request scheduler at a fixed offered rate regardless of how fast answers
// come back — the arrival process a live service sees — and the latency from
// submit to result that produces.

#include "brolm/laya.h"
#include "brolm/laya_scheduler.h"

#include <functional>
#include <string>
#include <vector>

namespace bench_laya {

struct OpenLoopConfig {
    int clients = 8;           // submitting threads; each a Poisson process at rate / clients
    double rate_rps = 100;     // offered requests per second, all clients together
    double seconds = 3.0;      // measured window
    double warmup_s = 0.5;     // arrivals before the window, not measured
    unsigned seed = 1234;
};

// One measured request, for tail analysis (--tail).
struct OpenLoopSample {
    double arrival_ms = 0;  // its Poisson arrival time, since the window opened
    double late_ms = 0;     // how far after that arrival the client actually submitted
    double total_ms = 0, tokenize_ms = 0, queue_ms = 0, forward_ms = 0;
    int forwards = 0, batch_tokens = 0, batch_requests = 0, device = 0;
};

struct OpenLoopResult {
    int submitted = 0;      // in the window
    int completed = 0;      // of those
    int tokens = 0;         // mean input tokens per request
    double p50 = 0, p95 = 0, p99 = 0, max = 0, mean = 0, min = 0;  // submit -> result, ms
    double offered_rps = 0, achieved_rps = 0;
    double deadline_missed_pct = 0;  // of window requests, against the scheduler's deadline
    brolm::laya::SchedulerStats stats;  // over the window
    std::vector<OpenLoopSample> samples;  // every completed window request
};

// The slowest 1 % of a cell's requests, taken apart: where their time went
// (submit lateness, tokenize, queue, forward) and what else arrived in the
// few ms before them, against the median request.
void print_tail(const OpenLoopResult& r, const char* key);

// Run one cell. `state_of(i)` is request i's state; every request asks `questions`.
OpenLoopResult run_open_loop(brolm::laya::Scheduler& sched, const std::function<std::string(int)>& state_of,
                             const std::vector<brolm::LayaQuestion>& questions, const OpenLoopConfig& cfg);

// Highest offered rate (stepping by `step` from `start`) whose p99 stays at
// or under `p99_ms` with the achieved rate within 5 % of offered. Returns
// every cell tried, last one the first failure (or the cap).
std::vector<OpenLoopResult> sweep_open_loop(brolm::laya::Scheduler& sched,
                                            const std::function<std::string(int)>& state_of,
                                            const std::vector<brolm::LayaQuestion>& questions, OpenLoopConfig cfg,
                                            double start_rps, double step, double p99_ms, double max_rps);

}  // namespace bench_laya
