#include "bench_laya_load.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <numeric>
#include <random>
#include <thread>

namespace bench_laya {

using Clock = std::chrono::steady_clock;

namespace {

double pct(const std::vector<double>& sorted, double p) {
    if (sorted.empty()) return 0;
    const double idx = p / 100.0 * static_cast<double>(sorted.size() - 1);
    const std::size_t lo = static_cast<std::size_t>(idx);
    const std::size_t hi = std::min(lo + 1, sorted.size() - 1);
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - static_cast<double>(lo));
}

// Sleep to `t` without the OS timer's millisecond-plus granularity: sleep
// while far, then spin (yielding) for the last stretch.
void wait_until(Clock::time_point t) {
    for (;;) {
        const auto left = t - Clock::now();
        if (left <= Clock::duration::zero()) return;
        if (left > std::chrono::milliseconds(3)) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        else std::this_thread::yield();
    }
}

}  // namespace

OpenLoopResult run_open_loop(brolm::laya::Scheduler& sched, const std::function<std::string(int)>& state_of,
                             const std::vector<brolm::LayaQuestion>& questions, const OpenLoopConfig& cfg) {
    const int clients = std::max(1, cfg.clients);
    const double per_client = cfg.rate_rps / clients;
    // Pre-built states, so client threads only submit.
    const int expected = static_cast<int>(cfg.rate_rps * (cfg.warmup_s + cfg.seconds) * 1.2) + 64;
    std::vector<std::string> states;
    states.reserve(static_cast<std::size_t>(expected));
    for (int i = 0; i < expected; ++i) states.push_back(state_of(i));

    std::mutex mu;
    std::condition_variable cv;
    std::vector<double> lat;
    int window_submitted = 0, window_done = 0, window_missed = 0;
    long long tokens = 0;
    std::atomic<int> next{0};

    const Clock::time_point t0 = Clock::now() + std::chrono::milliseconds(20);
    const Clock::time_point w0 = t0 + std::chrono::microseconds(static_cast<long long>(cfg.warmup_s * 1e6));
    const Clock::time_point w1 = w0 + std::chrono::microseconds(static_cast<long long>(cfg.seconds * 1e6));

    std::vector<std::thread> threads;
    for (int c = 0; c < clients; ++c) {
        threads.emplace_back([&, c] {
            std::mt19937_64 rng(cfg.seed + 7919u * static_cast<unsigned>(c));
            std::exponential_distribution<double> gap(per_client);
            Clock::time_point t = t0 + std::chrono::microseconds(static_cast<long long>(gap(rng) * 1e6));
            while (t < w1) {
                wait_until(t);
                const int i = next.fetch_add(1) % expected;
                const bool in_window = t >= w0;
                if (in_window) {
                    std::lock_guard<std::mutex> lk(mu);
                    ++window_submitted;
                }
                sched.submit(states[static_cast<std::size_t>(i)], questions, {},
                             [&, in_window](brolm::laya::ScheduledResult&& r, std::exception_ptr e) {
                                 if (!in_window) return;
                                 std::lock_guard<std::mutex> lk(mu);
                                 if (!e) {
                                     lat.push_back(r.timing.total_ms);
                                     tokens += r.result.input_tokens;
                                     if (r.timing.deadline_missed) ++window_missed;
                                 }
                                 ++window_done;
                                 cv.notify_all();
                             });
                t += std::chrono::microseconds(static_cast<long long>(gap(rng) * 1e6));
            }
        });
    }
    // Open the stats window with the measured arrivals, close it with them.
    wait_until(w0);
    sched.reset_stats();
    wait_until(w1);
    OpenLoopResult res;
    res.stats = sched.stats();
    for (auto& th : threads) th.join();
    {
        std::unique_lock<std::mutex> lk(mu);
        // No timeout: every submitted request completes, and a completion
        // reaching this frame's locals after return would be a use-after-free.
        cv.wait(lk, [&] { return window_done >= window_submitted; });
    }

    std::lock_guard<std::mutex> lk(mu);
    std::sort(lat.begin(), lat.end());
    res.submitted = window_submitted;
    res.completed = static_cast<int>(lat.size());
    res.tokens = lat.empty() ? 0 : static_cast<int>(tokens / static_cast<long long>(lat.size()));
    res.p50 = pct(lat, 50);
    res.p95 = pct(lat, 95);
    res.p99 = pct(lat, 99);
    res.max = lat.empty() ? 0 : lat.back();
    res.min = lat.empty() ? 0 : lat.front();
    res.mean = lat.empty() ? 0 : std::accumulate(lat.begin(), lat.end(), 0.0) / static_cast<double>(lat.size());
    res.offered_rps = cfg.rate_rps;
    // Achieved: completions inside the measured window per second of it (the
    // scheduler's own window counter). Below the offered rate = a growing backlog.
    res.achieved_rps = res.stats.throughput_rps;
    res.deadline_missed_pct = res.submitted ? 100.0 * window_missed / res.submitted : 0;
    return res;
}

std::vector<OpenLoopResult> sweep_open_loop(brolm::laya::Scheduler& sched,
                                            const std::function<std::string(int)>& state_of,
                                            const std::vector<brolm::LayaQuestion>& questions, OpenLoopConfig cfg,
                                            double start_rps, double step, double p99_ms, double max_rps) {
    std::vector<OpenLoopResult> out;
    for (double r = start_rps; r <= max_rps; r += step) {
        cfg.rate_rps = r;
        out.push_back(run_open_loop(sched, state_of, questions, cfg));
        const OpenLoopResult& c = out.back();
        // Sustained = the stats window's own throughput keeps up with the
        // offered rate and the tail holds.
        const bool keeps_up = c.achieved_rps >= 0.95 * r;
        if (c.p99 > p99_ms || !keeps_up) break;
    }
    return out;
}

}  // namespace bench_laya
