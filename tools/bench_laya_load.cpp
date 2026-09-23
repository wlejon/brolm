#include "bench_laya_load.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <mutex>
#include <numeric>
#include <random>
#include <thread>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <timeapi.h>
#endif

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
// while far, then spin (yielding) for the last stretch. On Windows a 1 ms
// sleep can last a whole 15.6 ms system tick unless the timer period is
// raised (TimerResolution below), releasing a late client's arrivals in a
// clump. `late_ms` in each sample checks that the offered process held.
void wait_until(Clock::time_point t) {
    for (;;) {
        const auto left = t - Clock::now();
        if (left <= Clock::duration::zero()) return;
        if (left > std::chrono::milliseconds(3)) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        else std::this_thread::yield();
    }
}

// A 1 ms system timer period for the duration of one open-loop run.
struct TimerResolution {
#ifdef _WIN32
    TimerResolution() { timeBeginPeriod(1); }
    ~TimerResolution() { timeEndPeriod(1); }
#endif
};

}  // namespace

OpenLoopResult run_open_loop(brolm::laya::Scheduler& sched, const std::function<std::string(int)>& state_of,
                             const std::vector<brolm::LayaQuestion>& questions, const OpenLoopConfig& cfg) {
    const TimerResolution timer_resolution;
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
    std::vector<OpenLoopSample> samples;
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
                OpenLoopSample s;
                s.arrival_ms = std::chrono::duration<double, std::milli>(t - w0).count();
                s.late_ms = std::chrono::duration<double, std::milli>(Clock::now() - t).count();
                sched.submit(states[static_cast<std::size_t>(i)], questions, {},
                             [&, in_window, s](brolm::laya::ScheduledResult&& r, std::exception_ptr e) mutable {
                                 if (!in_window) return;
                                 std::lock_guard<std::mutex> lk(mu);
                                 if (!e) {
                                     lat.push_back(r.timing.total_ms);
                                     tokens += r.result.input_tokens;
                                     if (r.timing.deadline_missed) ++window_missed;
                                     const auto& tm = r.timing;
                                     s.total_ms = tm.total_ms;
                                     s.tokenize_ms = tm.tokenize_ms;
                                     s.queue_ms = tm.queue_ms;
                                     s.forward_ms = tm.forward_ms;
                                     s.forwards = tm.forwards;
                                     s.batch_tokens = tm.batch_tokens;
                                     s.batch_requests = tm.batch_requests;
                                     s.device = tm.device;
                                     samples.push_back(s);
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
    res.samples = std::move(samples);
    return res;
}

void print_tail(const OpenLoopResult& r, const char* key) {
    std::vector<OpenLoopSample> s = r.samples;
    if (s.size() < 50) return;
    std::sort(s.begin(), s.end(), [](const auto& a, const auto& b) { return a.total_ms < b.total_ms; });
    const std::size_t n = s.size(), cut = n - std::max<std::size_t>(1, n / 100);
    // Arrivals in the 10 ms before each request (a clump ahead of it).
    std::vector<double> arrivals;
    for (const auto& x : r.samples) arrivals.push_back(x.arrival_ms);
    std::sort(arrivals.begin(), arrivals.end());
    auto ahead = [&](double at) {
        return static_cast<int>(std::lower_bound(arrivals.begin(), arrivals.end(), at) -
                                std::lower_bound(arrivals.begin(), arrivals.end(), at - 10.0));
    };
    struct Agg {
        double total = 0, late = 0, tok = 0, queue = 0, fwd = 0, fwds = 0, rows = 0, reqs = 0, clump = 0;
        int n = 0;
        void add(const OpenLoopSample& x, int c) {
            total += x.total_ms; late += x.late_ms; tok += x.tokenize_ms; queue += x.queue_ms;
            fwd += x.forward_ms; fwds += x.forwards; rows += x.batch_tokens; reqs += x.batch_requests; clump += c;
            ++n;
        }
    } mid, tail;
    for (std::size_t i = n * 45 / 100; i < n * 55 / 100; ++i) mid.add(s[i], ahead(s[i].arrival_ms));
    for (std::size_t i = cut; i < n; ++i) tail.add(s[i], ahead(s[i].arrival_ms));
    auto row = [](const char* what, const Agg& a) {
        const double k = a.n ? 1.0 / a.n : 0;
        std::printf("  %-8s %7.2f %7.2f %7.2f %7.2f %7.2f %6.2f %6.0f %6.1f %6.1f\n", what, a.total * k, a.late * k,
                    a.tok * k, a.queue * k, a.fwd * k, a.fwds * k, a.rows * k, a.reqs * k, a.clump * k);
    };
    // Late submits: the load generator's own timer (a client that overslept
    // releases its backlog of arrivals at once).
    double late_max = 0;
    int late_over_2ms = 0;
    for (const auto& x : r.samples) {
        late_max = std::max(late_max, x.late_ms);
        late_over_2ms += x.late_ms > 2.0;
    }
    std::printf("\ntail of %s: %zu requests, slowest %zu (>= %.1f ms) vs the middle 10 %%\n", key, n, n - cut,
                s[cut].total_ms);
    std::printf("  %-8s %7s %7s %7s %7s %7s %6s %6s %6s %6s\n", "", "total", "late", "token", "queue", "fwd",
                "fwds", "rows", "reqs", "ahead");
    row("median", mid);
    row("tail", tail);
    std::printf("  (ms; late = submit after its Poisson arrival; fwds = forwards it spanned; rows / reqs = its "
                "last forward; ahead = arrivals in the 10 ms before it)\n");
    std::printf("  submits > 2 ms late: %d of %zu, worst %.1f ms\n", late_over_2ms, r.samples.size(), late_max);
    std::printf("  slowest:");
    for (std::size_t i = n; i-- > cut && i + 5 >= n;) {
        const auto& x = s[i];
        std::printf(" [%.1f ms @%.0f: late %.1f tok %.1f queue %.1f fwd %.1f x%d dev%d]", x.total_ms, x.arrival_ms,
                    x.late_ms, x.tokenize_ms, x.queue_ms, x.forward_ms, x.forwards, x.device);
    }
    std::printf("\n");
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
