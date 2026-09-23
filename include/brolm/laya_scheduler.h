#pragma once

// Laya request scheduler: many concurrent callers, one packed forward at a
// time per device.
//
// Callers submit (state, questions, options) from any thread and get a
// future or a completion callback. Each device runs one model replica owned
// by one thread; whichever replica is idle packs what is queued into its
// next forward_items() call, so load spreads across devices by itself and a
// burst of small requests becomes one forward.
//
// Latency first, throughput second:
//  * Continuous batching — an idle device launches at once with whatever is
//    queued; it never waits to fill a batch.
//  * Token budget — a forward carries at most `token_budget()` packed rows,
//    derived from a per-device cost model (measured at pre-warm, corrected
//    online) so one forward's estimate stays under `target_forward_ms`. A
//    request that arrives while a forward runs waits at most about one
//    budget-sized forward before its own starts.
//  * Admission order — priority (higher first), then deadline (earliest
//    first), then arrival. The work unit is the (state, question) ITEM, not
//    the request: a large request is split across forwards and, with more
//    than one device idle, across devices, and a batch is shrunk when the
//    most urgent queued item would otherwise miss its deadline behind it.
//  * Pre-warm — every CUDA-graph bucket up to the budget is captured at load,
//    so no live request pays a capture.
//
// Missed deadlines are counted, never dropped: a late answer is still an
// answer, and the caller decides what late means.

#include "brolm/laya.h"

#include <chrono>
#include <cstdint>
#include <exception>
#include <functional>
#include <future>
#include <memory>
#include <string>
#include <vector>

namespace brolm::laya {

struct SchedulerOptions {
    // Device indices to run one replica on each. Empty = the default device
    // only. all_devices() lists every CUDA device (or the default device when
    // there is no CUDA).
    std::vector<int> devices;
    // Hard cap on packed rows per forward, and the size pre-warm covers.
    int max_batch_tokens = 2048;
    // The forward-time target the token budget is derived from (<= 0: the
    // budget is max_batch_tokens).
    double target_forward_ms = 12.0;
    // Deadline of a request that names none, relative to its submit.
    double default_deadline_ms = 30.0;
    // Capture every graph bucket up to max_batch_tokens at load.
    bool prewarm = true;
    // Load-time overrides of the checkpoint's max_len / head_max_len (<= 0: keep).
    int max_len = 0;
    int head_max_len = 0;
};

struct RequestOptions {
    PredictOptions predict;
    int priority = 0;          // higher runs first
    double deadline_ms = 0.0;  // relative to submit; <= 0: SchedulerOptions::default_deadline_ms
};

struct RequestTiming {
    double tokenize_ms = 0;  // on the submitting thread, before queueing
    double queue_ms = 0;     // enqueue -> the first forward carrying one of its items starts
    double forward_ms = 0;   // the forward(s) carrying it, first start -> last end
    double total_ms = 0;     // submit -> result ready (callback / future)
    int forwards = 0;        // forwards its items were spread over
    // The (last) forward that carried it: how full it was.
    int batch_items = 0;
    int batch_requests = 0;
    int batch_tokens = 0;
    int device = 0;          // device index of that forward
    bool deadline_missed = false;
};

struct ScheduledResult {
    LayaResult result;
    std::vector<std::string> order;  // question ids in submit order
    RequestTiming timing;
};

// Called exactly once, on a scheduler thread: `error` is null on success.
using Completion = std::function<void(ScheduledResult&& result, std::exception_ptr error)>;

struct SchedulerBatchRecord {
    uint64_t seq = 0;
    int device = 0;
    int requests = 0;
    int items = 0;
    int tokens = 0;
    int budget = 0;
    double start_ms = 0;  // since the scheduler started
    double ms = 0;
};

struct SchedulerDeviceStats {
    int device = 0;
    std::string name;
    uint64_t forwards = 0;
    double busy_ms = 0;
    double busy_fraction = 0;  // busy_ms / wall time since the stats window opened
    std::size_t graphs = 0;
};

struct SchedulerStats {
    uint64_t submitted = 0, completed = 0, failed = 0, deadline_missed = 0;
    int queued_requests = 0, queued_items = 0, in_flight_requests = 0;
    uint64_t forwards = 0, items_run = 0, tokens_run = 0;
    double mean_batch_items = 0, mean_batch_tokens = 0, mean_batch_requests = 0;
    double mean_occupancy = 0;  // mean batch tokens / token budget
    int token_budget = 0;
    double target_forward_ms = 0;
    double est_fixed_ms = 0, est_ms_per_1k_tokens = 0;  // the live cost model
    // Submit -> result over the most recent completions (up to 4096).
    double latency_p50 = 0, latency_p95 = 0, latency_p99 = 0, latency_max = 0, queue_mean = 0;
    double window_s = 0;         // seconds since the stats window opened
    double throughput_rps = 0;   // completed requests / window_s
    std::vector<SchedulerDeviceStats> devices;
    std::vector<SchedulerBatchRecord> recent_batches;  // oldest first, up to 128
};

class Scheduler {
public:
    // Load the checkpoint at `model_dir` onto every device of `opts`, one
    // replica per device, in parallel on the device threads. Returns at once;
    // wait_ready() / ready() report the load.
    Scheduler(std::string model_dir, SchedulerOptions opts = {});
    // Same with a custom per-replica initialiser (runs on the device thread
    // under that device's DeviceScope) — tests use init_synthetic.
    Scheduler(std::function<void(DecisionModel&)> init, SchedulerOptions opts = {});
    // shutdown().
    ~Scheduler();

    // Fail every request still queued ("shut down before the request ran"),
    // let the running forwards finish, join the device threads and free the
    // replicas' device memory. Idempotent; later submits throw. Must not be
    // called from a completion callback (those run on the device threads),
    // nor race a submit() still tokenizing on another thread.
    void shutdown();
    bool is_shut_down() const;

    Scheduler(const Scheduler&) = delete;
    Scheduler& operator=(const Scheduler&) = delete;

    static std::vector<int> all_devices();

    // Block until every replica is loaded and pre-warmed; rethrows a load error.
    void wait_ready();
    // Non-blocking: true once loaded (or failed — then `error` is set).
    bool poll_ready(std::string* error = nullptr) const;

    // Tokenize on the calling thread (so an option-fit error throws here,
    // before anything is queued) and queue the request. Blocks until ready.
    void submit(const std::string& state, std::vector<LayaQuestion> questions, const RequestOptions& opts,
                Completion done);
    std::future<ScheduledResult> submit(const std::string& state, std::vector<LayaQuestion> questions,
                                        const RequestOptions& opts = {});
    // submit + wait.
    ScheduledResult predict(const std::string& state, std::vector<LayaQuestion> questions,
                            const RequestOptions& opts = {});

    SchedulerStats stats() const;
    void reset_stats();

    int replicas() const;
    std::vector<int> devices() const;
    int token_budget() const;
    const SchedulerOptions& options() const;
    // Replica 0 (config, tokenizer). Valid once ready.
    const DecisionModel& model() const;

    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};

}  // namespace brolm::laya

namespace brolm {
using LayaScheduler = laya::Scheduler;
using LayaSchedulerOptions = laya::SchedulerOptions;
using LayaRequestOptions = laya::RequestOptions;
}  // namespace brolm
