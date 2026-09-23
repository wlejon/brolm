// Laya request scheduler — see include/brolm/laya_scheduler.h for the policy.
//
// Threads: callers tokenize and enqueue under `mu`; one thread per device
// owns its replica (DecisionModel is single-owner) and loops pick -> forward
// -> complete. Only picking and the stats touch `mu`; the forward, the
// calibration and the completion callbacks run unlocked. A request's items
// may be in several forwards at once (on several devices): each writes only
// its own answer slots, and the forward that drops `remaining` to zero
// completes the request.

#include "brolm/laya_scheduler.h"

#include "brotensor/runtime.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

namespace brolm::laya {

namespace bt = ::brotensor;
using Clock = std::chrono::steady_clock;

namespace {

double ms_between(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

// Below this many rows a split across idle devices costs more in fixed
// per-forward overhead than it saves.
constexpr int kMinSplitTokens = 512;
constexpr std::size_t kLatencyWindow = 4096;
constexpr std::size_t kBatchWindow = 128;

}  // namespace

// One submitted call (this TU only).
struct SchedRequest {
    uint64_t seq = 0;
    std::vector<LayaQuestion> questions;
    std::vector<SequenceResult> seqs;
    std::vector<int> item_tokens;
    std::vector<LayaAnswer> answers;  // one slot per item, written by the forward that ran it
    int next_item = 0;                // first item not yet taken (under Impl::mu)
    int input_tokens = 0;
    int priority = 0;
    Clock::time_point submit_t, deadline;
    double tokenize_ms = 0;
    Completion done;

    std::atomic<int> remaining{0};
    std::mutex mu;  // everything below
    std::exception_ptr error;
    bool started = false;
    Clock::time_point first_start, last_end;
    RequestTiming timing;
};
using Request = SchedRequest;

struct Scheduler::Impl {
    struct Worker {
        int device = 0;
        std::string name;
        DecisionModel model;
        std::thread thread;
        std::vector<DecisionModel::WarmPoint> warm;
        uint64_t forwards = 0;
        double busy_ms = 0;
        std::size_t graphs = 0;
    };

    SchedulerOptions opts;
    std::function<void(DecisionModel&)> init;
    std::vector<std::unique_ptr<Worker>> workers;

    mutable std::mutex mu;
    std::condition_variable work_cv, ready_cv;
    std::vector<std::shared_ptr<Request>> queue;  // requests with untaken items
    int queued_tokens = 0, queued_items = 0;
    int idle = 0, loaded = 0;
    bool ready = false, stop = false, shut = false;
    std::string load_error;
    uint64_t next_seq = 0, batch_seq = 0;

    // Cost model: forward ms ~= corr * (fit_a + fit_b * bucket(rows)).
    double fit_a = 2.0, fit_b = 0.008, corr = 1.0;
    int budget = 2048;

    // Stats window.
    Clock::time_point t_start = Clock::now(), window_start = Clock::now();
    uint64_t submitted = 0, completed = 0, failed = 0, missed = 0, outstanding = 0;
    uint64_t forwards = 0, items_run = 0, tokens_run = 0, requests_run = 0;
    double occupancy_sum = 0, queue_sum = 0;
    std::vector<double> latencies;
    std::size_t latency_at = 0;
    std::vector<SchedulerBatchRecord> batches;
    std::size_t batch_at = 0;

    double est_ms(int rows) const { return corr * (fit_a + fit_b * DecisionModel::token_bucket(rows)); }
    int rows_for_ms(double ms) const {
        const double r = (ms / std::max(corr, 1e-3) - fit_a) / std::max(fit_b, 1e-6);
        return r <= 0 ? 0 : static_cast<int>(std::min(r, 1e9));
    }
    // The largest row count whose estimate fits the forward target, never
    // below one full-length item (an item is never split). Recomputed as the
    // online correction learns what real batches cost: the pre-warm fit uses
    // synthetic long items, and real batches of many short items run slower
    // per token (more scorer rows, more attention segments).
    int floor_rows = 512;
    void refresh_budget() {
        budget = opts.target_forward_ms > 0
                     ? std::clamp(rows_for_ms(opts.target_forward_ms), floor_rows, opts.max_batch_tokens)
                     : opts.max_batch_tokens;
    }

    void start(std::vector<int> devs);
    void run(Worker& w);
    void finish_load();
    struct Taken {
        std::shared_ptr<Request> req;
        int item;
    };
    std::vector<Taken> pick(int& budget_out);
    void forward(Worker& w, std::vector<Taken>& batch, int batch_budget);
    void release(const std::shared_ptr<Request>& req, int items, std::exception_ptr err);
    void complete(Request& req);
};

// ─── Load ─────────────────────────────────────────────────────────────────

std::vector<int> Scheduler::all_devices() {
    bt::init();
    const int n = bt::default_device().type == bt::DeviceType::CUDA ? bt::cuda_device_count() : 0;
    std::vector<int> out;
    for (int i = 0; i < n; ++i) out.push_back(i);
    if (out.empty()) out.push_back(bt::default_device().index);
    return out;
}

Scheduler::Scheduler(std::string model_dir, SchedulerOptions opts)
    : Scheduler([dir = std::move(model_dir)](DecisionModel& m) { m.load_model(dir); }, std::move(opts)) {}

Scheduler::Scheduler(std::function<void(DecisionModel&)> init, SchedulerOptions opts) : impl_(new Impl) {
    bt::init();
    impl_->opts = std::move(opts);
    impl_->init = std::move(init);
    std::vector<int> devs = impl_->opts.devices;
    if (devs.empty()) devs.push_back(bt::default_device().index);
    std::sort(devs.begin(), devs.end());
    devs.erase(std::unique(devs.begin(), devs.end()), devs.end());
    if (bt::default_device().type == bt::DeviceType::CUDA) {
        for (int d : devs) {
            if (d < 0 || d >= bt::cuda_device_count()) {
                throw std::invalid_argument("laya::Scheduler: no CUDA device " + std::to_string(d) + " (" +
                                            std::to_string(bt::cuda_device_count()) + " present)");
            }
        }
    } else if (devs.size() > 1) {
        devs.resize(1);  // one CPU replica
    }
    impl_->opts.devices = devs;
    impl_->opts.max_batch_tokens = std::max(64, impl_->opts.max_batch_tokens);
    impl_->budget = impl_->opts.max_batch_tokens;
    impl_->latencies.reserve(kLatencyWindow);
    impl_->start(devs);
}

void Scheduler::Impl::start(std::vector<int> devs) {
    for (int d : devs) {
        auto w = std::make_unique<Worker>();
        w->device = d;
        workers.push_back(std::move(w));
    }
    for (auto& w : workers) {
        Worker* wp = w.get();
        wp->thread = std::thread([this, wp] { run(*wp); });
    }
}

void Scheduler::Impl::finish_load() {
    // Least-squares line through every replica's pre-warm points.
    double n = 0, sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (auto& w : workers) {
        for (const auto& p : w->warm) {
            n += 1;
            sx += p.tokens;
            sy += p.ms;
            sxx += static_cast<double>(p.tokens) * p.tokens;
            sxy += p.tokens * p.ms;
        }
    }
    if (n >= 2 && n * sxx - sx * sx > 0) {
        fit_b = std::max(1e-6, (n * sxy - sx * sy) / (n * sxx - sx * sx));
        fit_a = std::max(0.0, (sy - fit_b * sx) / n);
    }
    floor_rows = std::min(DecisionModel::token_bucket(workers.front()->model.config().max_len), opts.max_batch_tokens);
    refresh_budget();
    t_start = window_start = Clock::now();
    ready = true;
}

void Scheduler::Impl::run(Worker& w) {
    try {
        std::unique_ptr<bt::DeviceScope> scope;
        if (bt::default_device().type == bt::DeviceType::CUDA) {
            scope = std::make_unique<bt::DeviceScope>(bt::Device::cuda(w.device));
        }
        w.name = bt::device_product_name(bt::default_device());
        init(w.model);
        if (opts.max_len > 0) w.model.mutable_config().max_len = opts.max_len;
        if (opts.head_max_len > 0) w.model.mutable_config().head_max_len = opts.head_max_len;
        if (opts.prewarm) w.warm = w.model.prewarm_graphs(opts.max_batch_tokens);
        w.graphs = w.model.cached_graphs();

        {
            std::unique_lock<std::mutex> lk(mu);
            if (++loaded == static_cast<int>(workers.size())) {
                if (load_error.empty()) finish_load();
                else ready = true;
                ready_cv.notify_all();
            }
            ready_cv.wait(lk, [&] { return ready; });
            if (!load_error.empty()) return;
        }

        for (;;) {
            std::vector<Taken> batch;
            int batch_budget = 0;
            {
                std::unique_lock<std::mutex> lk(mu);
                ++idle;
                work_cv.wait(lk, [&] { return stop || !queue.empty(); });
                if (stop) return;  // the destructor fails what is still queued
                batch = pick(batch_budget);
                --idle;
                if (!queue.empty() && idle > 0) work_cv.notify_one();
            }
            forward(w, batch, batch_budget);
        }
    } catch (const std::exception& e) {
        std::lock_guard<std::mutex> lk(mu);
        if (load_error.empty()) load_error = "device " + std::to_string(w.device) + ": " + e.what();
        if (!ready && ++loaded == static_cast<int>(workers.size())) {
            ready = true;
            ready_cv.notify_all();
        }
    }
}

// ─── Picking ──────────────────────────────────────────────────────────────

std::vector<Scheduler::Impl::Taken> Scheduler::Impl::pick(int& budget_out) {
    std::sort(queue.begin(), queue.end(), [](const auto& a, const auto& b) {
        if (a->priority != b->priority) return a->priority > b->priority;
        if (a->deadline != b->deadline) return a->deadline < b->deadline;
        return a->seq < b->seq;
    });

    int cap = budget;
    // More than one device idle (this one included): take a fair share so
    // the others can start on the rest now instead of after this forward.
    const int idle_now = std::max(1, idle);
    if (idle_now > 1) cap = std::min(cap, std::max(kMinSplitTokens, (queued_tokens + idle_now - 1) / idle_now));
    // The most urgent item must not miss its deadline behind its batch-mates:
    // shrink the batch when that saves it. When even a forward of its own
    // cannot make the deadline it is late either way, and shrinking would
    // only cut throughput while the queue is already behind — so don't.
    const Request& head = *queue.front();
    const int head_rows = head.item_tokens[static_cast<std::size_t>(head.next_item)];
    const double slack = ms_between(Clock::now(), head.deadline);
    if (est_ms(cap) > slack && est_ms(head_rows) <= slack) cap = std::min(cap, rows_for_ms(slack));
    cap = std::max(cap, head_rows);
    budget_out = budget;

    std::vector<Taken> out;
    int rows = 0;
    for (auto& r : queue) {
        const int n = static_cast<int>(r->item_tokens.size());
        // Past the first request, take a request whole or leave it for the
        // next forward. A request cut at the budget waits for that next
        // forward with its answers half done — two forwards of latency, the
        // open-loop p99 tail — while the rows it would have filled are at
        // most one request's worth of occupancy. The head still splits (it
        // may be bigger than any budget, and a fair-share cap spreads it
        // across idle devices).
        if (!out.empty()) {
            int left = 0;
            for (int i = r->next_item; i < n; ++i) left += r->item_tokens[static_cast<std::size_t>(i)];
            if (rows + left > cap) break;
        }
        while (r->next_item < n) {
            const int t = r->item_tokens[static_cast<std::size_t>(r->next_item)];
            if (!out.empty() && rows + t > cap) break;
            out.push_back(Taken{r, r->next_item});
            ++r->next_item;
            rows += t;
            queued_tokens -= t;
            --queued_items;
        }
        if (rows >= cap) break;
    }
    queue.erase(std::remove_if(queue.begin(), queue.end(),
                               [](const auto& r) { return r->next_item >= static_cast<int>(r->item_tokens.size()); }),
                queue.end());
    return out;
}

// ─── Running ──────────────────────────────────────────────────────────────

void Scheduler::Impl::forward(Worker& w, std::vector<Taken>& batch, int batch_budget) {
    std::vector<LayaItem> items;
    items.reserve(batch.size());
    int rows = 0;
    for (const Taken& t : batch) {
        const Request& r = *t.req;
        const auto i = static_cast<std::size_t>(t.item);
        items.push_back(LayaItem::of(r.seqs[i], r.questions[i].qtype_index()));
        rows += r.item_tokens[i];
    }
    // Distinct requests in this batch, with how many of their items it carries.
    std::vector<std::pair<std::shared_ptr<Request>, int>> reqs;
    for (const Taken& t : batch) {
        if (reqs.empty() || reqs.back().first != t.req) reqs.emplace_back(t.req, 0);
        ++reqs.back().second;
    }

    const Clock::time_point t0 = Clock::now();
    std::exception_ptr err;
    try {
        const std::vector<LayaItemLogits> outs = w.model.forward_items(items);
        for (std::size_t k = 0; k < batch.size(); ++k) {
            Request& r = *batch[k].req;
            const auto i = static_cast<std::size_t>(batch[k].item);
            r.answers[i] = w.model.answer_from_logits(r.questions[i], outs[k]);
        }
    } catch (...) {
        err = std::current_exception();
    }
    const Clock::time_point t1 = Clock::now();
    const double ms = ms_between(t0, t1);

    {
        std::lock_guard<std::mutex> lk(mu);
        ++forwards;
        items_run += batch.size();
        tokens_run += static_cast<uint64_t>(rows);
        requests_run += reqs.size();
        occupancy_sum += static_cast<double>(rows) / std::max(1, batch_budget);
        ++w.forwards;
        w.busy_ms += ms;
        w.graphs = w.model.cached_graphs();
        if (!err) {
            const double ratio = ms / std::max(1e-3, fit_a + fit_b * DecisionModel::token_bucket(rows));
            corr = std::clamp(0.95 * corr + 0.05 * ratio, 0.25, 4.0);
            refresh_budget();
        }
        SchedulerBatchRecord rec{++batch_seq, w.device, static_cast<int>(reqs.size()),
                                 static_cast<int>(batch.size()), rows, batch_budget, ms_between(t_start, t0), ms};
        if (batches.size() < kBatchWindow) batches.push_back(rec);
        else batches[batch_at] = rec;
        batch_at = (batch_at + 1) % kBatchWindow;
    }

    for (auto& [req, n] : reqs) {
        {
            std::lock_guard<std::mutex> lk(req->mu);
            if (!req->started) {
                req->started = true;
                req->first_start = t0;
            }
            req->last_end = t1;
            ++req->timing.forwards;
            req->timing.batch_items = static_cast<int>(batch.size());
            req->timing.batch_requests = static_cast<int>(reqs.size());
            req->timing.batch_tokens = rows;
            req->timing.device = w.device;
        }
        release(req, n, err);
    }
}

void Scheduler::Impl::release(const std::shared_ptr<Request>& req, int items, std::exception_ptr err) {
    if (err) {
        std::lock_guard<std::mutex> lk(req->mu);
        if (!req->error) req->error = err;
    }
    if (req->remaining.fetch_sub(items, std::memory_order_acq_rel) == items) complete(*req);
}

void Scheduler::Impl::complete(Request& req) {
    const Clock::time_point now = Clock::now();
    ScheduledResult out;
    std::exception_ptr err;
    {
        std::lock_guard<std::mutex> lk(req.mu);
        err = req.error;
        RequestTiming& t = req.timing;
        t.tokenize_ms = req.tokenize_ms;
        t.queue_ms = req.started ? ms_between(req.submit_t, req.first_start) - req.tokenize_ms : 0;
        t.forward_ms = req.started ? ms_between(req.first_start, req.last_end) : 0;
        t.total_ms = ms_between(req.submit_t, now);
        t.deadline_missed = now > req.deadline;
        out.timing = t;
    }
    if (!err) {
        out.result.input_tokens = req.input_tokens;
        for (std::size_t i = 0; i < req.questions.size(); ++i) {
            out.order.push_back(req.questions[i].id);
            out.result.answers[req.questions[i].id] = std::move(req.answers[i]);
        }
    }
    {
        std::lock_guard<std::mutex> lk(mu);
        --outstanding;
        if (err) {
            ++failed;
        } else {
            ++completed;
            if (out.timing.deadline_missed) ++missed;
            queue_sum += out.timing.queue_ms;
            if (latencies.size() < kLatencyWindow) latencies.push_back(out.timing.total_ms);
            else latencies[latency_at] = out.timing.total_ms;
            latency_at = (latency_at + 1) % kLatencyWindow;
        }
    }
    Completion done = std::move(req.done);
    try {
        done(std::move(out), err);
    } catch (...) {
        // A throwing completion must not take the device thread down.
    }
}

// ─── Public surface ───────────────────────────────────────────────────────

Scheduler::~Scheduler() { shutdown(); }

void Scheduler::shutdown() {
    for (auto& w : impl_->workers) {
        if (w->thread.get_id() == std::this_thread::get_id()) {
            throw std::logic_error("laya::Scheduler: shutdown from a completion callback (a device thread)");
        }
    }
    std::vector<std::shared_ptr<Request>> orphans;
    {
        std::unique_lock<std::mutex> lk(impl_->mu);
        if (impl_->shut) return;
        impl_->ready_cv.wait(lk, [&] { return impl_->ready; });  // never tear down mid-load
        impl_->stop = impl_->shut = true;
        orphans.swap(impl_->queue);
        impl_->queued_items = impl_->queued_tokens = 0;
    }
    impl_->work_cv.notify_all();
    for (auto& w : impl_->workers) {
        if (w->thread.joinable()) w->thread.join();
    }
    const auto err = std::make_exception_ptr(std::runtime_error("laya::Scheduler: shut down before the request ran"));
    for (auto& r : orphans) {
        const int untaken = static_cast<int>(r->item_tokens.size()) - r->next_item;
        impl_->release(r, untaken, err);
    }
    // Free the replicas' device memory now, not whenever the last owner of
    // this object lets go (for a JS handle, a GC that may come after the
    // device runtime is gone). The config stays readable through model().
    for (auto& w : impl_->workers) {
        const Config cfg = w->model.config();
        w->model = DecisionModel();
        w->model.mutable_config() = cfg;
    }
}

bool Scheduler::is_shut_down() const {
    std::lock_guard<std::mutex> lk(impl_->mu);
    return impl_->shut;
}

void Scheduler::wait_ready() {
    std::unique_lock<std::mutex> lk(impl_->mu);
    impl_->ready_cv.wait(lk, [&] { return impl_->ready; });
    if (!impl_->load_error.empty()) throw std::runtime_error("laya::Scheduler: load failed: " + impl_->load_error);
}

bool Scheduler::poll_ready(std::string* error) const {
    std::lock_guard<std::mutex> lk(impl_->mu);
    if (error) *error = impl_->load_error;
    return impl_->ready;
}

void Scheduler::submit(const std::string& state, std::vector<LayaQuestion> questions, const RequestOptions& opts,
                       Completion done) {
    wait_ready();
    if (is_shut_down()) throw std::runtime_error("laya::Scheduler: shut down");
    auto req = std::make_shared<Request>();
    req->submit_t = Clock::now();
    const double deadline_ms = opts.deadline_ms > 0 ? opts.deadline_ms : impl_->opts.default_deadline_ms;
    req->deadline = req->submit_t + std::chrono::microseconds(static_cast<int64_t>(deadline_ms * 1000.0));
    req->priority = opts.priority;
    req->seqs = model().build_sequences(state, questions, opts.predict);  // throws before queueing
    req->questions = std::move(questions);
    req->done = std::move(done);
    for (const auto& s : req->seqs) {
        req->item_tokens.push_back(static_cast<int>(s.input_ids.size()));
        req->input_tokens += req->item_tokens.back();
    }
    req->answers.resize(req->seqs.size());
    const int n = static_cast<int>(req->seqs.size());
    req->remaining.store(n, std::memory_order_relaxed);
    req->tokenize_ms = ms_between(req->submit_t, Clock::now());

    {
        std::lock_guard<std::mutex> lk(impl_->mu);
        if (impl_->stop) throw std::runtime_error("laya::Scheduler: shut down");
        req->seq = ++impl_->next_seq;
        ++impl_->submitted;
        ++impl_->outstanding;
        if (n > 0) {
            impl_->queue.push_back(req);
            impl_->queued_tokens += req->input_tokens;
            impl_->queued_items += n;
        }
    }
    if (n == 0) {
        impl_->complete(*req);
        return;
    }
    impl_->work_cv.notify_one();
}

std::future<ScheduledResult> Scheduler::submit(const std::string& state, std::vector<LayaQuestion> questions,
                                               const RequestOptions& opts) {
    auto p = std::make_shared<std::promise<ScheduledResult>>();
    std::future<ScheduledResult> f = p->get_future();
    submit(state, std::move(questions), opts, [p](ScheduledResult&& r, std::exception_ptr e) {
        if (e) p->set_exception(e);
        else p->set_value(std::move(r));
    });
    return f;
}

ScheduledResult Scheduler::predict(const std::string& state, std::vector<LayaQuestion> questions,
                                   const RequestOptions& opts) {
    return submit(state, std::move(questions), opts).get();
}

SchedulerStats Scheduler::stats() const {
    const Impl& m = *impl_;
    std::lock_guard<std::mutex> lk(m.mu);
    SchedulerStats s;
    const Clock::time_point now = Clock::now();
    s.submitted = m.submitted;
    s.completed = m.completed;
    s.failed = m.failed;
    s.deadline_missed = m.missed;
    s.queued_requests = static_cast<int>(m.queue.size());
    s.queued_items = m.queued_items;
    s.in_flight_requests = static_cast<int>(m.outstanding);
    s.forwards = m.forwards;
    s.items_run = m.items_run;
    s.tokens_run = m.tokens_run;
    if (m.forwards) {
        const double f = static_cast<double>(m.forwards);
        s.mean_batch_items = static_cast<double>(m.items_run) / f;
        s.mean_batch_tokens = static_cast<double>(m.tokens_run) / f;
        s.mean_batch_requests = static_cast<double>(m.requests_run) / f;
        s.mean_occupancy = m.occupancy_sum / f;
    }
    s.token_budget = m.budget;
    s.target_forward_ms = m.opts.target_forward_ms;
    s.est_fixed_ms = m.corr * m.fit_a;
    s.est_ms_per_1k_tokens = m.corr * m.fit_b * 1000.0;
    if (!m.latencies.empty()) {
        std::vector<double> v = m.latencies;
        std::sort(v.begin(), v.end());
        auto pct = [&](double p) {
            return v[std::min(v.size() - 1, static_cast<std::size_t>(p / 100.0 * static_cast<double>(v.size())))];
        };
        s.latency_p50 = pct(50);
        s.latency_p95 = pct(95);
        s.latency_p99 = pct(99);
        s.latency_max = v.back();
    }
    if (m.completed) s.queue_mean = m.queue_sum / static_cast<double>(m.completed);
    s.window_s = ms_between(m.window_start, now) / 1000.0;
    if (s.window_s > 0) s.throughput_rps = static_cast<double>(m.completed) / s.window_s;
    for (const auto& w : m.workers) {
        SchedulerDeviceStats d;
        d.device = w->device;
        d.name = w->name;
        d.forwards = w->forwards;
        d.busy_ms = w->busy_ms;
        d.busy_fraction = s.window_s > 0 ? w->busy_ms / (s.window_s * 1000.0) : 0;
        d.graphs = w->graphs;
        s.devices.push_back(std::move(d));
    }
    s.recent_batches.reserve(m.batches.size());
    for (std::size_t i = 0; i < m.batches.size(); ++i) {
        s.recent_batches.push_back(m.batches[(m.batch_at + i) % m.batches.size()]);
    }
    return s;
}

void Scheduler::reset_stats() {
    Impl& m = *impl_;
    std::lock_guard<std::mutex> lk(m.mu);
    m.window_start = Clock::now();
    m.submitted = m.completed = m.failed = m.missed = 0;
    m.forwards = m.items_run = m.tokens_run = m.requests_run = 0;
    m.occupancy_sum = m.queue_sum = 0;
    m.latencies.clear();
    m.latency_at = 0;
    m.batches.clear();
    m.batch_at = 0;
    for (auto& w : m.workers) {
        w->forwards = 0;
        w->busy_ms = 0;
    }
}

int Scheduler::replicas() const { return static_cast<int>(impl_->workers.size()); }

std::vector<int> Scheduler::devices() const { return impl_->opts.devices; }

int Scheduler::token_budget() const {
    std::lock_guard<std::mutex> lk(impl_->mu);
    return impl_->budget;
}

const SchedulerOptions& Scheduler::options() const { return impl_->opts; }

const DecisionModel& Scheduler::model() const { return impl_->workers.front()->model; }

}  // namespace brolm::laya
