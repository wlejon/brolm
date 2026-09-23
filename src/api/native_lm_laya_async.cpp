// Promise delivery for the Laya binding (native_lm_laya.cpp).
//
// A LayaModel's scheduler completes requests on its device threads, which
// must never touch the bronze heap. The JS thread that created a promise
// registers it here and gets a LayaPost: a copyable handle any thread may
// call with a settle function. The settle is queued in that JS thread's
// mailbox and run by tickLayaAsync() — on the engine's frame pump (through
// tickLMAsync) or a Worker's tick — with the promise it belongs to, where it
// builds the result values and resolves or rejects.
//
// Per JS thread: promises are Persistents of the realm that made them, so a
// Worker's requests settle on the Worker's tick, never the main thread's.
// A pending entry also holds its model's scheduler alive, so a LayaModel
// collected with requests in flight still answers them, and the last
// reference to a scheduler is always dropped here, on the JS thread — never
// on one of its own device threads, which could not join themselves.

#include "host_lm_internal.h"

#include <brolm/laya_scheduler.h>

#include <mutex>
#include <unordered_map>

namespace brolm::api {

using LayaSettle = std::function<void(const ev::Persistent& promise)>;

struct LayaMailbox {
    std::mutex mu;
    std::vector<std::pair<uint64_t, LayaSettle>> ready;
};

namespace {

struct Pending {
    ev::Persistent promise;
    std::shared_ptr<void> keepalive;
};

thread_local std::shared_ptr<LayaMailbox> t_box;
thread_local std::unordered_map<uint64_t, Pending> t_pending;
thread_local uint64_t t_next_id = 0;

// Loads started by loadLayaAsync, polled each tick until ready.
struct PendingLoad {
    ev::Persistent promise;
    std::shared_ptr<brolm::laya::Scheduler> sched;
};
thread_local std::vector<PendingLoad> t_loads;

// Every live scheduler, so shutdownLM() can stop them before the device
// runtime goes away (process-wide: schedulers of every realm).
std::mutex g_live_mu;
std::vector<std::weak_ptr<brolm::laya::Scheduler>>& liveSchedulers() {
    static std::vector<std::weak_ptr<brolm::laya::Scheduler>> v;
    return v;
}

Value makeError(const std::string& msg) {
    ev::Persistent text(ev::fromUtf8(msg));
    auto ctor = ev::globalValue("Error");
    if (ctor.found && ev::isFunction(ctor.value)) {
        ev::Persistent c(ctor.value);
        const Value arg = text.get();
        auto r = ev::construct(c.get(), std::span<const Value>(&arg, 1));
        if (!r.thrown) return r.value;
    }
    return text.get();
}

}  // namespace

void LayaPost::operator()(LayaSettle settle) const {
    if (!box) return;
    std::lock_guard<std::mutex> lk(box->mu);
    box->ready.emplace_back(id, std::move(settle));
}

LayaPost layaTrackPromise(Value promise, std::shared_ptr<void> keepalive) {
    if (!t_box) t_box = std::make_shared<LayaMailbox>();
    const uint64_t id = ++t_next_id;
    Pending p;
    p.promise = ev::Persistent(promise);
    p.keepalive = std::move(keepalive);
    t_pending.emplace(id, std::move(p));
    return LayaPost{t_box, id};
}

void layaTrackLoad(Value promise, std::shared_ptr<brolm::laya::Scheduler> sched) {
    PendingLoad l;
    l.promise = ev::Persistent(promise);
    l.sched = std::move(sched);
    t_loads.push_back(std::move(l));
}

void layaRegisterScheduler(const std::shared_ptr<brolm::laya::Scheduler>& sched) {
    std::lock_guard<std::mutex> lk(g_live_mu);
    auto& v = liveSchedulers();
    v.erase(std::remove_if(v.begin(), v.end(), [](const auto& w) { return w.expired(); }), v.end());
    v.push_back(sched);
}

Value layaRejectWith(Value promise, const std::string& message) {
    ev::Persistent p(promise);
    ev::Persistent err(makeError(message));
    ev::rejectPromise(p.get(), err.get());
    return p.get();
}

bool tickLayaAsync() {
    bool settled = false;

    // Finished loads: resolve with a LayaModel, or reject with the load error.
    if (!t_loads.empty()) {
        std::vector<PendingLoad> still;
        std::vector<PendingLoad> done;
        for (auto& l : t_loads) {
            std::string err;
            if (l.sched->poll_ready(&err)) done.push_back(std::move(l));
            else still.push_back(std::move(l));
        }
        t_loads = std::move(still);
        for (auto& l : done) {
            std::string err;
            l.sched->poll_ready(&err);
            if (err.empty()) {
                ev::Persistent model(makeLayaModelValue(l.sched));
                ev::resolvePromise(l.promise.get(), model.get());
            } else {
                layaRejectWith(l.promise.get(), "loadLayaAsync: " + err);
            }
            settled = true;
        }
    }

    if (!t_box) return settled;
    std::vector<std::pair<uint64_t, LayaSettle>> ready;
    {
        std::lock_guard<std::mutex> lk(t_box->mu);
        ready.swap(t_box->ready);
    }
    for (auto& [id, settle] : ready) {
        auto it = t_pending.find(id);
        if (it == t_pending.end()) continue;
        Pending p = std::move(it->second);
        t_pending.erase(it);
        settle(p.promise);
        settled = true;
        // p (the promise root and the scheduler keepalive) drops here, on the JS thread.
    }
    return settled;
}

void shutdownLaya() {
    // This thread's unsettled promises and loads are released while the
    // runtime is still alive; the schedulers of every realm stop.
    t_loads.clear();
    t_pending.clear();
    if (t_box) {
        std::lock_guard<std::mutex> lk(t_box->mu);
        t_box->ready.clear();
    }
    std::vector<std::shared_ptr<brolm::laya::Scheduler>> live;
    {
        std::lock_guard<std::mutex> lk(g_live_mu);
        for (auto& w : liveSchedulers()) {
            if (auto s = w.lock()) live.push_back(std::move(s));
        }
        liveSchedulers().clear();
    }
    for (auto& s : live) s->shutdown();
}

void shutdownLM() { shutdownLaya(); }

}  // namespace brolm::api
