// bro.lm.loadLaya / loadLayaAsync and the LayaModel class.
//
// A LayaModel is a handle on a laya::Scheduler: one model replica per device,
// requests from this realm (and any other caller) packed into shared
// forwards. predict() is the blocking form (submit + wait); predictAsync()
// returns a Promise settled on this JS thread's next LM tick
// (native_lm_laya_async.cpp), so many calls in flight from one app batch
// together. Reading the arguments and building the results happen here, on
// the JS thread; the device threads only ever see C++ values.

#include "host_lm_internal.h"

namespace brolm::api {

HostClass g_layaModelClass;

namespace {

namespace laya = brolm::laya;

struct HostLaya {
    std::shared_ptr<laya::Scheduler> sched;
};

HostLaya* hostLayaOf(Value v) {
    if (!ev::isObject(v)) return nullptr;
    return static_cast<HostLaya*>(g_layaModelClass.unwrap(v));
}

std::vector<std::string> getObjectKeys(Value obj) {
    std::vector<std::string> keys;
    // obj and Object are rooted: getProperty allocates (embed.h GC contract).
    ev::Persistent target(obj);
    auto g = ev::globalValue("Object");
    if (!g.found || !ev::isObject(g.value)) return keys;
    ev::Persistent objectCtor(g.value);
    ev::Persistent keysFn(ev::getProperty(objectCtor.get(), "keys"));
    if (!ev::isFunction(keysFn.get())) return keys;
    const Value arg = target.get();
    auto res = ev::call(keysFn.get(), objectCtor.get(), std::span<const Value>(&arg, 1));
    if (res.thrown || !ev::isObject(res.value)) return keys;
    ev::Persistent arr(res.value);
    Value lenVal = ev::getProperty(arr.get(), "length");
    if (!ev::isNumber(lenVal)) return keys;
    uint32_t len = static_cast<uint32_t>(ev::toDouble(lenVal));
    keys.reserve(len);
    for (uint32_t i = 0; i < len; ++i) {
        Value k = ev::getElement(arr.get(), i);
        if (ev::isString(k)) keys.push_back(ev::toUtf8(k));
    }
    return keys;
}

// JSON.stringify(v) re-spaced the way Python's json.dumps separates items
// (", " / ": ") — the reference serialises dict/list states and non-string
// instructions with json.dumps, and the model was trained on that spacing.
bool pythonStyleJson(Value v, std::string& out) {
    ev::Persistent value(v);
    auto g = ev::globalValue("JSON");
    if (!g.found || !ev::isObject(g.value)) return false;
    ev::Persistent json(g.value);
    ev::Persistent stringifyFn(ev::getProperty(json.get(), "stringify"));
    if (!ev::isFunction(stringifyFn.get())) return false;
    const Value arg = value.get();
    auto res = ev::call(stringifyFn.get(), json.get(), std::span<const Value>(&arg, 1));
    if (res.thrown || !ev::isString(res.value)) return false;
    out = brolm::laya::python_json_spacing(ev::toUtf8(res.value));
    return true;
}

// Optional numeric property under a camelCase or snake_case name.
bool numProp(Value obj, const char* camel, const char* snake, double& out) {
    ev::Persistent o(obj);  // the first read allocates; the second needs o current
    Value v = ev::getProperty(o.get(), camel);
    if (!ev::isNumber(v) && snake) v = ev::getProperty(o.get(), snake);
    if (!ev::isNumber(v)) return false;
    out = ev::toDouble(v);
    return true;
}

bool boolProp(Value obj, const char* camel, const char* snake, bool& out) {
    ev::Persistent o(obj);
    Value v = ev::getProperty(o.get(), camel);
    if (!ev::isBool(v) && snake) v = ev::getProperty(o.get(), snake);
    if (!ev::isBool(v)) return false;
    out = ev::toBool(v);
    return true;
}

// state: a string is used as is; an object/array is serialised like the
// reference's json.dumps(state).
bool readState(Value v, std::string& out, std::string& err) {
    if (ev::isString(v)) {
        out = ev::toUtf8(v);
        return true;
    }
    if (ev::isObject(v)) {
        if (pythonStyleJson(v, out)) return true;
        err = "failed to serialize state to JSON";
        return false;
    }
    err = "state must be a string or object";
    return false;
}

// questions: { id: { type, instructions, criteria } }, in key order.
bool readQuestions(Value qsVal, std::vector<LayaQuestion>& out, std::string& err) {
    if (!ev::isObject(qsVal)) {
        err = "questions must be an object";
        return false;
    }
    ev::Persistent qsObj(qsVal);
    for (const std::string& qId : getObjectKeys(qsObj.get())) {
        ev::Persistent qDef(ev::getProperty(qsObj.get(), qId));
        if (!ev::isObject(qDef.get())) continue;

        LayaQuestion q;
        q.id = qId;
        Value typeVal = ev::getProperty(qDef.get(), "type");
        if (ev::isString(typeVal)) q.type = ev::toUtf8(typeVal);

        Value insVal = ev::getProperty(qDef.get(), "instructions");
        if (ev::isString(insVal)) {
            q.instructions = ev::toUtf8(insVal);
        } else if (!ev::isUndefined(insVal)) {
            pythonStyleJson(insVal, q.instructions);  // reference: json.dumps
        }

        ev::Persistent crit(ev::getProperty(qDef.get(), "criteria"));
        if (ev::isObject(crit.get())) {
            Value critLenVal = ev::getProperty(crit.get(), "length");
            if (ev::isNumber(critLenVal)) {
                const uint32_t critLen = static_cast<uint32_t>(ev::toDouble(critLenVal));
                for (uint32_t c = 0; c < critLen; ++c) {
                    ev::Persistent item(ev::getElement(crit.get(), c));
                    if (q.type == "choice" || q.type.empty()) {
                        if (ev::isString(item.get())) {
                            q.criteria_choice.emplace_back(ev::toUtf8(item.get()), "");
                        } else if (ev::isObject(item.get())) {
                            // [key, description] pairs
                            Value itemLen = ev::getProperty(item.get(), "length");
                            if (ev::isNumber(itemLen) && ev::toDouble(itemLen) >= 2) {
                                Value k = ev::getElement(item.get(), 0);
                                std::string ks = ev::isString(k) ? ev::toUtf8(k) : "";
                                Value d = ev::getElement(item.get(), 1);
                                q.criteria_choice.emplace_back(ks, ev::isString(d) ? ev::toUtf8(d) : "");
                            }
                        }
                    } else if (q.type == "score") {
                        if (ev::isString(item.get())) q.criteria_score.push_back(ev::toUtf8(item.get()));
                    } else if (ev::isString(item.get())) {  // noul: [false, true]
                        (c == 0 ? q.criteria_noul_false : q.criteria_noul_true) = ev::toUtf8(item.get());
                    }
                }
            } else {
                // Object criteria, e.g. { billing: "...", technical: "..." }
                for (const std::string& kStr : getObjectKeys(crit.get())) {
                    Value vVal = ev::getProperty(crit.get(), kStr);
                    std::string vStr;
                    if (ev::isString(vVal)) {
                        vStr = ev::toUtf8(vVal);
                    } else if (!ev::isUndefined(vVal) && !ev::isNull(vVal)) {
                        pythonStyleJson(vVal, vStr);  // structured criterion -> JSON text
                    }
                    if (q.type == "choice" || q.type.empty()) {
                        q.criteria_choice.emplace_back(kStr, vStr);
                    } else if (q.type == "score") {
                        q.criteria_score.push_back(kStr);  // reference: enumerate(dict) walks the keys
                    } else if (kStr == "false" || kStr == "0") {
                        q.criteria_noul_false = vStr;
                    } else if (kStr == "true" || kStr == "1") {
                        q.criteria_noul_true = vStr;
                    }
                }
            }
        }
        if (q.type.empty()) {
            if (!q.criteria_choice.empty()) q.type = "choice";
            else if (!q.criteria_score.empty()) q.type = "score";
            else q.type = "noul";
        }
        out.push_back(std::move(q));
    }
    return true;
}

// { maxLen, headMaxLen, truncateLeft, priority, deadlineMs } (snake_case accepted).
bool readRequestOptions(Value opts, laya::RequestOptions& ro, std::string& err) {
    if (!ev::isObject(opts)) return true;
    ev::Persistent o(opts);  // each read below allocates; o.get() is always current
    double d = 0;
    if (numProp(o.get(), "maxLen", "max_len", d)) ro.predict.max_len = static_cast<int>(d);
    if (numProp(o.get(), "headMaxLen", "head_max_len", d)) ro.predict.head_max_len = static_cast<int>(d);
    if (ro.predict.max_len < 0 || ro.predict.head_max_len < 0) {
        err = "maxLen / headMaxLen must be positive";
        return false;
    }
    boolProp(o.get(), "truncateLeft", "truncate_left", ro.predict.truncate_left);
    if (numProp(o.get(), "priority", nullptr, d)) ro.priority = static_cast<int>(d);
    if (numProp(o.get(), "deadlineMs", "deadline_ms", d)) ro.deadline_ms = d;
    return true;
}

Value buildTiming(const laya::RequestTiming& t) {
    ObjectBuilder o;
    o.set("tokenizeMs", t.tokenize_ms);
    o.set("queueMs", t.queue_ms);
    o.set("forwardMs", t.forward_ms);
    o.set("totalMs", t.total_ms);
    o.set("forwards", static_cast<double>(t.forwards));
    o.set("batchItems", static_cast<double>(t.batch_items));
    o.set("batchRequests", static_cast<double>(t.batch_requests));
    o.set("batchTokens", static_cast<double>(t.batch_tokens));
    o.set("device", static_cast<double>(t.device));
    o.set("deadlineMissed", t.deadline_missed);
    return o.build();
}

// The reference response shape, plus `timing`.
Value buildResult(const laya::ScheduledResult& r, const std::vector<LayaQuestion>& questions) {
    ObjectBuilder out;
    out.set("model", r.result.model);
    ObjectBuilder answersObj;
    for (const LayaQuestion& q : questions) {
        auto it = r.result.answers.find(q.id);
        if (it == r.result.answers.end()) continue;
        const LayaAnswer& ans = it->second;
        ObjectBuilder a;
        a.set("type", ans.type);
        if (ans.type == "choice") {
            a.set("choice", ans.choice);
            a.set("confidence", static_cast<double>(ans.confidence));
        } else if (ans.type == "score") {
            a.set("score", static_cast<double>(ans.score));
            a.set("confidence", static_cast<double>(ans.confidence));
            ObjectBuilder legend;
            for (size_t c = 0; c < q.criteria_score.size(); ++c) legend.set(std::to_string(c), q.criteria_score[c]);
            a.set("legend", legend.build());
        } else {
            a.set("noul", static_cast<double>(ans.noul));
            // Not in the reference's noul answer; the same entropy-based
            // confidence, which is what escalation has to gate on.
            a.set("confidence", static_cast<double>(ans.confidence));
        }
        if (ans.type != "noul") {
            ObjectBuilder probs;
            for (const auto& [k, p] : ans.probabilities) probs.set(k, static_cast<double>(p));
            a.set("probabilities", probs.build());
        }
        // Raw pre-temperature scorer logits (option order) and the temperature
        // applied — enough to refit calibration downstream.
        a.set("logits", makeFloat32Array(ans.logits.data(), ans.logits.size()));
        a.set("temperature", static_cast<double>(ans.temperature));
        ObjectBuilder rl;
        rl.set("act_probability", static_cast<double>(ans.act_probability));
        rl.set("act_logits", makeFloat32Array(ans.act_logits.data(), ans.act_logits.size()));
        a.set("rl_agent", rl.build());
        answersObj.set(q.id, a.build());
    }
    out.set("answers", answersObj.build());
    ObjectBuilder usage;
    usage.set("input_tokens", static_cast<double>(r.result.input_tokens));
    usage.set("output_tokens", 0.0);
    out.set("usage", usage.build());
    out.set("timing", buildTiming(r.timing));
    return out.build();
}

Value buildStats(const laya::SchedulerStats& s) {
    ObjectBuilder o;
    o.set("submitted", static_cast<double>(s.submitted));
    o.set("completed", static_cast<double>(s.completed));
    o.set("failed", static_cast<double>(s.failed));
    o.set("deadlineMissed", static_cast<double>(s.deadline_missed));
    o.set("queuedRequests", static_cast<double>(s.queued_requests));
    o.set("queuedItems", static_cast<double>(s.queued_items));
    o.set("inFlightRequests", static_cast<double>(s.in_flight_requests));
    o.set("forwards", static_cast<double>(s.forwards));
    o.set("itemsRun", static_cast<double>(s.items_run));
    o.set("tokensRun", static_cast<double>(s.tokens_run));
    o.set("meanBatchItems", s.mean_batch_items);
    o.set("meanBatchTokens", s.mean_batch_tokens);
    o.set("meanBatchRequests", s.mean_batch_requests);
    o.set("meanOccupancy", s.mean_occupancy);
    o.set("tokenBudget", static_cast<double>(s.token_budget));
    o.set("targetForwardMs", s.target_forward_ms);
    o.set("estFixedMs", s.est_fixed_ms);
    o.set("estMsPer1kTokens", s.est_ms_per_1k_tokens);
    o.set("latencyP50", s.latency_p50);
    o.set("latencyP95", s.latency_p95);
    o.set("latencyP99", s.latency_p99);
    o.set("latencyMax", s.latency_max);
    o.set("queueMeanMs", s.queue_mean);
    o.set("windowS", s.window_s);
    o.set("throughputRps", s.throughput_rps);
    o.set("devices", hostArrayOf(s.devices.size(), [&](size_t i) {
              const auto& d = s.devices[i];
              ObjectBuilder e;
              e.set("device", static_cast<double>(d.device));
              e.set("name", d.name);
              e.set("forwards", static_cast<double>(d.forwards));
              e.set("busyMs", d.busy_ms);
              e.set("busyFraction", d.busy_fraction);
              e.set("graphs", static_cast<double>(d.graphs));
              return e.build();
          }));
    o.set("recentBatches", hostArrayOf(s.recent_batches.size(), [&](size_t i) {
              const auto& b = s.recent_batches[i];
              ObjectBuilder e;
              e.set("seq", static_cast<double>(b.seq));
              e.set("device", static_cast<double>(b.device));
              e.set("requests", static_cast<double>(b.requests));
              e.set("items", static_cast<double>(b.items));
              e.set("tokens", static_cast<double>(b.tokens));
              e.set("budget", static_cast<double>(b.budget));
              e.set("startMs", b.start_ms);
              e.set("ms", b.ms);
              return e.build();
          }));
    return o.build();
}

// Everything a predict call reads from its arguments, read on the JS thread.
struct Call {
    std::string state;
    std::vector<LayaQuestion> questions;
    laya::RequestOptions opts;
};

bool readCall(std::span<const Value> a, Call& c, std::string& err) {
    if (a.size() < 2) {
        err = "(state, questions): 2 arguments required";
        return false;
    }
    if (!readState(a[0], c.state, err)) return false;
    if (!readQuestions(a[1], c.questions, err)) return false;
    return a.size() < 3 || readRequestOptions(a[2], c.opts, err);
}

laya::Scheduler* liveScheduler(Value self, const char* what, std::string& err) {
    HostLaya* h = hostLayaOf(self);
    if (!h || !h->sched) {
        err = std::string(what) + ": not a LayaModel";
        return nullptr;
    }
    if (h->sched->is_shut_down()) {
        err = std::string(what) + ": LayaModel is disposed";
        return nullptr;
    }
    return h->sched.get();
}

void decorateLayaModel(ObjectBuilder& b) {
    // config() -> { checkpoint, model_dir, encoder, model_name, tokenizer,
    //               vocab_size, hidden_size, num_layers, max_prefixes,
    //               max_len, head_max_len, temperature, temperature_by_options,
    //               devices, tokenBudget, maxBatchTokens, targetForwardMs, deadlineMs }
    b.def("config", 0, [](Value self, std::span<const Value>) -> Value {
        HostLaya* h = hostLayaOf(self);
        if (!h || !h->sched) return ev::throwTypeError("config: not a LayaModel");
        const laya::Config& c = h->sched->model().config();
        const laya::SchedulerOptions& so = h->sched->options();
        const std::vector<int> devs = h->sched->devices();
        const laya::DecisionModel& m = h->sched->model();
        const auto& enc = m.encoder().config();
        ObjectBuilder o;
        // Which checkpoint of the family this is, and its shape.
        o.set("checkpoint", ev::fromUtf8(c.variant));
        o.set("model_dir", ev::fromUtf8(c.model_dir));
        o.set("encoder", ev::fromUtf8(c.encoder));
        o.set("model_name", ev::fromUtf8(c.model_name));
        o.set("tokenizer", ev::fromUtf8(m.tokenizer().kind_name()));
        o.set("vocab_size", static_cast<double>(enc.vocab_size));
        o.set("hidden_size", static_cast<double>(enc.hidden_size));
        o.set("num_layers", static_cast<double>(enc.num_hidden_layers));
        o.set("max_prefixes", static_cast<double>(c.max_prefixes));
        o.set("max_len", static_cast<double>(c.max_len));
        o.set("head_max_len", static_cast<double>(c.head_max_len));
        o.set("temperature", makeFloat32Array(c.temperature.data(), c.temperature.size()));
        ObjectBuilder tbo;
        for (const auto& [k, t] : c.temperature_by_options) tbo.set(k, static_cast<double>(t));
        o.set("temperature_by_options", tbo.build());
        o.set("devices", hostArrayOf(devs.size(), [&](size_t i) { return ev::fromDouble(devs[i]); }));
        o.set("tokenBudget", static_cast<double>(h->sched->token_budget()));
        o.set("maxBatchTokens", static_cast<double>(so.max_batch_tokens));
        o.set("targetForwardMs", so.target_forward_ms);
        o.set("deadlineMs", so.default_deadline_ms);
        return o.build();
    });

    // predict(state, questions, options?) — blocking: queued with everything
    // else in flight, and this thread waits for its answer.
    b.def("predict", 3, [](Value self, std::span<const Value> a) -> Value {
        std::string err;
        laya::Scheduler* sched = liveScheduler(self, "predict", err);
        if (!sched) return ev::throwTypeError(err);
        Call c;
        if (!readCall(a, c, err)) return ev::throwTypeError("predict: " + err);
        try {
            const laya::ScheduledResult r = sched->predict(c.state, c.questions, c.opts);
            return buildResult(r, c.questions);
        } catch (const std::exception& e) {
            return ev::throwError(std::string("predict: ") + e.what());
        }
    });

    // predictAsync(state, questions, options?) -> Promise<result>, settled on
    // this thread's next LM tick.
    b.def("predictAsync", 3, [](Value self, std::span<const Value> a) -> Value {
        // `self` is a plain copy, current only until the first allocation:
        // unwrap it before createPromise.
        std::string err;
        laya::Scheduler* sched = liveScheduler(self, "predictAsync", err);
        std::shared_ptr<laya::Scheduler> keep = sched ? hostLayaOf(self)->sched : nullptr;
        ev::Persistent promise(ev::createPromise());
        if (!sched) return layaRejectWith(promise.get(), err);
        auto call = std::make_shared<Call>();
        if (!readCall(a, *call, err)) return layaRejectWith(promise.get(), "predictAsync: " + err);

        const LayaPost post = layaTrackPromise(promise.get(), keep);
        try {
            sched->submit(call->state, call->questions, call->opts,
                          [post, call](laya::ScheduledResult&& r, std::exception_ptr e) {
                              std::string failure;
                              if (e) {
                                  try {
                                      std::rethrow_exception(e);
                                  } catch (const std::exception& x) {
                                      failure = x.what();
                                  } catch (...) {
                                      failure = "unknown error";
                                  }
                              }
                              auto res = std::make_shared<laya::ScheduledResult>(std::move(r));
                              post([res, call, failure](const ev::Persistent& p) {
                                  if (!failure.empty()) {
                                      layaRejectWith(p.get(), "predictAsync: " + failure);
                                      return;
                                  }
                                  ev::Persistent v(buildResult(*res, call->questions));
                                  ev::resolvePromise(p.get(), v.get());
                              });
                          });
        } catch (const std::exception& e) {
            // Tokenize / option-fit errors: rejected on the next tick like any other failure.
            const std::string msg = std::string("predictAsync: ") + e.what();
            post([msg](const ev::Persistent& p) { layaRejectWith(p.get(), msg); });
        }
        return promise.get();
    });

    // stats() -> scheduler counters since the last resetStats().
    b.def("stats", 0, [](Value self, std::span<const Value>) -> Value {
        HostLaya* h = hostLayaOf(self);
        if (!h || !h->sched) return ev::throwTypeError("stats: not a LayaModel");
        return buildStats(h->sched->stats());
    });

    b.def("resetStats", 0, [](Value self, std::span<const Value>) -> Value {
        HostLaya* h = hostLayaOf(self);
        if (!h || !h->sched) return ev::throwTypeError("resetStats: not a LayaModel");
        h->sched->reset_stats();
        return ev::undefined();
    });

    // dispose(): stop the device threads and free every replica now. Pending
    // predictAsync promises reject; later calls throw.
    b.def("dispose", 0, [](Value self, std::span<const Value>) -> Value {
        HostLaya* h = hostLayaOf(self);
        if (!h || !h->sched) return ev::throwTypeError("dispose: not a LayaModel");
        h->sched->shutdown();
        return ev::undefined();
    });
}

// loadLaya(path | { path|modelPath, ... }, options?) — the options may ride
// on the first argument or come second:
//   devices: "all" | number[] | number   (default: the default device)
//   maxBatchTokens, targetForwardMs, deadlineMs, prewarm, maxLen, headMaxLen
bool readLoadArgs(std::span<const Value> a, const char* what, std::string& path, laya::SchedulerOptions& so,
                  std::string& err) {
    if (a.empty()) {
        err = std::string(what) + ": path string required";
        return false;
    }
    Value opts = ev::undefined();
    if (ev::isString(a[0])) {
        path = ev::toUtf8(a[0]);
        if (a.size() > 1 && ev::isObject(a[1])) opts = a[1];
    } else if (ev::isObject(a[0])) {
        Value p = ev::getProperty(a[0], "modelPath");
        if (!ev::isString(p)) p = ev::getProperty(a[0], "path");
        if (ev::isString(p)) path = ev::toUtf8(p);
        opts = a[0];
    }
    if (path.empty()) {
        err = std::string(what) + ": path string required";
        return false;
    }
    if (ev::isObject(opts)) {
        ev::Persistent o(opts);
        double d = 0;
        if (numProp(o.get(), "maxBatchTokens", "max_batch_tokens", d)) so.max_batch_tokens = static_cast<int>(d);
        if (numProp(o.get(), "targetForwardMs", "target_forward_ms", d)) so.target_forward_ms = d;
        if (numProp(o.get(), "deadlineMs", "deadline_ms", d)) so.default_deadline_ms = d;
        if (numProp(o.get(), "maxLen", "max_len", d)) so.max_len = static_cast<int>(d);
        if (numProp(o.get(), "headMaxLen", "head_max_len", d)) so.head_max_len = static_cast<int>(d);
        boolProp(o.get(), "prewarm", nullptr, so.prewarm);
        ev::Persistent devs(ev::getProperty(o.get(), "devices"));
        if (ev::isString(devs.get())) {
            if (ev::toUtf8(devs.get()) != "all") {
                err = std::string(what) + ": devices must be \"all\", an index or an array of indices";
                return false;
            }
            so.devices = laya::Scheduler::all_devices();
        } else if (ev::isNumber(devs.get())) {
            so.devices = {static_cast<int>(ev::toDouble(devs.get()))};
        } else if (ev::isObject(devs.get())) {
            Value lenV = ev::getProperty(devs.get(), "length");
            const uint32_t n = ev::isNumber(lenV) ? static_cast<uint32_t>(ev::toDouble(lenV)) : 0;
            for (uint32_t i = 0; i < n; ++i) {
                Value e = ev::getElement(devs.get(), i);
                if (ev::isNumber(e)) so.devices.push_back(static_cast<int>(ev::toDouble(e)));
            }
        }
    }
    const std::string resolved = resolvePath(path);
    if (!std::filesystem::exists(resolved + "/model.safetensors")) {
        err = std::string(what) + ": no Laya checkpoint (model.safetensors) at " + resolved;
        return false;
    }
    path = resolved;
    return true;
}

}  // namespace

Value makeLayaModelValue(std::shared_ptr<laya::Scheduler> sched) {
    layaRegisterScheduler(sched);
    auto h = std::make_unique<HostLaya>();
    h->sched = std::move(sched);
    return g_layaModelClass.createInstance(std::move(h));
}

// Blocking: returns once every replica is loaded and pre-warmed.
Value js_loadLaya(Value, std::span<const Value> a) {
    std::string path, err;
    laya::SchedulerOptions so;
    if (!readLoadArgs(a, "loadLaya", path, so, err)) {
        return err.find("path string") != std::string::npos ? ev::throwTypeError(err) : ev::throwError(err);
    }
    try {
        auto sched = std::make_shared<laya::Scheduler>(path, so);
        sched->wait_ready();
        return makeLayaModelValue(std::move(sched));
    } catch (const std::exception& e) {
        return ev::throwError(std::string("loadLaya: ") + e.what());
    }
}

// Promise<LayaModel>: the replicas load and pre-warm on their device threads.
Value js_loadLayaAsync(Value, std::span<const Value> a) {
    ev::Persistent promise(ev::createPromise());
    std::string path, err;
    laya::SchedulerOptions so;
    if (!readLoadArgs(a, "loadLayaAsync", path, so, err)) return layaRejectWith(promise.get(), err);
    try {
        layaTrackLoad(promise.get(), std::make_shared<laya::Scheduler>(path, so));
    } catch (const std::exception& e) {
        return layaRejectWith(promise.get(), std::string("loadLayaAsync: ") + e.what());
    }
    return promise.get();
}

void registerLMLayaClasses() {
    g_layaModelClass.install("LayaModel", 1, js_loadLaya, decorateLayaModel);
}

}  // namespace brolm::api
