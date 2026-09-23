#include "host_lm_internal.h"

namespace brolm::api {

HostClass g_layaModelClass;

brolm::LayaModel* hostLayaModelOf(Value v) {
    if (!ev::isObject(v)) return nullptr;
    return static_cast<brolm::LayaModel*>(g_layaModelClass.unwrap(v));
}

Value makeLayaModelValue(std::unique_ptr<brolm::LayaModel> model) {
    return g_layaModelClass.createInstance(std::move(model));
}

namespace {

std::vector<std::string> getObjectKeys(Value obj) {
    std::vector<std::string> keys;
    auto g = ev::globalValue("Object");
    if (!g.found || !ev::isObject(g.value)) return keys;
    Value keysFn = ev::getProperty(g.value, "keys");
    if (!ev::isFunction(keysFn)) return keys;
    auto res = ev::call(keysFn, g.value, std::span<const Value>(&obj, 1));
    if (res.thrown || !ev::isObject(res.value)) return keys;
    Value lenVal = ev::getProperty(res.value, "length");
    if (!ev::isNumber(lenVal)) return keys;
    uint32_t len = static_cast<uint32_t>(ev::toDouble(lenVal));
    keys.reserve(len);
    for (uint32_t i = 0; i < len; ++i) {
        Value k = ev::getElement(res.value, i);
        if (ev::isString(k)) keys.push_back(ev::toUtf8(k));
    }
    return keys;
}

// JSON.stringify(v) re-spaced the way Python's json.dumps separates items
// (", " / ": ") — the reference serialises dict/list states and non-string
// instructions with json.dumps, and the model was trained on that spacing.
bool pythonStyleJson(Value v, std::string& out) {
    auto g = ev::globalValue("JSON");
    if (!g.found || !ev::isObject(g.value)) return false;
    Value stringifyFn = ev::getProperty(g.value, "stringify");
    if (!ev::isFunction(stringifyFn)) return false;
    auto res = ev::call(stringifyFn, g.value, std::span<const Value>(&v, 1));
    if (res.thrown || !ev::isString(res.value)) return false;
    out = brolm::laya::python_json_spacing(ev::toUtf8(res.value));
    return true;
}

// Optional integer property; `found` reports whether it was a number.
int intProp(Value obj, const char* camel, const char* snake, bool& found) {
    Value v = ev::getProperty(obj, camel);
    if (!ev::isNumber(v)) v = ev::getProperty(obj, snake);
    found = ev::isNumber(v);
    return found ? static_cast<int>(ev::toDouble(v)) : 0;
}

static void decorateLayaModel(ObjectBuilder& b) {
    // config() -> { max_len, head_max_len, temperature: [choice, score, noul],
    //               temperature_by_options: { "choice:3-5": T, ... } }
    b.def("config", 0, [](Value self, std::span<const Value>) -> Value {
        auto* model = hostLayaModelOf(self);
        if (!model) return ev::throwTypeError("config: not a LayaModel");
        const brolm::laya::Config& c = model->config();
        ObjectBuilder o;
        o.set("max_len", static_cast<double>(c.max_len));
        o.set("head_max_len", static_cast<double>(c.head_max_len));
        o.set("temperature", makeFloat32Array(c.temperature.data(), c.temperature.size()));
        ObjectBuilder tbo;
        for (const auto& [k, t] : c.temperature_by_options) tbo.set(k, static_cast<double>(t));
        o.set("temperature_by_options", tbo.build());
        return o.build();
    });

    // predict(state, questions, options?)
    //   options: { maxLen, headMaxLen, truncateLeft } (snake_case accepted) —
    //   per-call overrides of the checkpoint's max_len / head_max_len, and
    //   truncateLeft keeps the newest state tokens (multi-turn conversations).
    b.def("predict", 3, [](Value self, std::span<const Value> a) -> Value {
        auto* model = hostLayaModelOf(self);
        if (!model) return ev::throwTypeError("predict: not a LayaModel");
        if (a.size() < 2) return ev::throwTypeError("predict(state, questions): 2 arguments required");

        brolm::laya::PredictOptions opts;
        if (a.size() >= 3 && ev::isObject(a[2])) {
            bool found = false;
            opts.max_len = intProp(a[2], "maxLen", "max_len", found);
            opts.head_max_len = intProp(a[2], "headMaxLen", "head_max_len", found);
            Value tl = ev::getProperty(a[2], "truncateLeft");
            if (!ev::isBool(tl)) tl = ev::getProperty(a[2], "truncate_left");
            opts.truncate_left = ev::isBool(tl) && ev::toBool(tl);
            if (opts.max_len < 0 || opts.head_max_len < 0) {
                return ev::throwRangeError("predict: maxLen / headMaxLen must be positive");
            }
        }

        // 1. Read state: a string is used as is; an object/array is
        //    serialised like the reference's json.dumps(state).
        std::string state;
        if (ev::isString(a[0])) {
            state = ev::toUtf8(a[0]);
        } else if (ev::isObject(a[0])) {
            if (!pythonStyleJson(a[0], state)) {
                return ev::throwTypeError("predict: failed to serialize state to JSON");
            }
        } else {
            return ev::throwTypeError("predict: state must be a string or object");
        }

        // 2. Parse questions JS object into std::unordered_map<std::string, LayaQuestion>.
        if (!ev::isObject(a[1])) {
            return ev::throwTypeError("predict: questions must be an object");
        }

        std::vector<std::string> keys = getObjectKeys(a[1]);
        std::unordered_map<std::string, brolm::LayaQuestion> questions;
        std::vector<std::string> questionOrder;
        questionOrder.reserve(keys.size());

        for (const std::string& qId : keys) {
            Value qDef = ev::getProperty(a[1], qId);
            if (!ev::isObject(qDef)) continue;

            brolm::LayaQuestion q;
            q.id = qId;

            Value typeVal = ev::getProperty(qDef, "type");
            if (ev::isString(typeVal)) {
                q.type = ev::toUtf8(typeVal);
            }

            Value insVal = ev::getProperty(qDef, "instructions");
            if (ev::isString(insVal)) {
                q.instructions = ev::toUtf8(insVal);
            } else if (!ev::isUndefined(insVal)) {
                pythonStyleJson(insVal, q.instructions);  // reference: json.dumps
            }

            Value critVal = ev::getProperty(qDef, "criteria");
            if (ev::isObject(critVal)) {
                Value critLenVal = ev::getProperty(critVal, "length");
                if (ev::isNumber(critLenVal)) {
                    // Array of criteria
                    uint32_t critLen = static_cast<uint32_t>(ev::toDouble(critLenVal));
                    if (q.type == "choice") {
                        for (uint32_t c = 0; c < critLen; ++c) {
                            Value item = ev::getElement(critVal, c);
                            if (ev::isString(item)) {
                                q.criteria_choice.emplace_back(ev::toUtf8(item), "");
                            } else if (ev::isObject(item)) {
                                Value itemLen = ev::getProperty(item, "length");
                                if (ev::isNumber(itemLen) && ev::toDouble(itemLen) >= 2) {
                                    Value k = ev::getElement(item, 0);
                                    Value v = ev::getElement(item, 1);
                                    q.criteria_choice.emplace_back(
                                        ev::isString(k) ? ev::toUtf8(k) : "",
                                        ev::isString(v) ? ev::toUtf8(v) : "");
                                }
                            }
                        }
                    } else if (q.type == "score") {
                        for (uint32_t c = 0; c < critLen; ++c) {
                            Value item = ev::getElement(critVal, c);
                            if (ev::isString(item)) {
                                q.criteria_score.push_back(ev::toUtf8(item));
                            }
                        }
                    } else { // noul
                        if (critLen > 0) {
                            Value v0 = ev::getElement(critVal, 0);
                            if (ev::isString(v0)) q.criteria_noul_false = ev::toUtf8(v0);
                        }
                        if (critLen > 1) {
                            Value v1 = ev::getElement(critVal, 1);
                            if (ev::isString(v1)) q.criteria_noul_true = ev::toUtf8(v1);
                        }
                    }
                } else {
                    // Object criteria (e.g. { billing: "...", technical: "..." })
                    std::vector<std::string> cKeys = getObjectKeys(critVal);
                    for (const std::string& kStr : cKeys) {
                        Value vVal = ev::getProperty(critVal, kStr);
                        std::string vStr;
                        if (ev::isString(vVal)) {
                            vStr = ev::toUtf8(vVal);
                        } else if (!ev::isUndefined(vVal) && !ev::isNull(vVal)) {
                            pythonStyleJson(vVal, vStr);  // structured criterion -> JSON text
                        }
                        if (q.type == "choice" || q.type.empty()) {
                            q.criteria_choice.emplace_back(kStr, vStr);
                        } else if (q.type == "score") {
                            // reference: enumerate(dict) walks the keys
                            q.criteria_score.push_back(kStr);
                        } else { // noul
                            if (kStr == "false" || kStr == "0") q.criteria_noul_false = vStr;
                            else if (kStr == "true" || kStr == "1") q.criteria_noul_true = vStr;
                        }
                    }
                }
            }

            if (q.type.empty()) {
                if (!q.criteria_choice.empty()) q.type = "choice";
                else if (!q.criteria_score.empty()) q.type = "score";
                else q.type = "noul";
            }

            questionOrder.push_back(qId);
            questions[qId] = std::move(q);
        }

        // 3. Call model->predict in the caller's question order.
        std::vector<brolm::LayaQuestion> ordered;
        ordered.reserve(questionOrder.size());
        for (const std::string& qId : questionOrder) ordered.push_back(questions[qId]);

        try {
            brolm::LayaResult res = model->predict(state, ordered, opts);

            // 4. Build and return JS object using bronze::embed APIs.
            ObjectBuilder out;
            out.set("model", res.model);

            ObjectBuilder answersObj;
            for (const std::string& qId : questionOrder) {
                auto it = res.answers.find(qId);
                if (it == res.answers.end()) continue;
                const brolm::LayaAnswer& ans = it->second;
                const brolm::LayaQuestion& q = questions[qId];

                ObjectBuilder aObj;
                aObj.set("type", ans.type);

                if (ans.type == "choice") {
                    aObj.set("choice", ans.choice);
                    aObj.set("confidence", static_cast<double>(ans.confidence));
                } else if (ans.type == "score") {
                    aObj.set("score", static_cast<double>(ans.score));
                    aObj.set("confidence", static_cast<double>(ans.confidence));
                    ObjectBuilder legendObj;
                    for (size_t c = 0; c < q.criteria_score.size(); ++c) {
                        legendObj.set(std::to_string(c), q.criteria_score[c]);
                    }
                    aObj.set("legend", legendObj.build());
                } else {  // "noul"
                    aObj.set("noul", static_cast<double>(ans.noul));
                }
                if (ans.type != "noul") {
                    ObjectBuilder probObj;
                    for (const auto& [k, p] : ans.probabilities) {
                        probObj.set(k, static_cast<double>(p));
                    }
                    aObj.set("probabilities", probObj.build());
                }
                // Raw, pre-temperature scorer logits (option order) and the
                // temperature applied — enough to refit calibration downstream.
                aObj.set("logits", makeFloat32Array(ans.logits.data(), ans.logits.size()));
                aObj.set("temperature", static_cast<double>(ans.temperature));

                ObjectBuilder rlObj;
                rlObj.set("act_probability", static_cast<double>(ans.act_probability));
                rlObj.set("act_logits", makeFloat32Array(ans.act_logits.data(), ans.act_logits.size()));
                aObj.set("rl_agent", rlObj.build());

                answersObj.set(qId, aObj.build());
            }
            out.set("answers", answersObj.build());

            ObjectBuilder usageObj;
            usageObj.set("input_tokens", static_cast<double>(res.input_tokens));
            usageObj.set("output_tokens", 0.0);
            out.set("usage", usageObj.build());

            return out.build();
        } catch (const std::exception& e) {
            return ev::throwError(std::string("predict: ") + e.what());
        }
    });
}

} // namespace

Value js_loadLaya(Value, std::span<const Value> a) {
    if (a.empty()) {
        return ev::throwTypeError("loadLaya: path string required");
    }

    std::string path;
    if (ev::isString(a[0])) {
        path = ev::toUtf8(a[0]);
    } else if (ev::isObject(a[0])) {
        Value p = ev::getProperty(a[0], "modelPath");
        if (!ev::isString(p)) p = ev::getProperty(a[0], "path");
        if (ev::isString(p)) path = ev::toUtf8(p);
    }

    if (path.empty()) {
        return ev::throwTypeError("loadLaya: path string required");
    }

    std::string resolved = resolvePath(path);
    if (!std::filesystem::exists(resolved)) {
        return ev::throwError("loadLaya: model path not found: " + resolved);
    }

    brotensor::init();

    try {
        auto model = std::make_unique<brolm::LayaModel>();
        model->load_model(resolved);
        // Optional load-time defaults: { maxLen, headMaxLen } replace the
        // checkpoint's max_len / head_max_len for every later predict().
        if (ev::isObject(a[0])) {
            bool found = false;
            const int ml = intProp(a[0], "maxLen", "max_len", found);
            if (found && ml > 0) model->mutable_config().max_len = ml;
            const int hml = intProp(a[0], "headMaxLen", "head_max_len", found);
            if (found && hml > 0) model->mutable_config().head_max_len = hml;
        }
        return g_layaModelClass.createInstance(std::move(model));
    } catch (const std::exception& e) {
        return ev::throwError(std::string("loadLaya: ") + e.what());
    }
}

void registerLMLayaClasses() {
    g_layaModelClass.install("LayaModel", 1, js_loadLaya, decorateLayaModel);
}

} // namespace brolm::api
