// bro.lm.Grammar: constrained decoding (brolm/grammar.h).
//
// A Grammar is built by one of the static factories and passed as
// opts.grammar to LMModel.generate / generateStream / bro.lm.generate, which
// mask every decode step so the output conforms. Generation works on a copy
// of the grammar's state, so one Grammar object serves any number of calls;
// accept() / canAccept() / isAccepted() / reset() drive the object's own
// state for callers that validate text themselves.

#include "host_lm_internal.h"

#include <brolm/grammar.h>

namespace brolm::api {

HostClass g_grammarClass;

namespace {

struct HostGrammar {
    brolm::Grammar g;
};

HostGrammar* hostGrammarPtr(Value v) {
    if (!ev::isObject(v)) return nullptr;
    return static_cast<HostGrammar*>(g_grammarClass.unwrap(v));
}

Value makeGrammarValue(brolm::Grammar g) {
    auto w = std::make_unique<HostGrammar>();
    w->g = std::move(g);
    return g_grammarClass.createInstance(std::move(w));
}

// A factory taking one string argument.
template <class F>
ev::NativeFn stringFactory(const char* name, F make) {
    std::string label = name;
    return [label, make](Value, std::span<const Value> a) -> Value {
        if (a.empty() || !ev::isString(a[0]))
            return ev::throwTypeError("Grammar." + label + "(text): string required");
        try {
            return makeGrammarValue(make(ev::toUtf8(a[0])));
        } catch (const std::exception& e) {
            return ev::throwError("Grammar." + label + ": " + e.what());
        }
    };
}

template <class F>
ev::NativeFn plainFactory(const char* name, F make) {
    std::string label = name;
    return [label, make](Value, std::span<const Value>) -> Value {
        try {
            return makeGrammarValue(make());
        } catch (const std::exception& e) {
            return ev::throwError("Grammar." + label + ": " + e.what());
        }
    };
}

void decorateGrammar(ObjectBuilder& b) {
    b.def("canAccept", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* w = hostGrammarPtr(self);
        if (!w) return ev::throwTypeError("canAccept: not a Grammar");
        if (a.empty() || !ev::isString(a[0])) return ev::throwTypeError("canAccept(text): string required");
        return ev::fromBool(w->g.can_accept(ev::toUtf8(a[0])));
    });
    b.def("accept", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* w = hostGrammarPtr(self);
        if (!w) return ev::throwTypeError("accept: not a Grammar");
        if (a.empty() || !ev::isString(a[0])) return ev::throwTypeError("accept(text): string required");
        return ev::fromBool(w->g.accept(ev::toUtf8(a[0])));
    });
    b.def("isAccepted", 0, [](Value self, std::span<const Value>) -> Value {
        auto* w = hostGrammarPtr(self);
        if (!w) return ev::throwTypeError("isAccepted: not a Grammar");
        return ev::fromBool(w->g.is_accepted());
    });
    b.def("reset", 0, [](Value self, std::span<const Value>) -> Value {
        auto* w = hostGrammarPtr(self);
        if (!w) return ev::throwTypeError("reset: not a Grammar");
        w->g.reset();
        return ev::undefined();
    });
    b.def("clone", 0, [](Value self, std::span<const Value>) -> Value {
        auto* w = hostGrammarPtr(self);
        if (!w) return ev::throwTypeError("clone: not a Grammar");
        return makeGrammarValue(w->g.clone());
    });
}

// JSON.stringify(v) for a schema handed over as an object.
bool stringifyJson(Value v, std::string& out) {
    ev::Persistent value(v);
    auto g = ev::globalValue("JSON");
    if (!g.found || !ev::isObject(g.value)) return false;
    ev::Persistent json(g.value);
    ev::Persistent stringify(ev::getProperty(json.get(), "stringify"));
    if (!ev::isFunction(stringify.get())) return false;
    const Value arg = value.get();
    auto res = ev::call(stringify.get(), json.get(), std::span<const Value>(&arg, 1));
    if (res.thrown || !ev::isString(res.value)) return false;
    out = ev::toUtf8(res.value);
    return true;
}

}  // namespace

const brolm::Grammar* hostGrammarOf(Value v) {
    HostGrammar* w = hostGrammarPtr(v);
    return w ? &w->g : nullptr;
}

void registerLMGrammarClass() {
    g_grammarClass.install("Grammar", 0, nullptr, decorateGrammar);

    g_grammarClass.setStatic("regex", ev::makeFunction(
        stringFactory("regex", [](const std::string& s) { return brolm::Grammar::regex(s); }), 1, "regex"));
    g_grammarClass.setStatic("bnf", ev::makeFunction(
        stringFactory("bnf", [](const std::string& s) { return brolm::Grammar::bnf(s); }), 1, "bnf"));
    g_grammarClass.setStatic("exact", ev::makeFunction(
        stringFactory("exact", [](const std::string& s) { return brolm::Grammar::exact(s); }), 1, "exact"));
    g_grammarClass.setStatic("jsonObject", ev::makeFunction(
        plainFactory("jsonObject", [] { return brolm::Grammar::json_object(); }), 0, "jsonObject"));
    g_grammarClass.setStatic("jsonArray", ev::makeFunction(
        plainFactory("jsonArray", [] { return brolm::Grammar::json_array(); }), 0, "jsonArray"));
    g_grammarClass.setStatic("jsonValue", ev::makeFunction(
        plainFactory("jsonValue", [] { return brolm::Grammar::json_value(); }), 0, "jsonValue"));
    g_grammarClass.setStatic("jsonString", ev::makeFunction(
        plainFactory("jsonString", [] { return brolm::Grammar::json_string(); }), 0, "jsonString"));
    g_grammarClass.setStatic("jsonNumber", ev::makeFunction(
        plainFactory("jsonNumber", [] { return brolm::Grammar::json_number(); }), 0, "jsonNumber"));
    g_grammarClass.setStatic("jsonInteger", ev::makeFunction(
        plainFactory("jsonInteger", [] { return brolm::Grammar::json_integer(); }), 0, "jsonInteger"));
    g_grammarClass.setStatic("jsonBoolean", ev::makeFunction(
        plainFactory("jsonBoolean", [] { return brolm::Grammar::json_boolean(); }), 0, "jsonBoolean"));
    g_grammarClass.setStatic("jsonNull", ev::makeFunction(
        plainFactory("jsonNull", [] { return brolm::Grammar::json_null(); }), 0, "jsonNull"));

    // jsonSchema(schema): a JSON Schema as text or as an object.
    g_grammarClass.setStatic("jsonSchema", ev::makeFunction(
        [](Value, std::span<const Value> a) -> Value {
            std::string schema;
            if (!a.empty() && ev::isString(a[0])) {
                schema = ev::toUtf8(a[0]);
            } else if (a.empty() || !ev::isObject(a[0]) || !stringifyJson(a[0], schema)) {
                return ev::throwTypeError("Grammar.jsonSchema(schema): schema string or object required");
            }
            try {
                return makeGrammarValue(brolm::Grammar::json_schema(schema));
            } catch (const std::exception& e) {
                return ev::throwError(std::string("Grammar.jsonSchema: ") + e.what());
            }
        }, 1, "jsonSchema"));

    // choice(options): exactly one of a fixed set of strings.
    g_grammarClass.setStatic("choice", ev::makeFunction(
        [](Value, std::span<const Value> a) -> Value {
            if (a.empty() || !ev::isObject(a[0]))
                return ev::throwTypeError("Grammar.choice(options): array of strings required");
            std::vector<std::string> options = readStringArray(a[0]);
            if (options.empty())
                return ev::throwTypeError("Grammar.choice(options): at least one string required");
            try {
                return makeGrammarValue(brolm::Grammar::choice(options));
            } catch (const std::exception& e) {
                return ev::throwError(std::string("Grammar.choice: ") + e.what());
            }
        }, 1, "choice"));
}

}  // namespace brolm::api
