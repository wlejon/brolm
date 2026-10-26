#pragma once

#include "embed/embed.h"
#include "embed/embed_host_class.h"
#include "object_builder.h"

#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>

namespace bronze::embed {
inline bool isDouble(Value v) { return isNumber(v); }
// `val` is rooted before registerGlobal / globalValue, either of which may
// allocate and move it (embed.h GC contract).
inline void setGlobalValue(std::string_view name, Value val) {
    Persistent v(val);
    registerGlobal(name, v.get());
    auto g = globalValue("globalThis");
    if (g.found && isObject(g.value)) {
        Persistent global(g.value);
        setProperty(global.get(), name, v.get());
    }
}
inline void setGlobalFunction(std::string_view name, uint32_t arity, NativeFn fn) {
    setGlobalValue(name, makeFunction(std::move(fn), arity, name));
}
inline void registerFunction(std::string_view name, NativeFn fn, uint32_t arity = 0) {
    setGlobalValue(name, makeFunction(std::move(fn), arity, name));
}
inline Value getGlobal(std::string_view name) {
    auto gv = globalValue(name);
    return gv.found ? gv.value : undefined();
}
} // namespace bronze::embed

namespace brolm::api {

namespace ev = bronze::embed;
using Value = bronze::Value;

// bronze's HostClass (embed/embed_host_class.h): a constructor and prototype
// per realm, branded handles, and unwrap() checked across inherit() lineages.
// It used to be a copy kept per library, one constructor and prototype per
// thread; the realms of one thread then shared a class's mutable prototype,
// so one realm could change what another's instances do.
using HostClass = bronze::embed::HostClass;
using bronze::embed::hostArrayOf;

} // namespace brolm::api
