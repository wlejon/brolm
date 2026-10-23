// bro.lm.VerifiedValidator and bro.lm.createVerifiedValidator(predicateOrSpec, opts):
// SMT-Guided Neuro-Symbolic Constraint Solving & Repair for LLM Structured Outputs.

#include "host_lm_internal.h"

#include <brolm/verified_validator.h>
#include <brotensor/json.h>

#include <cmath>
#include <unordered_set>

namespace brolm::api {

HostClass g_verifiedValidatorClass;

namespace {

struct HostVerifiedValidator {
    brolm::VerifiedValidator validator;
    ev::Persistent predicateFn;
    bool positionalArgs = false;
    bool autoDiscoverFields = true;
    bool hasDefaultMin = false;
    int64_t defaultMin = -1000000;
    bool hasDefaultMax = false;
    int64_t defaultMax = 1000000;
    uint32_t defaultBitWidth = 32;
    ev::Persistent activeTemplateObj;
};

HostVerifiedValidator* hostValidatorPtr(Value v) {
    if (!ev::isObject(v)) return nullptr;
    return static_cast<HostVerifiedValidator*>(g_verifiedValidatorClass.unwrap(v));
}

// Obtain Object.keys(obj) safely under the moving GC contract.
std::vector<std::string> getObjectKeys(Value objVal) {
    if (!ev::isObject(objVal)) return {};
    ev::Persistent obj(objVal);
    auto objGlobal = ev::globalValue("Object");
    if (!objGlobal.found || !ev::isObject(objGlobal.value)) return {};
    ev::Persistent objCtor(objGlobal.value);
    ev::Persistent keysFn(ev::getProperty(objCtor.get(), "keys"));
    if (!ev::isFunction(keysFn.get())) return {};
    const Value arg = obj.get();
    ev::CallResult res = ev::call(keysFn.get(), objCtor.get(), std::span<const Value>(&arg, 1));
    if (res.thrown || !ev::isObject(res.value)) return {};
    return readStringArray(res.value);
}

// Call fn.toString() to inspect JS function source when available.
std::string getFunctionSource(Value fnVal) {
    if (!ev::isFunction(fnVal)) return {};
    ev::Persistent fn(fnVal);
    ev::Persistent toStr(ev::getProperty(fn.get(), "toString"));
    if (!ev::isFunction(toStr.get())) return {};
    ev::CallResult res = ev::call(toStr.get(), fn.get(), {});
    if (res.thrown || !ev::isString(res.value)) return {};
    return ev::toUtf8(res.value);
}

bool stringifyJsonValue(Value v, std::string& out) {
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

Value buildJsObjectFromFields(const HostVerifiedValidator& w,
                              const brolm::FieldMap& fields,
                              Value templateObjVal) {
    ev::Persistent templateObj;
    if (ev::isObject(templateObjVal)) templateObj.set(templateObjVal);

    ev::Persistent outObj(ev::createObject());
    std::unordered_set<std::string> written;

    // Copy all properties from templateObj first, updating any repaired fields
    if (ev::isObject(templateObj.get())) {
        std::vector<std::string> keys = getObjectKeys(templateObj.get());
        for (const auto& k : keys) {
            written.insert(k);
            ev::Persistent origVal(ev::getProperty(templateObj.get(), k));
            auto it = fields.find(k);
            if (it != fields.end()) {
                const brolm::FieldConstraint* fc = w.validator.find_field(k);
                bool as_bool = (fc && fc->type == brolm::FieldType::Bool) || ev::isBool(origVal.get());
                ev::Persistent repVal(as_bool ? ev::fromBool(it->second != 0)
                                              : ev::fromDouble(static_cast<double>(it->second)));
                outObj.set(ev::setProperty(outObj.get(), k, repVal.get()));
            } else {
                outObj.set(ev::setProperty(outObj.get(), k, origVal.get()));
            }
        }
    }

    // Write declared fields in declaration order
    for (const auto& fc : w.validator.fields()) {
        if (written.count(fc.name)) continue;
        auto it = fields.find(fc.name);
        if (it != fields.end()) {
            written.insert(fc.name);
            ev::Persistent repVal(fc.type == brolm::FieldType::Bool
                ? ev::fromBool(it->second != 0)
                : ev::fromDouble(static_cast<double>(it->second)));
            outObj.set(ev::setProperty(outObj.get(), fc.name, repVal.get()));
        }
    }

    // Write any remaining fields in `fields`
    for (const auto& [k, v] : fields) {
        if (written.count(k)) continue;
        ev::Persistent repVal(ev::fromDouble(static_cast<double>(v)));
        outObj.set(ev::setProperty(outObj.get(), k, repVal.get()));
    }

    return outObj.get();
}

void bindJsPredicate(HostVerifiedValidator* w) {
    if (!ev::isFunction(w->predicateFn.get())) return;
    w->validator.set_predicate([w](const brolm::FieldMap& fmap) -> bool {
        if (!ev::isFunction(w->predicateFn.get())) return true;
        if (w->positionalArgs && !w->validator.fields().empty()) {
            const auto& fspecs = w->validator.fields();
            std::vector<ev::Persistent> rootedArgs;
            rootedArgs.reserve(fspecs.size());
            for (const auto& fc : fspecs) {
                auto it = fmap.find(fc.name);
                int64_t val = (it != fmap.end()) ? it->second : 0;
                Value jv = (fc.type == brolm::FieldType::Bool)
                    ? ev::fromBool(val != 0)
                    : ev::fromDouble(static_cast<double>(val));
                rootedArgs.emplace_back(jv);
            }
            std::vector<Value> rawArgs;
            rawArgs.reserve(rootedArgs.size());
            for (const auto& p : rootedArgs) rawArgs.push_back(p.get());
            ev::CallResult r = ev::call(w->predicateFn.get(), ev::undefined(),
                                        std::span<const Value>(rawArgs.data(), rawArgs.size()));
            return !r.thrown && ev::toBool(r.value);
        }

        ev::Persistent candObj(buildJsObjectFromFields(*w, fmap, w->activeTemplateObj.get()));
        const Value arg = candObj.get();
        ev::CallResult r = ev::call(w->predicateFn.get(), ev::undefined(),
                                    std::span<const Value>(&arg, 1));
        return !r.thrown && ev::toBool(r.value);
    }, "predicateFn");
}

void parseFieldDescriptorObject(HostVerifiedValidator& w, const std::string& fallbackName, Value descVal) {
    ev::Persistent desc(descVal);
    brolm::FieldConstraint fc;
    fc.name = fallbackName;
    fc.bit_width = w.defaultBitWidth;
    if (w.hasDefaultMin) { fc.has_min = true; fc.min_val = w.defaultMin; }
    if (w.hasDefaultMax) { fc.has_max = true; fc.max_val = w.defaultMax; }

    if (ev::isString(desc.get())) {
        std::string t = ev::toUtf8(desc.get());
        if (t == "bool" || t == "boolean") {
            fc = brolm::FieldConstraint::boolean(fallbackName);
        } else {
            fc.name = fallbackName.empty() ? t : fallbackName;
        }
        if (!fc.name.empty()) w.validator.add_field(fc);
        return;
    }

    if (!ev::isObject(desc.get())) return;

    // Support [min, max] array tuple shorthand: { port: [1024, 65535] }
    double arrLen = 0;
    if (!fallbackName.empty() && propNumber(desc.get(), "length", arrLen) && arrLen >= 2) {
        Value loV = ev::getElement(desc.get(), 0);
        if (ev::isNumber(loV)) { fc.has_min = true; fc.min_val = static_cast<int64_t>(ev::toDouble(loV)); }
        Value hiV = ev::getElement(desc.get(), 1);
        if (ev::isNumber(hiV)) { fc.has_max = true; fc.max_val = static_cast<int64_t>(ev::toDouble(hiV)); }
        w.validator.add_field(fc);
        return;
    }

    std::string parsedName;
    if (propString(desc.get(), "name", parsedName) && !parsedName.empty()) {
        fc.name = parsedName;
    }
    if (fc.name.empty()) return;

    std::string typeStr;
    if (propString(desc.get(), "type", typeStr)) {
        if (typeStr == "bool" || typeStr == "boolean") {
            fc.type = brolm::FieldType::Bool;
            fc.has_min = true;
            fc.min_val = 0;
            fc.has_max = true;
            fc.max_val = 1;
        } else if (typeStr == "uint" || typeStr == "unsigned") {
            fc.type = brolm::FieldType::UInt;
            if (!fc.has_min) { fc.has_min = true; fc.min_val = 0; }
        }
    }

    double d = 0;
    if (propNumber(desc.get(), "min", d) || propNumber(desc.get(), "minimum", d)) {
        fc.has_min = true;
        fc.min_val = static_cast<int64_t>(d);
    }
    if (propNumber(desc.get(), "max", d) || propNumber(desc.get(), "maximum", d)) {
        fc.has_max = true;
        fc.max_val = static_cast<int64_t>(d);
    }
    if (propNumber(desc.get(), "bitWidth", d) || propNumber(desc.get(), "width", d)) {
        uint32_t bw = static_cast<uint32_t>(d);
        if (bw == 8 || bw == 16 || bw == 32 || bw == 64) fc.bit_width = bw;
    }
    if (propNumber(desc.get(), "alignment", d) || propNumber(desc.get(), "step", d) ||
        propNumber(desc.get(), "multipleOf", d)) {
        if (d > 1) fc.alignment = static_cast<uint64_t>(d);
    }
    if (propNumber(desc.get(), "nonzeroMask", d) || propNumber(desc.get(), "bitmask", d)) {
        fc.nonzero_mask = static_cast<uint64_t>(d);
    }
    if (propNumber(desc.get(), "zeroMask", d)) {
        fc.zero_mask = static_cast<uint64_t>(d);
    }

    w.validator.add_field(fc);
}

void parseValidatorOptions(HostVerifiedValidator& w, Value optsVal) {
    if (!ev::isObject(optsVal)) return;
    ev::Persistent opts(optsVal);

    double d = 0;
    if (propNumber(opts.get(), "min", d)) {
        w.hasDefaultMin = true;
        w.defaultMin = static_cast<int64_t>(d);
    }
    if (propNumber(opts.get(), "max", d)) {
        w.hasDefaultMax = true;
        w.defaultMax = static_cast<int64_t>(d);
    }
    if (propNumber(opts.get(), "bitWidth", d) || propNumber(opts.get(), "width", d)) {
        uint32_t bw = static_cast<uint32_t>(d);
        if (bw == 8 || bw == 16 || bw == 32 || bw == 64) w.defaultBitWidth = bw;
    }
    bool b = false;
    if (propBool(opts.get(), "positional", b)) {
        w.positionalArgs = b;
    }

    // Parse opts.fields (array of names/descriptors or object map of descriptors)
    ev::Persistent fieldsVal(ev::getProperty(opts.get(), "fields"));
    if (ev::isObject(fieldsVal.get())) {
        double len = 0;
        if (propNumber(fieldsVal.get(), "length", len)) {
            uint32_t n = static_cast<uint32_t>(len);
            for (uint32_t i = 0; i < n; ++i) {
                Value elem = ev::getElement(fieldsVal.get(), i);
                parseFieldDescriptorObject(w, "", elem);
            }
        } else {
            std::vector<std::string> keys = getObjectKeys(fieldsVal.get());
            for (const auto& k : keys) {
                Value prop = ev::getProperty(fieldsVal.get(), k);
                parseFieldDescriptorObject(w, k, prop);
            }
        }
    }

    // Parse opts.constraints / opts.constraint / opts.expression
    for (std::string_view key : {"constraints", "constraint", "expression", "invariant"}) {
        ev::Persistent cval(ev::getProperty(opts.get(), key));
        if (ev::isString(cval.get())) {
            w.validator.add_constraint(ev::toUtf8(cval.get()));
        } else if (ev::isObject(cval.get())) {
            for (const auto& s : readStringArray(cval.get())) {
                w.validator.add_constraint(s);
            }
        }
    }

    // Parse opts.predicate if provided on the spec object
    ev::Persistent predVal(ev::getProperty(opts.get(), "predicate"));
    if (!ev::isFunction(predVal.get())) {
        predVal.set(ev::getProperty(opts.get(), "predicateFn"));
    }
    if (ev::isFunction(predVal.get())) {
        w.predicateFn.set(predVal.get());
    }
}

bool extractCandidateFields(HostVerifiedValidator& w, Value inputVal,
                            brolm::FieldMap& outFields,
                            ev::Persistent& outTemplateObj,
                            std::string& outSyntaxErr) {
    ev::Persistent input(inputVal);
    if (ev::isString(input.get())) {
        std::string text = ev::toUtf8(input.get());
        ev::CallResult parsed = ev::parseJson(text);
        if (parsed.thrown || !ev::isObject(parsed.value)) {
            outSyntaxErr = "JSON syntax error";
            return false;
        }
        input.set(parsed.value);
    }
    if (!ev::isObject(input.get())) {
        outSyntaxErr = "Candidate must be an object or JSON string";
        return false;
    }

    outTemplateObj.set(input.get());
    std::vector<std::string> keys = getObjectKeys(input.get());
    bool hadPredeclaredFields = !w.validator.fields().empty();

    for (const auto& k : keys) {
        Value pv = ev::getProperty(input.get(), k);
        if (ev::isNumber(pv)) {
            double dv = ev::toDouble(pv);
            if (std::isfinite(dv)) {
                outFields[k] = static_cast<int64_t>(std::llround(dv));
                if (!hadPredeclaredFields && w.autoDiscoverFields && !w.validator.has_field(k)) {
                    brolm::FieldConstraint fc = brolm::FieldConstraint::unbounded_int(k, w.defaultBitWidth);
                    if (w.hasDefaultMin) { fc.has_min = true; fc.min_val = w.defaultMin; }
                    if (w.hasDefaultMax) { fc.has_max = true; fc.max_val = w.defaultMax; }
                    w.validator.add_field(fc);
                }
            }
        } else if (ev::isBool(pv)) {
            outFields[k] = ev::toBool(pv) ? 1 : 0;
            if (!hadPredeclaredFields && w.autoDiscoverFields && !w.validator.has_field(k)) {
                w.validator.add_field(brolm::FieldConstraint::boolean(k));
            }
        }
    }
    return true;
}

Value makeStringArrayValue(const std::vector<std::string>& items) {
    ev::Persistent arr(ev::makeArray(static_cast<uint32_t>(items.size())));
    for (size_t i = 0; i < items.size(); ++i) {
        ev::Persistent s(ev::fromUtf8(items[i]));
        arr.set(ev::setElement(arr.get(), static_cast<uint32_t>(i), s.get()));
    }
    return arr.get();
}

Value makeInt64ArrayValue(const std::vector<int64_t>& items) {
    ev::Persistent arr(ev::makeArray(static_cast<uint32_t>(items.size())));
    for (size_t i = 0; i < items.size(); ++i) {
        ev::Persistent n(ev::fromDouble(static_cast<double>(items[i])));
        arr.set(ev::setElement(arr.get(), static_cast<uint32_t>(i), n.get()));
    }
    return arr.get();
}

brolm::RepairOptions parseRepairOpts(Value optsVal) {
    brolm::RepairOptions ropts;
    if (!ev::isObject(optsVal)) return ropts;
    ev::Persistent opts(optsVal);
    double d = 0;
    if (propNumber(opts.get(), "maxIterations", d) || propNumber(opts.get(), "maxCegisIterations", d)) {
        if (d >= 1) ropts.max_cegis_iterations = static_cast<uint32_t>(d);
    }
    if (propNumber(opts.get(), "maxConflicts", d) || propNumber(opts.get(), "maxSatConflicts", d)) {
        if (d >= 100) ropts.max_sat_conflicts = static_cast<uint32_t>(d);
    }
    bool b = true;
    if (propBool(opts.get(), "minimizePerturbation", b)) {
        ropts.minimize_perturbation = b;
    }
    ev::Persistent fixedVal(ev::getProperty(opts.get(), "fixedFields"));
    if (!ev::isObject(fixedVal.get())) {
        fixedVal.set(ev::getProperty(opts.get(), "fixed"));
    }
    if (ev::isObject(fixedVal.get())) {
        ropts.fixed_fields = readStringArray(fixedVal.get());
    }
    return ropts;
}

Value buildValidatorInstance(std::span<const Value> a) {
    if (a.empty() || ev::isUndefined(a[0]) || ev::isNull(a[0])) {
        return ev::throwTypeError("createVerifiedValidator(predicateOrSpec, opts): predicate function, constraint string, or spec object required");
    }

    auto w = std::make_unique<HostVerifiedValidator>();

    // Parse options (2nd arg, or 1st arg if 1st arg is a spec object)
    if (a.size() >= 2 && ev::isObject(a[1])) {
        parseValidatorOptions(*w, a[1]);
    }

    if (ev::isFunction(a[0])) {
        w->predicateFn.set(a[0]);
    } else if (ev::isString(a[0])) {
        std::string s = ev::toUtf8(a[0]);
        if (s.find("=>") != std::string::npos || s.rfind("function", 0) == 0) {
            w->validator.add_constraints_from_js_source(s, &w->positionalArgs);
        } else {
            w->validator.add_constraint(s);
        }
    } else if (ev::isObject(a[0])) {
        parseValidatorOptions(*w, a[0]);
    } else {
        return ev::throwTypeError("createVerifiedValidator: first argument must be a function, string, or spec object");
    }

    // If a JS predicate function is attached, inspect its source text via fn.toString()
    // so symbolic conjuncts are automatically lowered into BVContext too!
    if (ev::isFunction(w->predicateFn.get())) {
        std::string src = getFunctionSource(w->predicateFn.get());
        if (!src.empty()) {
            bool pos = w->positionalArgs;
            w->validator.add_constraints_from_js_source(src, &pos);
            w->positionalArgs = pos;
        }
        // Apply default bounds to any auto-discovered unbounded fields
        if (w->hasDefaultMin || w->hasDefaultMax || w->defaultBitWidth != 32) {
            std::vector<brolm::FieldConstraint> updated = w->validator.fields();
            for (auto& fc : updated) {
                fc.bit_width = w->defaultBitWidth;
                if (w->hasDefaultMin && !fc.has_min) { fc.has_min = true; fc.min_val = w->defaultMin; }
                if (w->hasDefaultMax && !fc.has_max) { fc.has_max = true; fc.max_val = w->defaultMax; }
                w->validator.add_field(fc);
            }
        }
        bindJsPredicate(w.get());
    }

    // Also auto-discover fields from symbolic constraints if fields were not explicitly listed
    if (w->validator.fields().empty() && !w->validator.constraints().empty()) {
        for (const auto& sc : w->validator.constraints()) {
            // Scan identifiers in sc.expression
            const std::string& expr = sc.expression;
            size_t i = 0;
            while (i < expr.size()) {
                if (std::isalpha(static_cast<unsigned char>(expr[i])) || expr[i] == '_') {
                    size_t start = i++;
                    while (i < expr.size() &&
                           (std::isalnum(static_cast<unsigned char>(expr[i])) || expr[i] == '_')) {
                        ++i;
                    }
                    std::string id = expr.substr(start, i - start);
                    if (id != "true" && id != "false" && id != "Math" &&
                        id != "abs" && id != "min" && id != "max" && id != "imul" && id != "clz32" &&
                        !w->validator.has_field(id)) {
                        brolm::FieldConstraint fc = brolm::FieldConstraint::unbounded_int(id, w->defaultBitWidth);
                        if (w->hasDefaultMin) { fc.has_min = true; fc.min_val = w->defaultMin; }
                        if (w->hasDefaultMax) { fc.has_max = true; fc.max_val = w->defaultMax; }
                        w->validator.add_field(fc);
                    }
                } else {
                    ++i;
                }
            }
        }
    }

    ev::Persistent inst(g_verifiedValidatorClass.createInstance(std::move(w)));
    // Attach .grammarMask getter/property for seamless composition with brolm's grammar sampler
    auto* ptr = hostValidatorPtr(inst.get());
    if (ptr) {
        ev::Persistent gramVal(makeGrammarValue(ptr->validator.build_grammar()));
        inst.set(ev::setProperty(inst.get(), "grammarMask", gramVal.get()));
    }
    return inst.get();
}

void decorateVerifiedValidator(ObjectBuilder& b) {
    b.def("validate", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* w = hostValidatorPtr(self);
        if (!w) return ev::throwTypeError("validate: not a VerifiedValidator");
        if (a.empty()) return ev::throwTypeError("validate(obj): candidate object or JSON string required");

        brolm::FieldMap fmap;
        ev::Persistent tmpl;
        std::string synErr;
        if (!extractCandidateFields(*w, a[0], fmap, tmpl, synErr)) {
            ObjectBuilder out;
            out.set("valid", false);
            ev::Persistent vArr(makeStringArrayValue({synErr}));
            out.set("violations", vArr.get());
            return out.build();
        }

        w->activeTemplateObj.set(tmpl.get());
        brolm::ValidationResult vres = w->validator.validate(fmap);
        w->activeTemplateObj.set(ev::undefined());

        ObjectBuilder out;
        out.set("valid", vres.valid);
        ev::Persistent vArr(makeStringArrayValue(vres.violations));
        out.set("violations", vArr.get());
        return out.build();
    });

    auto repairFn = [](Value self, std::span<const Value> a) -> Value {
        auto* w = hostValidatorPtr(self);
        if (!w) return ev::throwTypeError("repair: not a VerifiedValidator");
        if (a.empty()) return ev::throwTypeError("repair(obj, opts): candidate object or JSON string required");

        brolm::FieldMap fmap;
        ev::Persistent tmpl;
        std::string synErr;
        if (!extractCandidateFields(*w, a[0], fmap, tmpl, synErr)) {
            ObjectBuilder out;
            out.set("valid", false);
            out.set("repaired", ev::null());
            out.set("smtSolved", false);
            out.set("iterations", 0.0);
            ev::Persistent vArr(makeStringArrayValue({synErr}));
            out.set("violations", vArr.get());
            return out.build();
        }

        brolm::RepairOptions ropts = (a.size() >= 2) ? parseRepairOpts(a[1]) : brolm::RepairOptions{};
        w->activeTemplateObj.set(tmpl.get());
        brolm::RepairResult rres = w->validator.repair_with_smt(fmap, ropts);
        w->activeTemplateObj.set(ev::undefined());

        ObjectBuilder out;
        out.set("valid", rres.valid);
        out.set("smtSolved", rres.smt_solved);
        out.set("iterations", static_cast<double>(rres.iterations));
        ev::Persistent vArr(makeStringArrayValue(rres.violations));
        out.set("violations", vArr.get());

        if (rres.valid) {
            ev::Persistent repObj(buildJsObjectFromFields(*w, rres.repaired_fields, tmpl.get()));
            out.set("repaired", repObj.get());
            std::string jsonStr;
            if (stringifyJsonValue(repObj.get(), jsonStr)) {
                out.set("repairedJson", jsonStr);
            }
        } else {
            out.set("repaired", ev::null());
            out.set("repairedJson", "");
        }
        return out.build();
    };

    b.def("repair", 2, repairFn);
    b.def("repairWithSmt", 2, repairFn);

    b.def("generateValid", 1, [](Value self, std::span<const Value> a) -> Value {
        auto* w = hostValidatorPtr(self);
        if (!w) return ev::throwTypeError("generateValid: not a VerifiedValidator");

        brolm::FieldMap seedMap;
        ev::Persistent tmpl;
        if (!a.empty() && ev::isObject(a[0])) {
            std::string dummyErr;
            extractCandidateFields(*w, a[0], seedMap, tmpl, dummyErr);
        }

        brolm::RepairOptions ropts = (a.size() >= 2) ? parseRepairOpts(a[1]) : brolm::RepairOptions{};
        w->activeTemplateObj.set(tmpl.get());
        brolm::RepairResult rres = w->validator.generate_valid(seedMap, ropts);
        w->activeTemplateObj.set(ev::undefined());

        ObjectBuilder out;
        out.set("valid", rres.valid);
        out.set("smtSolved", rres.smt_solved);
        out.set("iterations", static_cast<double>(rres.iterations));
        if (rres.valid) {
            ev::Persistent repObj(buildJsObjectFromFields(*w, rres.repaired_fields, tmpl.get()));
            out.set("repaired", repObj.get());
            out.set("value", repObj.get());
            out.set("object", repObj.get());
        } else {
            out.set("repaired", ev::null());
            out.set("value", ev::null());
            out.set("object", ev::null());
        }
        return out.build();
    });

    b.def("validDomainForField", 2, [](Value self, std::span<const Value> a) -> Value {
        auto* w = hostValidatorPtr(self);
        if (!w) return ev::throwTypeError("validDomainForField: not a VerifiedValidator");
        if (a.size() < 2 || !ev::isString(a[1])) {
            return ev::throwTypeError("validDomainForField(partialObj, fieldName): fieldName string required");
        }
        std::string fieldName = ev::toUtf8(a[1]);
        brolm::FieldMap partialMap;
        ev::Persistent tmpl;
        if (ev::isObject(a[0]) || ev::isString(a[0])) {
            std::string dummyErr;
            extractCandidateFields(*w, a[0], partialMap, tmpl, dummyErr);
        }

        uint32_t maxSamples = 8;
        if (a.size() >= 3 && ev::isObject(a[2])) {
            double ms = 0;
            if (propNumber(a[2], "maxSamples", ms) && ms >= 1) {
                maxSamples = static_cast<uint32_t>(ms);
            }
        }

        w->activeTemplateObj.set(tmpl.get());
        brolm::FieldDomainResult dom = w->validator.valid_domain_for_field(partialMap, fieldName, maxSamples);
        w->activeTemplateObj.set(ev::undefined());

        ObjectBuilder out;
        out.set("feasible", dom.feasible);
        out.set("valid", dom.feasible);
        out.set("field", fieldName);
        if (dom.feasible) {
            out.set("min", static_cast<double>(dom.min_val));
            out.set("max", static_cast<double>(dom.max_val));
            out.set("sample", static_cast<double>(dom.sample_val));
            ev::Persistent valsArr(makeInt64ArrayValue(dom.sample_values));
            out.set("values", valsArr.get());
        } else {
            out.set("min", ev::null());
            out.set("max", ev::null());
            out.set("sample", ev::null());
            ev::Persistent emptyArr(ev::makeArray(0));
            out.set("values", emptyArr.get());
        }
        return out.build();
    });

    b.def("grammar", 0, [](Value self, std::span<const Value>) -> Value {
        auto* w = hostValidatorPtr(self);
        if (!w) return ev::throwTypeError("grammar: not a VerifiedValidator");
        return makeGrammarValue(w->validator.build_grammar());
    });

    b.def("fieldGrammar", 2, [](Value self, std::span<const Value> a) -> Value {
        auto* w = hostValidatorPtr(self);
        if (!w) return ev::throwTypeError("fieldGrammar: not a VerifiedValidator");
        if (a.size() < 2 || !ev::isString(a[1])) {
            return ev::throwTypeError("fieldGrammar(partialObj, fieldName): fieldName string required");
        }
        std::string fieldName = ev::toUtf8(a[1]);
        brolm::FieldMap partialMap;
        ev::Persistent tmpl;
        if (ev::isObject(a[0]) || ev::isString(a[0])) {
            std::string dummyErr;
            extractCandidateFields(*w, a[0], partialMap, tmpl, dummyErr);
        }
        w->activeTemplateObj.set(tmpl.get());
        brolm::Grammar g = w->validator.build_field_grammar(partialMap, fieldName);
        w->activeTemplateObj.set(ev::undefined());
        return makeGrammarValue(std::move(g));
    });
}

}  // namespace

Value js_createVerifiedValidator(Value, std::span<const Value> a) {
    return buildValidatorInstance(a);
}

void registerLMVerifiedValidatorClass() {
    g_verifiedValidatorClass.install(
        "VerifiedValidator", 2,
        [](Value, std::span<const Value> a) -> Value { return buildValidatorInstance(a); },
        decorateVerifiedValidator);
}

}  // namespace brolm::api
