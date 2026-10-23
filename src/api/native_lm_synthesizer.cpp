// bro.lm.synthesizeVerifiedFunction(spec, opts) and bro.lm.solveWithProver(problemSpec, opts):
// Neuro-Symbolic CEGIS Function Synthesizer and Prover-Guided Problem Solver (Issue #150).

#include "host_lm_internal.h"

#include <brolm/verified_validator.h>

#include <cmath>
#include <unordered_set>

namespace brolm::api {

namespace {

std::vector<std::string> getObjKeys(Value objVal) {
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

std::string getFnSource(Value fnVal) {
    if (!ev::isFunction(fnVal)) return {};
    ev::Persistent fn(fnVal);
    ev::Persistent toStr(ev::getProperty(fn.get(), "toString"));
    if (!ev::isFunction(toStr.get())) return {};
    ev::CallResult res = ev::call(toStr.get(), fn.get(), {});
    if (res.thrown || !ev::isString(res.value)) return {};
    return ev::toUtf8(res.value);
}

Value makeStringArray(const std::vector<std::string>& items) {
    return hostArrayOf(items.size(), [&](size_t i) {
        return ev::fromUtf8(items[i]);
    });
}

Value makeInt64Array(const std::vector<int64_t>& items) {
    return hostArrayOf(items.size(), [&](size_t i) {
        return ev::fromDouble(static_cast<double>(items[i]));
    });
}

Value makeFieldMapObject(const brolm::FieldMap& fmap,
                         const std::vector<brolm::FieldConstraint>& fields) {
    ev::Persistent out(ev::createObject());
    std::unordered_set<std::string> written;
    for (const auto& fc : fields) {
        auto it = fmap.find(fc.name);
        if (it != fmap.end()) {
            written.insert(fc.name);
            ev::Persistent v(fc.type == brolm::FieldType::Bool
                ? ev::fromBool(it->second != 0)
                : ev::fromDouble(static_cast<double>(it->second)));
            out.set(ev::setProperty(out.get(), fc.name, v.get()));
        }
    }
    for (const auto& [k, val] : fmap) {
        if (written.count(k)) continue;
        ev::Persistent v(ev::fromDouble(static_cast<double>(val)));
        out.set(ev::setProperty(out.get(), k, v.get()));
    }
    return out.get();
}

Value makeCounterexampleObject(const brolm::CounterexampleRecord& cex,
                               const std::vector<std::string>& paramOrder) {
    ObjectBuilder b;
    ev::Persistent inArr(makeInt64Array(cex.inputs));
    b.set("inputs", inArr.get());

    std::vector<brolm::FieldConstraint> orderSpecs;
    for (const auto& p : paramOrder) {
        orderSpecs.push_back(brolm::FieldConstraint::unbounded_int(p));
    }
    ev::Persistent paramObj(makeFieldMapObject(cex.params, orderSpecs));
    b.set("params", paramObj.get());
    b.set("candidateOutput", static_cast<double>(cex.candidate_output));
    if (cex.has_expected_output) {
        b.set("expectedOutput", static_cast<double>(cex.expected_output));
    } else {
        b.set("expectedOutput", ev::null());
    }
    b.set("candidateSource", cex.candidate_source);
    b.set("violation", cex.violation);
    return b.build();
}

void parseFieldDomainsFromJs(Value fieldsVal,
                             int64_t defMin,
                             int64_t defMax,
                             uint32_t defWidth,
                             std::vector<brolm::FieldConstraint>& outDomains,
                             std::vector<std::string>& outParams) {
    if (!ev::isObject(fieldsVal)) return;
    ev::Persistent fobj(fieldsVal);

    auto parseOne = [&](const std::string& fallbackName, Value descVal) {
        ev::Persistent desc(descVal);
        brolm::FieldConstraint fc = brolm::FieldConstraint::integer(fallbackName, defMin, defMax, defWidth);
        if (ev::isString(desc.get())) {
            std::string s = ev::toUtf8(desc.get());
            if (s == "bool" || s == "boolean") {
                fc = brolm::FieldConstraint::boolean(fallbackName);
            } else if (fallbackName.empty()) {
                fc.name = s;
            }
        } else if (ev::isObject(desc.get())) {
            double arrLen = 0;
            if (!fallbackName.empty() && propNumber(desc.get(), "length", arrLen) && arrLen >= 2) {
                Value loV = ev::getElement(desc.get(), 0);
                Value hiV = ev::getElement(desc.get(), 1);
                if (ev::isNumber(loV)) { fc.has_min = true; fc.min_val = static_cast<int64_t>(ev::toDouble(loV)); }
                if (ev::isNumber(hiV)) { fc.has_max = true; fc.max_val = static_cast<int64_t>(ev::toDouble(hiV)); }
            } else {
                std::string n;
                if (propString(desc.get(), "name", n) && !n.empty()) fc.name = n;
                std::string t;
                if (propString(desc.get(), "type", t) && (t == "bool" || t == "boolean")) {
                    fc.type = brolm::FieldType::Bool;
                    fc.has_min = true; fc.min_val = 0;
                    fc.has_max = true; fc.max_val = 1;
                }
                double d = 0;
                if (propNumber(desc.get(), "min", d) || propNumber(desc.get(), "minimum", d)) {
                    fc.has_min = true; fc.min_val = static_cast<int64_t>(d);
                }
                if (propNumber(desc.get(), "max", d) || propNumber(desc.get(), "maximum", d)) {
                    fc.has_max = true; fc.max_val = static_cast<int64_t>(d);
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
            }
        }
        if (!fc.name.empty()) {
            outDomains.push_back(fc);
            if (std::find(outParams.begin(), outParams.end(), fc.name) == outParams.end()) {
                outParams.push_back(fc.name);
            }
        }
    };

    double len = 0;
    if (propNumber(fobj.get(), "length", len)) {
        uint32_t n = static_cast<uint32_t>(len);
        for (uint32_t i = 0; i < n; ++i) {
            parseOne("", ev::getElement(fobj.get(), i));
        }
    } else {
        for (const auto& k : getObjKeys(fobj.get())) {
            parseOne(k, ev::getProperty(fobj.get(), k));
        }
    }
}

bool readFieldMapFromJsObject(Value objVal, brolm::FieldMap& outMap) {
    if (!ev::isObject(objVal)) return false;
    ev::Persistent obj(objVal);
    std::vector<std::string> keys = getObjKeys(obj.get());
    for (const auto& k : keys) {
        Value v = ev::getProperty(obj.get(), k);
        if (ev::isNumber(v)) {
            double d = ev::toDouble(v);
            if (std::isfinite(d)) outMap[k] = static_cast<int64_t>(std::llround(d));
        } else if (ev::isBool(v)) {
            outMap[k] = ev::toBool(v) ? 1 : 0;
        }
    }
    return !outMap.empty();
}

}  // namespace

Value js_synthesizeVerifiedFunction(Value, std::span<const Value> a) {
    if (a.empty()) {
        return ev::throwTypeError("synthesizeVerifiedFunction(spec, opts): specification required");
    }

    brolm::SynthesizeSpec spec;
    ev::Persistent postFnP;
    ev::Persistent refFnP;
    ev::Persistent synthFnP;

    auto applySpecObj = [&](Value objVal) {
        if (!ev::isObject(objVal)) return;
        ev::Persistent o(objVal);

        propString(o.get(), "name", spec.name);
        propString(o.get(), "outputVar", spec.output_var);

        double d = 0;
        if (propNumber(o.get(), "min", d)) spec.default_min = static_cast<int64_t>(d);
        if (propNumber(o.get(), "max", d)) spec.default_max = static_cast<int64_t>(d);
        if (propNumber(o.get(), "bitWidth", d) || propNumber(o.get(), "width", d)) {
            uint32_t bw = static_cast<uint32_t>(d);
            if (bw == 32 || bw == 64) spec.bit_width = bw;
        }
        if (propNumber(o.get(), "maxIterations", d) || propNumber(o.get(), "maxCegisIterations", d)) {
            if (d >= 1) spec.max_cegis_iterations = static_cast<uint32_t>(d);
        }
        if (propNumber(o.get(), "maxConflicts", d) || propNumber(o.get(), "maxSatConflicts", d)) {
            if (d >= 100) spec.max_sat_conflicts = static_cast<uint32_t>(d);
        }

        // Domain shorthand: domain: [min, max]
        ev::Persistent domV(ev::getProperty(o.get(), "domain"));
        double domLen = 0;
        if (ev::isObject(domV.get()) && propNumber(domV.get(), "length", domLen) && domLen >= 2) {
            Value loV = ev::getElement(domV.get(), 0);
            Value hiV = ev::getElement(domV.get(), 1);
            if (ev::isNumber(loV)) spec.default_min = static_cast<int64_t>(ev::toDouble(loV));
            if (ev::isNumber(hiV)) spec.default_max = static_cast<int64_t>(ev::toDouble(hiV));
        }

        for (std::string_view pkey : {"params", "parameters", "inputs", "args", "inputNames"}) {
            Value pv = ev::getProperty(o.get(), pkey);
            if (ev::isObject(pv)) {
                auto arr = readStringArray(pv);
                if (!arr.empty()) spec.params = std::move(arr);
            }
        }

        for (std::string_view fkey : {"fields", "domains", "paramDomains"}) {
            Value fv = ev::getProperty(o.get(), fkey);
            if (ev::isObject(fv)) {
                parseFieldDomainsFromJs(fv, spec.default_min, spec.default_max, spec.bit_width,
                                        spec.param_domains, spec.params);
            }
        }

        for (std::string_view preKey : {"precondition", "requires", "assume"}) {
            std::string preStr;
            if (propString(o.get(), preKey, preStr) && !preStr.empty()) {
                spec.precondition = preStr;
            }
        }

        for (std::string_view postKey : {"postcondition", "ensure", "ensures", "invariant",
                                         "constraint", "constraints", "spec"}) {
            ev::Persistent pv(ev::getProperty(o.get(), postKey));
            if (ev::isString(pv.get())) {
                spec.postcondition = ev::toUtf8(pv.get());
            } else if (ev::isFunction(pv.get())) {
                postFnP.set(pv.get());
                std::string fsrc = getFnSource(pv.get());
                std::vector<std::string> fparams;
                std::string fexpr;
                if (brolm::extract_js_function_expression(fsrc, &fparams, &fexpr) && !fexpr.empty()) {
                    if (fparams.size() >= 2) {
                        spec.output_var = fparams.back();
                        if (spec.params.empty()) {
                            spec.params.assign(fparams.begin(), fparams.end() - 1);
                        }
                    }
                    spec.postcondition = fexpr;
                }
            }
        }

        for (std::string_view refKey : {"referenceFn", "reference", "oracle", "targetFn"}) {
            ev::Persistent rv(ev::getProperty(o.get(), refKey));
            if (ev::isString(rv.get())) {
                spec.reference_expression = ev::toUtf8(rv.get());
            } else if (ev::isFunction(rv.get())) {
                refFnP.set(rv.get());
                std::string rsrc = getFnSource(rv.get());
                if (!rsrc.empty()) spec.reference_expression = rsrc;
            }
        }

        for (std::string_view candKey : {"candidates", "initialCandidates", "candidate"}) {
            ev::Persistent cv(ev::getProperty(o.get(), candKey));
            if (ev::isString(cv.get())) {
                spec.candidates.push_back(ev::toUtf8(cv.get()));
            } else if (ev::isFunction(cv.get())) {
                std::string s = getFnSource(cv.get());
                if (!s.empty()) spec.candidates.push_back(std::move(s));
            } else if (ev::isObject(cv.get())) {
                double clen = 0;
                if (propNumber(cv.get(), "length", clen)) {
                    uint32_t n = static_cast<uint32_t>(clen);
                    for (uint32_t i = 0; i < n; ++i) {
                        ev::Persistent elem(ev::getElement(cv.get(), i));
                        if (ev::isString(elem.get())) {
                            spec.candidates.push_back(ev::toUtf8(elem.get()));
                        } else if (ev::isFunction(elem.get())) {
                            std::string s = getFnSource(elem.get());
                            if (!s.empty()) spec.candidates.push_back(std::move(s));
                        }
                    }
                }
            }
        }

        for (std::string_view synKey : {"synthesizer", "candidateGenerator", "llmSynthesizer", "propose"}) {
            Value sv = ev::getProperty(o.get(), synKey);
            if (ev::isFunction(sv)) synthFnP.set(sv);
        }

        for (std::string_view exKey : {"examples", "ioExamples"}) {
            ev::Persistent exArr(ev::getProperty(o.get(), exKey));
            double exLen = 0;
            if (ev::isObject(exArr.get()) && propNumber(exArr.get(), "length", exLen)) {
                uint32_t n = static_cast<uint32_t>(exLen);
                for (uint32_t i = 0; i < n; ++i) {
                    ev::Persistent item(ev::getElement(exArr.get(), i));
                    if (!ev::isObject(item.get())) continue;
                    brolm::SynthesizeExample ex;
                    double outD = 0;
                    if (propNumber(item.get(), "output", outD) ||
                        propNumber(item.get(), "expected", outD) ||
                        propNumber(item.get(), "out", outD)) {
                        ex.expected_output = static_cast<int64_t>(std::llround(outD));
                    }
                    ev::Persistent inVal(ev::getProperty(item.get(), "inputs"));
                    if (ev::isUndefined(inVal.get())) inVal.set(ev::getProperty(item.get(), "input"));
                    if (ev::isNumber(inVal.get())) {
                        ex.inputs.push_back(static_cast<int64_t>(std::llround(ev::toDouble(inVal.get()))));
                    } else if (ev::isObject(inVal.get())) {
                        double ilen = 0;
                        if (propNumber(inVal.get(), "length", ilen)) {
                            for (uint32_t j = 0; j < static_cast<uint32_t>(ilen); ++j) {
                                Value iv = ev::getElement(inVal.get(), j);
                                ex.inputs.push_back(ev::isNumber(iv)
                                    ? static_cast<int64_t>(std::llround(ev::toDouble(iv))) : 0);
                            }
                        }
                    }
                    if (!ex.inputs.empty()) spec.examples.push_back(std::move(ex));
                }
            }
        }
    };

    if (ev::isString(a[0])) {
        spec.postcondition = ev::toUtf8(a[0]);
    } else if (ev::isFunction(a[0])) {
        refFnP.set(a[0]);
        spec.reference_expression = getFnSource(a[0]);
    } else if (ev::isObject(a[0])) {
        applySpecObj(a[0]);
    }
    if (a.size() >= 2 && ev::isObject(a[1])) {
        applySpecObj(a[1]);
    }

    if (ev::isFunction(refFnP.get())) {
        spec.concrete_reference = [&refFnP](const std::vector<int64_t>& inputs) -> int64_t {
            std::vector<ev::Persistent> rooted;
            rooted.reserve(inputs.size());
            for (int64_t v : inputs) rooted.emplace_back(ev::fromDouble(static_cast<double>(v)));
            std::vector<Value> raw;
            raw.reserve(rooted.size());
            for (const auto& p : rooted) raw.push_back(p.get());
            ev::CallResult r = ev::call(refFnP.get(), ev::undefined(),
                                        std::span<const Value>(raw.data(), raw.size()));
            if (r.thrown || !ev::isNumber(r.value)) return 0;
            return static_cast<int64_t>(std::llround(ev::toDouble(r.value)));
        };
    }

    if (ev::isFunction(postFnP.get())) {
        spec.concrete_postcondition = [&postFnP](const std::vector<int64_t>& inputs, int64_t outVal) -> bool {
            std::vector<ev::Persistent> rooted;
            rooted.reserve(inputs.size() + 1);
            for (int64_t v : inputs) rooted.emplace_back(ev::fromDouble(static_cast<double>(v)));
            rooted.emplace_back(ev::fromDouble(static_cast<double>(outVal)));
            std::vector<Value> raw;
            raw.reserve(rooted.size());
            for (const auto& p : rooted) raw.push_back(p.get());
            ev::CallResult r = ev::call(postFnP.get(), ev::undefined(),
                                        std::span<const Value>(raw.data(), raw.size()));
            return !r.thrown && ev::toBool(r.value);
        };
    }

    if (ev::isFunction(synthFnP.get())) {
        spec.synthesizer = [&synthFnP, &spec](const std::vector<brolm::CounterexampleRecord>& cexs,
                                              uint32_t iter) -> std::string {
            ev::Persistent cexArr(hostArrayOf(cexs.size(), [&](size_t i) {
                return makeCounterexampleObject(cexs[i], spec.params);
            }));
            ev::Persistent iterVal(ev::fromDouble(static_cast<double>(iter)));
            const Value callArgs[2] = {cexArr.get(), iterVal.get()};
            ev::CallResult r = ev::call(synthFnP.get(), ev::undefined(),
                                        std::span<const Value>(callArgs, 2));
            if (r.thrown || ev::isUndefined(r.value) || ev::isNull(r.value)) return {};
            if (ev::isString(r.value)) return ev::toUtf8(r.value);
            if (ev::isFunction(r.value)) return getFnSource(r.value);
            return {};
        };
    }

    brolm::SynthesizeResult sres = brolm::synthesize_verified_function(spec);

    ObjectBuilder out;
    out.set("verified", sres.verified);
    out.set("valid", sres.verified);
    out.set("synthesizedSource", sres.synthesized_source);
    out.set("expression", sres.expression);
    out.set("iterations", static_cast<double>(sres.iterations));
    out.set("smtQueries", static_cast<double>(sres.smt_queries));
    out.set("certificateHash", sres.certificate_hash);
    if (!sres.error.empty()) {
        out.set("error", sres.error);
    }

    ev::Persistent paramsArr(makeStringArray(sres.params));
    out.set("params", paramsArr.get());

    ev::Persistent cexArr(hostArrayOf(sres.counterexamples_resolved.size(), [&](size_t i) {
        return makeCounterexampleObject(sres.counterexamples_resolved[i], sres.params);
    }));
    out.set("counterexamplesResolved", cexArr.get());

    if (sres.verified) {
        std::string verifiedExpr = sres.expression;
        std::vector<std::string> verifiedParams = sres.params;
        uint32_t verifiedWidth = spec.bit_width;
        ev::Persistent fnVal(ev::makeFunction(
            [verifiedExpr, verifiedParams, verifiedWidth](Value, std::span<const Value> fnArgs) -> Value {
                std::vector<int64_t> inVals(verifiedParams.size(), 0);
                for (size_t i = 0; i < verifiedParams.size() && i < fnArgs.size(); ++i) {
                    if (ev::isNumber(fnArgs[i])) {
                        inVals[i] = static_cast<int64_t>(std::llround(ev::toDouble(fnArgs[i])));
                    } else if (ev::isBool(fnArgs[i])) {
                        inVals[i] = ev::toBool(fnArgs[i]) ? 1 : 0;
                    }
                }
                int64_t outVal = 0;
                if (!brolm::evaluate_bv_expression(verifiedExpr, verifiedParams, inVals, &outVal, verifiedWidth)) {
                    return ev::throwError("synthesized function runtime evaluation failed");
                }
                return ev::fromDouble(static_cast<double>(outVal));
            },
            static_cast<uint32_t>(verifiedParams.size()),
            spec.name));
        out.set("fn", fnVal.get());
    } else {
        out.set("fn", ev::null());
    }

    return out.build();
}

Value js_solveWithProver(Value, std::span<const Value> a) {
    if (a.empty() || !ev::isObject(a[0])) {
        return ev::throwTypeError("solveWithProver(problemSpec, opts): problemSpec object required");
    }

    brolm::ProverProblemSpec problem;
    int64_t defMin = -100000;
    int64_t defMax = 100000;
    uint32_t defWidth = 32;
    ev::Persistent predFnP;
    ev::Persistent proposerFnP;

    auto applyProblemObj = [&](Value objVal) {
        if (!ev::isObject(objVal)) return;
        ev::Persistent o(objVal);

        double d = 0;
        if (propNumber(o.get(), "min", d)) defMin = static_cast<int64_t>(d);
        if (propNumber(o.get(), "max", d)) defMax = static_cast<int64_t>(d);
        if (propNumber(o.get(), "bitWidth", d) || propNumber(o.get(), "width", d)) {
            uint32_t bw = static_cast<uint32_t>(d);
            if (bw == 32 || bw == 64) defWidth = bw;
        }
        if (propNumber(o.get(), "maxIterations", d) || propNumber(o.get(), "maxCegisIterations", d)) {
            if (d >= 1) problem.repair_opts.max_cegis_iterations = static_cast<uint32_t>(d);
        }
        if (propNumber(o.get(), "maxConflicts", d) || propNumber(o.get(), "maxSatConflicts", d)) {
            if (d >= 100) problem.repair_opts.max_sat_conflicts = static_cast<uint32_t>(d);
        }
        bool minPerturb = true;
        if (propBool(o.get(), "minimizePerturbation", minPerturb)) {
            problem.repair_opts.minimize_perturbation = minPerturb;
        }

        for (std::string_view fkey : {"fields", "variables", "vars"}) {
            Value fv = ev::getProperty(o.get(), fkey);
            if (ev::isObject(fv)) {
                std::vector<brolm::FieldConstraint> fcs;
                std::vector<std::string> dummyParams;
                parseFieldDomainsFromJs(fv, defMin, defMax, defWidth, fcs, dummyParams);
                for (const auto& fc : fcs) problem.validator.add_field(fc);
            }
        }

        for (std::string_view ckey : {"constraints", "constraint", "invariant",
                                      "rules", "postcondition", "expression"}) {
            ev::Persistent cv(ev::getProperty(o.get(), ckey));
            if (ev::isString(cv.get())) {
                problem.validator.add_constraint(ev::toUtf8(cv.get()));
            } else if (ev::isObject(cv.get())) {
                for (const auto& s : readStringArray(cv.get())) {
                    problem.validator.add_constraint(s);
                }
            }
        }

        for (std::string_view pkey : {"predicate", "predicateFn"}) {
            Value pv = ev::getProperty(o.get(), pkey);
            if (ev::isFunction(pv)) {
                predFnP.set(pv);
                std::string fsrc = getFnSource(pv);
                if (!fsrc.empty()) {
                    problem.validator.add_constraints_from_js_source(fsrc);
                }
            }
        }

        propString(o.get(), "minimize", problem.minimize_expr);
        propString(o.get(), "maximize", problem.maximize_expr);
        if (problem.minimize_expr.empty() && problem.maximize_expr.empty()) {
            propString(o.get(), "objective", problem.minimize_expr);
        }

        for (std::string_view candKey : {"candidate", "initial", "seed"}) {
            Value cv = ev::getProperty(o.get(), candKey);
            brolm::FieldMap cmap;
            if (readFieldMapFromJsObject(cv, cmap)) {
                problem.candidates.push_back(std::move(cmap));
            }
        }

        ev::Persistent candsArr(ev::getProperty(o.get(), "candidates"));
        double clen = 0;
        if (ev::isObject(candsArr.get()) && propNumber(candsArr.get(), "length", clen)) {
            uint32_t n = static_cast<uint32_t>(clen);
            for (uint32_t i = 0; i < n; ++i) {
                brolm::FieldMap cmap;
                if (readFieldMapFromJsObject(ev::getElement(candsArr.get(), i), cmap)) {
                    problem.candidates.push_back(std::move(cmap));
                }
            }
        }

        for (std::string_view propKey : {"synthesizer", "proposer", "candidateGenerator"}) {
            Value sv = ev::getProperty(o.get(), propKey);
            if (ev::isFunction(sv)) proposerFnP.set(sv);
        }
    };

    applyProblemObj(a[0]);
    if (a.size() >= 2 && ev::isObject(a[1])) {
        applyProblemObj(a[1]);
    }

    if (ev::isFunction(predFnP.get())) {
        problem.validator.set_predicate([&predFnP, &problem](const brolm::FieldMap& fmap) -> bool {
            ev::Persistent obj(makeFieldMapObject(fmap, problem.validator.fields()));
            const Value arg = obj.get();
            ev::CallResult r = ev::call(predFnP.get(), ev::undefined(), std::span<const Value>(&arg, 1));
            return !r.thrown && ev::toBool(r.value);
        }, "predicate");
    }

    if (ev::isFunction(proposerFnP.get())) {
        problem.proposer = [&proposerFnP, &problem](const std::vector<brolm::FieldMap>& rejected,
                                                    const std::vector<std::string>& violations,
                                                    uint32_t iter) -> brolm::FieldMap {
            ev::Persistent rejArr(hostArrayOf(rejected.size(), [&](size_t i) {
                return makeFieldMapObject(rejected[i], problem.validator.fields());
            }));
            ev::Persistent violArr(makeStringArray(violations));
            ev::Persistent iterVal(ev::fromDouble(static_cast<double>(iter)));
            const Value callArgs[3] = {rejArr.get(), violArr.get(), iterVal.get()};
            ev::CallResult r = ev::call(proposerFnP.get(), ev::undefined(),
                                        std::span<const Value>(callArgs, 3));
            brolm::FieldMap cmap;
            if (!r.thrown && ev::isObject(r.value)) {
                readFieldMapFromJsObject(r.value, cmap);
            }
            return cmap;
        };
    }

    brolm::ProverResult pres = brolm::solve_with_prover(problem);

    ObjectBuilder out;
    out.set("solved", pres.solved);
    out.set("valid", pres.valid);
    out.set("hasObjective", pres.has_objective);
    out.set("objectiveValue", static_cast<double>(pres.objective_value));
    out.set("iterations", static_cast<double>(pres.iterations));
    out.set("smtQueries", static_cast<double>(pres.smt_queries));
    out.set("certificateHash", pres.certificate_hash);

    ev::Persistent violArr(makeStringArray(pres.violations));
    out.set("violations", violArr.get());

    ev::Persistent rejArr(hostArrayOf(pres.rejected_candidates.size(), [&](size_t i) {
        return makeFieldMapObject(pres.rejected_candidates[i], problem.validator.fields());
    }));
    out.set("counterexamplesResolved", rejArr.get());

    if (pres.solved) {
        ev::Persistent solObj(makeFieldMapObject(pres.solution, problem.validator.fields()));
        out.set("solution", solObj.get());
        out.set("repaired", solObj.get());
    } else {
        out.set("solution", ev::null());
        out.set("repaired", ev::null());
    }

    return out.build();
}

}  // namespace brolm::api
