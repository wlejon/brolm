// Comprehensive C++ and Bronze JS test suite for brolm::VerifiedValidator and
// bro.lm.createVerifiedValidator(predicateFn, opts) (Issue #136).

#include <brolm/verified_validator.h>
#include "api/api.h"
#include "api/host_lm_internal.h"
#include "embed/embed.h"
#include "eval/eval.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

namespace ev = bronze::embed;
using Value = bronze::Value;

#define TEST_CHECK(cond) do { \
    if (!(cond)) { \
        std::cerr << "CHECK FAILED: " #cond " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::exit(1); \
    } \
} while (0)

// ─── 1. Symbolic Expression Parsing into brass::verify::BVContext ─────────────

static void test_bv_expression_parser() {
    std::cout << "[1/6] symbolic expression parsing into brass::verify::BVContext..." << std::endl;

    using namespace brass::verify;
    BVContext ctx;
    std::unordered_map<std::string, BVNodeId> vars;
    vars["a"] = ctx.make_var("a", 32);
    vars["b"] = ctx.make_var("b", 32);
    vars["c"] = ctx.make_var("c", 32);
    vars["flags"] = ctx.make_var("flags", 32);

    std::string err;
    BVNodeId div_guard = 0;
    // Test arithmetic (+, -, *, %, /), bitwise (&, |, ^, <<, >>, >>>, ~),
    // comparisons (==, ===, !=, !==, <, <=, >, >=), logical (&&, ||, !), ternary, Math.*
    std::string expr =
        "a >= 10 && a <= 50 && "
        "b == a * 2 - 5 && "
        "((a ^ b) & 0xff) === c && "
        "(flags & 0x3) !== 0 && "
        "((flags << 2) >> 1) >= 2 && "
        "!(a == 0 || b < 0) && "
        "(a > 20 ? a - 20 : 20 - a) == Math.abs(a - 20) && "
        "Math.min(a, b) == a && Math.max(a, b) == b && "
        "(b - a) % 8 === 3";

    BVNodeId cond = brolm::VerifiedValidator::parse_constraint_to_bv(
        ctx, expr, vars, 32, &div_guard, &err);
    TEST_CHECK(cond != UINT32_MAX);
    TEST_CHECK(err.empty());
    if (div_guard != 0) {
        cond = ctx.make_binary(BVOp::And, cond, div_guard, 1);
    }

    SolveOutcome so = solve_nonzero(ctx, cond, 20000);
    TEST_CHECK(so.status == SolveOutcome::Sat);
    TEST_CHECK(so.model.size() == 4);
    TEST_CHECK(ctx.eval_concrete(cond, so.model) == 1);

    int64_t va = sign_extend_bits(so.model[0], 32);
    int64_t vb = sign_extend_bits(so.model[1], 32);
    int64_t vc = sign_extend_bits(so.model[2], 32);
    int64_t vf = sign_extend_bits(so.model[3], 32);

    TEST_CHECK(va >= 10 && va <= 50);
    TEST_CHECK(vb == va * 2 - 5);
    TEST_CHECK(((va ^ vb) & 0xff) == vc);
    TEST_CHECK((vf & 0x3) != 0);
    TEST_CHECK((vb - va) % 8 == 3);
}

// ─── 2. Single-Shot SMT Repair of Invalid LLM Config Objects ──────────────────

static void test_cpp_smt_repair() {
    std::cout << "[2/6] C++ SMT repair of invalid LLM config objects..." << std::endl;

    // Case A: Network & range config (out-of-range port, min > max, broken alignment, flags = 0, bad checksum)
    brolm::VerifiedValidator v;
    v.add_field(brolm::FieldConstraint::integer("port", 1024, 65535));
    v.add_field(brolm::FieldConstraint::integer("minVal", 0, 1000));
    v.add_field(brolm::FieldConstraint::integer("maxVal", 0, 1000));
    v.add_field(brolm::FieldConstraint::integer("flags", 0, 255));
    v.add_field(brolm::FieldConstraint::integer("a", 0, 255));
    v.add_field(brolm::FieldConstraint::integer("b", 0, 255));
    v.add_field(brolm::FieldConstraint::integer("checksum", 0, 255));

    v.add_constraint("minVal < maxVal && (maxVal - minVal) % 16 === 0");
    v.add_constraint("(flags & 0x3) !== 0");
    v.add_constraint("checksum === ((a ^ b) & 0xff)");

    brolm::FieldMap bad_candidate = {
        {"port", 80},        // violates port >= 1024
        {"minVal", 100},
        {"maxVal", 40},      // violates minVal < maxVal and % 16
        {"flags", 4},        // violates (flags & 0x3) != 0
        {"a", 0x12},
        {"b", 0x34},
        {"checksum", 0x99}   // violates checksum == (0x12 ^ 0x34) == 0x26
    };

    brolm::ValidationResult vres = v.validate(bad_candidate);
    TEST_CHECK(!vres.valid);
    TEST_CHECK(vres.violations.size() >= 4);

    brolm::RepairResult rres = v.repairWithSmt(bad_candidate);
    TEST_CHECK(rres.valid);
    TEST_CHECK(rres.smt_solved);
    TEST_CHECK(rres.iterations == 1);

    // Verify repaired assignment passes validate() and preserves valid fields
    brolm::ValidationResult post_val = v.validate(rres.repaired_fields);
    TEST_CHECK(post_val.valid);
    TEST_CHECK(rres.repaired_fields.at("port") == 1024);
    TEST_CHECK(rres.repaired_fields.at("minVal") == 100);
    TEST_CHECK(rres.repaired_fields.at("maxVal") == 116);
    TEST_CHECK(rres.repaired_fields.at("a") == 0x12);
    TEST_CHECK(rres.repaired_fields.at("b") == 0x34);
    TEST_CHECK(rres.repaired_fields.at("checksum") == (0x12 ^ 0x34));
    TEST_CHECK((rres.repaired_fields.at("flags") & 0x3) != 0);

    // Case B: RGB color sum + layout bounding box JSON repair
    brolm::VerifiedValidator layout_v;
    layout_v.add_field(brolm::FieldConstraint::integer("r", 0, 255));
    layout_v.add_field(brolm::FieldConstraint::integer("g", 0, 255));
    layout_v.add_field(brolm::FieldConstraint::integer("b", 0, 255));
    layout_v.add_field(brolm::FieldConstraint::integer("x", 0, 500));
    layout_v.add_field(brolm::FieldConstraint::integer("width", 16, 500));
    layout_v.add_field(brolm::FieldConstraint::integer("maxWidth", 100, 500));
    layout_v.add_constraint("r + g + b === 255");
    layout_v.add_constraint("x + width <= maxWidth");

    std::string bad_json = R"({"name":"panel","r":100,"g":100,"b":100,"x":150,"width":120,"maxWidth":200})";
    TEST_CHECK(!layout_v.validate_json(bad_json).valid);

    brolm::RepairResult jrep = layout_v.repair_json_with_smt(bad_json);
    TEST_CHECK(jrep.valid);
    TEST_CHECK(jrep.smt_solved);
    TEST_CHECK(jrep.repaired_fields.at("r") + jrep.repaired_fields.at("g") + jrep.repaired_fields.at("b") == 255);
    TEST_CHECK(jrep.repaired_fields.at("x") + jrep.repaired_fields.at("width") <= jrep.repaired_fields.at("maxWidth"));
    TEST_CHECK(jrep.repaired_json.find("\"name\":\"panel\"") != std::string::npos);
    TEST_CHECK(layout_v.validate_json(jrep.repaired_json).valid);
}

// ─── 3. C++ CEGIS Loop with Blocking Clauses & Domain Deduction ───────────────

static void test_cpp_cegis_and_domain() {
    std::cout << "[3/6] C++ CEGIS blocking-clause refinement and domain deduction..." << std::endl;

    brolm::VerifiedValidator v;
    v.add_field(brolm::FieldConstraint::integer("x", 10, 30));
    v.add_field(brolm::FieldConstraint::integer("y", 10, 30));
    v.add_constraint("x + y == 40");

    // Black-box C++ predicate that only accepts x == 17 (a specific prime)
    int pred_calls = 0;
    v.set_predicate([&](const brolm::FieldMap& m) {
        ++pred_calls;
        return m.at("x") == 17 && m.at("y") == 23;
    }, "prime_pair_predicate");

    brolm::RepairResult r = v.repair_with_smt({{"x", 12}, {"y", 12}});
    TEST_CHECK(r.valid);
    TEST_CHECK(r.smt_solved);
    TEST_CHECK(r.repaired_fields.at("x") == 17);
    TEST_CHECK(r.repaired_fields.at("y") == 23);
    TEST_CHECK(pred_calls >= 1);

    // Domain deduction test
    brolm::VerifiedValidator dv;
    dv.add_field(brolm::FieldConstraint::integer("min", 0, 100));
    dv.add_field(brolm::FieldConstraint::integer("max", 0, 100));
    dv.add_constraint("min < max && (max - min) % 16 == 0");

    brolm::FieldDomainResult dom = dv.valid_domain_for_field({{"min", 32}}, "max", 8);
    TEST_CHECK(dom.feasible);
    TEST_CHECK(dom.min_val == 48);
    TEST_CHECK(dom.max_val == 96);
    TEST_CHECK((dom.sample_values == std::vector<int64_t>{48, 64, 80, 96}));

    // Unsatisfiable partial domain
    brolm::FieldDomainResult dom_unsat = dv.valid_domain_for_field({{"min", 96}}, "max", 8);
    TEST_CHECK(!dom_unsat.feasible);
}

// ─── 4. Unsatisfiable (UNSAT) Detection ───────────────────────────────────────

static void test_cpp_unsat_detection() {
    std::cout << "[4/6] UNSAT specification detection..." << std::endl;

    brolm::VerifiedValidator v;
    v.add_field(brolm::FieldConstraint::integer("a", 0, 50));
    v.add_field(brolm::FieldConstraint::integer("b", 0, 50));
    v.add_constraint("a + b >= 200 && a < b && a > b");

    brolm::RepairResult r = v.repair_with_smt({{"a", 25}, {"b", 25}});
    TEST_CHECK(!r.valid);
    TEST_CHECK(!r.smt_solved);

    brolm::RepairResult gen = v.generate_valid();
    TEST_CHECK(!gen.valid);
}

// ─── 5. Bronze JS API: bro.lm.createVerifiedValidator ─────────────────────────

static void run_js_checked(const char* label, const char* script) {
    ev::CallResult res = bronze::eval::evalScript(script);
    if (res.thrown) {
        std::string msg = ev::isObject(res.value)
            ? ev::toUtf8(ev::getProperty(res.value, "message"))
            : ev::toUtf8(res.value);
        std::cerr << "JS test failed in " << label << ": " << msg << std::endl;
        std::exit(1);
    }
}

static void test_js_verified_validator_api() {
    std::cout << "[5/6] Bronze JS API (bro.lm.createVerifiedValidator)..." << std::endl;

    run_js_checked("js_validator_object_and_repair", R"JS(
        function assert(cond, msg) {
            if (!cond) throw new Error("Assertion failed: " + msg);
        }

        assert(typeof bro.lm.createVerifiedValidator === "function", "createVerifiedValidator mounted");
        assert(typeof bro.lm.VerifiedValidator === "function", "VerifiedValidator class mounted");

        // 1. JS arrow predicate function automatically lowered to BVContext + live verification
        const v1 = bro.lm.createVerifiedValidator(
            (cfg) => cfg.min < cfg.max && (cfg.max - cfg.min) % 16 === 0 && (cfg.flags & 3) !== 0,
            {
                fields: {
                    min: { min: 0, max: 256 },
                    max: { min: 0, max: 256 },
                    flags: { min: 0, max: 15 }
                }
            }
        );

        const bad1 = { name: "llm_cfg", min: 64, max: 20, flags: 8 };
        const check1 = v1.validate(bad1);
        assert(check1.valid === false, "bad1 should be invalid");
        assert(check1.violations.length >= 2, "bad1 violations reported");

        const rep1 = v1.repair(bad1);
        assert(rep1.valid === true, "rep1 should be valid");
        assert(rep1.smtSolved === true, "rep1 solved by SMT");
        assert(rep1.repaired !== null, "rep1.repaired present");
        assert(rep1.repaired.name === "llm_cfg", "extra string field preserved");
        assert(rep1.repaired.min < rep1.repaired.max, "repaired min < max");
        assert((rep1.repaired.max - rep1.repaired.min) % 16 === 0, "repaired span % 16 === 0");
        assert((rep1.repaired.flags & 3) !== 0, "repaired flags & 3 !== 0");
        assert(v1.validate(rep1.repaired).valid === true, "repaired object validates");

        // 2. Domain deduction via .validDomainForField(partialObj, fieldName)
        const dom = v1.validDomainForField({ min: 32, flags: 1 }, "max");
        assert(dom.feasible === true, "domain for max given min=32 is feasible");
        assert(dom.min === 48, "min feasible max is 48 (got " + dom.min + ")");
        assert(dom.max === 256, "max feasible max is 256 (got " + dom.max + ")");
        assert(dom.values.length >= 3, "sample values populated");

        // 3. Grammar composition (.grammarMask and .fieldGrammar)
        assert(v1.grammarMask !== undefined && v1.grammarMask !== null, "grammarMask attached");
        assert(v1.grammarMask.canAccept('{"min":16,') === true, "grammarMask accepts JSON prefix");
        const fg = v1.fieldGrammar({ min: 224, flags: 1 }, "max");
        assert(fg.canAccept("240") === true, "fieldGrammar accepts valid SMT value 240");
        assert(fg.canAccept("225") === false, "fieldGrammar rejects invalid value 225");
    )JS");

    run_js_checked("js_cegis_and_unsat", R"JS(
        function assert(cond, msg) {
            if (!cond) throw new Error("Assertion failed: " + msg);
        }

        // 4. CEGIS blocking-clause loop with a black-box JS closure predicate
        const secretAllowed = [13, 19, 23];
        function isSecretAllowed(x) {
            for (let i = 0; i < secretAllowed.length; ++i) {
                if (secretAllowed[i] === x) return true;
            }
            return false;
        }

        let jsPredCalls = 0;
        const cegisValidator = bro.lm.createVerifiedValidator(
            (obj) => {
                jsPredCalls = jsPredCalls + 1;
                return isSecretAllowed(obj.token) && obj.token + obj.offset === 30;
            },
            {
                fields: [
                    { name: "token", min: 10, max: 25 },
                    { name: "offset", min: 0, max: 25 }
                ],
                constraints: "token + offset === 30"
            }
        );

        const cegisRes = cegisValidator.repair({ token: 10, offset: 10 });
        assert(cegisRes.valid === true, "CEGIS repair succeeded");
        assert(cegisRes.smtSolved === true, "CEGIS used SMT");
        assert(cegisRes.iterations >= 1, "CEGIS recorded iterations");
        assert(isSecretAllowed(cegisRes.repaired.token), "CEGIS satisfied black-box JS predicate");
        assert(cegisRes.repaired.token + cegisRes.repaired.offset === 30, "CEGIS satisfied symbolic sum");
        assert(jsPredCalls >= 1, "live JS predicate was called");

        // 5. Synthesizing from scratch via .generateValid()
        const synth = cegisValidator.generateValid();
        assert(synth.valid === true, "generateValid succeeded");
        assert(isSecretAllowed(synth.value.token), "generateValid value satisfies black-box predicate");
        assert(synth.value.token + synth.value.offset === 30, "generateValid value satisfies sum");

        // 6. Positional predicate function (a, b, c) => ...
        const posValidator = bro.lm.createVerifiedValidator(
            (r, g, b) => r + g + b === 255 && r >= 50 && g >= 50 && b >= 50,
            { fields: ["r", "g", "b"], min: 0, max: 255 }
        );
        const posRep = posValidator.repair({ r: 100, g: 100, b: 100 });
        assert(posRep.valid === true, "positional predicate repaired");
        assert(posRep.repaired.r + posRep.repaired.g + posRep.repaired.b === 255, "RGB sum == 255");

        // 7. UNSAT detection from JS
        const unsatValidator = bro.lm.createVerifiedValidator({
            fields: [
                { name: "x", min: 0, max: 10 },
                { name: "y", min: 0, max: 10 }
            ],
            constraints: ["x + y >= 50", "x > y", "y > x"]
        });
        const unsatRes = unsatValidator.repair({ x: 5, y: 5 });
        assert(unsatRes.valid === false, "UNSAT returns valid: false");
        assert(unsatRes.smtSolved === false, "UNSAT returns smtSolved: false");
        assert(unsatRes.repaired === null, "UNSAT returns repaired: null");
    )JS");
}

// ─── 6. Grammar + VerifiedValidator End-to-End Pipeline ───────────────────────

static void test_grammar_and_validator_composition() {
    std::cout << "[6/6] Grammar + VerifiedValidator composition..." << std::endl;

    brolm::VerifiedValidator v;
    v.add_field(brolm::FieldConstraint::integer("port", 1024, 65535));
    v.add_field(brolm::FieldConstraint::boolean("tls"));
    v.add_constraint("tls == 1 ? port == 443 || port >= 8443 : port >= 1024");

    brolm::Grammar g = v.build_grammar();
    std::string candidate_json = R"({"port":1080,"tls":true})";
    TEST_CHECK(g.accept(candidate_json));
    TEST_CHECK(g.is_accepted());

    // Syntactically valid per Grammar, but semantically invalid (tls=true requires port >= 8443)
    TEST_CHECK(!v.validate_json(candidate_json).valid);

    // Repair via SMT and verify both Grammar acceptance and semantic validity
    brolm::RepairResult rep = v.repair_json_with_smt(candidate_json);
    TEST_CHECK(rep.valid);
    TEST_CHECK(rep.repaired_fields.at("tls") == 1);
    TEST_CHECK(rep.repaired_fields.at("port") >= 8443);

    g.reset();
    TEST_CHECK(g.accept(rep.repaired_json));
    TEST_CHECK(g.is_accepted());
    TEST_CHECK(v.validate_json(rep.repaired_json).valid);
}

int main() {
    test_bv_expression_parser();
    test_cpp_smt_repair();
    test_cpp_cegis_and_domain();
    test_cpp_unsat_detection();

    ev::Realm* realm = ev::createRealm();
    {
        ev::RealmScope scope(realm);
        brolm::api::installLM();
        test_js_verified_validator_api();
    }
    brolm::api::shutdownLM();
    ev::destroyRealm(realm);

    test_grammar_and_validator_composition();

    std::cout << "All brolm_test_verified_validator checks passed!" << std::endl;
    return 0;
}
