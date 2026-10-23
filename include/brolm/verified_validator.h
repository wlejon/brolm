#pragma once

// SMT-Guided Neuro-Symbolic Constraint Validator & Repair Engine for LLM
// Structured Outputs.
//
// Grammar-constrained decoding (brolm::Grammar) guarantees syntactic validity
// (e.g., well-formed JSON matching a schema), while brolm::VerifiedValidator
// enforces semantic, arithmetic, bitwise, and relational invariants across
// fields using brass::verify::BVContext + solve_nonzero and counterexample-
// guided inductive synthesis (CEGIS) with custom predicate callbacks.

#include <brolm/grammar.h>
#include <verify/bv_solver.hpp>

#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace brolm {

enum class FieldType : uint8_t {
    Int,
    UInt,
    Bool
};

struct FieldConstraint {
    std::string name;
    FieldType type = FieldType::Int;
    uint32_t bit_width = 32;
    bool has_min = false;
    int64_t min_val = INT64_MIN;
    bool has_max = false;
    int64_t max_val = INT64_MAX;
    uint64_t alignment = 0;     // if > 1, requires (val % alignment) == 0
    uint64_t nonzero_mask = 0;  // if != 0, requires (val & nonzero_mask) != 0
    uint64_t zero_mask = 0;     // if != 0, requires (val & zero_mask) == 0

    static FieldConstraint integer(std::string field_name, int64_t min_v, int64_t max_v,
                                   uint32_t width = 32) {
        FieldConstraint fc;
        fc.name = std::move(field_name);
        fc.type = FieldType::Int;
        fc.bit_width = width;
        fc.has_min = true;
        fc.min_val = min_v;
        fc.has_max = true;
        fc.max_val = max_v;
        return fc;
    }

    static FieldConstraint unbounded_int(std::string field_name, uint32_t width = 32) {
        FieldConstraint fc;
        fc.name = std::move(field_name);
        fc.type = FieldType::Int;
        fc.bit_width = width;
        return fc;
    }

    static FieldConstraint boolean(std::string field_name) {
        FieldConstraint fc;
        fc.name = std::move(field_name);
        fc.type = FieldType::Bool;
        fc.bit_width = 32;
        fc.has_min = true;
        fc.min_val = 0;
        fc.has_max = true;
        fc.max_val = 1;
        return fc;
    }
};

struct SymbolicConstraint {
    std::string expression;
    std::string label;
};

using FieldMap = std::unordered_map<std::string, int64_t>;
using PredicateCallback = std::function<bool(const FieldMap&)>;
using BVBuilderCallback = std::function<brass::verify::BVNodeId(
    brass::verify::BVContext&,
    const std::unordered_map<std::string, brass::verify::BVNodeId>&)>;

struct ValidationResult {
    bool valid = false;
    std::vector<std::string> violations;
    FieldMap extracted_fields;
};

struct RepairOptions {
    std::vector<std::string> fixed_fields;
    uint32_t max_cegis_iterations = 64;
    uint32_t max_sat_conflicts = 20000;
    bool minimize_perturbation = true;
};

struct RepairResult {
    bool valid = false;
    bool smt_solved = false;
    uint32_t iterations = 0;
    FieldMap repaired_fields;
    std::string repaired_json;
    std::vector<std::string> violations;
};

struct FieldDomainResult {
    bool feasible = false;
    int64_t min_val = 0;
    int64_t max_val = 0;
    int64_t sample_val = 0;
    std::vector<int64_t> sample_values;
};

class VerifiedValidator {
public:
    VerifiedValidator() = default;

    // ─── Specification Builders ─────────────────────────────────────────────

    void add_field(const FieldConstraint& field);
    bool has_field(std::string_view name) const noexcept;
    const FieldConstraint* find_field(std::string_view name) const noexcept;

    // Add a symbolic constraint expression (e.g., "min <= max && (max - min) % 16 == 0").
    // Top-level '&&' conjuncts are automatically registered so validate() reports
    // granular per-conjunct violations.
    void add_constraint(std::string expression, std::string label = "");

    // Parse a JS arrow/function source string (e.g. "(cfg) => cfg.min < cfg.max && ...")
    // into symbolic constraints and auto-discover referenced fields when possible.
    // Returns true if the function body was fully lowered into symbolic constraints.
    bool add_constraints_from_js_source(std::string_view js_source,
                                        bool* out_positional_args = nullptr);

    // Programmatic BVContext constraint builder callback.
    void add_bv_constraint(BVBuilderCallback builder, std::string label = "bv_constraint");

    // Attach an arbitrary concrete predicate (e.g. a live Bronze JS callback).
    void set_predicate(PredicateCallback pred, std::string label = "predicate");
    bool has_predicate() const noexcept { return static_cast<bool>(predicate_); }

    const std::vector<FieldConstraint>& fields() const noexcept { return fields_; }
    const std::vector<SymbolicConstraint>& constraints() const noexcept { return constraints_; }

    // ─── Validation, SMT Repair & Synthesis ─────────────────────────────────

    ValidationResult validate(const FieldMap& fields) const;
    ValidationResult validate_json(std::string_view json_text) const;

    RepairResult repair_with_smt(const FieldMap& candidate,
                                 const RepairOptions& opts = {}) const;
    RepairResult repairWithSmt(const FieldMap& candidate,
                               const RepairOptions& opts = {}) const {
        return repair_with_smt(candidate, opts);
    }

    RepairResult repair_json_with_smt(std::string_view json_text,
                                      const RepairOptions& opts = {}) const;
    RepairResult repairJsonWithSmt(std::string_view json_text,
                                   const RepairOptions& opts = {}) const {
        return repair_json_with_smt(json_text, opts);
    }

    RepairResult generate_valid(const FieldMap& seed = {},
                                const RepairOptions& opts = {}) const;

    FieldDomainResult valid_domain_for_field(const FieldMap& partial_fields,
                                             std::string_view field_name,
                                             uint32_t max_samples = 8) const;

    // ─── Grammar Integration ────────────────────────────────────────────────

    // Build a JSON Schema Grammar covering the validator's declared fields.
    Grammar build_grammar() const;

    // Build a Grammar restricting a single field's next value to SMT-feasible choices.
    Grammar build_field_grammar(const FieldMap& partial_fields,
                                std::string_view field_name,
                                uint32_t max_choices = 16) const;

    // ─── Symbolic Expression Parser into BVContext ──────────────────────────

    // Parse an arithmetic/bitwise/relational/boolean expression string directly
    // into a 1-bit or N-bit node in `ctx`. Returns UINT32_MAX on parse error.
    static brass::verify::BVNodeId parse_expression_to_bv(
        brass::verify::BVContext& ctx,
        std::string_view expr,
        const std::unordered_map<std::string, brass::verify::BVNodeId>& vars,
        uint32_t default_width = 32,
        brass::verify::BVNodeId* out_div_guards = nullptr,
        std::string* out_error = nullptr);

    // Same as parse_expression_to_bv, coerced to a 1-bit boolean condition (expr != 0).
    static brass::verify::BVNodeId parse_constraint_to_bv(
        brass::verify::BVContext& ctx,
        std::string_view expr,
        const std::unordered_map<std::string, brass::verify::BVNodeId>& vars,
        uint32_t default_width = 32,
        brass::verify::BVNodeId* out_div_guards = nullptr,
        std::string* out_error = nullptr);

private:
    std::vector<FieldConstraint> fields_;
    std::vector<SymbolicConstraint> constraints_;
    std::vector<std::pair<BVBuilderCallback, std::string>> bv_builders_;
    PredicateCallback predicate_;
    std::string predicate_label_ = "predicate";
};

// ─── Neuro-Symbolic Verified Function Synthesizer & Prover (Issue #150) ───────

struct SynthesizeExample {
    std::vector<int64_t> inputs;
    int64_t expected_output = 0;
};

struct CounterexampleRecord {
    std::vector<int64_t> inputs;
    FieldMap params;
    int64_t candidate_output = 0;
    int64_t expected_output = 0;
    bool has_expected_output = false;
    std::string candidate_source;
    std::string violation;
};

struct SynthesizeSpec {
    std::string name = "synthesized";
    std::vector<std::string> params;
    std::vector<FieldConstraint> param_domains;
    int64_t default_min = -4096;
    int64_t default_max = 4096;
    uint32_t bit_width = 32;

    std::string precondition;
    std::string postcondition;
    std::string reference_expression;
    std::string output_var = "out";

    std::vector<SynthesizeExample> examples;
    std::vector<std::string> candidates;

    // Optional callback invoked on each CEGIS iteration with accumulated counterexamples
    std::function<std::string(const std::vector<CounterexampleRecord>&, uint32_t)> synthesizer;

    // Optional concrete oracle / postcondition checker (inputs, candidate_out) -> valid
    std::function<bool(const std::vector<int64_t>&, int64_t)> concrete_postcondition;
    std::function<int64_t(const std::vector<int64_t>&)> concrete_reference;

    uint32_t max_cegis_iterations = 32;
    uint32_t max_sat_conflicts = 20000;
};

struct SynthesizeResult {
    bool verified = false;
    std::string synthesized_source;
    std::string expression;
    std::vector<std::string> params;
    uint32_t iterations = 0;
    uint32_t smt_queries = 0;
    std::vector<CounterexampleRecord> counterexamples_resolved;
    std::string certificate_hash;
    std::string error;
};

struct ProverProblemSpec {
    VerifiedValidator validator;
    std::string minimize_expr;
    std::string maximize_expr;
    std::vector<FieldMap> candidates;
    std::function<FieldMap(const std::vector<FieldMap>&,
                           const std::vector<std::string>&,
                           uint32_t)> proposer;
    RepairOptions repair_opts;
};

struct ProverResult {
    bool solved = false;
    bool valid = false;
    FieldMap solution;
    int64_t objective_value = 0;
    bool has_objective = false;
    uint32_t iterations = 0;
    uint32_t smt_queries = 0;
    std::vector<FieldMap> rejected_candidates;
    std::vector<std::string> violations;
    std::string certificate_hash;
};

// Extract the body expression and optional parameter names from a JS arrow/function
// string (e.g. "(x, min, max) => x < min ? min : (x > max ? max : x)").
bool extract_js_function_expression(std::string_view js_source,
                                    std::vector<std::string>* out_params,
                                    std::string* out_expr);

// Evaluate a parsed symbolic expression on concrete parameter values.
bool evaluate_bv_expression(std::string_view expr,
                            const std::vector<std::string>& param_names,
                            const std::vector<int64_t>& param_values,
                            int64_t* out_value,
                            uint32_t width = 32);

// Run the Neuro-Symbolic CEGIS loop to synthesize and formally prove a JS function
// against a bitvector postcondition, reference specification, and I/O examples.
SynthesizeResult synthesize_verified_function(const SynthesizeSpec& spec);

// Solve a neuro-symbolic constraint satisfaction / optimization problem using
// VerifiedValidator + brass::verify::BVContext.
ProverResult solve_with_prover(const ProverProblemSpec& problem);

}  // namespace brolm
