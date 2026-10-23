#include "brolm/verified_validator.h"
#include "brotensor/json.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <unordered_set>

namespace brolm {

using brass::verify::BVContext;
using brass::verify::BVNodeId;
using brass::verify::BVOp;
using brass::verify::SolveOutcome;
using brass::verify::sign_extend_bits;
using brass::verify::solve_nonzero;
using brass::verify::width_mask;

namespace {

int64_t clamp_to_field(int64_t val, const FieldConstraint& fc) {
    if (fc.type == FieldType::Bool) return (val != 0) ? 1 : 0;
    if (fc.has_min && val < fc.min_val) val = fc.min_val;
    if (fc.has_max && val > fc.max_val) val = fc.max_val;
    return val;
}

uint32_t choose_context_width(const std::vector<FieldConstraint>& fields,
                              const FieldMap& candidate) {
    auto fits_i32 = [](int64_t v) {
        return v >= -1073741824LL && v <= 1073741823LL;
    };
    for (const auto& f : fields) {
        if (f.bit_width > 32) return 64;
        if (f.has_min && !fits_i32(f.min_val)) return 64;
        if (f.has_max && !fits_i32(f.max_val)) return 64;
    }
    for (const auto& [k, v] : candidate) {
        if (!fits_i32(v)) return 64;
    }
    return 32;
}

std::string serialize_json_value(const brotensor::json::Value& v) {
    using brotensor::json::Type;
    switch (v.type()) {
        case Type::Null: return "null";
        case Type::Bool: return v.as_bool() ? "true" : "false";
        case Type::Number: {
            double d = v.as_number();
            if (std::isfinite(d) && std::floor(d) == d &&
                d >= static_cast<double>(INT64_MIN) && d <= static_cast<double>(INT64_MAX)) {
                return std::to_string(static_cast<int64_t>(d));
            }
            std::ostringstream os;
            os << d;
            return os.str();
        }
        case Type::String: {
            std::string out = "\"";
            for (char c : v.as_string()) {
                switch (c) {
                    case '"':  out += "\\\""; break;
                    case '\\': out += "\\\\"; break;
                    case '\n': out += "\\n";  break;
                    case '\r': out += "\\r";  break;
                    case '\t': out += "\\t";  break;
                    default:   out += c;      break;
                }
            }
            out += "\"";
            return out;
        }
        case Type::Array: {
            std::string out = "[";
            const auto& arr = v.as_array();
            for (size_t i = 0; i < arr.size(); ++i) {
                if (i > 0) out += ",";
                out += serialize_json_value(arr[i]);
            }
            out += "]";
            return out;
        }
        case Type::Object: {
            std::string out = "{";
            const auto& obj = v.as_object();
            for (size_t i = 0; i < obj.size(); ++i) {
                if (i > 0) out += ",";
                out += serialize_json_value(brotensor::json::Value::make_string(obj[i].first));
                out += ":";
                out += serialize_json_value(obj[i].second);
            }
            out += "}";
            return out;
        }
    }
    return "null";
}

}  // namespace

// ─── Validation ───────────────────────────────────────────────────────────────

ValidationResult VerifiedValidator::validate(const FieldMap& fields) const {
    ValidationResult res;
    res.extracted_fields = fields;

    for (const auto& fc : fields_) {
        auto it = fields.find(fc.name);
        if (it == fields.end()) {
            res.violations.push_back("missing required field '" + fc.name + "'");
            continue;
        }
        int64_t val = it->second;
        if (fc.type == FieldType::Bool && val != 0 && val != 1) {
            res.violations.push_back(fc.name + " must be boolean (0 or 1)");
        }
        if (fc.has_min && val < fc.min_val) {
            res.violations.push_back(fc.name + " >= " + std::to_string(fc.min_val));
        }
        if (fc.has_max && val > fc.max_val) {
            res.violations.push_back(fc.name + " <= " + std::to_string(fc.max_val));
        }
        if (fc.alignment > 1 && (val % static_cast<int64_t>(fc.alignment)) != 0) {
            res.violations.push_back("(" + fc.name + " % " + std::to_string(fc.alignment) + ") == 0");
        }
        if (fc.nonzero_mask != 0 && (static_cast<uint64_t>(val) & fc.nonzero_mask) == 0) {
            res.violations.push_back("(" + fc.name + " & " + std::to_string(fc.nonzero_mask) + ") != 0");
        }
        if (fc.zero_mask != 0 && (static_cast<uint64_t>(val) & fc.zero_mask) != 0) {
            res.violations.push_back("(" + fc.name + " & " + std::to_string(fc.zero_mask) + ") == 0");
        }
    }

    if (!constraints_.empty() || !bv_builders_.empty()) {
        uint32_t w = choose_context_width(fields_, fields);
        BVContext ctx;
        std::unordered_map<std::string, BVNodeId> var_map;
        std::vector<uint64_t> var_vals;

        auto ensure_var = [&](const std::string& name, int64_t val) {
            if (var_map.find(name) == var_map.end()) {
                var_map[name] = ctx.make_var(name, w);
                var_vals.push_back(static_cast<uint64_t>(val) & width_mask(w));
            }
        };
        for (const auto& fc : fields_) {
            auto it = fields.find(fc.name);
            ensure_var(fc.name, (it != fields.end()) ? it->second : 0);
        }
        for (const auto& [k, v] : fields) ensure_var(k, v);

        for (const auto& sc : constraints_) {
            BVNodeId div_guard = 0;
            std::string err;
            BVNodeId cid = parse_constraint_to_bv(ctx, sc.expression, var_map, w, &div_guard, &err);
            if (cid == UINT32_MAX) {
                res.violations.push_back("invalid constraint '" + sc.label + "': " + err);
                continue;
            }
            if (div_guard != 0 && ctx.eval_concrete(div_guard, var_vals) == 0) {
                res.violations.push_back(sc.label + " (division by zero)");
                continue;
            }
            if (ctx.eval_concrete(cid, var_vals) == 0) {
                res.violations.push_back(sc.label);
            }
        }
        for (const auto& [builder, label] : bv_builders_) {
            BVNodeId bid = builder(ctx, var_map);
            uint32_t bw = ctx.width(bid);
            BVNodeId bcond = (bw == 1) ? bid : ctx.make_binary(BVOp::Ne, bid, ctx.make_const(0, bw), 1);
            if (ctx.eval_concrete(bcond, var_vals) == 0) {
                res.violations.push_back(label);
            }
        }
    }

    if (predicate_) {
        if (!predicate_(fields)) {
            if (res.violations.empty()) {
                res.violations.push_back(predicate_label_);
            }
        }
    }

    res.valid = res.violations.empty();
    return res;
}

ValidationResult VerifiedValidator::validate_json(std::string_view json_text) const {
    ValidationResult res;
    brotensor::json::Value doc;
    try {
        doc = brotensor::json::parse(std::string(json_text));
    } catch (const std::exception& e) {
        res.valid = false;
        res.violations.push_back(std::string("JSON syntax error: ") + e.what());
        return res;
    }
    if (!doc.is_object()) {
        res.valid = false;
        res.violations.push_back("JSON root must be an object");
        return res;
    }
    FieldMap fmap;
    for (const auto& [k, v] : doc.as_object()) {
        if (v.is_number()) fmap[k] = static_cast<int64_t>(std::llround(v.as_number()));
        else if (v.is_bool()) fmap[k] = v.as_bool() ? 1 : 0;
    }
    return validate(fmap);
}

// ─── SMT-Guided Repair & CEGIS Engine ─────────────────────────────────────────

RepairResult VerifiedValidator::repair_with_smt(const FieldMap& candidate,
                                                const RepairOptions& opts) const {
    RepairResult out;

    std::vector<FieldConstraint> active_fields = fields_;
    if (active_fields.empty()) {
        for (const auto& [k, _] : candidate) {
            active_fields.push_back(FieldConstraint::unbounded_int(k));
        }
        std::sort(active_fields.begin(), active_fields.end(),
                  [](const FieldConstraint& a, const FieldConstraint& b) { return a.name < b.name; });
    }

    bool has_all_fields = !active_fields.empty();
    for (const auto& fc : active_fields) {
        if (candidate.find(fc.name) == candidate.end()) {
            has_all_fields = false;
            break;
        }
    }
    ValidationResult init_val = validate(candidate);
    out.violations = init_val.violations;
    if (has_all_fields && init_val.valid) {
        out.valid = true;
        out.smt_solved = false;
        out.iterations = 0;
        out.repaired_fields = candidate;
        return out;
    }

    if (active_fields.empty()) {
        out.valid = false;
        return out;
    }

    const uint32_t w = choose_context_width(active_fields, candidate);
    const uint64_t mask = width_mask(w);
    BVContext ctx;

    std::unordered_map<std::string, BVNodeId> var_map;
    std::vector<BVNodeId> var_nodes;
    var_nodes.reserve(active_fields.size());

    for (const auto& fc : active_fields) {
        BVNodeId vid = ctx.make_var(fc.name, w);
        var_map[fc.name] = vid;
        var_nodes.push_back(vid);
    }

    BVNodeId base_cond = ctx.make_const(1, 1);
    auto conjoin = [&](BVNodeId c) {
        base_cond = ctx.make_binary(BVOp::And, base_cond, c, 1);
    };

    for (size_t i = 0; i < active_fields.size(); ++i) {
        const auto& fc = active_fields[i];
        BVNodeId v = var_nodes[i];
        if (fc.type == FieldType::Bool) {
            conjoin(ctx.make_binary(BVOp::Sge, v, ctx.make_const(0, w), 1));
            conjoin(ctx.make_binary(BVOp::Sle, v, ctx.make_const(1, w), 1));
        } else {
            int64_t lo = fc.has_min ? fc.min_val : (w == 32 ? -1000000LL : -1000000000LL);
            int64_t hi = fc.has_max ? fc.max_val : (w == 32 ?  1000000LL :  1000000000LL);
            auto cit = candidate.find(fc.name);
            if (cit != candidate.end()) {
                if (!fc.has_min) lo = std::min<int64_t>(lo, cit->second - 65536LL);
                if (!fc.has_max) hi = std::max<int64_t>(hi, cit->second + 65536LL);
            }
            conjoin(ctx.make_binary(BVOp::Sge, v, ctx.make_const(static_cast<uint64_t>(lo), w), 1));
            conjoin(ctx.make_binary(BVOp::Sle, v, ctx.make_const(static_cast<uint64_t>(hi), w), 1));
        }
        if (fc.alignment > 1) {
            BVNodeId align_c = ctx.make_const(fc.alignment, w);
            BVNodeId rem = (std::has_single_bit(fc.alignment))
                ? ctx.make_binary(BVOp::And, v, ctx.make_const(fc.alignment - 1, w), w)
                : ctx.make_binary(BVOp::SRem, v, align_c, w);
            conjoin(ctx.make_binary(BVOp::Eq, rem, ctx.make_const(0, w), 1));
        }
        if (fc.nonzero_mask != 0) {
            BVNodeId masked = ctx.make_binary(BVOp::And, v, ctx.make_const(fc.nonzero_mask, w), w);
            conjoin(ctx.make_binary(BVOp::Ne, masked, ctx.make_const(0, w), 1));
        }
        if (fc.zero_mask != 0) {
            BVNodeId masked = ctx.make_binary(BVOp::And, v, ctx.make_const(fc.zero_mask, w), w);
            conjoin(ctx.make_binary(BVOp::Eq, masked, ctx.make_const(0, w), 1));
        }
    }

    for (const auto& sc : constraints_) {
        BVNodeId div_guard = 0;
        std::string err;
        BVNodeId cid = parse_constraint_to_bv(ctx, sc.expression, var_map, w, &div_guard, &err);
        if (cid == UINT32_MAX) {
            out.valid = false;
            return out;
        }
        if (div_guard != 0) conjoin(div_guard);
        conjoin(cid);
    }

    for (const auto& [builder, _] : bv_builders_) {
        BVNodeId bid = builder(ctx, var_map);
        uint32_t bw = ctx.width(bid);
        conjoin(bw == 1 ? bid : ctx.make_binary(BVOp::Ne, bid, ctx.make_const(0, bw), 1));
    }

    std::unordered_set<std::string> fixed_set(opts.fixed_fields.begin(), opts.fixed_fields.end());
    for (size_t i = 0; i < active_fields.size(); ++i) {
        const auto& fname = active_fields[i].name;
        if (fixed_set.count(fname)) {
            auto it = candidate.find(fname);
            if (it != candidate.end()) {
                conjoin(ctx.make_binary(BVOp::Eq, var_nodes[i],
                                        ctx.make_const(static_cast<uint64_t>(it->second), w), 1));
            }
        }
    }

    auto model_to_map = [&](const std::vector<uint64_t>& m) {
        FieldMap fmap = candidate;
        for (size_t i = 0; i < active_fields.size() && i < m.size(); ++i) {
            fmap[active_fields[i].name] = sign_extend_bits(m[i], w);
        }
        return fmap;
    };

    auto make_blocking_clause = [&](const std::vector<uint64_t>& m) {
        BVNodeId all_eq = ctx.make_const(1, 1);
        for (size_t i = 0; i < active_fields.size() && i < m.size(); ++i) {
            BVNodeId eq_i = ctx.make_binary(BVOp::Eq, var_nodes[i],
                                            ctx.make_const(m[i] & mask, w), 1);
            all_eq = ctx.make_binary(BVOp::And, all_eq, eq_i, 1);
        }
        return ctx.make_unary(BVOp::Not, all_eq, 1);
    };

    std::vector<int64_t> clamped_vals(active_fields.size(), 0);
    std::vector<uint64_t> clamped_model(active_fields.size(), 0);
    for (size_t i = 0; i < active_fields.size(); ++i) {
        auto it = candidate.find(active_fields[i].name);
        int64_t raw = (it != candidate.end()) ? it->second : 0;
        clamped_vals[i] = clamp_to_field(raw, active_fields[i]);
        clamped_model[i] = static_cast<uint64_t>(clamped_vals[i]) & mask;
    }

    if (!candidate.empty() && ctx.eval_concrete(base_cond, clamped_model) != 0) {
        FieldMap cmap = model_to_map(clamped_model);
        if (!predicate_ || predicate_(cmap)) {
            out.valid = true;
            out.smt_solved = true;
            out.iterations = 1;
            out.repaired_fields = std::move(cmap);
            return out;
        }
    }

    SolveOutcome base_solve = solve_nonzero(ctx, base_cond, opts.max_sat_conflicts);
    out.iterations = 1;
    if (base_solve.status != SolveOutcome::Sat) {
        out.valid = false;
        out.smt_solved = false;
        return out;
    }

    std::vector<uint64_t> best_model = base_solve.model;
    BVNodeId pinned_cond = base_cond;

    if (opts.minimize_perturbation && !candidate.empty()) {
        std::vector<size_t> order;
        for (size_t i = 0; i < active_fields.size(); ++i) {
            if (!fixed_set.count(active_fields[i].name) &&
                candidate.find(active_fields[i].name) != candidate.end()) {
                order.push_back(i);
            }
        }
        std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            bool a_in = (candidate.at(active_fields[a].name) == clamped_vals[a]);
            bool b_in = (candidate.at(active_fields[b].name) == clamped_vals[b]);
            if (a_in != b_in) return a_in > b_in;
            bool a_bool = (active_fields[a].type == FieldType::Bool);
            bool b_bool = (active_fields[b].type == FieldType::Bool);
            return a_bool > b_bool;
        });

        std::vector<bool> is_pinned(active_fields.size(), false);
        for (size_t idx : order) {
            BVNodeId eq_c = ctx.make_binary(BVOp::Eq, var_nodes[idx],
                                            ctx.make_const(clamped_model[idx], w), 1);
            BVNodeId try_c = ctx.make_binary(BVOp::And, pinned_cond, eq_c, 1);
            SolveOutcome so = solve_nonzero(ctx, try_c, opts.max_sat_conflicts);
            if (so.status == SolveOutcome::Sat) {
                pinned_cond = try_c;
                best_model = so.model;
                is_pinned[idx] = true;
            }
        }

        for (size_t idx : order) {
            if (is_pinned[idx]) continue;
            int64_t center = clamped_vals[idx];
            int64_t cur_val = sign_extend_bits(best_model[idx], w);
            int64_t best_dist = std::abs(cur_val - center);
            if (best_dist == 0) continue;

            auto try_window = [&](int64_t delta, std::vector<uint64_t>& model_out) -> bool {
                int64_t lo = center - delta;
                int64_t hi = center + delta;
                if (active_fields[idx].has_min) lo = std::max(lo, active_fields[idx].min_val);
                if (active_fields[idx].has_max) hi = std::min(hi, active_fields[idx].max_val);
                if (lo > hi) return false;
                BVNodeId ge = ctx.make_binary(BVOp::Sge, var_nodes[idx],
                                              ctx.make_const(static_cast<uint64_t>(lo), w), 1);
                BVNodeId le = ctx.make_binary(BVOp::Sle, var_nodes[idx],
                                              ctx.make_const(static_cast<uint64_t>(hi), w), 1);
                BVNodeId wcond = ctx.make_binary(BVOp::And, pinned_cond,
                                                 ctx.make_binary(BVOp::And, ge, le, 1), 1);
                SolveOutcome so = solve_nonzero(ctx, wcond, opts.max_sat_conflicts);
                if (so.status == SolveOutcome::Sat) {
                    model_out = std::move(so.model);
                    return true;
                }
                return false;
            };

            int64_t hi_d = std::min<int64_t>(best_dist - 1, 4096);
            std::vector<uint64_t> win_model;
            if (hi_d >= 0 && try_window(hi_d, win_model)) {
                best_model = win_model;
                int64_t lo_d = 0;
                int64_t cur_hi = std::abs(sign_extend_bits(best_model[idx], w) - center) - 1;
                for (int step = 0; step < 6 && lo_d <= cur_hi; ++step) {
                    int64_t mid = lo_d + (cur_hi - lo_d) / 2;
                    if (try_window(mid, win_model)) {
                        best_model = win_model;
                        cur_hi = std::abs(sign_extend_bits(best_model[idx], w) - center) - 1;
                    } else {
                        lo_d = mid + 1;
                    }
                }
                BVNodeId fix_best = ctx.make_binary(BVOp::Eq, var_nodes[idx],
                                                    ctx.make_const(best_model[idx] & mask, w), 1);
                pinned_cond = ctx.make_binary(BVOp::And, pinned_cond, fix_best, 1);
            }
        }
    }

    FieldMap best_map = model_to_map(best_model);
    if (!predicate_ || predicate_(best_map)) {
        out.valid = true;
        out.smt_solved = true;
        out.repaired_fields = std::move(best_map);
        return out;
    }

    if (!candidate.empty()) {
        std::vector<std::vector<uint64_t>> probes;
        const int64_t deltas[] = {1, -1, 2, -2, 4, -4, 8, -8, 16, -16, 32, -32, 64, -64};
        for (size_t i = 0; i < active_fields.size(); ++i) {
            if (fixed_set.count(active_fields[i].name)) continue;
            for (int64_t d : deltas) {
                auto m = clamped_model;
                int64_t nv = clamp_to_field(clamped_vals[i] + d, active_fields[i]);
                m[i] = static_cast<uint64_t>(nv) & mask;
                probes.push_back(std::move(m));
            }
        }
        for (size_t i = 0; i < active_fields.size(); ++i) {
            for (size_t j = i + 1; j < active_fields.size(); ++j) {
                if (fixed_set.count(active_fields[i].name) || fixed_set.count(active_fields[j].name))
                    continue;
                auto m = clamped_model;
                std::swap(m[i], m[j]);
                probes.push_back(m);
                for (int64_t step : {1, 8, 16}) {
                    auto m2 = m;
                    int64_t v_i = sign_extend_bits(m2[i], w);
                    m2[j] = static_cast<uint64_t>(clamp_to_field(v_i + step, active_fields[j])) & mask;
                    probes.push_back(std::move(m2));
                }
            }
        }
        for (const auto& pm : probes) {
            if (ctx.eval_concrete(base_cond, pm) != 0) {
                FieldMap pmap = model_to_map(pm);
                if (predicate_(pmap)) {
                    out.valid = true;
                    out.smt_solved = true;
                    out.iterations += 1;
                    out.repaired_fields = std::move(pmap);
                    return out;
                }
            }
        }
    }

    BVNodeId block0 = make_blocking_clause(best_model);
    base_cond = ctx.make_binary(BVOp::And, base_cond, block0, 1);
    BVNodeId active_cond = ctx.make_binary(BVOp::And, pinned_cond, block0, 1);
    bool using_pinned = (pinned_cond != base_cond);

    for (uint32_t iter = 1; iter < opts.max_cegis_iterations; ++iter) {
        if (using_pinned && iter > 6) {
            active_cond = base_cond;
            using_pinned = false;
        }
        SolveOutcome so = solve_nonzero(ctx, active_cond, opts.max_sat_conflicts);
        if (so.status != SolveOutcome::Sat) {
            if (using_pinned) {
                active_cond = base_cond;
                using_pinned = false;
                so = solve_nonzero(ctx, active_cond, opts.max_sat_conflicts);
            }
            if (so.status != SolveOutcome::Sat) break;
        }
        out.iterations = iter + 1;
        FieldMap cmap = model_to_map(so.model);
        if (predicate_(cmap)) {
            out.valid = true;
            out.smt_solved = true;
            out.repaired_fields = std::move(cmap);
            return out;
        }
        BVNodeId blk = make_blocking_clause(so.model);
        base_cond = ctx.make_binary(BVOp::And, base_cond, blk, 1);
        active_cond = ctx.make_binary(BVOp::And, active_cond, blk, 1);
    }

    out.valid = false;
    out.smt_solved = false;
    return out;
}

RepairResult VerifiedValidator::repair_json_with_smt(std::string_view json_text,
                                                     const RepairOptions& opts) const {
    RepairResult out;
    brotensor::json::Value doc;
    try {
        doc = brotensor::json::parse(std::string(json_text));
    } catch (const std::exception& e) {
        out.valid = false;
        out.violations.push_back(std::string("JSON syntax error: ") + e.what());
        return out;
    }
    if (!doc.is_object()) {
        out.valid = false;
        out.violations.push_back("JSON root must be an object");
        return out;
    }

    FieldMap fmap;
    for (const auto& [k, v] : doc.as_object()) {
        if (v.is_number()) fmap[k] = static_cast<int64_t>(std::llround(v.as_number()));
        else if (v.is_bool()) fmap[k] = v.as_bool() ? 1 : 0;
    }

    out = repair_with_smt(fmap, opts);
    if (!out.valid) return out;

    brotensor::json::Value::Object members = doc.as_object();
    std::unordered_set<std::string> seen;
    for (auto& [k, v] : members) {
        seen.insert(k);
        auto it = out.repaired_fields.find(k);
        if (it != out.repaired_fields.end()) {
            const FieldConstraint* fc = find_field(k);
            if ((fc && fc->type == FieldType::Bool) || v.is_bool()) {
                v = brotensor::json::Value::make_bool(it->second != 0);
            } else {
                v = brotensor::json::Value::make_number(static_cast<double>(it->second));
            }
        }
    }
    for (const auto& fc : fields_) {
        if (!seen.count(fc.name)) {
            auto it = out.repaired_fields.find(fc.name);
            if (it != out.repaired_fields.end()) {
                brotensor::json::Value val = (fc.type == FieldType::Bool)
                    ? brotensor::json::Value::make_bool(it->second != 0)
                    : brotensor::json::Value::make_number(static_cast<double>(it->second));
                members.emplace_back(fc.name, std::move(val));
            }
        }
    }
    out.repaired_json = serialize_json_value(brotensor::json::Value::make_object(std::move(members)));
    return out;
}

RepairResult VerifiedValidator::generate_valid(const FieldMap& seed,
                                               const RepairOptions& opts) const {
    if (seed.empty() && !fields_.empty()) {
        FieldMap init_seed;
        for (const auto& fc : fields_) {
            init_seed[fc.name] = clamp_to_field(0, fc);
        }
        RepairResult r = repair_with_smt(init_seed, opts);
        if (r.valid && !r.smt_solved) {
            r.smt_solved = true;
            r.iterations = 1;
        }
        return r;
    }
    RepairResult r = repair_with_smt(seed, opts);
    if (r.valid && !r.smt_solved) {
        r.smt_solved = true;
        r.iterations = 1;
    }
    return r;
}

// ─── Domain Deduction for Neuro-Symbolic Token Filtering ──────────────────────

FieldDomainResult VerifiedValidator::valid_domain_for_field(const FieldMap& partial_fields,
                                                            std::string_view field_name,
                                                            uint32_t max_samples) const {
    FieldDomainResult res;
    std::string target(field_name);
    if (!has_field(target)) return res;

    RepairOptions ropts;
    ropts.minimize_perturbation = false;
    for (const auto& [k, _] : partial_fields) {
        if (k != target && has_field(k)) ropts.fixed_fields.push_back(k);
    }

    RepairResult first = repair_with_smt(partial_fields, ropts);
    if (!first.valid) return res;

    int64_t v0 = first.repaired_fields[target];
    res.feasible = true;
    res.sample_val = v0;
    res.min_val = v0;
    res.max_val = v0;
    res.sample_values.push_back(v0);

    const FieldConstraint* fc = find_field(target);
    int64_t bound_lo = (fc && fc->has_min) ? fc->min_val : -65536;
    int64_t bound_hi = (fc && fc->has_max) ? fc->max_val :  65536;

    auto probe_range = [&](int64_t lo, int64_t hi, int64_t& found_val) -> bool {
        if (lo > hi) return false;
        VerifiedValidator sub = *this;
        sub.add_constraint(target + " >= " + std::to_string(lo) + " && " +
                           target + " <= " + std::to_string(hi));
        RepairResult r = sub.repair_with_smt(partial_fields, ropts);
        if (r.valid) {
            found_val = r.repaired_fields[target];
            return true;
        }
        return false;
    };

    int64_t cur_min = v0;
    int64_t probe_v = 0;
    if (probe_range(bound_lo, cur_min - 1, probe_v)) {
        cur_min = probe_v;
        int64_t lo = bound_lo, hi = cur_min - 1;
        for (int step = 0; step < 16 && lo <= hi; ++step) {
            int64_t mid = lo + (hi - lo) / 2;
            if (probe_range(lo, mid, probe_v)) {
                cur_min = probe_v;
                hi = cur_min - 1;
            } else {
                lo = mid + 1;
            }
        }
    }
    res.min_val = cur_min;

    int64_t cur_max = v0;
    if (probe_range(cur_max + 1, bound_hi, probe_v)) {
        cur_max = probe_v;
        int64_t lo = cur_max + 1, hi = bound_hi;
        for (int step = 0; step < 16 && lo <= hi; ++step) {
            int64_t mid = lo + (hi - lo + 1) / 2;
            if (probe_range(mid, hi, probe_v)) {
                cur_max = probe_v;
                lo = cur_max + 1;
            } else {
                hi = mid - 1;
            }
        }
    }
    res.max_val = cur_max;

    auto add_sample = [&](int64_t v) {
        if (std::find(res.sample_values.begin(), res.sample_values.end(), v) == res.sample_values.end()) {
            res.sample_values.push_back(v);
        }
    };
    add_sample(res.min_val);
    add_sample(res.max_val);

    VerifiedValidator enum_val = *this;
    for (int64_t existing : res.sample_values) {
        enum_val.add_constraint(target + " != " + std::to_string(existing));
    }
    while (res.sample_values.size() < max_samples) {
        RepairResult r = enum_val.repair_with_smt(partial_fields, ropts);
        if (!r.valid) break;
        int64_t nv = r.repaired_fields[target];
        add_sample(nv);
        enum_val.add_constraint(target + " != " + std::to_string(nv));
    }
    std::sort(res.sample_values.begin(), res.sample_values.end());
    return res;
}

// ─── Grammar Integration ──────────────────────────────────────────────────────

Grammar VerifiedValidator::build_grammar() const {
    if (fields_.empty()) return Grammar::json_object();
    std::ostringstream schema;
    schema << "{\"type\":\"object\",\"properties\":{";
    for (size_t i = 0; i < fields_.size(); ++i) {
        if (i > 0) schema << ",";
        const char* jtype = (fields_[i].type == FieldType::Bool) ? "boolean" : "integer";
        schema << "\"" << fields_[i].name << "\":{\"type\":\"" << jtype << "\"}";
    }
    schema << "},\"required\":[";
    for (size_t i = 0; i < fields_.size(); ++i) {
        if (i > 0) schema << ",";
        schema << "\"" << fields_[i].name << "\"";
    }
    schema << "]}";
    return Grammar::json_schema(schema.str());
}

Grammar VerifiedValidator::build_field_grammar(const FieldMap& partial_fields,
                                               std::string_view field_name,
                                               uint32_t max_choices) const {
    FieldDomainResult dom = valid_domain_for_field(partial_fields, field_name, max_choices);
    if (!dom.feasible || dom.sample_values.empty()) return Grammar::json_integer();
    const FieldConstraint* fc = find_field(field_name);
    std::vector<std::string> choices;
    choices.reserve(dom.sample_values.size());
    for (int64_t v : dom.sample_values) {
        if (fc && fc->type == FieldType::Bool) choices.push_back(v != 0 ? "true" : "false");
        else choices.push_back(std::to_string(v));
    }
    return Grammar::choice(choices);
}

}  // namespace brolm
