#include "brolm/verified_validator.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <iomanip>
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

std::string trim_ws(std::string_view sv) {
    size_t b = 0, e = sv.size();
    while (b < e && std::isspace(static_cast<unsigned char>(sv[b]))) ++b;
    while (e > b && std::isspace(static_cast<unsigned char>(sv[e - 1]))) --e;
    return std::string(sv.substr(b, e - b));
}

bool is_ident_char(char c) {
    return std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '$';
}

std::string replace_ident(std::string src, std::string_view from, std::string_view to) {
    if (from.empty() || from == to) return src;
    size_t pos = 0;
    while ((pos = src.find(from, pos)) != std::string::npos) {
        bool left_ok = (pos == 0) || (!is_ident_char(src[pos - 1]) && src[pos - 1] != '.');
        bool right_ok = (pos + from.size() >= src.size()) || !is_ident_char(src[pos + from.size()]);
        if (left_ok && right_ok) {
            src.replace(pos, from.size(), to);
            pos += to.size();
        } else {
            pos += from.size();
        }
    }
    return src;
}

std::string compute_certificate_hash(std::string_view tag,
                                     std::string_view expr,
                                     const std::vector<std::string>& params,
                                     std::string_view postcond) {
    auto fnv64 = [](std::string_view data, uint64_t seed) {
        uint64_t h = seed;
        for (unsigned char c : data) {
            h ^= static_cast<uint64_t>(c);
            h *= 1099511628211ULL;
        }
        return h;
    };
    std::string payload(tag);
    payload += "|";
    for (const auto& p : params) {
        payload += p;
        payload += ",";
    }
    payload += "|";
    payload += expr;
    payload += "|";
    payload += postcond;

    uint64_t h1 = fnv64(payload, 14695981039346656037ULL);
    uint64_t h2 = fnv64(payload, 0x9e3779b97f4a7c15ULL ^ h1);
    uint64_t h3 = fnv64(payload, 0xc2b2ae3d27d4eb4fULL ^ h2);
    uint64_t h4 = fnv64(payload, 0x165667b19e3779f9ULL ^ h3);

    std::ostringstream os;
    os << std::hex << std::setfill('0')
       << std::setw(16) << h1
       << std::setw(16) << h2
       << std::setw(16) << h3
       << std::setw(16) << h4;
    return os.str();
}

std::string format_synthesized_source(const std::vector<std::string>& params,
                                      std::string_view expr) {
    std::ostringstream os;
    os << "(";
    for (size_t i = 0; i < params.size(); ++i) {
        if (i > 0) os << ", ";
        os << params[i];
    }
    os << ") => (" << trim_ws(expr) << ")";
    return os.str();
}

// Convert a block body `{ if (c1) return e1; const k = v; return e2; }` into a single expression.
std::string block_body_to_expression(std::string_view inner_raw) {
    std::string inner = trim_ws(inner_raw);
    std::unordered_map<std::string, std::string> locals;
    auto subst_locals = [&](std::string e) {
        for (const auto& [k, v] : locals) {
            e = replace_ident(std::move(e), k, "(" + v + ")");
        }
        return e;
    };

    std::vector<std::pair<std::string, std::string>> branches;
    std::string final_return;

    size_t cur = 0;
    while (cur < inner.size()) {
        while (cur < inner.size() &&
               (std::isspace(static_cast<unsigned char>(inner[cur])) || inner[cur] == ';')) {
            ++cur;
        }
        if (cur >= inner.size()) break;

        if (inner.compare(cur, 3, "if ") == 0 || inner.compare(cur, 3, "if(") == 0) {
            size_t lp = inner.find('(', cur);
            if (lp == std::string::npos) break;
            int depth = 1;
            size_t rp = lp + 1;
            while (rp < inner.size() && depth > 0) {
                if (inner[rp] == '(') ++depth;
                else if (inner[rp] == ')') --depth;
                ++rp;
            }
            std::string cond = subst_locals(trim_ws(inner.substr(lp + 1, rp - lp - 2)));
            std::string rest = trim_ws(inner.substr(rp));
            std::string then_stmt;
            size_t adv = rp;
            if (!rest.empty() && rest.front() == '{') {
                int bdepth = 1;
                size_t bp = rp + inner.substr(rp).find('{') + 1;
                size_t bstart = bp;
                while (bp < inner.size() && bdepth > 0) {
                    if (inner[bp] == '{') ++bdepth;
                    else if (inner[bp] == '}') --bdepth;
                    ++bp;
                }
                then_stmt = trim_ws(inner.substr(bstart, bp - bstart - 1));
                if (!then_stmt.empty() && then_stmt.back() == ';') then_stmt.pop_back();
                adv = bp;
            } else {
                size_t semi = inner.find(';', rp);
                if (semi == std::string::npos) break;
                then_stmt = trim_ws(inner.substr(rp, semi - rp));
                adv = semi + 1;
            }
            if (then_stmt.rfind("return", 0) == 0) {
                std::string ret_expr = subst_locals(trim_ws(then_stmt.substr(6)));
                if (!ret_expr.empty() && ret_expr.back() == ';') ret_expr.pop_back();
                branches.emplace_back(std::move(cond), std::move(ret_expr));
                cur = adv;
                continue;
            }
        }

        size_t semi = inner.find(';', cur);
        std::string stmt = trim_ws(
            inner.substr(cur, (semi == std::string::npos) ? std::string::npos : (semi - cur)));
        cur = (semi == std::string::npos) ? inner.size() : (semi + 1);

        bool matched_decl = false;
        for (std::string_view kw : {"const ", "let ", "var "}) {
            if (stmt.rfind(kw, 0) == 0) {
                std::string decl = trim_ws(stmt.substr(kw.size()));
                size_t eq = decl.find('=');
                if (eq != std::string::npos) {
                    std::string vname = trim_ws(decl.substr(0, eq));
                    std::string rhs = subst_locals(trim_ws(decl.substr(eq + 1)));
                    locals[vname] = rhs;
                }
                matched_decl = true;
                break;
            }
        }
        if (matched_decl) continue;

        if (stmt.rfind("return", 0) == 0) {
            final_return = subst_locals(trim_ws(stmt.substr(6)));
            break;
        }
    }

    if (final_return.empty() && branches.empty()) {
        return inner;
    }
    if (final_return.empty()) final_return = "0";

    std::string acc = final_return;
    for (auto it = branches.rbegin(); it != branches.rend(); ++it) {
        acc = "((" + it->first + ") ? (" + it->second + ") : (" + acc + "))";
    }
    return acc;
}

std::vector<std::string> build_template_candidates(const std::vector<std::string>& params,
                                                   std::string_view postcondition,
                                                   std::string_view ref_expr) {
    std::vector<std::string> out;
    auto add = [&](std::string s) {
        s = trim_ws(s);
        if (s.empty()) return;
        if (std::find(out.begin(), out.end(), s) == out.end()) {
            out.push_back(std::move(s));
        }
    };

    if (!ref_expr.empty()) {
        std::string r_expr;
        if (extract_js_function_expression(ref_expr, nullptr, &r_expr)) add(r_expr);
        else add(std::string(ref_expr));
    }

    // Check if postcondition contains `out === <rhs>` or `out == <rhs>`
    std::string post(postcondition);
    for (std::string_view pat : {"out === ", "out == ", "out===", "out=="}) {
        size_t p = post.find(pat);
        if (p != std::string::npos) {
            std::string rhs = trim_ws(post.substr(p + pat.size()));
            size_t amp = rhs.find("&&");
            if (amp != std::string::npos) rhs = trim_ws(rhs.substr(0, amp));
            if (!rhs.empty() && rhs.back() == ')') {
                int bal = 0;
                for (char c : rhs) {
                    if (c == '(') ++bal;
                    else if (c == ')') --bal;
                }
                while (bal < 0 && !rhs.empty() && rhs.back() == ')') {
                    rhs.pop_back();
                    ++bal;
                }
            }
            if (!rhs.empty() && rhs.find("out") == std::string::npos) {
                add(rhs);
            }
        }
    }

    if (params.size() == 1) {
        const std::string& x = params[0];
        add(x);
        add("(" + x + " >= 0 ? " + x + " : -" + x + ")");
        add("Math.abs(" + x + ")");
        add("(" + x + " > 0 ? " + x + " : 0)");
        add("(" + x + " < 0 ? -1 : (" + x + " > 0 ? 1 : 0))");
        add("-" + x);
        add("~" + x);
        add(x + " + 1");
        add(x + " - 1");
        add(x + " * 2");
        add(x + " * " + x);
        add("0");
        add("1");
    } else if (params.size() == 2) {
        const std::string& a = params[0];
        const std::string& b = params[1];
        add(a + " + " + b);
        add(a + " - " + b);
        add(b + " - " + a);
        add("(" + a + " < " + b + " ? " + a + " : " + b + ")");
        add("(" + a + " > " + b + " ? " + a + " : " + b + ")");
        add("Math.min(" + a + ", " + b + ")");
        add("Math.max(" + a + ", " + b + ")");
        add("Math.abs(" + a + " - " + b + ")");
        add("(" + a + " >= " + b + " ? " + a + " - " + b + " : " + b + " - " + a + ")");
        add("((" + a + " + " + b + ") >> 1)");
        add(a + " * " + b);
        add(a + " ^ " + b);
        add(a + " & " + b);
        add(a + " | " + b);
        add("(" + b + " != 0 ? " + a + " / " + b + " : 0)");
        add("(" + b + " != 0 ? " + a + " % " + b + " : 0)");
    } else if (params.size() >= 3) {
        const std::string& x = params[0];
        const std::string& lo = params[1];
        const std::string& hi = params[2];
        add("(" + x + " < " + lo + " ? " + lo + " : (" + x + " > " + hi + " ? " + hi + " : " + x + "))");
        add("Math.min(Math.max(" + x + ", " + lo + "), " + hi + ")");
        add("(" + x + " ? " + lo + " : " + hi + ")");
        add(x + " + " + lo + " + " + hi);
        add(x + " * " + lo + " + " + hi);
    }

    return out;
}

}  // namespace

bool extract_js_function_expression(std::string_view js_source,
                                    std::vector<std::string>* out_params,
                                    std::string* out_expr) {
    std::string s = trim_ws(js_source);
    if (s.empty() || s.find("[native code]") != std::string::npos) return false;

    std::vector<std::string> params;
    std::string body;

    size_t arrow = s.find("=>");
    if (arrow != std::string::npos) {
        // Check that '=>' is at top level (not inside nested parens of a comparison)
        int depth = 0;
        size_t top_arrow = std::string::npos;
        for (size_t i = 0; i + 1 < s.size(); ++i) {
            if (s[i] == '(') ++depth;
            else if (s[i] == ')') --depth;
            else if (depth == 0 && s[i] == '=' && s[i + 1] == '>') {
                top_arrow = i;
                break;
            }
        }
        if (top_arrow != std::string::npos) {
            std::string head = trim_ws(s.substr(0, top_arrow));
            body = trim_ws(s.substr(top_arrow + 2));
            if (!head.empty() && head.front() == '(' && head.back() == ')') {
                head = trim_ws(head.substr(1, head.size() - 2));
            }
            std::stringstream ss(head);
            std::string item;
            while (std::getline(ss, item, ',')) {
                std::string p = trim_ws(item);
                if (!p.empty()) params.push_back(p);
            }
        } else {
            body = s;
        }
    } else if (s.rfind("function", 0) == 0) {
        size_t lp = s.find('(');
        size_t rp = (lp != std::string::npos) ? s.find(')', lp) : std::string::npos;
        if (lp == std::string::npos || rp == std::string::npos) return false;
        std::string head = trim_ws(s.substr(lp + 1, rp - lp - 1));
        body = trim_ws(s.substr(rp + 1));
        std::stringstream ss(head);
        std::string item;
        while (std::getline(ss, item, ',')) {
            std::string p = trim_ws(item);
            if (!p.empty()) params.push_back(p);
        }
    } else {
        body = s;
    }

    if (!body.empty() && body.front() == '{' && body.back() == '}') {
        body = block_body_to_expression(body.substr(1, body.size() - 2));
    }
    if (!body.empty() && body.back() == ';') {
        body = trim_ws(body.substr(0, body.size() - 1));
    }

    if (out_params && !params.empty()) *out_params = std::move(params);
    if (out_expr) *out_expr = std::move(body);
    return true;
}

bool evaluate_bv_expression(std::string_view expr,
                            const std::vector<std::string>& param_names,
                            const std::vector<int64_t>& param_values,
                            int64_t* out_value,
                            uint32_t width) {
    if (param_names.size() != param_values.size()) return false;
    BVContext ctx;
    std::unordered_map<std::string, BVNodeId> vars;
    std::vector<uint64_t> vals;
    vals.reserve(param_names.size());
    uint64_t mask = width_mask(width);
    for (size_t i = 0; i < param_names.size(); ++i) {
        vars[param_names[i]] = ctx.make_var(param_names[i], width);
        vals.push_back(static_cast<uint64_t>(param_values[i]) & mask);
    }
    BVNodeId div_guard = 0;
    std::string err;
    BVNodeId node = VerifiedValidator::parse_expression_to_bv(
        ctx, expr, vars, width, &div_guard, &err);
    if (node == UINT32_MAX) return false;
    if (div_guard != 0 && ctx.eval_concrete(div_guard, vals) == 0) return false;
    uint64_t raw = ctx.eval_concrete(node, vals);
    uint32_t nw = ctx.width(node);
    if (out_value) {
        *out_value = (nw == 1) ? static_cast<int64_t>(raw & 1ULL) : sign_extend_bits(raw, nw);
    }
    return true;
}

SynthesizeResult synthesize_verified_function(const SynthesizeSpec& spec) {
    SynthesizeResult res;
    std::vector<std::string> params = spec.params;
    if (params.empty()) {
        for (const auto& fd : spec.param_domains) {
            if (!fd.name.empty()) params.push_back(fd.name);
        }
    }
    if (params.empty() && !spec.reference_expression.empty()) {
        extract_js_function_expression(spec.reference_expression, &params, nullptr);
    }
    if (params.empty() && !spec.candidates.empty()) {
        extract_js_function_expression(spec.candidates[0], &params, nullptr);
    }
    if (params.empty()) {
        params.push_back("x");
    }
    res.params = params;

    const uint32_t w = (spec.bit_width == 64) ? 64 : 32;
    const uint64_t mask = width_mask(w);

    auto domain_for_param = [&](const std::string& pname) -> FieldConstraint {
        for (const auto& fd : spec.param_domains) {
            if (fd.name == pname) return fd;
        }
        return FieldConstraint::integer(pname, spec.default_min, spec.default_max, w);
    };

    // Helper: compute expected output for a concrete input vector
    auto compute_expected_output = [&](const std::vector<int64_t>& inputs,
                                       int64_t& expected_out) -> bool {
        if (spec.concrete_reference) {
            expected_out = spec.concrete_reference(inputs);
            return true;
        }
        for (const auto& ex : spec.examples) {
            if (ex.inputs == inputs) {
                expected_out = ex.expected_output;
                return true;
            }
        }
        if (!spec.reference_expression.empty()) {
            std::vector<std::string> rparams;
            std::string rexpr;
            if (extract_js_function_expression(spec.reference_expression, &rparams, &rexpr)) {
                if (rparams.size() == params.size()) {
                    for (size_t i = 0; i < params.size(); ++i) {
                        rexpr = replace_ident(std::move(rexpr), rparams[i], params[i]);
                    }
                }
                if (evaluate_bv_expression(rexpr, params, inputs, &expected_out, w)) {
                    return true;
                }
            }
        }
        if (!spec.postcondition.empty()) {
            BVContext sctx;
            std::unordered_map<std::string, BVNodeId> svars;
            BVNodeId out_var = sctx.make_var(spec.output_var, w);
            svars[spec.output_var] = out_var;
            svars["out"] = out_var;
            svars["result"] = out_var;
            svars["ret"] = out_var;
            for (size_t i = 0; i < params.size() && i < inputs.size(); ++i) {
                svars[params[i]] = sctx.make_const(static_cast<uint64_t>(inputs[i]), w);
            }
            BVNodeId dguard = 0;
            std::string perr;
            BVNodeId pcond = VerifiedValidator::parse_constraint_to_bv(
                sctx, spec.postcondition, svars, w, &dguard, &perr);
            if (pcond != UINT32_MAX) {
                if (dguard != 0) pcond = sctx.make_binary(BVOp::And, pcond, dguard, 1);
                ++res.smt_queries;
                SolveOutcome so = solve_nonzero(sctx, pcond, spec.max_sat_conflicts);
                if (so.status == SolveOutcome::Sat && !so.model.empty()) {
                    expected_out = sign_extend_bits(so.model[0], w);
                    return true;
                }
            }
        }
        return false;
    };

    // Helper: normalize a candidate string into an expression over `params`
    auto normalize_candidate = [&](std::string_view raw_cand, std::string& norm_expr) -> bool {
        std::vector<std::string> cparams;
        if (!extract_js_function_expression(raw_cand, &cparams, &norm_expr)) return false;
        if (cparams.size() == params.size()) {
            for (size_t i = 0; i < params.size(); ++i) {
                norm_expr = replace_ident(std::move(norm_expr), cparams[i], params[i]);
            }
        }
        return !norm_expr.empty();
    };

    // Helper: formally verify a normalized candidate expression using BVContext + solve_nonzero
    auto verify_candidate_expr = [&](const std::string& cand_expr,
                                     CounterexampleRecord* out_cex) -> bool {
        BVContext ctx;
        std::unordered_map<std::string, BVNodeId> vars;
        std::vector<BVNodeId> param_nodes;
        param_nodes.reserve(params.size());

        BVNodeId dom_cond = ctx.make_const(1, 1);
        auto conjoin_dom = [&](BVNodeId c) {
            dom_cond = ctx.make_binary(BVOp::And, dom_cond, c, 1);
        };

        for (const auto& pname : params) {
            FieldConstraint fc = domain_for_param(pname);
            BVNodeId v = ctx.make_var(pname, w);
            vars[pname] = v;
            param_nodes.push_back(v);

            int64_t lo = fc.has_min ? fc.min_val : spec.default_min;
            int64_t hi = fc.has_max ? fc.max_val : spec.default_max;
            conjoin_dom(ctx.make_binary(BVOp::Sge, v, ctx.make_const(static_cast<uint64_t>(lo), w), 1));
            conjoin_dom(ctx.make_binary(BVOp::Sle, v, ctx.make_const(static_cast<uint64_t>(hi), w), 1));
        }

        if (!spec.precondition.empty()) {
            BVNodeId pre_div = 0;
            std::string pre_err;
            BVNodeId pre_id = VerifiedValidator::parse_constraint_to_bv(
                ctx, spec.precondition, vars, w, &pre_div, &pre_err);
            if (pre_id == UINT32_MAX) {
                res.error = "Invalid precondition: " + pre_err;
                return false;
            }
            if (pre_div != 0) conjoin_dom(pre_div);
            conjoin_dom(pre_id);
        }

        BVNodeId cand_div_guard = 0;
        std::string cand_err;
        BVNodeId cand_node = VerifiedValidator::parse_expression_to_bv(
            ctx, cand_expr, vars, w, &cand_div_guard, &cand_err);
        if (cand_node == UINT32_MAX) {
            if (out_cex) {
                out_cex->candidate_source = cand_expr;
                out_cex->violation = "parse error: " + cand_err;
            }
            return false;
        }
        uint32_t cw = ctx.width(cand_node);
        if (cw < w) {
            cand_node = (cw == 1) ? ctx.make_zext(cand_node, w - 1)
                                  : ctx.make_sext(cand_node, w - cw);
        } else if (cw > w) {
            cand_node = ctx.make_extract(cand_node, w - 1, 0);
        }

        auto vars_with_out = vars;
        vars_with_out[spec.output_var] = cand_node;
        vars_with_out["out"] = cand_node;
        vars_with_out["result"] = cand_node;
        vars_with_out["ret"] = cand_node;

        BVNodeId spec_ok = ctx.make_const(1, 1);
        auto conjoin_spec = [&](BVNodeId c) {
            spec_ok = ctx.make_binary(BVOp::And, spec_ok, c, 1);
        };

        if (cand_div_guard != 0) {
            conjoin_spec(cand_div_guard);
        }

        if (!spec.postcondition.empty()) {
            BVNodeId post_div = 0;
            std::string post_err;
            BVNodeId post_id = VerifiedValidator::parse_constraint_to_bv(
                ctx, spec.postcondition, vars_with_out, w, &post_div, &post_err);
            if (post_id == UINT32_MAX) {
                res.error = "Invalid postcondition: " + post_err;
                return false;
            }
            if (post_div != 0) conjoin_spec(post_div);
            conjoin_spec(post_id);
        }

        if (!spec.reference_expression.empty()) {
            std::string rexpr;
            std::vector<std::string> rparams;
            if (extract_js_function_expression(spec.reference_expression, &rparams, &rexpr)) {
                if (rparams.size() == params.size()) {
                    for (size_t i = 0; i < params.size(); ++i) {
                        rexpr = replace_ident(std::move(rexpr), rparams[i], params[i]);
                    }
                }
                BVNodeId ref_div = 0;
                std::string ref_err;
                BVNodeId ref_node = VerifiedValidator::parse_expression_to_bv(
                    ctx, rexpr, vars, w, &ref_div, &ref_err);
                if (ref_node != UINT32_MAX) {
                    uint32_t rw = ctx.width(ref_node);
                    if (rw < w) {
                        ref_node = (rw == 1) ? ctx.make_zext(ref_node, w - 1)
                                             : ctx.make_sext(ref_node, w - rw);
                    }
                    if (ref_div != 0) conjoin_dom(ref_div);
                    conjoin_spec(ctx.make_binary(BVOp::Eq, cand_node, ref_node, 1));
                }
            }
        }

        for (const auto& ex : spec.examples) {
            if (ex.inputs.size() != params.size()) continue;
            BVNodeId match_in = ctx.make_const(1, 1);
            for (size_t i = 0; i < params.size(); ++i) {
                BVNodeId eq_i = ctx.make_binary(
                    BVOp::Eq, param_nodes[i],
                    ctx.make_const(static_cast<uint64_t>(ex.inputs[i]), w), 1);
                match_in = ctx.make_binary(BVOp::And, match_in, eq_i, 1);
            }
            BVNodeId match_out = ctx.make_binary(
                BVOp::Eq, cand_node,
                ctx.make_const(static_cast<uint64_t>(ex.expected_output), w), 1);
            BVNodeId impl = ctx.make_binary(
                BVOp::Or, ctx.make_unary(BVOp::Not, match_in, 1), match_out, 1);
            conjoin_spec(impl);
        }

        // Search for any input in domain where !spec_ok holds
        BVNodeId violation = ctx.make_binary(
            BVOp::And, dom_cond, ctx.make_unary(BVOp::Not, spec_ok, 1), 1);

        ++res.smt_queries;
        SolveOutcome so = solve_nonzero(ctx, violation, spec.max_sat_conflicts);
        if (so.status == SolveOutcome::Sat) {
            if (out_cex) {
                out_cex->candidate_source = format_synthesized_source(params, cand_expr);
                out_cex->inputs.resize(params.size(), 0);
                for (size_t i = 0; i < params.size() && i < so.model.size(); ++i) {
                    int64_t v = sign_extend_bits(so.model[i], w);
                    out_cex->inputs[i] = v;
                    out_cex->params[params[i]] = v;
                }
                uint64_t raw_out = ctx.eval_concrete(cand_node, so.model);
                out_cex->candidate_output = sign_extend_bits(raw_out, w);
                if (cand_div_guard != 0 && ctx.eval_concrete(cand_div_guard, so.model) == 0) {
                    out_cex->violation = "division/remainder by zero";
                } else {
                    out_cex->violation = spec.postcondition.empty()
                        ? "reference/example mismatch"
                        : spec.postcondition;
                }
                int64_t exp_out = 0;
                if (compute_expected_output(out_cex->inputs, exp_out)) {
                    out_cex->expected_output = exp_out;
                    out_cex->has_expected_output = true;
                }
            }
            return false;
        }
        if (so.status != SolveOutcome::Unsat) {
            if (out_cex) {
                out_cex->candidate_source = format_synthesized_source(params, cand_expr);
                out_cex->violation = "solver conflict budget exceeded";
            }
            return false;
        }

        // Also check concrete_postcondition / concrete_reference if provided
        if (spec.concrete_postcondition || spec.concrete_reference) {
            std::vector<std::vector<int64_t>> test_vectors;
            for (const auto& ex : spec.examples) {
                if (ex.inputs.size() == params.size()) test_vectors.push_back(ex.inputs);
            }
            const int64_t sample_scalars[] = {-100, -10, -1, 0, 1, 2, 10, 50, 100};
            for (int64_t s : sample_scalars) {
                std::vector<int64_t> pt(params.size(), s);
                for (size_t i = 0; i < params.size(); ++i) {
                    FieldConstraint fc = domain_for_param(params[i]);
                    int64_t lo = fc.has_min ? fc.min_val : spec.default_min;
                    int64_t hi = fc.has_max ? fc.max_val : spec.default_max;
                    pt[i] = std::clamp(pt[i] + static_cast<int64_t>(i * 3), lo, hi);
                }
                test_vectors.push_back(std::move(pt));
            }
            for (const auto& vec : test_vectors) {
                std::vector<uint64_t> uvec(vec.size());
                for (size_t i = 0; i < vec.size(); ++i) uvec[i] = static_cast<uint64_t>(vec[i]) & mask;
                if (ctx.eval_concrete(dom_cond, uvec) == 0) continue;
                int64_t cout = sign_extend_bits(ctx.eval_concrete(cand_node, uvec), w);
                bool ok = true;
                int64_t exp = 0;
                bool has_exp = false;
                if (spec.concrete_reference) {
                    exp = spec.concrete_reference(vec);
                    has_exp = true;
                    if (cout != exp) ok = false;
                }
                if (ok && spec.concrete_postcondition) {
                    if (!spec.concrete_postcondition(vec, cout)) ok = false;
                }
                if (!ok) {
                    if (out_cex) {
                        out_cex->candidate_source = format_synthesized_source(params, cand_expr);
                        out_cex->inputs = vec;
                        for (size_t i = 0; i < params.size(); ++i) out_cex->params[params[i]] = vec[i];
                        out_cex->candidate_output = cout;
                        out_cex->expected_output = exp;
                        out_cex->has_expected_output = has_exp;
                        out_cex->violation = "concrete oracle failed";
                    }
                    return false;
                }
            }
        }

        return true;
    };

    std::vector<std::string> template_bank = build_template_candidates(
        params, spec.postcondition, spec.reference_expression);
    size_t next_cand_idx = 0;
    size_t next_tmpl_idx = 0;
    std::string last_expr;
    std::unordered_set<std::string> tried_exprs;

    for (uint32_t iter = 0; iter < spec.max_cegis_iterations; ++iter) {
        std::string raw_cand;

        if (spec.synthesizer) {
            raw_cand = spec.synthesizer(res.counterexamples_resolved, iter);
        }
        if (raw_cand.empty() && next_cand_idx < spec.candidates.size()) {
            raw_cand = spec.candidates[next_cand_idx++];
        }
        if (raw_cand.empty()) {
            // Filter template bank by all accumulated counterexamples with known expected_output
            while (next_tmpl_idx < template_bank.size()) {
                const std::string& tmpl = template_bank[next_tmpl_idx++];
                if (tried_exprs.count(tmpl)) continue;
                bool matches_cexs = true;
                for (const auto& cex : res.counterexamples_resolved) {
                    if (!cex.has_expected_output) continue;
                    int64_t tv = 0;
                    if (!evaluate_bv_expression(tmpl, params, cex.inputs, &tv, w) ||
                        tv != cex.expected_output) {
                        matches_cexs = false;
                        break;
                    }
                }
                if (matches_cexs) {
                    raw_cand = tmpl;
                    break;
                }
            }
        }
        if (raw_cand.empty() && !res.counterexamples_resolved.empty()) {
            // Counterexample-guided guard patch on the latest expression
            const auto& last_cex = res.counterexamples_resolved.back();
            if (last_cex.has_expected_output && !last_expr.empty()) {
                std::ostringstream cond;
                for (size_t i = 0; i < params.size() && i < last_cex.inputs.size(); ++i) {
                    if (i > 0) cond << " && ";
                    cond << params[i] << " == " << last_cex.inputs[i];
                }
                raw_cand = "((" + cond.str() + ") ? " +
                           std::to_string(last_cex.expected_output) +
                           " : (" + last_expr + "))";
            }
        }
        if (raw_cand.empty()) break;

        std::string norm_expr;
        if (!normalize_candidate(raw_cand, norm_expr)) continue;
        tried_exprs.insert(norm_expr);
        last_expr = norm_expr;
        res.iterations = iter + 1;

        CounterexampleRecord cex;
        if (verify_candidate_expr(norm_expr, &cex)) {
            res.verified = true;
            res.expression = norm_expr;
            res.synthesized_source = format_synthesized_source(params, norm_expr);
            res.certificate_hash = compute_certificate_hash(
                "BROLM_SYNTH_V1", norm_expr, params, spec.postcondition);
            return res;
        }
        if (!res.error.empty()) {
            return res;
        }
        res.counterexamples_resolved.push_back(std::move(cex));
    }

    res.verified = false;
    return res;
}

ProverResult solve_with_prover(const ProverProblemSpec& problem) {
    ProverResult res;
    VerifiedValidator val = problem.validator;
    RepairOptions ropts = problem.repair_opts;

    // 1. Check any initial candidates or proposer callbacks
    size_t cand_idx = 0;
    for (uint32_t iter = 0; iter < ropts.max_cegis_iterations; ++iter) {
        FieldMap cand;
        bool has_cand = false;
        if (cand_idx < problem.candidates.size()) {
            cand = problem.candidates[cand_idx++];
            has_cand = true;
        } else if (problem.proposer) {
            cand = problem.proposer(res.rejected_candidates, res.violations, iter);
            has_cand = !cand.empty();
        }
        if (!has_cand) break;

        ValidationResult vres = val.validate(cand);
        ++res.iterations;
        if (vres.valid && problem.minimize_expr.empty() && problem.maximize_expr.empty()) {
            res.solved = true;
            res.valid = true;
            res.solution = std::move(cand);
            res.violations.clear();
            break;
        }
        res.violations = vres.violations;
        res.rejected_candidates.push_back(cand);
    }

    // 2. Run SMT repair / synthesis
    if (!res.solved) {
        FieldMap seed = res.rejected_candidates.empty() ? FieldMap{} : res.rejected_candidates.back();
        RepairResult rep = seed.empty() ? val.generate_valid({}, ropts)
                                        : val.repair_with_smt(seed, ropts);
        res.smt_queries += std::max<uint32_t>(1, rep.iterations);
        res.iterations += std::max<uint32_t>(1, rep.iterations);
        if (!rep.valid) {
            res.solved = false;
            res.valid = false;
            res.violations = rep.violations;
            return res;
        }
        res.solved = true;
        res.valid = true;
        res.solution = rep.repaired_fields;
        res.violations.clear();
    }

    // 3. Optional SMT objective optimization (minimize / maximize)
    std::string obj_expr = !problem.minimize_expr.empty() ? problem.minimize_expr : problem.maximize_expr;
    bool is_minimize = !problem.minimize_expr.empty();
    if (!obj_expr.empty() && res.solved) {
        std::vector<std::string> fnames;
        std::vector<int64_t> fvals;
        for (const auto& fc : val.fields()) {
            fnames.push_back(fc.name);
            auto it = res.solution.find(fc.name);
            fvals.push_back(it != res.solution.end() ? it->second : 0);
        }
        if (fnames.empty()) {
            for (const auto& [k, v] : res.solution) {
                fnames.push_back(k);
                fvals.push_back(v);
            }
        }
        int64_t best_obj = 0;
        if (evaluate_bv_expression(obj_expr, fnames, fvals, &best_obj, 32)) {
            res.has_objective = true;
            res.objective_value = best_obj;

            auto try_improve = [&](int64_t bound, FieldMap& improved_out, int64_t& obj_out) -> bool {
                VerifiedValidator sub = val;
                std::string cmp = is_minimize
                    ? ("(" + obj_expr + ") <= " + std::to_string(bound))
                    : ("(" + obj_expr + ") >= " + std::to_string(bound));
                sub.add_constraint(cmp);
                RepairOptions opt_ropts = ropts;
                opt_ropts.minimize_perturbation = false;
                RepairResult r = sub.repair_with_smt(res.solution, opt_ropts);
                ++res.smt_queries;
                if (!r.valid) return false;
                std::vector<int64_t> nv;
                for (const auto& fn : fnames) nv.push_back(r.repaired_fields[fn]);
                int64_t nobj = 0;
                if (!evaluate_bv_expression(obj_expr, fnames, nv, &nobj, 32)) return false;
                improved_out = std::move(r.repaired_fields);
                obj_out = nobj;
                return true;
            };

            FieldMap cand_map;
            int64_t cand_obj = best_obj;
            if (is_minimize) {
                int64_t lo = best_obj - 4096;
                int64_t hi = best_obj - 1;
                for (int step = 0; step < 14 && lo <= hi; ++step) {
                    int64_t mid = lo + (hi - lo) / 2;
                    if (try_improve(mid, cand_map, cand_obj)) {
                        res.solution = cand_map;
                        best_obj = cand_obj;
                        hi = best_obj - 1;
                    } else {
                        lo = mid + 1;
                    }
                }
            } else {
                int64_t lo = best_obj + 1;
                int64_t hi = best_obj + 4096;
                for (int step = 0; step < 14 && lo <= hi; ++step) {
                    int64_t mid = lo + (hi - lo + 1) / 2;
                    if (try_improve(mid, cand_map, cand_obj)) {
                        res.solution = cand_map;
                        best_obj = cand_obj;
                        lo = best_obj + 1;
                    } else {
                        hi = mid - 1;
                    }
                }
            }
            res.objective_value = best_obj;
        }
    }

    std::vector<std::string> cert_keys;
    std::ostringstream sol_stream;
    for (const auto& [k, _] : res.solution) cert_keys.push_back(k);
    std::sort(cert_keys.begin(), cert_keys.end());
    for (const auto& k : cert_keys) {
        sol_stream << k << "=" << res.solution.at(k) << ";";
    }
    res.certificate_hash = compute_certificate_hash(
        "BROLM_PROVER_V1", sol_stream.str(), cert_keys, obj_expr);
    return res;
}

}  // namespace brolm
