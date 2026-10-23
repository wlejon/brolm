#include "brolm/verified_validator.h"

#include <algorithm>
#include <bit>
#include <cctype>
#include <cstdlib>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace brolm {

using brass::verify::BVContext;
using brass::verify::BVNodeId;
using brass::verify::BVOp;
using brass::verify::sign_extend_bits;

namespace {

std::string trim_ws(std::string_view sv) {
    size_t b = 0, e = sv.size();
    while (b < e && std::isspace(static_cast<unsigned char>(sv[b]))) ++b;
    while (e > b && std::isspace(static_cast<unsigned char>(sv[e - 1]))) --e;
    return std::string(sv.substr(b, e - b));
}

// Split an expression on top-level `&&` (outside parentheses/braces/brackets).
std::vector<std::string> split_top_level_and(std::string_view expr) {
    std::vector<std::string> parts;
    int depth = 0;
    size_t start = 0;
    for (size_t i = 0; i < expr.size(); ++i) {
        char c = expr[i];
        if (c == '(' || c == '[' || c == '{') ++depth;
        else if (c == ')' || c == ']' || c == '}') --depth;
        else if (depth == 0 && c == '&' && i + 1 < expr.size() && expr[i + 1] == '&') {
            std::string sub = trim_ws(expr.substr(start, i - start));
            if (!sub.empty()) parts.push_back(std::move(sub));
            ++i;
            start = i + 1;
        }
    }
    std::string tail = trim_ws(expr.substr(start));
    if (!tail.empty()) parts.push_back(std::move(tail));
    return parts;
}

enum class TokKind {
    Eof, Ident, IntLit,
    Plus, Minus, Star, Slash, Percent,
    Amp, Pipe, Caret, Tilde, Bang,
    AmpAmp, PipePipe,
    Shl, Shr, LShr,
    Eq, Ne, Lt, Le, Gt, Ge,
    LParen, RParen, Question, Colon, Comma, Dot
};

struct Token {
    TokKind kind = TokKind::Eof;
    std::string text;
    int64_t int_val = 0;
};

class Lexer {
public:
    explicit Lexer(std::string_view src) : src_(src) {}

    std::vector<Token> tokenize(std::string* err) {
        std::vector<Token> out;
        while (pos_ < src_.size()) {
            char c = src_[pos_];
            if (std::isspace(static_cast<unsigned char>(c))) { ++pos_; continue; }
            if (std::isalpha(static_cast<unsigned char>(c)) || c == '_' || c == '$') {
                size_t start = pos_++;
                while (pos_ < src_.size() &&
                       (std::isalnum(static_cast<unsigned char>(src_[pos_])) ||
                        src_[pos_] == '_' || src_[pos_] == '$')) {
                    ++pos_;
                }
                std::string id(src_.substr(start, pos_ - start));
                if (id == "true") out.push_back({TokKind::IntLit, id, 1});
                else if (id == "false") out.push_back({TokKind::IntLit, id, 0});
                else out.push_back({TokKind::Ident, std::move(id), 0});
                continue;
            }
            if (std::isdigit(static_cast<unsigned char>(c))) {
                size_t start = pos_;
                int base = 10;
                if (c == '0' && pos_ + 1 < src_.size()) {
                    char n = src_[pos_ + 1];
                    if (n == 'x' || n == 'X') { base = 16; pos_ += 2; }
                    else if (n == 'b' || n == 'B') { base = 2; pos_ += 2; }
                }
                size_t digits_start = pos_;
                while (pos_ < src_.size()) {
                    char d = src_[pos_];
                    bool ok = (base == 16) ? std::isxdigit(static_cast<unsigned char>(d)) != 0
                            : (base == 2)  ? (d == '0' || d == '1')
                                           : std::isdigit(static_cast<unsigned char>(d)) != 0;
                    if (!ok) break;
                    ++pos_;
                }
                if (pos_ < src_.size() && src_[pos_] == 'n') ++pos_;
                std::string digits(src_.substr(digits_start, pos_ - digits_start));
                if (digits.empty()) {
                    if (err) *err = "Invalid integer literal";
                    return {};
                }
                uint64_t uval = std::strtoull(digits.c_str(), nullptr, base);
                out.push_back({TokKind::IntLit, std::string(src_.substr(start, pos_ - start)),
                               static_cast<int64_t>(uval)});
                continue;
            }
            auto match = [&](std::string_view s) {
                if (src_.substr(pos_, s.size()) == s) { pos_ += s.size(); return true; }
                return false;
            };
            if (match("===")) { out.push_back({TokKind::Eq, "===", 0}); continue; }
            if (match("!==")) { out.push_back({TokKind::Ne, "!==", 0}); continue; }
            if (match(">>>")) { out.push_back({TokKind::LShr, ">>>", 0}); continue; }
            if (match("=="))  { out.push_back({TokKind::Eq, "==", 0}); continue; }
            if (match("!="))  { out.push_back({TokKind::Ne, "!=", 0}); continue; }
            if (match("<="))  { out.push_back({TokKind::Le, "<=", 0}); continue; }
            if (match(">="))  { out.push_back({TokKind::Ge, ">=", 0}); continue; }
            if (match("<<"))  { out.push_back({TokKind::Shl, "<<", 0}); continue; }
            if (match(">>"))  { out.push_back({TokKind::Shr, ">>", 0}); continue; }
            if (match("&&"))  { out.push_back({TokKind::AmpAmp, "&&", 0}); continue; }
            if (match("||"))  { out.push_back({TokKind::PipePipe, "||", 0}); continue; }

            ++pos_;
            switch (c) {
                case '+': out.push_back({TokKind::Plus, "+", 0}); break;
                case '-': out.push_back({TokKind::Minus, "-", 0}); break;
                case '*': out.push_back({TokKind::Star, "*", 0}); break;
                case '/': out.push_back({TokKind::Slash, "/", 0}); break;
                case '%': out.push_back({TokKind::Percent, "%", 0}); break;
                case '&': out.push_back({TokKind::Amp, "&", 0}); break;
                case '|': out.push_back({TokKind::Pipe, "|", 0}); break;
                case '^': out.push_back({TokKind::Caret, "^", 0}); break;
                case '~': out.push_back({TokKind::Tilde, "~", 0}); break;
                case '!': out.push_back({TokKind::Bang, "!", 0}); break;
                case '<': out.push_back({TokKind::Lt, "<", 0}); break;
                case '>': out.push_back({TokKind::Gt, ">", 0}); break;
                case '(': out.push_back({TokKind::LParen, "(", 0}); break;
                case ')': out.push_back({TokKind::RParen, ")", 0}); break;
                case '?': out.push_back({TokKind::Question, "?", 0}); break;
                case ':': out.push_back({TokKind::Colon, ":", 0}); break;
                case ',': out.push_back({TokKind::Comma, ",", 0}); break;
                case '.': out.push_back({TokKind::Dot, ".", 0}); break;
                default:
                    if (err) *err = std::string("Unexpected character '") + c + "'";
                    return {};
            }
        }
        out.push_back({TokKind::Eof, "", 0});
        return out;
    }

private:
    std::string_view src_;
    size_t pos_ = 0;
};

class ExprParser {
public:
    ExprParser(BVContext& ctx,
               const std::vector<Token>& toks,
               const std::unordered_map<std::string, BVNodeId>& vars,
               uint32_t default_width,
               BVNodeId* div_guards,
               std::string* err)
        : ctx_(ctx), toks_(toks), vars_(vars), w_(default_width),
          div_guards_(div_guards), err_(err) {}

    BVNodeId parse_full() {
        BVNodeId res = parse_ternary();
        if (res == UINT32_MAX) return UINT32_MAX;
        if (peek().kind != TokKind::Eof) {
            fail("Unexpected trailing token '" + peek().text + "'");
            return UINT32_MAX;
        }
        return res;
    }

    BVNodeId to_bool(BVNodeId id) {
        if (id == UINT32_MAX) return UINT32_MAX;
        uint32_t nw = ctx_.width(id);
        if (nw == 1) return id;
        return ctx_.make_binary(BVOp::Ne, id, ctx_.make_const(0, nw), 1);
    }

    BVNodeId to_word(BVNodeId id, uint32_t target_w) {
        if (id == UINT32_MAX) return UINT32_MAX;
        uint32_t nw = ctx_.width(id);
        if (nw == target_w) return id;
        if (nw == 1) return ctx_.make_zext(id, target_w - 1);
        if (nw < target_w) return ctx_.make_sext(id, target_w - nw);
        return ctx_.make_extract(id, target_w - 1, 0);
    }

private:
    const Token& peek() const { return toks_[idx_]; }
    const Token& get() { return toks_[idx_++]; }

    void fail(std::string msg) {
        if (err_ && err_->empty()) *err_ = std::move(msg);
    }

    void match_widths(BVNodeId& a, BVNodeId& b) {
        uint32_t wa = ctx_.width(a);
        uint32_t wb = ctx_.width(b);
        if (wa == wb) {
            if (wa == 1) {
                a = to_word(a, w_);
                b = to_word(b, w_);
            }
            return;
        }
        uint32_t target = std::max({wa, wb, w_});
        a = to_word(a, target);
        b = to_word(b, target);
    }

    void add_div_guard(BVNodeId divisor) {
        if (!div_guards_) return;
        uint32_t dw = ctx_.width(divisor);
        BVNodeId nz = ctx_.make_binary(BVOp::Ne, divisor, ctx_.make_const(0, dw), 1);
        if (path_cond_ != 0 && path_cond_ != UINT32_MAX) {
            nz = ctx_.make_binary(BVOp::Or, ctx_.make_unary(BVOp::Not, path_cond_, 1), nz, 1);
        }
        if (*div_guards_ == 0 || *div_guards_ == UINT32_MAX) {
            *div_guards_ = nz;
        } else {
            *div_guards_ = ctx_.make_binary(BVOp::And, *div_guards_, nz, 1);
        }
    }

    BVNodeId push_path_cond(BVNodeId extra_cond) {
        BVNodeId saved = path_cond_;
        if (extra_cond != 0 && extra_cond != UINT32_MAX) {
            BVNodeId b = to_bool(extra_cond);
            if (path_cond_ == 0 || path_cond_ == UINT32_MAX) path_cond_ = b;
            else path_cond_ = ctx_.make_binary(BVOp::And, path_cond_, b, 1);
        }
        return saved;
    }

    BVNodeId parse_ternary() {
        BVNodeId c = parse_logical_or();
        if (c == UINT32_MAX) return UINT32_MAX;
        if (peek().kind == TokKind::Question) {
            get();
            BVNodeId cb = to_bool(c);
            BVNodeId saved_t = push_path_cond(cb);
            BVNodeId t = parse_ternary();
            path_cond_ = saved_t;
            if (t == UINT32_MAX) return UINT32_MAX;
            if (peek().kind != TokKind::Colon) {
                fail("Expected ':' in ternary expression");
                return UINT32_MAX;
            }
            get();
            BVNodeId saved_f = push_path_cond(ctx_.make_unary(BVOp::Not, cb, 1));
            BVNodeId f = parse_ternary();
            path_cond_ = saved_f;
            if (f == UINT32_MAX) return UINT32_MAX;
            uint32_t wt = ctx_.width(t), wf = ctx_.width(f);
            if (wt != wf) {
                uint32_t tw = std::max(wt, wf);
                t = to_word(t, tw);
                f = to_word(f, tw);
            }
            return ctx_.make_ite(cb, t, f);
        }
        return c;
    }

    BVNodeId parse_logical_or() {
        BVNodeId lhs = parse_logical_and();
        while (lhs != UINT32_MAX && peek().kind == TokKind::PipePipe) {
            get();
            BVNodeId lb = to_bool(lhs);
            BVNodeId saved = push_path_cond(ctx_.make_unary(BVOp::Not, lb, 1));
            BVNodeId rhs = parse_logical_and();
            path_cond_ = saved;
            if (rhs == UINT32_MAX) return UINT32_MAX;
            lhs = ctx_.make_binary(BVOp::Or, lb, to_bool(rhs), 1);
        }
        return lhs;
    }

    BVNodeId parse_logical_and() {
        BVNodeId lhs = parse_bit_or();
        while (lhs != UINT32_MAX && peek().kind == TokKind::AmpAmp) {
            get();
            BVNodeId lb = to_bool(lhs);
            BVNodeId saved = push_path_cond(lb);
            BVNodeId rhs = parse_bit_or();
            path_cond_ = saved;
            if (rhs == UINT32_MAX) return UINT32_MAX;
            lhs = ctx_.make_binary(BVOp::And, lb, to_bool(rhs), 1);
        }
        return lhs;
    }

    BVNodeId parse_bit_or() {
        BVNodeId lhs = parse_bit_xor();
        while (lhs != UINT32_MAX && peek().kind == TokKind::Pipe) {
            get();
            BVNodeId rhs = parse_bit_xor();
            if (rhs == UINT32_MAX) return UINT32_MAX;
            match_widths(lhs, rhs);
            lhs = ctx_.make_binary(BVOp::Or, lhs, rhs, ctx_.width(lhs));
        }
        return lhs;
    }

    BVNodeId parse_bit_xor() {
        BVNodeId lhs = parse_bit_and();
        while (lhs != UINT32_MAX && peek().kind == TokKind::Caret) {
            get();
            BVNodeId rhs = parse_bit_and();
            if (rhs == UINT32_MAX) return UINT32_MAX;
            match_widths(lhs, rhs);
            lhs = ctx_.make_binary(BVOp::Xor, lhs, rhs, ctx_.width(lhs));
        }
        return lhs;
    }

    BVNodeId parse_bit_and() {
        BVNodeId lhs = parse_equality();
        while (lhs != UINT32_MAX && peek().kind == TokKind::Amp) {
            get();
            BVNodeId rhs = parse_equality();
            if (rhs == UINT32_MAX) return UINT32_MAX;
            match_widths(lhs, rhs);
            lhs = ctx_.make_binary(BVOp::And, lhs, rhs, ctx_.width(lhs));
        }
        return lhs;
    }

    BVNodeId parse_equality() {
        BVNodeId lhs = parse_relational();
        while (lhs != UINT32_MAX && (peek().kind == TokKind::Eq || peek().kind == TokKind::Ne)) {
            TokKind k = get().kind;
            BVNodeId rhs = parse_relational();
            if (rhs == UINT32_MAX) return UINT32_MAX;
            if (ctx_.width(lhs) != ctx_.width(rhs)) match_widths(lhs, rhs);
            lhs = ctx_.make_binary(k == TokKind::Eq ? BVOp::Eq : BVOp::Ne, lhs, rhs, 1);
        }
        return lhs;
    }

    BVNodeId parse_relational() {
        BVNodeId lhs = parse_shift();
        while (lhs != UINT32_MAX &&
               (peek().kind == TokKind::Lt || peek().kind == TokKind::Le ||
                peek().kind == TokKind::Gt || peek().kind == TokKind::Ge)) {
            TokKind k = get().kind;
            BVNodeId rhs = parse_shift();
            if (rhs == UINT32_MAX) return UINT32_MAX;
            match_widths(lhs, rhs);
            BVOp op = (k == TokKind::Lt) ? BVOp::Slt
                    : (k == TokKind::Le) ? BVOp::Sle
                    : (k == TokKind::Gt) ? BVOp::Sgt : BVOp::Sge;
            lhs = ctx_.make_binary(op, lhs, rhs, 1);
        }
        return lhs;
    }

    BVNodeId parse_shift() {
        BVNodeId lhs = parse_additive();
        while (lhs != UINT32_MAX &&
               (peek().kind == TokKind::Shl || peek().kind == TokKind::Shr ||
                peek().kind == TokKind::LShr)) {
            TokKind k = get().kind;
            BVNodeId rhs = parse_additive();
            if (rhs == UINT32_MAX) return UINT32_MAX;
            match_widths(lhs, rhs);
            BVOp op = (k == TokKind::Shl) ? BVOp::Shl
                    : (k == TokKind::Shr) ? BVOp::AShr : BVOp::LShr;
            lhs = ctx_.make_binary(op, lhs, rhs, ctx_.width(lhs));
        }
        return lhs;
    }

    BVNodeId parse_additive() {
        BVNodeId lhs = parse_multiplicative();
        while (lhs != UINT32_MAX && (peek().kind == TokKind::Plus || peek().kind == TokKind::Minus)) {
            TokKind k = get().kind;
            BVNodeId rhs = parse_multiplicative();
            if (rhs == UINT32_MAX) return UINT32_MAX;
            match_widths(lhs, rhs);
            lhs = ctx_.make_binary(k == TokKind::Plus ? BVOp::Add : BVOp::Sub,
                                   lhs, rhs, ctx_.width(lhs));
        }
        return lhs;
    }

    BVNodeId make_signed_rem_or_div(BVOp op, BVNodeId lhs, BVNodeId rhs) {
        add_div_guard(rhs);
        uint32_t w = ctx_.width(lhs);
        const auto& rnode = ctx_.nodes()[rhs];
        if (rnode.op == BVOp::Const) {
            int64_t cval = sign_extend_bits(rnode.imm, w);
            if (cval > 0 && std::has_single_bit(static_cast<uint64_t>(cval))) {
                uint64_t mask = static_cast<uint64_t>(cval) - 1ULL;
                if (op == BVOp::SRem) {
                    BVNodeId zero = ctx_.make_const(0, w);
                    BVNodeId non_neg = ctx_.make_binary(BVOp::Sge, lhs, zero, 1);
                    BVNodeId pos_rem = ctx_.make_binary(BVOp::And, lhs, ctx_.make_const(mask, w), w);
                    BVNodeId neg_lhs = ctx_.make_unary(BVOp::Neg, lhs, w);
                    BVNodeId neg_rem = ctx_.make_unary(
                        BVOp::Neg, ctx_.make_binary(BVOp::And, neg_lhs, ctx_.make_const(mask, w), w), w);
                    return ctx_.make_ite(non_neg, pos_rem, neg_rem);
                }
            }
            if (w == 64 && cval > 0 && cval <= 0x7FFFFFFFLL) {
                BVNodeId a32 = ctx_.make_extract(lhs, 31, 0);
                BVNodeId b32 = ctx_.make_const(static_cast<uint64_t>(cval), 32);
                BVNodeId r32 = ctx_.make_binary(op, a32, b32, 32);
                return ctx_.make_sext(r32, 32);
            }
        }
        return ctx_.make_binary(op, lhs, rhs, w);
    }

    BVNodeId parse_multiplicative() {
        BVNodeId lhs = parse_unary();
        while (lhs != UINT32_MAX &&
               (peek().kind == TokKind::Star || peek().kind == TokKind::Slash ||
                peek().kind == TokKind::Percent)) {
            TokKind k = get().kind;
            BVNodeId rhs = parse_unary();
            if (rhs == UINT32_MAX) return UINT32_MAX;
            match_widths(lhs, rhs);
            if (k == TokKind::Star) {
                lhs = ctx_.make_binary(BVOp::Mul, lhs, rhs, ctx_.width(lhs));
            } else {
                lhs = make_signed_rem_or_div(k == TokKind::Slash ? BVOp::SDiv : BVOp::SRem, lhs, rhs);
            }
        }
        return lhs;
    }

    BVNodeId parse_unary() {
        if (peek().kind == TokKind::Plus) {
            get();
            return parse_unary();
        }
        if (peek().kind == TokKind::Minus) {
            get();
            BVNodeId v = parse_unary();
            if (v == UINT32_MAX) return UINT32_MAX;
            v = to_word(v, std::max(ctx_.width(v), w_));
            return ctx_.make_unary(BVOp::Neg, v, ctx_.width(v));
        }
        if (peek().kind == TokKind::Tilde) {
            get();
            BVNodeId v = parse_unary();
            if (v == UINT32_MAX) return UINT32_MAX;
            v = to_word(v, std::max(ctx_.width(v), w_));
            return ctx_.make_unary(BVOp::Not, v, ctx_.width(v));
        }
        if (peek().kind == TokKind::Bang) {
            get();
            BVNodeId v = parse_unary();
            if (v == UINT32_MAX) return UINT32_MAX;
            return ctx_.make_unary(BVOp::Not, to_bool(v), 1);
        }
        return parse_primary();
    }

    BVNodeId parse_math_call(const std::string& fn) {
        if (peek().kind != TokKind::LParen) {
            fail("Expected '(' after Math." + fn);
            return UINT32_MAX;
        }
        get();
        std::vector<BVNodeId> args;
        if (peek().kind != TokKind::RParen) {
            while (true) {
                BVNodeId a = parse_ternary();
                if (a == UINT32_MAX) return UINT32_MAX;
                args.push_back(to_word(a, std::max(ctx_.width(a), w_)));
                if (peek().kind == TokKind::Comma) { get(); continue; }
                break;
            }
        }
        if (peek().kind != TokKind::RParen) {
            fail("Expected ')' after Math." + fn + " arguments");
            return UINT32_MAX;
        }
        get();
        if (fn == "abs" && args.size() == 1) {
            uint32_t aw = ctx_.width(args[0]);
            BVNodeId neg = ctx_.make_binary(BVOp::Slt, args[0], ctx_.make_const(0, aw), 1);
            return ctx_.make_ite(neg, ctx_.make_unary(BVOp::Neg, args[0], aw), args[0]);
        }
        if ((fn == "min" || fn == "max") && args.size() >= 2) {
            BVNodeId acc = args[0];
            for (size_t i = 1; i < args.size(); ++i) {
                BVNodeId b = args[i];
                match_widths(acc, b);
                BVNodeId cmp = ctx_.make_binary(fn == "min" ? BVOp::Slt : BVOp::Sgt, acc, b, 1);
                acc = ctx_.make_ite(cmp, acc, b);
            }
            return acc;
        }
        if (fn == "imul" && args.size() == 2) {
            BVNodeId a = args[0], b = args[1];
            match_widths(a, b);
            return ctx_.make_binary(BVOp::Mul, a, b, ctx_.width(a));
        }
        if (fn == "clz32" && args.size() == 1) {
            return ctx_.make_unary(BVOp::Clz, args[0], ctx_.width(args[0]));
        }
        fail("Unsupported Math function: Math." + fn);
        return UINT32_MAX;
    }

    BVNodeId parse_primary() {
        if (peek().kind == TokKind::LParen) {
            get();
            BVNodeId inner = parse_ternary();
            if (inner == UINT32_MAX) return UINT32_MAX;
            if (peek().kind != TokKind::RParen) {
                fail("Expected ')'");
                return UINT32_MAX;
            }
            get();
            return inner;
        }
        if (peek().kind == TokKind::IntLit) {
            int64_t val = get().int_val;
            return ctx_.make_const(static_cast<uint64_t>(val), w_);
        }
        if (peek().kind == TokKind::Ident) {
            std::string name = get().text;
            if (name == "Math" && peek().kind == TokKind::Dot) {
                get();
                if (peek().kind != TokKind::Ident) {
                    fail("Expected method name after 'Math.'");
                    return UINT32_MAX;
                }
                return parse_math_call(get().text);
            }
            while (peek().kind == TokKind::Dot) {
                get();
                if (peek().kind != TokKind::Ident) {
                    fail("Expected property name after '.'");
                    return UINT32_MAX;
                }
                std::string prop = get().text;
                std::string combined = name + "." + prop;
                if (vars_.find(combined) != vars_.end()) name = std::move(combined);
                else name = std::move(prop);
            }
            if (peek().kind == TokKind::LParen) {
                fail("Unsupported function call in symbolic expression: " + name);
                return UINT32_MAX;
            }
            auto it = vars_.find(name);
            if (it == vars_.end()) {
                fail("Unknown field variable '" + name + "'");
                return UINT32_MAX;
            }
            return it->second;
        }
        fail("Unexpected token '" + peek().text + "'");
        return UINT32_MAX;
    }

    BVContext& ctx_;
    const std::vector<Token>& toks_;
    const std::unordered_map<std::string, BVNodeId>& vars_;
    uint32_t w_ = 32;
    BVNodeId* div_guards_ = nullptr;
    BVNodeId path_cond_ = 0;
    std::string* err_ = nullptr;
    size_t idx_ = 0;
};

}  // namespace

// ─── Public Static Expression Parser ──────────────────────────────────────────

BVNodeId VerifiedValidator::parse_expression_to_bv(
    BVContext& ctx,
    std::string_view expr,
    const std::unordered_map<std::string, BVNodeId>& vars,
    uint32_t default_width,
    BVNodeId* out_div_guards,
    std::string* out_error) {
    std::string lex_err;
    Lexer lex(expr);
    std::vector<Token> toks = lex.tokenize(&lex_err);
    if (!lex_err.empty()) {
        if (out_error) *out_error = lex_err;
        return UINT32_MAX;
    }
    ExprParser parser(ctx, toks, vars, default_width, out_div_guards, out_error);
    return parser.parse_full();
}

BVNodeId VerifiedValidator::parse_constraint_to_bv(
    BVContext& ctx,
    std::string_view expr,
    const std::unordered_map<std::string, BVNodeId>& vars,
    uint32_t default_width,
    BVNodeId* out_div_guards,
    std::string* out_error) {
    BVNodeId node = parse_expression_to_bv(ctx, expr, vars, default_width, out_div_guards, out_error);
    if (node == UINT32_MAX) return UINT32_MAX;
    uint32_t nw = ctx.width(node);
    if (nw == 1) return node;
    return ctx.make_binary(BVOp::Ne, node, ctx.make_const(0, nw), 1);
}

// ─── Specification Builders ───────────────────────────────────────────────────

void VerifiedValidator::add_field(const FieldConstraint& field) {
    for (auto& existing : fields_) {
        if (existing.name == field.name) {
            existing = field;
            return;
        }
    }
    fields_.push_back(field);
}

bool VerifiedValidator::has_field(std::string_view name) const noexcept {
    return find_field(name) != nullptr;
}

const FieldConstraint* VerifiedValidator::find_field(std::string_view name) const noexcept {
    for (const auto& f : fields_) {
        if (f.name == name) return &f;
    }
    return nullptr;
}

void VerifiedValidator::add_constraint(std::string expression, std::string label) {
    std::vector<std::string> conjuncts = split_top_level_and(expression);
    if (conjuncts.size() <= 1) {
        std::string trimmed = trim_ws(expression);
        if (!trimmed.empty()) {
            constraints_.push_back({trimmed, label.empty() ? trimmed : std::move(label)});
        }
        return;
    }
    for (auto& c : conjuncts) {
        constraints_.push_back({c, c});
    }
}

void VerifiedValidator::add_bv_constraint(BVBuilderCallback builder, std::string label) {
    if (builder) bv_builders_.emplace_back(std::move(builder), std::move(label));
}

void VerifiedValidator::set_predicate(PredicateCallback pred, std::string label) {
    predicate_ = std::move(pred);
    predicate_label_ = std::move(label);
}

bool VerifiedValidator::add_constraints_from_js_source(std::string_view js_source,
                                                       bool* out_positional_args) {
    std::string s = trim_ws(js_source);
    if (s.empty() || s.find("[native code]") != std::string::npos) return false;

    std::vector<std::string> params;
    std::string body;

    size_t arrow = s.find("=>");
    if (arrow != std::string::npos) {
        std::string head = trim_ws(s.substr(0, arrow));
        body = trim_ws(s.substr(arrow + 2));
        if (!head.empty() && head.front() == '(' && head.back() == ')') {
            head = trim_ws(head.substr(1, head.size() - 2));
        }
        if (!head.empty() && head.front() == '{' && head.back() == '}') {
            head = trim_ws(head.substr(1, head.size() - 2));
        }
        std::stringstream ss(head);
        std::string item;
        while (std::getline(ss, item, ',')) {
            std::string p = trim_ws(item);
            if (!p.empty()) params.push_back(p);
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

    bool positional = false;
    std::string obj_param;
    if (params.size() > 1) {
        positional = true;
    } else if (params.size() == 1) {
        if (body.find(params[0] + ".") != std::string::npos) {
            obj_param = params[0];
            positional = false;
        } else if (has_field(params[0]) && fields_.size() == 1) {
            positional = true;
        } else {
            obj_param = params[0];
        }
    }
    if (out_positional_args) *out_positional_args = positional;

    if (fields_.empty()) {
        if (positional && !params.empty()) {
            for (const auto& p : params) add_field(FieldConstraint::unbounded_int(p));
        } else if (!obj_param.empty()) {
            std::string prefix = obj_param + ".";
            size_t pos = 0;
            while ((pos = body.find(prefix, pos)) != std::string::npos) {
                pos += prefix.size();
                size_t end = pos;
                while (end < body.size() &&
                       (std::isalnum(static_cast<unsigned char>(body[end])) || body[end] == '_')) {
                    ++end;
                }
                if (end > pos) {
                    std::string fname = body.substr(pos, end - pos);
                    if (!has_field(fname)) add_field(FieldConstraint::unbounded_int(fname));
                }
                pos = end;
            }
        }
    }

    auto strip_obj_prefix = [&](std::string expr) {
        if (obj_param.empty()) return expr;
        std::string prefix = obj_param + ".";
        size_t pos = 0;
        while ((pos = expr.find(prefix, pos)) != std::string::npos) {
            if (pos > 0 && (std::isalnum(static_cast<unsigned char>(expr[pos - 1])) ||
                            expr[pos - 1] == '_')) {
                pos += prefix.size();
                continue;
            }
            expr.erase(pos, prefix.size());
        }
        return expr;
    };

    std::vector<std::string> candidate_exprs;
    if (!body.empty() && body.front() == '{' && body.back() == '}') {
        std::string inner = trim_ws(body.substr(1, body.size() - 2));
        std::unordered_map<std::string, std::string> locals;
        auto subst_locals = [&](std::string e) {
            for (const auto& [k, v] : locals) {
                size_t p = 0;
                while ((p = e.find(k, p)) != std::string::npos) {
                    bool left_ok = (p == 0) || (!std::isalnum(static_cast<unsigned char>(e[p - 1])) && e[p - 1] != '_');
                    bool right_ok = (p + k.size() >= e.size()) ||
                                    (!std::isalnum(static_cast<unsigned char>(e[p + k.size()])) && e[p + k.size()] != '_');
                    if (left_ok && right_ok) {
                        std::string repl = "(" + v + ")";
                        e.replace(p, k.size(), repl);
                        p += repl.size();
                    } else {
                        p += k.size();
                    }
                }
            }
            return e;
        };

        size_t cur = 0;
        while (cur < inner.size()) {
            while (cur < inner.size() && (std::isspace(static_cast<unsigned char>(inner[cur])) || inner[cur] == ';')) ++cur;
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
                std::string cond = subst_locals(strip_obj_prefix(trim_ws(inner.substr(lp + 1, rp - lp - 2))));
                size_t semi = inner.find(';', rp);
                if (semi == std::string::npos) break;
                std::string then_stmt = trim_ws(inner.substr(rp, semi - rp));
                if (then_stmt.rfind("return", 0) == 0) {
                    std::string ret_val = trim_ws(then_stmt.substr(6));
                    if (ret_val == "false" || ret_val == "0") {
                        candidate_exprs.push_back("!(" + cond + ")");
                    }
                }
                cur = semi + 1;
                continue;
            }
            size_t semi = inner.find(';', cur);
            std::string stmt = trim_ws(inner.substr(cur, (semi == std::string::npos) ? std::string::npos : (semi - cur)));
            cur = (semi == std::string::npos) ? inner.size() : (semi + 1);
            for (std::string_view kw : {"const ", "let ", "var "}) {
                if (stmt.rfind(kw, 0) == 0) {
                    std::string decl = trim_ws(stmt.substr(kw.size()));
                    size_t eq = decl.find('=');
                    if (eq != std::string::npos) {
                        std::string vname = trim_ws(decl.substr(0, eq));
                        std::string rhs = subst_locals(strip_obj_prefix(trim_ws(decl.substr(eq + 1))));
                        locals[vname] = rhs;
                    }
                    stmt.clear();
                    break;
                }
            }
            if (stmt.rfind("return", 0) == 0) {
                std::string ret_expr = subst_locals(strip_obj_prefix(trim_ws(stmt.substr(6))));
                for (auto& c : split_top_level_and(ret_expr)) candidate_exprs.push_back(std::move(c));
            }
        }
    } else {
        std::string norm = strip_obj_prefix(body);
        if (!norm.empty() && norm.back() == ';') norm.pop_back();
        for (auto& c : split_top_level_and(norm)) candidate_exprs.push_back(std::move(c));
    }

    if (candidate_exprs.empty()) return false;

    BVContext test_ctx;
    std::unordered_map<std::string, BVNodeId> test_vars;
    for (const auto& f : fields_) {
        test_vars[f.name] = test_ctx.make_var(f.name, 32);
    }
    bool all_ok = true;
    size_t added = 0;
    for (const auto& c : candidate_exprs) {
        std::string err;
        BVNodeId id = parse_constraint_to_bv(test_ctx, c, test_vars, 32, nullptr, &err);
        if (id != UINT32_MAX && err.empty()) {
            constraints_.push_back({c, c});
            ++added;
        } else {
            all_ok = false;
        }
    }
    return added > 0 && all_ok;
}

}  // namespace brolm
