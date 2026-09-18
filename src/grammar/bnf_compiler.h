#pragma once

#include "grammar/grammar_types.h"

#include <cctype>
#include <cstddef>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace brolm {

class BNFCompiler {
public:
    explicit BNFCompiler(std::string_view bnf_text, GrammarGraph& graph)
        : text_(bnf_text), p_(0), graph_(graph) {}

    void compile() {
        // Collect all rules: rule_name ::= expression
        while (p_ < text_.size()) {
            skip_ws_and_comments();
            if (p_ >= text_.size()) break;

            std::string rule_name = parse_identifier();
            if (rule_name.empty()) {
                ++p_;
                continue;
            }

            skip_ws_and_comments();
            if (peek() == ':' && p_ + 2 < text_.size() && text_[p_ + 1] == ':' && text_[p_ + 2] == '=') {
                p_ += 3;
            } else if (peek() == ':' && p_ + 1 < text_.size() && text_[p_ + 1] == '=') {
                p_ += 2;
            } else if (peek() == '=') {
                p_ += 1;
            }

            int rule_id = graph_.add_rule(rule_name);
            rule_texts_[rule_name] = {rule_id, parse_rule_body()};
        }

        if (rule_texts_.empty()) {
            int root_id = graph_.add_rule("root");
            graph_.root_rule_id = root_id;
            int s = graph_.add_state(root_id);
            graph_.rules[root_id].entry_state = s;
            graph_.rules[root_id].exit_state  = s;
            return;
        }

        // Set root rule
        auto it = rule_texts_.find("root");
        if (it != rule_texts_.end()) {
            graph_.root_rule_id = it->second.first;
        } else {
            graph_.root_rule_id = rule_texts_.begin()->second.first;
        }

        // Compile each rule's body into states
        for (const auto& [name, entry] : rule_texts_) {
            int rule_id = entry.first;
            std::string body = entry.second;
            auto pair = compile_expression(body, rule_id);
            graph_.rules[rule_id].entry_state = pair.first;
            graph_.rules[rule_id].exit_state  = pair.second;
        }
    }

private:
    std::string_view text_;
    std::size_t      p_ = 0;
    GrammarGraph&    graph_;
    std::unordered_map<std::string, std::pair<int, std::string>> rule_texts_;

    char peek() const { return p_ < text_.size() ? text_[p_] : '\0'; }
    char get() { return p_ < text_.size() ? text_[p_++] : '\0'; }

    void skip_ws_and_comments() {
        while (p_ < text_.size()) {
            char c = text_[p_];
            if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
                ++p_;
            } else if (c == '#' || (c == '/' && p_ + 1 < text_.size() && text_[p_ + 1] == '/')) {
                while (p_ < text_.size() && text_[p_] != '\n') ++p_;
            } else {
                break;
            }
        }
    }

    std::string parse_identifier() {
        if (peek() == '<') {
            get();
            std::size_t start = p_;
            while (p_ < text_.size() && text_[p_] != '>') ++p_;
            std::string id(text_.substr(start, p_ - start));
            if (peek() == '>') get();
            return id;
        }
        std::size_t start = p_;
        while (p_ < text_.size() && (std::isalnum(static_cast<unsigned char>(text_[p_])) || text_[p_] == '_' || text_[p_] == '-')) {
            ++p_;
        }
        return std::string(text_.substr(start, p_ - start));
    }

    std::string parse_rule_body() {
        std::size_t start = p_;
        while (p_ < text_.size()) {
            if (text_[p_] == '\n') {
                // Check if next non-empty line starts a new rule
                std::size_t saved = p_;
                ++p_;
                skip_ws_and_comments();
                if (p_ < text_.size()) {
                    std::string id = parse_identifier();
                    skip_ws_and_comments();
                    if (!id.empty() && (peek() == ':' || peek() == '=')) {
                        p_ = saved;
                        break;
                    }
                }
                p_ = saved + 1;
            } else {
                ++p_;
            }
        }
        return std::string(text_.substr(start, p_ - start));
    }

    std::pair<int, int> compile_expression(std::string_view expr, int rule_id);
};

class BNFExprParser {
public:
    BNFExprParser(std::string_view expr, int rule_id, GrammarGraph& graph)
        : expr_(expr), rule_id_(rule_id), graph_(graph) {}

    std::pair<int, int> parse() {
        return parse_alt();
    }

private:
    std::string_view expr_;
    std::size_t      idx_ = 0;
    int              rule_id_;
    GrammarGraph&    graph_;

    void skip_ws() {
        while (idx_ < expr_.size() && (expr_[idx_] == ' ' || expr_[idx_] == '\t' || expr_[idx_] == '\r' || expr_[idx_] == '\n')) {
            ++idx_;
        }
    }

    std::pair<int, int> parse_alt() {
        auto first = parse_seq();
        skip_ws();
        if (idx_ >= expr_.size() || expr_[idx_] != '|') return first;

        int entry = graph_.add_state(rule_id_);
        int exit  = graph_.add_state(rule_id_);

        graph_.add_epsilon(entry, first.first);
        graph_.add_epsilon(first.second, exit);

        while (idx_ < expr_.size() && expr_[idx_] == '|') {
            ++idx_;  // consume '|'
            auto next = parse_seq();
            graph_.add_epsilon(entry, next.first);
            graph_.add_epsilon(next.second, exit);
            skip_ws();
        }

        return {entry, exit};
    }

    std::pair<int, int> parse_seq() {
        int entry = -1;
        int last_exit = -1;

        while (true) {
            skip_ws();
            if (idx_ >= expr_.size() || expr_[idx_] == '|' || expr_[idx_] == ')') break;

            auto factor = parse_quantifier();
            if (entry == -1) {
                entry = factor.first;
                last_exit = factor.second;
            } else {
                graph_.add_epsilon(last_exit, factor.first);
                last_exit = factor.second;
            }
        }

        if (entry == -1) {
            int s = graph_.add_state(rule_id_);
            return {s, s};
        }
        return {entry, last_exit};
    }

    std::pair<int, int> parse_quantifier() {
        auto atom = parse_atom();
        skip_ws();
        if (idx_ < expr_.size()) {
            char c = expr_[idx_];
            if (c == '*' || c == '+' || c == '?') {
                ++idx_;
                int entry = graph_.add_state(rule_id_);
                int exit  = graph_.add_state(rule_id_);
                if (c == '*') {
                    graph_.add_epsilon(entry, atom.first);
                    graph_.add_epsilon(atom.second, atom.first);
                    graph_.add_epsilon(atom.second, exit);
                    graph_.add_epsilon(entry, exit);
                } else if (c == '+') {
                    graph_.add_epsilon(entry, atom.first);
                    graph_.add_epsilon(atom.second, atom.first);
                    graph_.add_epsilon(atom.second, exit);
                } else if (c == '?') {
                    graph_.add_epsilon(entry, atom.first);
                    graph_.add_epsilon(atom.second, exit);
                    graph_.add_epsilon(entry, exit);
                }
                return {entry, exit};
            }
        }
        return atom;
    }

    std::pair<int, int> parse_atom() {
        skip_ws();
        if (idx_ >= expr_.size()) {
            int s = graph_.add_state(rule_id_);
            return {s, s};
        }

        char c = expr_[idx_];

        if (c == '(') {
            ++idx_;
            auto sub = parse_alt();
            skip_ws();
            if (idx_ < expr_.size() && expr_[idx_] == ')') ++idx_;
            return sub;
        }

        if (c == '"' || c == '\'') {
            char quote = c;
            ++idx_;
            int entry = -1;
            int last_exit = -1;
            while (idx_ < expr_.size() && expr_[idx_] != quote) {
                char ch = expr_[idx_++];
                if (ch == '\\' && idx_ < expr_.size()) {
                    char esc = expr_[idx_++];
                    if (esc == 'n') ch = '\n';
                    else if (esc == 't') ch = '\t';
                    else if (esc == 'r') ch = '\r';
                    else if (esc == '0') ch = '\0';
                    else ch = esc;
                }
                CharSet cs;
                cs.set(static_cast<uint8_t>(ch));
                int e = graph_.add_state(rule_id_);
                int x = graph_.add_state(rule_id_);
                graph_.add_char_trans(e, x, cs);
                if (entry == -1) {
                    entry = e;
                    last_exit = x;
                } else {
                    graph_.add_epsilon(last_exit, e);
                    last_exit = x;
                }
            }
            if (idx_ < expr_.size() && expr_[idx_] == quote) ++idx_;
            if (entry == -1) {
                int s = graph_.add_state(rule_id_);
                return {s, s};
            }
            return {entry, last_exit};
        }

        if (c == '[') {
            ++idx_;
            CharSet cs;
            bool negate = false;
            if (idx_ < expr_.size() && expr_[idx_] == '^') {
                negate = true;
                ++idx_;
            }
            while (idx_ < expr_.size() && expr_[idx_] != ']') {
                char ch = expr_[idx_++];
                if (ch == '\\' && idx_ < expr_.size()) {
                    char esc = expr_[idx_++];
                    if (esc == 'n') ch = '\n';
                    else if (esc == 't') ch = '\t';
                    else if (esc == 'r') ch = '\r';
                    else ch = esc;
                }
                if (idx_ < expr_.size() && expr_[idx_] == '-' && idx_ + 1 < expr_.size() && expr_[idx_ + 1] != ']') {
                    ++idx_;  // consume '-'
                    char end_ch = expr_[idx_++];
                    if (end_ch == '\\' && idx_ < expr_.size()) end_ch = expr_[idx_++];
                    cs.set_range(static_cast<uint8_t>(ch), static_cast<uint8_t>(end_ch));
                } else {
                    cs.set(static_cast<uint8_t>(ch));
                }
            }
            if (idx_ < expr_.size() && expr_[idx_] == ']') ++idx_;
            if (negate) cs.invert();

            int entry = graph_.add_state(rule_id_);
            int exit  = graph_.add_state(rule_id_);
            graph_.add_char_trans(entry, exit, cs);
            return {entry, exit};
        }

        // Non-terminal identifier
        std::size_t start = idx_;
        while (idx_ < expr_.size() && (std::isalnum(static_cast<unsigned char>(expr_[idx_])) || expr_[idx_] == '_' || expr_[idx_] == '-')) {
            ++idx_;
        }
        std::string ref_name(expr_.substr(start, idx_ - start));
        if (!ref_name.empty()) {
            int target_rule = graph_.add_rule(ref_name);
            int entry = graph_.add_state(rule_id_);
            int exit  = graph_.add_state(rule_id_);
            graph_.add_rule_call(entry, exit, target_rule);
            return {entry, exit};
        }

        ++idx_;
        int s = graph_.add_state(rule_id_);
        return {s, s};
    }
};

inline std::pair<int, int> BNFCompiler::compile_expression(std::string_view expr, int rule_id) {
    BNFExprParser parser(expr, rule_id, graph_);
    return parser.parse();
}

}  // namespace brolm
