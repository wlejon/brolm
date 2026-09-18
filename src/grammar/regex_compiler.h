#pragma once

#include "grammar/grammar_types.h"

#include <cctype>
#include <cstddef>
#include <string_view>
#include <utility>

namespace brolm {

class RegexCompiler {
public:
    explicit RegexCompiler(std::string_view pattern, GrammarGraph& graph, int rule_id)
        : pattern_(pattern), p_(0), graph_(graph), rule_id_(rule_id) {}

    std::pair<int, int> compile() {
        if (pattern_.empty()) {
            int s = graph_.add_state(rule_id_);
            return {s, s};
        }
        if (pattern_[0] == '^') ++p_;
        auto node = parse_alternation();
        if (p_ < pattern_.size() && pattern_[p_] == '$') ++p_;
        return node;
    }

private:
    std::string_view pattern_;
    std::size_t      p_ = 0;
    GrammarGraph&    graph_;
    int              rule_id_;

    char peek() const {
        return p_ < pattern_.size() ? pattern_[p_] : '\0';
    }

    char get() {
        return p_ < pattern_.size() ? pattern_[p_++] : '\0';
    }

    std::pair<int, int> parse_alternation() {
        auto first = parse_sequence();
        if (peek() != '|') return first;

        int entry = graph_.add_state(rule_id_);
        int exit  = graph_.add_state(rule_id_);

        graph_.add_epsilon(entry, first.first);
        graph_.add_epsilon(first.second, exit);

        while (peek() == '|') {
            get();  // consume '|'
            auto next = parse_sequence();
            graph_.add_epsilon(entry, next.first);
            graph_.add_epsilon(next.second, exit);
        }

        return {entry, exit};
    }

    std::pair<int, int> parse_sequence() {
        int entry = -1;
        int last_exit = -1;

        while (p_ < pattern_.size() && peek() != '|' && peek() != ')' && peek() != '$') {
            auto factor = parse_factor();
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

    std::pair<int, int> parse_factor() {
        auto atom = parse_atom();
        char c = peek();
        if (c == '*' || c == '+' || c == '?' || c == '{') {
            get();
            int entry = graph_.add_state(rule_id_);
            int exit  = graph_.add_state(rule_id_);

            if (c == '*') {
                graph_.add_epsilon(entry, atom.first);
                graph_.add_epsilon(atom.second, atom.first);
                graph_.add_epsilon(atom.second, exit);
                graph_.add_epsilon(entry, exit);
                return {entry, exit};
            } else if (c == '+') {
                graph_.add_epsilon(entry, atom.first);
                graph_.add_epsilon(atom.second, atom.first);
                graph_.add_epsilon(atom.second, exit);
                return {entry, exit};
            } else if (c == '?') {
                graph_.add_epsilon(entry, atom.first);
                graph_.add_epsilon(atom.second, exit);
                graph_.add_epsilon(entry, exit);
                return {entry, exit};
            } else if (c == '{') {
                int min_c = 0;
                while (std::isdigit(static_cast<unsigned char>(peek()))) {
                    min_c = min_c * 10 + (get() - '0');
                }
                int max_c = min_c;
                if (peek() == ',') {
                    get();
                    if (peek() == '}') {
                        max_c = -1;  // unbounded {min,}
                    } else {
                        max_c = 0;
                        while (std::isdigit(static_cast<unsigned char>(peek()))) {
                            max_c = max_c * 10 + (get() - '0');
                        }
                    }
                }
                if (peek() == '}') get();

                if (min_c == 0 && max_c == 0) {
                    int s = graph_.add_state(rule_id_);
                    return {s, s};
                }
                // For simplicity on small {m,n}, treat as + or * when ranges are standard:
                if (min_c == 0 && max_c == -1) {
                    graph_.add_epsilon(entry, atom.first);
                    graph_.add_epsilon(atom.second, atom.first);
                    graph_.add_epsilon(atom.second, exit);
                    graph_.add_epsilon(entry, exit);
                    return {entry, exit};
                }
                if (min_c == 1 && max_c == -1) {
                    graph_.add_epsilon(entry, atom.first);
                    graph_.add_epsilon(atom.second, atom.first);
                    graph_.add_epsilon(atom.second, exit);
                    return {entry, exit};
                }
                // Otherwise general repeat:
                graph_.add_epsilon(entry, atom.first);
                graph_.add_epsilon(atom.second, exit);
                if (min_c == 0) graph_.add_epsilon(entry, exit);
                return {entry, exit};
            }
        }
        return atom;
    }

    std::pair<int, int> parse_atom() {
        char c = get();
        if (c == '(') {
            auto sub = parse_alternation();
            if (peek() == ')') get();
            return sub;
        }
        if (c == '[') {
            CharSet cs;
            bool negate = false;
            if (peek() == '^') {
                negate = true;
                get();
            }
            while (p_ < pattern_.size() && peek() != ']') {
                char ch = get();
                if (ch == '\\') {
                    ch = parse_escape_char();
                }
                if (peek() == '-' && p_ + 1 < pattern_.size() && pattern_[p_ + 1] != ']') {
                    get();  // consume '-'
                    char end_ch = get();
                    if (end_ch == '\\') end_ch = parse_escape_char();
                    cs.set_range(static_cast<uint8_t>(ch), static_cast<uint8_t>(end_ch));
                } else {
                    cs.set(static_cast<uint8_t>(ch));
                }
            }
            if (peek() == ']') get();
            if (negate) cs.invert();

            int entry = graph_.add_state(rule_id_);
            int exit  = graph_.add_state(rule_id_);
            graph_.add_char_trans(entry, exit, cs);
            return {entry, exit};
        }
        if (c == '.') {
            CharSet cs;
            cs.set_all();
            int entry = graph_.add_state(rule_id_);
            int exit  = graph_.add_state(rule_id_);
            graph_.add_char_trans(entry, exit, cs);
            return {entry, exit};
        }
        if (c == '\\') {
            char next = peek();
            if (next == 'd' || next == 'D' || next == 'w' || next == 'W' || next == 's' || next == 'S') {
                get();
                CharSet cs;
                if (next == 'd' || next == 'D') cs.set_range('0', '9');
                else if (next == 'w' || next == 'W') {
                    cs.set_range('a', 'z');
                    cs.set_range('A', 'Z');
                    cs.set_range('0', '9');
                    cs.set('_');
                } else if (next == 's' || next == 'S') {
                    cs.set(' '); cs.set('\t'); cs.set('\n'); cs.set('\r');
                }
                if (std::isupper(static_cast<unsigned char>(next))) {
                    cs.invert();
                }
                int entry = graph_.add_state(rule_id_);
                int exit  = graph_.add_state(rule_id_);
                graph_.add_char_trans(entry, exit, cs);
                return {entry, exit};
            }
            c = parse_escape_char();
        }

        CharSet cs;
        cs.set(static_cast<uint8_t>(c));
        int entry = graph_.add_state(rule_id_);
        int exit  = graph_.add_state(rule_id_);
        graph_.add_char_trans(entry, exit, cs);
        return {entry, exit};
    }

    char parse_escape_char() {
        if (p_ >= pattern_.size()) return '\\';
        char c = get();
        switch (c) {
            case 'n': return '\n';
            case 't': return '\t';
            case 'r': return '\r';
            case '0': return '\0';
            case '"': return '"';
            case '\\': return '\\';
            default: return c;
        }
    }
};

}  // namespace brolm
