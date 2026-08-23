#include "brolm/grammar.h"
#include "brolm/detail/json.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace brolm {

namespace {

// ─── Bitset of 256 byte values ───────────────────────────────────────────────

struct CharSet {
    std::array<uint64_t, 4> bits{};

    void set(uint8_t c) {
        bits[c / 64] |= (1ULL << (c % 64));
    }

    void set_range(uint8_t low, uint8_t high) {
        for (int c = low; c <= high; ++c) {
            set(static_cast<uint8_t>(c));
        }
    }

    void set_all() {
        bits[0] = bits[1] = bits[2] = bits[3] = ~0ULL;
    }

    void invert() {
        bits[0] = ~bits[0];
        bits[1] = ~bits[1];
        bits[2] = ~bits[2];
        bits[3] = ~bits[3];
    }

    bool test(uint8_t c) const {
        return (bits[c / 64] & (1ULL << (c % 64))) != 0;
    }

    bool empty() const {
        return bits[0] == 0 && bits[1] == 0 && bits[2] == 0 && bits[3] == 0;
    }
};

// ─── Automaton State & Transition Graph ───────────────────────────────────────

enum class TransType { Epsilon, Char, RuleCall };

struct Transition {
    TransType type;
    int       target_state = -1;
    CharSet   chars;
    int       target_rule  = -1;  // For RuleCall
};

struct State {
    int                     id = 0;
    int                     rule_id = 0;
    std::vector<Transition> transitions;
};

struct Rule {
    int         id = 0;
    std::string name;
    int         entry_state = -1;
    int         exit_state  = -1;
};

// Parser Thread / Configuration
struct Thread {
    int              state_id = 0;
    std::vector<int> call_stack;

    bool operator==(const Thread& o) const {
        return state_id == o.state_id && call_stack == o.call_stack;
    }
};

struct ThreadHash {
    std::size_t operator()(const Thread& t) const {
        std::size_t h = std::hash<int>()(t.state_id);
        for (int s : t.call_stack) {
            h ^= std::hash<int>()(s) + 0x9e3779b9 + (h << 6) + (h >> 2);
        }
        return h;
    }
};

// ─── Regex & BNF Grammar Builder ─────────────────────────────────────────────

class GrammarGraph {
public:
    std::vector<State> states;
    std::vector<Rule>  rules;
    std::unordered_map<std::string, int> rule_map;
    int root_rule_id = 0;

    int add_state(int rule_id) {
        int id = static_cast<int>(states.size());
        states.push_back(State{id, rule_id, {}});
        return id;
    }

    int add_rule(const std::string& name) {
        auto it = rule_map.find(name);
        if (it != rule_map.end()) return it->second;
        int id = static_cast<int>(rules.size());
        rules.push_back(Rule{id, name, -1, -1});
        rule_map[name] = id;
        return id;
    }

    void add_epsilon(int from, int to) {
        states[from].transitions.push_back({TransType::Epsilon, to, {}, -1});
    }

    void add_char_trans(int from, int to, const CharSet& cs) {
        states[from].transitions.push_back({TransType::Char, to, cs, -1});
    }

    void add_rule_call(int from, int to, int target_rule) {
        states[from].transitions.push_back({TransType::RuleCall, to, {}, target_rule});
    }
};

}  // namespace

class Grammar::Impl {
public:
    GrammarGraph                  graph_;
    std::vector<Thread>           active_threads_;
    int                           root_entry_ = 0;
    int                           root_exit_  = 0;

    void initialize() {
        if (graph_.rules.empty() || graph_.root_rule_id < 0 ||
            graph_.root_rule_id >= static_cast<int>(graph_.rules.size())) {
            root_entry_ = 0;
            root_exit_  = 0;
            active_threads_.clear();
            return;
        }
        root_entry_ = graph_.rules[graph_.root_rule_id].entry_state;
        root_exit_  = graph_.rules[graph_.root_rule_id].exit_state;
        reset();
    }

    void reset() {
        active_threads_.clear();
        if (root_entry_ >= 0 && root_entry_ < static_cast<int>(graph_.states.size())) {
            std::vector<Thread> init_set = { Thread{root_entry_, {}} };
            active_threads_ = compute_epsilon_closure(init_set);
        }
    }

    std::vector<Thread> compute_epsilon_closure(const std::vector<Thread>& initial) const {
        std::vector<Thread> result;
        std::unordered_set<Thread, ThreadHash> visited;
        std::queue<Thread> queue;

        for (const auto& t : initial) {
            if (visited.insert(t).second) {
                result.push_back(t);
                queue.push(t);
            }
        }

        constexpr std::size_t MAX_CALL_STACK = 64;

        while (!queue.empty()) {
            Thread cur = queue.front();
            queue.pop();

            if (cur.state_id < 0 || cur.state_id >= static_cast<int>(graph_.states.size())) {
                continue;
            }

            const State& st = graph_.states[cur.state_id];
            const Rule& rule = graph_.rules[st.rule_id];

            // If we reached the exit state of a sub-rule and have a return address on the stack:
            if (cur.state_id == rule.exit_state && !cur.call_stack.empty()) {
                Thread next_thread;
                next_thread.state_id = cur.call_stack.back();
                next_thread.call_stack = cur.call_stack;
                next_thread.call_stack.pop_back();

                if (visited.insert(next_thread).second) {
                    result.push_back(next_thread);
                    queue.push(next_thread);
                }
            }

            for (const auto& tr : st.transitions) {
                if (tr.type == TransType::Epsilon) {
                    Thread next_thread{tr.target_state, cur.call_stack};
                    if (visited.insert(next_thread).second) {
                        result.push_back(next_thread);
                        queue.push(next_thread);
                    }
                } else if (tr.type == TransType::RuleCall) {
                    if (cur.call_stack.size() < MAX_CALL_STACK &&
                        tr.target_rule >= 0 &&
                        tr.target_rule < static_cast<int>(graph_.rules.size())) {
                        int entry = graph_.rules[tr.target_rule].entry_state;
                        Thread next_thread;
                        next_thread.state_id = entry;
                        next_thread.call_stack = cur.call_stack;
                        next_thread.call_stack.push_back(tr.target_state);

                        if (visited.insert(next_thread).second) {
                            result.push_back(next_thread);
                            queue.push(next_thread);
                        }
                    }
                }
            }
        }

        return result;
    }

    std::vector<Thread> step_char(const std::vector<Thread>& threads, uint8_t c) const {
        std::vector<Thread> next_raw;
        for (const auto& t : threads) {
            if (t.state_id < 0 || t.state_id >= static_cast<int>(graph_.states.size())) continue;
            const State& st = graph_.states[t.state_id];
            for (const auto& tr : st.transitions) {
                if (tr.type == TransType::Char && tr.chars.test(c)) {
                    next_raw.push_back(Thread{tr.target_state, t.call_stack});
                }
            }
        }
        return compute_epsilon_closure(next_raw);
    }

    bool can_accept(std::string_view text) const {
        if (active_threads_.empty()) return false;
        std::vector<Thread> curr = active_threads_;
        for (char ch : text) {
            curr = step_char(curr, static_cast<uint8_t>(ch));
            if (curr.empty()) return false;
        }
        return !curr.empty();
    }

    bool accept(std::string_view text) {
        if (active_threads_.empty()) return false;
        std::vector<Thread> curr = active_threads_;
        for (char ch : text) {
            curr = step_char(curr, static_cast<uint8_t>(ch));
            if (curr.empty()) {
                active_threads_.clear();
                return false;
            }
        }
        active_threads_ = std::move(curr);
        return true;
    }

    bool is_accepted() const {
        for (const auto& t : active_threads_) {
            if (t.state_id == root_exit_ && t.call_stack.empty()) {
                return true;
            }
        }
        return false;
    }
};

// ─── Regex Parser & Compiler ──────────────────────────────────────────────────

namespace {

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

// ─── BNF Grammar Parser & Compiler ────────────────────────────────────────────

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

}  // namespace

// ─── Grammar Factory Implementations ──────────────────────────────────────────

Grammar Grammar::regex(const std::string& pattern) {
    Grammar g;
    g.impl_ = std::make_unique<Impl>();
    int rule_id = g.impl_->graph_.add_rule("root");
    g.impl_->graph_.root_rule_id = rule_id;

    RegexCompiler compiler(pattern, g.impl_->graph_, rule_id);
    auto [entry, exit] = compiler.compile();
    g.impl_->graph_.rules[rule_id].entry_state = entry;
    g.impl_->graph_.rules[rule_id].exit_state  = exit;

    g.impl_->initialize();
    return g;
}

Grammar Grammar::bnf(const std::string& bnf_text) {
    Grammar g;
    g.impl_ = std::make_unique<Impl>();
    BNFCompiler compiler(bnf_text, g.impl_->graph_);
    compiler.compile();
    g.impl_->initialize();
    return g;
}

Grammar Grammar::json_value() {
    static const std::string kJsonValBnf = R"(
root    ::= ws value ws
value   ::= object | array | string | number | boolean | null
object  ::= "{" ws (pair (ws "," ws pair)*)? ws "}"
pair    ::= string ws ":" ws value
array   ::= "[" ws (value (ws "," ws value)*)? ws "]"
string  ::= "\"" ([^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]))* "\""
number  ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
boolean ::= "true" | "false"
null    ::= "null"
ws      ::= [ \t\n\r]*
)";
    return Grammar::bnf(kJsonValBnf);
}

Grammar Grammar::json_object() {
    static const std::string kJsonObjBnf = R"(
root    ::= ws object ws
object  ::= "{" ws (pair (ws "," ws pair)*)? ws "}"
pair    ::= string ws ":" ws value
value   ::= object | array | string | number | boolean | null
array   ::= "[" ws (value (ws "," ws value)*)? ws "]"
string  ::= "\"" ([^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]))* "\""
number  ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
boolean ::= "true" | "false"
null    ::= "null"
ws      ::= [ \t\n\r]*
)";
    return Grammar::bnf(kJsonObjBnf);
}

Grammar Grammar::json_array() {
    static const std::string kJsonArrBnf = R"(
root    ::= ws array ws
object  ::= "{" ws (pair (ws "," ws pair)*)? ws "}"
pair    ::= string ws ":" ws value
value   ::= object | array | string | number | boolean | null
array   ::= "[" ws (value (ws "," ws value)*)? ws "]"
string  ::= "\"" ([^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]))* "\""
number  ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
boolean ::= "true" | "false"
null    ::= "null"
ws      ::= [ \t\n\r]*
)";
    return Grammar::bnf(kJsonArrBnf);
}

Grammar Grammar::json_string() {
    return Grammar::regex(R"("[^"\\]*(\\.[^"\\]*)*")");
}

Grammar Grammar::json_number() {
    return Grammar::regex(R"(-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?)");
}

Grammar Grammar::json_integer() {
    return Grammar::regex(R"(-?(0|[1-9][0-9]*))");
}

Grammar Grammar::json_boolean() {
    return Grammar::regex("true|false");
}

Grammar Grammar::json_null() {
    return Grammar::regex("null");
}

Grammar Grammar::choice(const std::vector<std::string>& choices) {
    if (choices.empty()) {
        return Grammar::exact("");
    }
    std::ostringstream oss;
    oss << "root ::= ";
    for (std::size_t i = 0; i < choices.size(); ++i) {
        if (i > 0) oss << " | ";
        oss << "\"";
        for (char c : choices[i]) {
            if (c == '"' || c == '\\') oss << '\\';
            oss << c;
        }
        oss << "\"";
    }
    oss << "\n";
    return Grammar::bnf(oss.str());
}

Grammar Grammar::exact(const std::string& str) {
    return Grammar::choice({str});
}

namespace {

std::string json_schema_node_to_bnf(const detail::json::Value& schema,
                                    std::ostringstream& bnf_out,
                                    int& rule_counter) {
    int rule_num = rule_counter++;
    std::string rule_name = "schema_rule_" + std::to_string(rule_num);

    if (schema.is_object() && schema.contains("enum")) {
        const auto& enum_arr = schema.at("enum").as_array();
        std::ostringstream rhs;
        for (std::size_t i = 0; i < enum_arr.size(); ++i) {
            if (i > 0) rhs << " | ";
            if (enum_arr[i].is_string()) {
                rhs << "\"\\\"" << enum_arr[i].as_string() << "\\\"\"";
            } else if (enum_arr[i].is_number()) {
                rhs << "\"" << enum_arr[i].as_number() << "\"";
            } else if (enum_arr[i].is_bool()) {
                rhs << (enum_arr[i].as_bool() ? "\"true\"" : "\"false\"");
            }
        }
        bnf_out << rule_name << " ::= " << rhs.str() << "\n";
        return rule_name;
    }

    std::string type = "object";
    if (schema.is_object() && schema.contains("type")) {
        type = schema.at("type").as_string();
    }

    if (type == "string") {
        if (schema.is_object() && schema.contains("pattern")) {
            std::string pat = schema.at("pattern").as_string();
            // Wrap regex pattern with quotes
            bnf_out << rule_name << " ::= \"\\\"\" (" << pat << ") \"\\\"\"\n";
        } else {
            bnf_out << rule_name << " ::= string\n";
        }
    } else if (type == "integer") {
        bnf_out << rule_name << " ::= integer\n";
    } else if (type == "number") {
        bnf_out << rule_name << " ::= number\n";
    } else if (type == "boolean") {
        bnf_out << rule_name << " ::= boolean\n";
    } else if (type == "null") {
        bnf_out << rule_name << " ::= null\n";
    } else if (type == "array") {
        std::string item_rule = "value";
        if (schema.is_object() && schema.contains("items")) {
            item_rule = json_schema_node_to_bnf(schema.at("items"), bnf_out, rule_counter);
        }
        bnf_out << rule_name << " ::= \"[\" ws (" << item_rule << " (ws \",\" ws " << item_rule << ")*)? ws \"]\"\n";
    } else if (type == "object") {
        if (schema.is_object() && schema.contains("properties")) {
            const auto& props = schema.at("properties").as_object();
            std::vector<std::string> req_names;
            if (schema.contains("required")) {
                const auto& req_arr = schema.at("required").as_array();
                for (const auto& r : req_arr) {
                    if (r.is_string()) req_names.push_back(r.as_string());
                }
            }

            std::ostringstream obj_body;
            obj_body << "\"{\" ws ";
            bool first = true;
            for (const auto& [pname, pschema] : props) {
                std::string p_rule = json_schema_node_to_bnf(pschema, bnf_out, rule_counter);
                if (!first) obj_body << " ws \",\" ws ";
                first = false;
                obj_body << "\"\\\"" << pname << "\\\"\" ws \":\" ws " << p_rule;
            }
            obj_body << " ws \"}\"";
            bnf_out << rule_name << " ::= " << obj_body.str() << "\n";
        } else {
            bnf_out << rule_name << " ::= object\n";
        }
    } else {
        bnf_out << rule_name << " ::= value\n";
    }

    return rule_name;
}

}  // namespace

Grammar Grammar::json_schema(const std::string& schema_json) {
    try {
        detail::json::Value schema = detail::json::parse(schema_json);
        std::ostringstream bnf;
        int rule_counter = 0;

        std::string root_sub = json_schema_node_to_bnf(schema, bnf, rule_counter);

        std::string full_bnf = "root ::= ws " + root_sub + " ws\n" +
            "object  ::= \"{\" ws (pair (ws \",\" ws pair)*)? ws \"}\"\n" +
            "pair    ::= string ws \":\" ws value\n" +
            "value   ::= object | array | string | number | boolean | null\n" +
            "array   ::= \"[\" ws (value (ws \",\" ws value)*)? ws \"]\"\n" +
            "string  ::= \"\\\"\" ([^\"\\\\] | \"\\\\\" ([\"\\\\/bfnrt] | \"u\" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]))* \"\\\"\"\n" +
            "integer ::= \"-\"? (\"0\" | [1-9] [0-9]*)\n" +
            "number  ::= \"-\"? (\"0\" | [1-9] [0-9]*) (\".\" [0-9]+)? ([eE] [+-]? [0-9]+)?\n" +
            "boolean ::= \"true\" | \"false\"\n" +
            "null    ::= \"null\"\n" +
            "ws      ::= [ \\t\\n\\r]*\n" +
            bnf.str();

        return Grammar::bnf(full_bnf);
    } catch (const std::exception&) {
        return Grammar::json_value();
    }
}

// ─── Grammar Lifecycle & API ──────────────────────────────────────────────────

Grammar::Grammar() : impl_(std::make_unique<Impl>()) {}

Grammar::Grammar(const Grammar& other) {
    if (other.impl_) {
        impl_ = std::make_unique<Impl>(*other.impl_);
    } else {
        impl_ = std::make_unique<Impl>();
    }
}

Grammar& Grammar::operator=(const Grammar& other) {
    if (this != &other) {
        if (other.impl_) {
            impl_ = std::make_unique<Impl>(*other.impl_);
        } else {
            impl_ = std::make_unique<Impl>();
        }
    }
    return *this;
}

Grammar::Grammar(Grammar&& other) noexcept = default;
Grammar& Grammar::operator=(Grammar&& other) noexcept = default;
Grammar::~Grammar() = default;

bool Grammar::can_accept(std::string_view text) const {
    if (!impl_) return false;
    return impl_->can_accept(text);
}

bool Grammar::accept(std::string_view text) {
    if (!impl_) return false;
    return impl_->accept(text);
}

bool Grammar::is_accepted() const {
    if (!impl_) return false;
    return impl_->is_accepted();
}

void Grammar::reset() {
    if (impl_) impl_->reset();
}

Grammar Grammar::clone() const {
    return Grammar(*this);
}

void Grammar::mask_logits(float* logits, int vocab_size,
                          const std::vector<std::string>& vocab_tokens,
                          int eos_id) const {
    if (!logits || vocab_size <= 0) return;
    const float neg_inf = -std::numeric_limits<float>::infinity();

    const int n = std::min(vocab_size, static_cast<int>(vocab_tokens.size()));
    for (int i = 0; i < n; ++i) {
        if (i == eos_id) {
            if (!is_accepted()) {
                logits[i] = neg_inf;
            }
            continue;
        }
        const std::string& tok = vocab_tokens[i];
        if (tok.empty()) continue;
        if (!can_accept(tok)) {
            logits[i] = neg_inf;
        }
    }
    for (int i = n; i < vocab_size; ++i) {
        if (i == eos_id && !is_accepted()) {
            logits[i] = neg_inf;
        }
    }
}

void Grammar::mask_logits(float* logits, int vocab_size,
                          const std::function<std::string_view(int)>& token_to_text,
                          int eos_id) const {
    if (!logits || vocab_size <= 0) return;
    const float neg_inf = -std::numeric_limits<float>::infinity();

    for (int i = 0; i < vocab_size; ++i) {
        if (i == eos_id) {
            if (!is_accepted()) {
                logits[i] = neg_inf;
            }
            continue;
        }
        std::string_view tok = token_to_text(i);
        if (tok.empty()) continue;
        if (!can_accept(tok)) {
            logits[i] = neg_inf;
        }
    }
}

}  // namespace brolm
