#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <queue>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace brolm {

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

    bool operator==(const CharSet& o) const {
        return bits == o.bits;
    }
};

// ─── Automaton State & Transition Graph ───────────────────────────────────────

enum class TransType { Epsilon, Char, RuleCall };

struct Transition {
    TransType type = TransType::Epsilon;
    int       target_state = -1;
    CharSet   chars{};
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

    bool operator<(const Thread& o) const {
        if (state_id != o.state_id) return state_id < o.state_id;
        return call_stack < o.call_stack;
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

inline std::vector<Thread> compute_epsilon_closure(const GrammarGraph& graph, const std::vector<Thread>& initial) {
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

        if (cur.state_id < 0 || cur.state_id >= static_cast<int>(graph.states.size())) {
            continue;
        }

        const State& st = graph.states[cur.state_id];
        const Rule& rule = graph.rules[st.rule_id];

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
                    tr.target_rule < static_cast<int>(graph.rules.size())) {
                    int entry = graph.rules[tr.target_rule].entry_state;
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

inline std::vector<Thread> step_char(const GrammarGraph& graph, const std::vector<Thread>& threads, uint8_t c) {
    std::vector<Thread> next_raw;
    for (const auto& t : threads) {
        if (t.state_id < 0 || t.state_id >= static_cast<int>(graph.states.size())) continue;
        const State& st = graph.states[t.state_id];
        for (const auto& tr : st.transitions) {
            if (tr.type == TransType::Char && tr.chars.test(c)) {
                next_raw.push_back(Thread{tr.target_state, t.call_stack});
            }
        }
    }
    return compute_epsilon_closure(graph, next_raw);
}

}  // namespace brolm
