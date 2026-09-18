#include "grammar/dfa_compiler.h"

#include <algorithm>
#include <array>
#include <map>
#include <unordered_map>
#include <utility>
#include <vector>

namespace brolm {

namespace {

struct VectorThreadHash {
    std::size_t operator()(const std::vector<Thread>& vec) const {
        std::size_t h = vec.size();
        for (const auto& t : vec) {
            h ^= std::hash<int>()(t.state_id) + 0x9e3779b9 + (h << 6) + (h >> 2);
            for (int s : t.call_stack) {
                h ^= std::hash<int>()(s) + 0x517cc1b7 + (h << 6) + (h >> 2);
            }
        }
        return h;
    }
};

}  // namespace

CompiledDfa DfaCompiler::subset_construct(const GrammarGraph& graph, int32_t max_states) {
    if (graph.rules.empty() || graph.root_rule_id < 0 ||
        graph.root_rule_id >= static_cast<int>(graph.rules.size())) {
        return CompiledDfa{};
    }

    int root_entry = graph.rules[graph.root_rule_id].entry_state;
    int root_exit  = graph.rules[graph.root_rule_id].exit_state;

    if (root_entry < 0 || root_entry >= static_cast<int>(graph.states.size())) {
        return CompiledDfa{};
    }

    std::vector<Thread> init_set = { Thread{root_entry, {}} };
    std::vector<Thread> start_threads = compute_epsilon_closure(graph, init_set);
    std::sort(start_threads.begin(), start_threads.end());
    start_threads.erase(std::unique(start_threads.begin(), start_threads.end()), start_threads.end());

    auto is_accepting = [&](const std::vector<Thread>& threads) -> bool {
        for (const auto& t : threads) {
            if (t.state_id == root_exit && t.call_stack.empty()) {
                return true;
            }
        }
        return false;
    };

    if (start_threads.empty()) {
        CompiledDfa dfa;
        dfa.num_states = 1;
        dfa.start_state = 0;
        dfa.transition_table.assign(256, -1);
        dfa.is_accept_state.assign(1, false);
        return dfa;
    }

    std::vector<std::vector<Thread>> dfa_states;
    std::map<std::vector<Thread>, int32_t> state_map;
    std::vector<int32_t> trans_table;
    std::vector<bool> accept_flags;

    dfa_states.push_back(start_threads);
    state_map[start_threads] = 0;
    accept_flags.push_back(is_accepting(start_threads));

    size_t head = 0;
    while (head < dfa_states.size()) {
        if (dfa_states.size() > static_cast<size_t>(max_states)) {
            return CompiledDfa{};
        }

        int32_t cur_dfa_id = static_cast<int32_t>(head);
        const std::vector<Thread> cur_threads = dfa_states[head++];

        size_t base_idx = static_cast<size_t>(cur_dfa_id) * 256;
        if (trans_table.size() < base_idx + 256) {
            trans_table.resize(base_idx + 256, -1);
        }

        std::unordered_map<std::vector<Thread>, int32_t, VectorThreadHash> local_next_cache;

        for (int b = 0; b < 256; ++b) {
            uint8_t byte_val = static_cast<uint8_t>(b);
            std::vector<Thread> next_raw;

            for (const auto& t : cur_threads) {
                if (t.state_id < 0 || t.state_id >= static_cast<int>(graph.states.size())) continue;
                const State& st = graph.states[t.state_id];
                for (const auto& tr : st.transitions) {
                    if (tr.type == TransType::Char && tr.chars.test(byte_val)) {
                        next_raw.push_back(Thread{tr.target_state, t.call_stack});
                    }
                }
            }

            if (next_raw.empty()) {
                trans_table[base_idx + b] = -1;
                continue;
            }

            std::sort(next_raw.begin(), next_raw.end());
            next_raw.erase(std::unique(next_raw.begin(), next_raw.end()), next_raw.end());

            auto cache_it = local_next_cache.find(next_raw);
            if (cache_it != local_next_cache.end()) {
                trans_table[base_idx + b] = cache_it->second;
                continue;
            }

            std::vector<Thread> next_threads = compute_epsilon_closure(graph, next_raw);
            std::sort(next_threads.begin(), next_threads.end());
            next_threads.erase(std::unique(next_threads.begin(), next_threads.end()), next_threads.end());

            int32_t target_dfa_id = -1;
            if (next_threads.empty()) {
                target_dfa_id = -1;
            } else {
                auto it = state_map.find(next_threads);
                if (it != state_map.end()) {
                    target_dfa_id = it->second;
                } else {
                    target_dfa_id = static_cast<int32_t>(dfa_states.size());
                    state_map[next_threads] = target_dfa_id;
                    dfa_states.push_back(next_threads);
                    accept_flags.push_back(is_accepting(next_threads));
                }
            }

            local_next_cache[next_raw] = target_dfa_id;
            trans_table[base_idx + b] = target_dfa_id;
        }
    }

    CompiledDfa dfa;
    dfa.num_states = static_cast<int32_t>(dfa_states.size());
    dfa.start_state = 0;
    dfa.transition_table = std::move(trans_table);
    dfa.is_accept_state = std::move(accept_flags);
    return dfa;
}

CompiledDfa DfaCompiler::minimize(const CompiledDfa& dfa) {
    if (!dfa.is_valid() || dfa.num_states <= 1) {
        return dfa;
    }

    // Moore's partition refinement algorithm
    std::vector<int32_t> partition(dfa.num_states, 0);
    bool has_accept = false;
    bool has_non_accept = false;

    for (int32_t s = 0; s < dfa.num_states; ++s) {
        if (dfa.is_accept_state[s]) {
            has_accept = true;
        } else {
            has_non_accept = true;
        }
    }

    int current_num_partitions = 1;
    if (has_accept && has_non_accept) {
        current_num_partitions = 2;
        for (int32_t s = 0; s < dfa.num_states; ++s) {
            partition[s] = dfa.is_accept_state[s] ? 1 : 0;
        }
    }

    bool changed = true;
    while (changed) {
        changed = false;
        int next_partition_id = 0;
        std::vector<int32_t> new_partition(dfa.num_states);

        for (int p = 0; p < current_num_partitions; ++p) {
            std::map<std::vector<int32_t>, int32_t> sig_to_part;

            for (int32_t s = 0; s < dfa.num_states; ++s) {
                if (partition[s] != p) continue;

                std::vector<int32_t> sig(256);
                for (int b = 0; b < 256; ++b) {
                    int32_t nxt = dfa.transition_table[static_cast<size_t>(s) * 256 + b];
                    sig[b] = (nxt == -1) ? -1 : partition[nxt];
                }

                auto it = sig_to_part.find(sig);
                if (it == sig_to_part.end()) {
                    int assigned = next_partition_id++;
                    sig_to_part[sig] = assigned;
                    new_partition[s] = assigned;
                } else {
                    new_partition[s] = it->second;
                }
            }
        }

        if (next_partition_id != current_num_partitions) {
            changed = true;
            current_num_partitions = next_partition_id;
            partition = std::move(new_partition);
        }
    }

    // Remap partitions so that start_state maps to minimal state 0
    int old_start_part = partition[dfa.start_state];
    std::vector<int32_t> remap(current_num_partitions, -1);
    remap[old_start_part] = 0;
    int next_id = 1;
    for (int p = 0; p < current_num_partitions; ++p) {
        if (p != old_start_part) {
            remap[p] = next_id++;
        }
    }

    CompiledDfa min_dfa;
    min_dfa.num_states = current_num_partitions;
    min_dfa.start_state = 0;
    min_dfa.transition_table.assign(static_cast<size_t>(min_dfa.num_states) * 256, -1);
    min_dfa.is_accept_state.assign(min_dfa.num_states, false);

    for (int p = 0; p < current_num_partitions; ++p) {
        int32_t rep_s = -1;
        for (int32_t s = 0; s < dfa.num_states; ++s) {
            if (partition[s] == p) {
                rep_s = s;
                break;
            }
        }
        if (rep_s == -1) continue;

        int32_t new_state = remap[p];
        min_dfa.is_accept_state[new_state] = dfa.is_accept_state[rep_s];

        for (int b = 0; b < 256; ++b) {
            int32_t nxt = dfa.transition_table[static_cast<size_t>(rep_s) * 256 + b];
            min_dfa.transition_table[static_cast<size_t>(new_state) * 256 + b] =
                (nxt == -1) ? -1 : remap[partition[nxt]];
        }
    }

    return min_dfa;
}

CompiledDfa DfaCompiler::compile(const GrammarGraph& graph, int32_t max_states) {
    CompiledDfa unmin = subset_construct(graph, max_states);
    if (!unmin.is_valid()) {
        return unmin;
    }
    return minimize(unmin);
}

}  // namespace brolm
