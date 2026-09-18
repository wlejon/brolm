#pragma once

#include "brolm/grammar_jit.h"
#include "grammar/grammar_types.h"

#include <cstdint>
#include <vector>

namespace brolm {

class DfaCompiler {
public:
    // Converts GrammarGraph (Thompson NFA) into a minimal DFA via subset construction
    // and Moore partition refinement minimization.
    // Returns CompiledDfa with start_state = 0, transition_table (num_states * 256),
    // and is_accept_state (num_states).
    static CompiledDfa compile(const GrammarGraph& graph, int32_t max_states = 65536);

    // Performs subset construction only (unminimized DFA)
    static CompiledDfa subset_construct(const GrammarGraph& graph, int32_t max_states = 65536);

    // Minimizes an existing DFA using Moore's partition refinement algorithm
    static CompiledDfa minimize(const CompiledDfa& dfa);
};

}  // namespace brolm
