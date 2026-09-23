#pragma once

// Constrained decoding and logit masking for brolm text generation.
//
// Provides state-machine based logit masking driven by Regular Expressions,
// BNF / EBNF grammars, JSON Schemas, or built-in primitive type constraints.
// Before each sampling step, invalid token continuations are masked with -INFINITY
// so the decoder is guaranteed to produce outputs conforming to the grammar.

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace brolm {

class Grammar {
public:
    // ─── Factory constructors ──────────────────────────────────────────────

    // Constrain output to match a regular expression pattern.
    // Supports literals, character classes [a-z], [^0-9], \\d, \\w, \\s, wildcards .,
    // alternations |, groups (...), and quantifiers *, +, ?, {n}, {min,max}.
    static Grammar regex(const std::string& pattern);

    // Constrain output using a standard BNF / EBNF grammar specification.
    // Rules have the format `rule_name ::= expression`.
    // Default start rule is "root" (or the first defined rule).
    static Grammar bnf(const std::string& bnf_text);

    // Standard JSON structural constraints:
    static Grammar json_object();     // Valid JSON object: { ... }
    static Grammar json_array();      // Valid JSON array:  [ ... ]
    static Grammar json_value();      // Any valid JSON value (object, array, string, number, bool, null)
    static Grammar json_string();     // Valid JSON quoted string: "..."
    static Grammar json_number();     // Valid JSON number (int or float)
    static Grammar json_integer();    // Valid JSON integer
    static Grammar json_boolean();    // "true" | "false"
    static Grammar json_null();       // "null"

    // Constrain output to conform to a JSON Schema specification string.
    // Handles object properties, required fields, types (string, number, integer, boolean,
    // array, object), enums, and string regex patterns.
    static Grammar json_schema(const std::string& schema_json);

    // Constrain output to match one of a fixed set of string choices.
    static Grammar choice(const std::vector<std::string>& choices);

    // Constrain output to match an exact string literal.
    static Grammar exact(const std::string& str);

    // ─── Lifecycle & State Management ──────────────────────────────────────

    Grammar();
    Grammar(const Grammar& other);
    Grammar& operator=(const Grammar& other);
    Grammar(Grammar&& other) noexcept;
    Grammar& operator=(Grammar&& other) noexcept;
    ~Grammar();

    // Check if appending `text` is a valid continuation from the current state.
    // Does NOT mutate internal state.
    bool can_accept(std::string_view text) const;

    // Advance the grammar state by consuming `text`.
    // Returns true if successfully advanced, or false if the continuation is invalid.
    bool accept(std::string_view text);

    // True if the current state is an accepting (valid finished) state.
    bool is_accepted() const;

    // Reset the state machine back to its initial starting state.
    void reset();

    // Returns a copy of this grammar in its current state.
    Grammar clone() const;

    // ─── Logit Masking ─────────────────────────────────────────────────────

    // Set logits of all invalid next tokens to -INFINITY.
    // When `is_accepted()` is false, `eos_id` is also masked with -INFINITY.
    // A token other than `eos_id` with empty text is always masked (it would
    // never advance the grammar), and so is every id at or past
    // vocab_tokens.size() except `eos_id`.
    void mask_logits(float* logits, int vocab_size,
                     const std::vector<std::string>& vocab_tokens,
                     int eos_id = -1) const;

    void mask_logits(float* logits, int vocab_size,
                     const std::function<std::string_view(int)>& token_to_text,
                     int eos_id = -1) const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace brolm
