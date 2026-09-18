#include "brolm/grammar.h"
#include "brolm/grammar_jit.h"
#include "brolm/detail/json.h"
#include "grammar/dfa_compiler.h"
#include "grammar/grammar_types.h"
#include "grammar/regex_compiler.h"
#include "grammar/bnf_compiler.h"

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

class Grammar::Impl {
public:
    GrammarGraph                  graph_;
    std::vector<Thread>           active_threads_;
    int                           root_entry_ = 0;
    int                           root_exit_  = 0;
    std::unique_ptr<JitGrammar>   jit_grammar_;

    Impl() = default;

    Impl(const Impl& other)
        : graph_(other.graph_),
          active_threads_(other.active_threads_),
          root_entry_(other.root_entry_),
          root_exit_(other.root_exit_) {
        if (other.jit_grammar_) {
            jit_grammar_ = std::make_unique<JitGrammar>(*other.jit_grammar_);
        }
    }

    Impl& operator=(const Impl& other) {
        if (this != &other) {
            graph_ = other.graph_;
            active_threads_ = other.active_threads_;
            root_entry_ = other.root_entry_;
            root_exit_ = other.root_exit_;
            if (other.jit_grammar_) {
                jit_grammar_ = std::make_unique<JitGrammar>(*other.jit_grammar_);
            } else {
                jit_grammar_.reset();
            }
        }
        return *this;
    }

    void initialize() {
        if (graph_.rules.empty() || graph_.root_rule_id < 0 ||
            graph_.root_rule_id >= static_cast<int>(graph_.rules.size())) {
            root_entry_ = 0;
            root_exit_  = 0;
            active_threads_.clear();
            jit_grammar_.reset();
            return;
        }
        root_entry_ = graph_.rules[graph_.root_rule_id].entry_state;
        root_exit_  = graph_.rules[graph_.root_rule_id].exit_state;

        // Attempt DFA compilation & JIT instantiation
        try {
            CompiledDfa dfa = DfaCompiler::compile(graph_);
            if (dfa.is_valid()) {
                jit_grammar_ = std::make_unique<JitGrammar>(std::move(dfa));
            }
        } catch (...) {
            jit_grammar_.reset();
        }

        reset();
    }

    void reset() {
        active_threads_.clear();
        if (root_entry_ >= 0 && root_entry_ < static_cast<int>(graph_.states.size())) {
            std::vector<Thread> init_set = { Thread{root_entry_, {}} };
            active_threads_ = compute_epsilon_closure(graph_, init_set);
        }
        if (jit_grammar_) {
            jit_grammar_->reset();
        }
    }

    bool can_accept(std::string_view text) const {
        if (jit_grammar_) {
            return jit_grammar_->can_accept(text);
        }
        if (active_threads_.empty()) return false;
        std::vector<Thread> curr = active_threads_;
        for (char ch : text) {
            curr = step_char(graph_, curr, static_cast<uint8_t>(ch));
            if (curr.empty()) return false;
        }
        return !curr.empty();
    }

    bool accept(std::string_view text) {
        if (jit_grammar_) {
            bool ok = jit_grammar_->accept(text);
            if (!ok) {
                active_threads_.clear();
                return false;
            }
            if (!active_threads_.empty()) {
                std::vector<Thread> curr = active_threads_;
                for (char ch : text) {
                    curr = step_char(graph_, curr, static_cast<uint8_t>(ch));
                    if (curr.empty()) break;
                }
                active_threads_ = std::move(curr);
            }
            return true;
        }

        if (active_threads_.empty()) return false;
        std::vector<Thread> curr = active_threads_;
        for (char ch : text) {
            curr = step_char(graph_, curr, static_cast<uint8_t>(ch));
            if (curr.empty()) {
                active_threads_.clear();
                return false;
            }
        }
        active_threads_ = std::move(curr);
        return true;
    }

    bool is_accepted() const {
        if (jit_grammar_) {
            return jit_grammar_->is_accepted();
        }
        for (const auto& t : active_threads_) {
            if (t.state_id == root_exit_ && t.call_stack.empty()) {
                return true;
            }
        }
        return false;
    }

    void mask_logits(float* logits, int vocab_size,
                     const std::vector<std::string>& vocab_tokens,
                     int eos_id) const {
        if (!logits || vocab_size <= 0) return;
        if (jit_grammar_) {
            jit_grammar_->mask_logits(logits, vocab_size, vocab_tokens, eos_id);
            return;
        }

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
};

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
    if (!impl_ || !logits || vocab_size <= 0) return;
    impl_->mask_logits(logits, vocab_size, vocab_tokens, eos_id);
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
