#include "brolm/qwen_tokenizer.h"

#include "brolm/detail/byte_level_bpe.h"
#include "brolm/detail/json.h"
#include "brotensor/gguf.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace brolm::qwen {

namespace bpe = brolm::detail::bpe;

// ─── BPE delegation ────────────────────────────────────────────────────────

std::vector<std::string> Tokenizer::bpe_(const std::string& token) const {
    // GPT-2/Qwen variant: no "</w>" end-of-word marker.
    return bpe::bpe_merge(token, merge_ranks_, /*append_end_of_word=*/false);
}

// ─── Pre-tokenization (GPT-2 / Qwen) ───────────────────────────────────────
//
// GPT-2 regex (simplified):
//   's | 't | 're | 've | 'm | 'll | 'd
//   | ?letters+ | ?digits+ | ?punctuation+ | whitespace-runs
//
// where "?" is an optional single leading space folded into the token. ASCII-
// focused (non-ASCII bytes lump into the punct run); unlike CLIP, whitespace
// is preserved (folded in or emitted as its own piece) rather than dropped.

namespace {

std::vector<std::string> pre_tokenize(std::string_view text) {
    std::vector<std::string> pieces;
    std::size_t i = 0;
    while (i < text.size()) {
        unsigned char c = static_cast<unsigned char>(text[i]);

        if (c == '\'') {
            if (bpe::starts_with(text, i, "'re")) { pieces.emplace_back("'re"); i += 3; continue; }
            if (bpe::starts_with(text, i, "'ve")) { pieces.emplace_back("'ve"); i += 3; continue; }
            if (bpe::starts_with(text, i, "'ll")) { pieces.emplace_back("'ll"); i += 3; continue; }
            if (bpe::starts_with(text, i, "'s"))  { pieces.emplace_back("'s");  i += 2; continue; }
            if (bpe::starts_with(text, i, "'t"))  { pieces.emplace_back("'t");  i += 2; continue; }
            if (bpe::starts_with(text, i, "'m"))  { pieces.emplace_back("'m");  i += 2; continue; }
            if (bpe::starts_with(text, i, "'d"))  { pieces.emplace_back("'d");  i += 2; continue; }
        }

        // Optional single leading space folds into the following run.
        std::size_t start = i;
        bool have_lead_space = false;
        if (c == ' ') {
            if (i + 1 < text.size()) {
                unsigned char d = static_cast<unsigned char>(text[i + 1]);
                if (!bpe::is_ascii_space(d)) {
                    have_lead_space = true;
                    ++i;
                    c = d;
                }
            }
            if (!have_lead_space) {
                pieces.emplace_back(text.substr(i, 1));
                ++i;
                continue;
            }
        } else if (bpe::is_ascii_space(c)) {
            // Non-space whitespace (tab/newline/...) emits as its own run.
            std::size_t j = i;
            while (j < text.size() &&
                   bpe::is_ascii_space(static_cast<unsigned char>(text[j])) &&
                   text[j] != ' ') ++j;
            pieces.emplace_back(text.substr(i, j - i));
            i = j;
            continue;
        }

        if (bpe::is_ascii_letter(c)) {
            std::size_t j = i;
            while (j < text.size() &&
                   bpe::is_ascii_letter(static_cast<unsigned char>(text[j]))) ++j;
            pieces.emplace_back(text.substr(start, j - start));
            i = j;
            continue;
        }

        if (bpe::is_ascii_digit(c)) {
            std::size_t j = i;
            while (j < text.size() &&
                   bpe::is_ascii_digit(static_cast<unsigned char>(text[j]))) ++j;
            pieces.emplace_back(text.substr(start, j - start));
            i = j;
            continue;
        }

        // Punct run.
        std::size_t j = i;
        while (j < text.size()) {
            unsigned char u = static_cast<unsigned char>(text[j]);
            if (bpe::is_ascii_space(u) || bpe::is_ascii_letter(u) ||
                bpe::is_ascii_digit(u)) break;
            if (u == '\'' && j != i) {
                std::string_view rest = text.substr(j);
                if (rest.substr(0, 3) == "'re" || rest.substr(0, 3) == "'ve" ||
                    rest.substr(0, 3) == "'ll" ||
                    rest.substr(0, 2) == "'s"  || rest.substr(0, 2) == "'t"  ||
                    rest.substr(0, 2) == "'m"  || rest.substr(0, 2) == "'d") {
                    break;
                }
            }
            ++j;
        }
        pieces.emplace_back(text.substr(start, j - start));
        i = j;
    }
    return pieces;
}

}  // namespace

// ─── Tokenizer driver ──────────────────────────────────────────────────────

Tokenizer Tokenizer::load(const std::string& vocab_json_path,
                          const std::string& merges_txt_path,
                          const std::vector<std::string>& extra_special_tokens) {
    Tokenizer t;
    t.vocab_       = bpe::load_vocab_json(vocab_json_path);
    t.merge_ranks_ = bpe::load_merges_txt(merges_txt_path);
    bpe::build_byte_unicode_maps(t.byte_to_unicode_, t.unicode_to_byte_);

    for (const auto& [tok, id] : t.vocab_) {
        t.id_to_token_.emplace(id, tok);
    }

    // Built-in Qwen3 specials plus caller extensions. Each is only registered
    // if it's actually present in vocab.json.
    std::vector<std::string> specials = {
        "<|endoftext|>", "<|im_start|>", "<|im_end|>"
    };
    specials.insert(specials.end(),
                    extra_special_tokens.begin(), extra_special_tokens.end());
    for (const auto& s : specials) {
        auto it = t.vocab_.find(s);
        if (it == t.vocab_.end()) continue;
        t.specials_.add(s, it->second);
    }

    auto lookup = [&](const char* s) -> int {
        auto it = t.vocab_.find(s);
        return it != t.vocab_.end() ? it->second : -1;
    };
    t.endoftext_id_ = lookup("<|endoftext|>");
    t.im_start_id_  = lookup("<|im_start|>");
    t.im_end_id_    = lookup("<|im_end|>");
    return t;
}

namespace {

namespace gg = ::brotensor::gguf;

const gg::Value& need_meta(const gg::File& f, const char* key) {
    const gg::Value* v = f.find_meta(key);
    if (!v) throw std::runtime_error(
        std::string("qwen::Tokenizer::from_gguf: missing metadata '") + key + "'");
    return *v;
}

const std::vector<gg::Value>& need_array(const gg::Value& v, const char* key,
                                         gg::ValueType elem) {
    if (v.type != gg::ValueType::Array || v.array_elem_type != elem) {
        throw std::runtime_error(
            std::string("qwen::Tokenizer::from_gguf: metadata '") + key +
            "' is not an array of the expected element type");
    }
    return v.array;
}

}  // namespace

Tokenizer Tokenizer::from_gguf(
    const gg::File& f,
    const std::vector<std::string>& extra_special_tokens) {
    Tokenizer t;
    bpe::build_byte_unicode_maps(t.byte_to_unicode_, t.unicode_to_byte_);

    // Vocab: tokens[i] is the byte-level-encoded string for id i.
    const auto& tokens =
        need_array(need_meta(f, "tokenizer.ggml.tokens"),
                   "tokenizer.ggml.tokens", gg::ValueType::String);
    t.vocab_.reserve(tokens.size());
    t.id_to_token_.reserve(tokens.size());
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        const std::int32_t id = static_cast<std::int32_t>(i);
        t.vocab_.emplace(tokens[i].str, id);
        t.id_to_token_.emplace(id, tokens[i].str);
    }

    // Merges: each entry is "a b" in priority order; lowest index = lowest rank.
    const auto& merges =
        need_array(need_meta(f, "tokenizer.ggml.merges"),
                   "tokenizer.ggml.merges", gg::ValueType::String);
    t.merge_ranks_.reserve(merges.size());
    for (std::size_t i = 0; i < merges.size(); ++i) {
        const std::string& line = merges[i].str;
        const auto sp = line.find(' ');
        if (sp == std::string::npos) {
            throw std::runtime_error(
                "qwen::Tokenizer::from_gguf: malformed merges entry '" + line + "'");
        }
        std::string key = line.substr(0, sp) + '\x01' + line.substr(sp + 1);
        t.merge_ranks_.emplace(std::move(key), static_cast<std::int32_t>(i));
    }

    // Optional token_type array: type 3 (CONTROL) ids become atomic specials.
    if (const auto* tt = f.find_meta("tokenizer.ggml.token_type")) {
        if (tt->type == gg::ValueType::Array &&
            tt->array_elem_type == gg::ValueType::I32) {
            for (std::size_t i = 0; i < tt->array.size() && i < tokens.size(); ++i) {
                if (tt->array[i].scalar.i32 == 3) {
                    t.specials_.add(tokens[i].str, static_cast<std::int32_t>(i));
                }
            }
        }
    }

    // Always make sure the Qwen3 ChatML specials and caller-supplied extras are
    // registered if they exist by name (some converters omit token_type).
    std::vector<std::string> by_name = {
        "<|endoftext|>", "<|im_start|>", "<|im_end|>"};
    by_name.insert(by_name.end(),
                   extra_special_tokens.begin(), extra_special_tokens.end());
    for (const auto& s : by_name) {
        auto it = t.vocab_.find(s);
        if (it == t.vocab_.end()) continue;
        t.specials_.add(s, it->second);
    }

    auto lookup = [&](const char* s) -> int {
        auto it = t.vocab_.find(s);
        return it != t.vocab_.end() ? it->second : -1;
    };
    t.endoftext_id_ = lookup("<|endoftext|>");
    t.im_start_id_  = lookup("<|im_start|>");
    t.im_end_id_    = lookup("<|im_end|>");
    return t;
}

namespace {

namespace j = brolm::detail::json;

std::string slurp_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error(
        "qwen::Tokenizer::from_tokenizer_json: cannot open '" + path + "'");
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

}  // namespace

Tokenizer Tokenizer::from_tokenizer_json(
    const std::string& tokenizer_json_path,
    const std::vector<std::string>& extra_special_tokens) {
    Tokenizer t;
    bpe::build_byte_unicode_maps(t.byte_to_unicode_, t.unicode_to_byte_);

    j::Value root;
    try {
        root = j::parse(slurp_file(tokenizer_json_path));
    } catch (const std::exception& e) {
        throw std::runtime_error(
            std::string("qwen::Tokenizer::from_tokenizer_json: json parse: ") +
            e.what());
    }
    if (!root.is_object()) {
        throw std::runtime_error(
            "qwen::Tokenizer::from_tokenizer_json: root is not a JSON object");
    }

    const j::Value* model = root.find("model");
    if (!model || !model->is_object()) {
        throw std::runtime_error(
            "qwen::Tokenizer::from_tokenizer_json: missing 'model' object");
    }

    // model.vocab: { "token": id, ... }
    const j::Value* vocab = model->find("vocab");
    if (!vocab || !vocab->is_object()) {
        throw std::runtime_error(
            "qwen::Tokenizer::from_tokenizer_json: model.vocab is not an object "
            "(not a byte-level BPE tokenizer.json)");
    }
    const auto& vocab_members = vocab->as_object();
    t.vocab_.reserve(vocab_members.size());
    for (const auto& [tok, idv] : vocab_members) {
        if (!idv.is_number()) continue;
        t.vocab_.emplace(tok, static_cast<int32_t>(idv.as_number()));
    }

    // model.merges: array of "a b" strings, or ["a","b"] pairs (newer HF).
    // Lowest index = highest priority = lowest rank.
    const j::Value* merges = model->find("merges");
    if (merges && merges->is_array()) {
        const auto& arr = merges->as_array();
        t.merge_ranks_.reserve(arr.size());
        for (std::size_t i = 0; i < arr.size(); ++i) {
            std::string a, b;
            if (arr[i].is_array()) {
                const auto& pair = arr[i].as_array();
                if (pair.size() != 2 || !pair[0].is_string() ||
                    !pair[1].is_string()) {
                    throw std::runtime_error(
                        "qwen::Tokenizer::from_tokenizer_json: malformed "
                        "model.merges pair entry");
                }
                a = pair[0].as_string();
                b = pair[1].as_string();
            } else if (arr[i].is_string()) {
                const std::string& line = arr[i].as_string();
                const auto sp = line.find(' ');
                if (sp == std::string::npos) {
                    throw std::runtime_error(
                        "qwen::Tokenizer::from_tokenizer_json: malformed "
                        "model.merges entry '" + line + "'");
                }
                a = line.substr(0, sp);
                b = line.substr(sp + 1);
            } else {
                continue;
            }
            t.merge_ranks_.emplace(a + '\x01' + b, static_cast<int32_t>(i));
        }
    }

    // added_tokens: [{ "id": N, "content": "<|...|>", "special": true }, ...].
    // These carry the control specials (Llama-3's live at 128000..128255) that
    // may not appear in model.vocab. Add each to the vocab so decode() renders
    // it, and register the special-flagged ones as atomic specials.
    if (const j::Value* at = root.find("added_tokens"); at && at->is_array()) {
        for (const auto& tok : at->as_array()) {
            if (!tok.is_object()) continue;
            const j::Value* content = tok.find("content");
            const j::Value* idv = tok.find("id");
            if (!content || !content->is_string() || !idv || !idv->is_number()) {
                continue;
            }
            const std::string& s = content->as_string();
            const int32_t id = static_cast<int32_t>(idv->as_number());
            t.vocab_[s] = id;
            bool special = true;  // added_tokens default to special in HF
            if (const j::Value* sp = tok.find("special"); sp && sp->is_bool()) {
                special = sp->as_bool();
            }
            if (special) t.specials_.add(s, id);
        }
    }

    for (const auto& [tok, id] : t.vocab_) {
        t.id_to_token_.emplace(id, tok);
    }

    // Caller extras + Qwen built-ins (registered only if present in the vocab;
    // for a Llama-3 tokenizer.json none of the Qwen names exist, so these are
    // no-ops and the accessor ids stay -1).
    std::vector<std::string> by_name = {
        "<|endoftext|>", "<|im_start|>", "<|im_end|>"};
    by_name.insert(by_name.end(),
                   extra_special_tokens.begin(), extra_special_tokens.end());
    for (const auto& s : by_name) {
        auto it = t.vocab_.find(s);
        if (it == t.vocab_.end()) continue;
        t.specials_.add(s, it->second);
    }

    auto lookup = [&](const char* s) -> int {
        auto it = t.vocab_.find(s);
        return it != t.vocab_.end() ? it->second : -1;
    };
    t.endoftext_id_ = lookup("<|endoftext|>");
    t.im_start_id_  = lookup("<|im_start|>");
    t.im_end_id_    = lookup("<|im_end|>");
    return t;
}

void Tokenizer::register_special_token(const std::string& token, int32_t id) {
    specials_.add(token, id);
    if (token == "<|endoftext|>")      endoftext_id_ = id;
    else if (token == "<|im_start|>")  im_start_id_  = id;
    else if (token == "<|im_end|>")    im_end_id_    = id;
}

void Tokenizer::encode_piece_(std::string_view piece,
                              std::vector<int32_t>& out) const {
    bpe::encode_piece(piece, byte_to_unicode_, vocab_, merge_ranks_,
                      /*append_end_of_word=*/false, out);
}

std::vector<int32_t> Tokenizer::encode(std::string_view text,
                                       bool add_special) const {
    std::vector<int32_t> ids;
    ids.reserve(text.size());

    // Walk the input, splitting out verbatim special-token substrings and
    // BPE-encoding the spans between them via GPT-2 pre-tokenization.
    bpe::encode_with_specials(
        text, specials_,
        [this](std::string_view span, std::vector<int32_t>& out) {
            for (const auto& p : pre_tokenize(span)) encode_piece_(p, out);
        },
        ids);

    // Qwen3 has no BOS. add_special only appends <|endoftext|> as an EOS hook.
    if (add_special && endoftext_id_ >= 0) {
        ids.push_back(endoftext_id_);
    }
    return ids;
}

std::string Tokenizer::decode(const std::vector<int32_t>& ids) const {
    std::string out;
    std::string encoded;  // pending run of byte-encoded pieces
    auto flush_encoded = [&]() {
        std::size_t i = 0;
        while (i < encoded.size()) {
            uint32_t cp = bpe::next_codepoint(encoded, i);
            auto it = unicode_to_byte_.find(cp);
            if (it != unicode_to_byte_.end()) {
                out += static_cast<char>(it->second);
            }
        }
        encoded.clear();
    };

    for (int32_t id : ids) {
        if (const std::string* sp = specials_.token_for_id(id)) {
            flush_encoded();
            out += *sp;
            continue;
        }
        auto it = id_to_token_.find(id);
        if (it != id_to_token_.end()) {
            encoded += it->second;
        }
    }
    flush_encoded();
    return out;
}

std::string Tokenizer::apply_chat_template(
    const std::vector<std::pair<std::string, std::string>>& messages,
    bool add_generation_prompt) const {
    std::string out;
    for (const auto& [role, content] : messages) {
        out += "<|im_start|>";
        out += role;
        out += '\n';
        out += content;
        out += "<|im_end|>\n";
    }
    if (add_generation_prompt) {
        out += "<|im_start|>assistant\n";
    }
    return out;
}

}  // namespace brolm::qwen
