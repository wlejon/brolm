#include "brolm/qwen_tokenizer.h"

#include "brolm/detail/byte_level_bpe.h"
#include "brolm/detail/json.h"
#include "brolm/detail/unicode.h"
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
namespace j = brolm::detail::json;

// ─── BPE delegation ────────────────────────────────────────────────────────

std::vector<std::string> Tokenizer::bpe_(const std::string& token) const {
    // GPT-2/Qwen variant: no "</w>" end-of-word marker.
    return bpe::bpe_merge(token, merge_ranks_, /*append_end_of_word=*/false);
}

// ─── Pre-tokenization (GPT-2 / Qwen2 / Llama-3) ────────────────────────────
//
// The Hugging Face `tokenizers` Split regex these families share, run over
// code points with Unicode properties (\p{L}, \p{N}, and \s = White_Space):
//
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)      contraction after an ASCII apostrophe
//   | [^\r\n\p{L}\p{N}]?\p{L}+        one optional non-letter/digit lead-in
//                                      (a space, tab, NBSP, CJK stop, ...),
//                                      then a letter run
//   | \p{N}                           one digit-like code point each (Qwen);
//                                      Llama-3 has \p{N}{1,3}
//   |  ?[^\s\p{L}\p{N}]+[\r\n]*       optional space, an "other" run, then
//                                      any trailing CR/LF
//   | \s*[\r\n]+                      whitespace ending in the run's last
//                                      newline
//   | \s+(?!\S)                       a whitespace run followed by non-space
//                                      keeps its last char for the next piece
//   | \s+                             the remaining whitespace
//
// Alternatives are tried in this order at every position, as a backtracking
// engine would; each branch below is the leftmost alternative that matches,
// with the backtracking cases worked out by hand. The (?i) uses Unicode simple
// case folding, whose only non-ASCII member for these letters is U+017F LATIN
// SMALL LETTER LONG S (folds to 's').

namespace {

namespace uni = brolm::detail::unicode;

// Lowercase the ASCII letters and the long s; everything else is unchanged.
uint32_t fold_contraction_char(uint32_t cp) {
    if (cp >= 'A' && cp <= 'Z') return cp + ('a' - 'A');
    if (cp == 0x017F) return 's';
    return cp;
}

bool is_newline(uint32_t cp) { return cp == '\r' || cp == '\n'; }

// Neither white space nor a letter nor a number: [^\s\p{L}\p{N}].
bool is_other(uint32_t cp) {
    return !uni::is_white_space(cp) && !uni::is_letter(cp) && !uni::is_number(cp);
}

// Split `text` into the regex's pre-tokens, each a view into `text`. Digit
// runs are capped at `digit_run_max` code points (1 for Qwen, 3 for Llama-3).
std::vector<std::string_view> pre_tokenize(std::string_view text,
                                           int digit_run_max) {
    // Decode once; offs[k] is the byte offset of code point k, offs[n] the end.
    std::vector<uint32_t> cps;
    std::vector<std::size_t> offs;
    cps.reserve(text.size());
    offs.reserve(text.size() + 1);
    for (std::size_t i = 0; i < text.size();) {
        offs.push_back(i);
        cps.push_back(uni::decode_utf8(text, i));
    }
    offs.push_back(text.size());
    const std::size_t n = cps.size();

    std::vector<std::string_view> pieces;
    auto emit = [&](std::size_t a, std::size_t b) {
        pieces.push_back(text.substr(offs[a], offs[b] - offs[a]));
    };

    std::size_t i = 0;
    while (i < n) {
        const uint32_t c = cps[i];

        // (?i:'s|'t|'re|'ve|'m|'ll|'d)
        if (c == '\'' && i + 1 < n) {
            const uint32_t d = fold_contraction_char(cps[i + 1]);
            if (d == 's' || d == 't' || d == 'm' || d == 'd') {
                emit(i, i + 2);
                i += 2;
                continue;
            }
            if (i + 2 < n) {
                const uint32_t e = fold_contraction_char(cps[i + 2]);
                if ((d == 'r' && e == 'e') || (d == 'v' && e == 'e') ||
                    (d == 'l' && e == 'l')) {
                    emit(i, i + 3);
                    i += 3;
                    continue;
                }
            }
        }

        // [^\r\n\p{L}\p{N}]?\p{L}+ — with the optional lead-in, the letter run
        // must start at i+1; without it, at i. Both cannot hold at once, so
        // there is nothing to backtrack over.
        {
            std::size_t j = i;
            if (!uni::is_letter(c) && !uni::is_number(c) && !is_newline(c)) ++j;
            if (j < n && uni::is_letter(cps[j])) {
                std::size_t k = j + 1;
                while (k < n && uni::is_letter(cps[k])) ++k;
                emit(i, k);
                i = k;
                continue;
            }
        }

        // \p{N} (or \p{N}{1,3})
        if (uni::is_number(c)) {
            std::size_t k = i + 1;
            while (k < n && k - i < static_cast<std::size_t>(digit_run_max) &&
                   uni::is_number(cps[k])) ++k;
            emit(i, k);
            i = k;
            continue;
        }

        //  ?[^\s\p{L}\p{N}]+[\r\n]*
        {
            std::size_t j = i;
            if (c == ' ') ++j;
            if (j < n && is_other(cps[j])) {
                std::size_t k = j + 1;
                while (k < n && is_other(cps[k])) ++k;
                while (k < n && is_newline(cps[k])) ++k;
                emit(i, k);
                i = k;
                continue;
            }
        }

        // The remaining alternatives all start with a whitespace run.
        if (!uni::is_white_space(c)) {
            emit(i, i + 1);  // unreachable: every code point is in some class
            ++i;
            continue;
        }
        std::size_t run_end = i + 1;
        while (run_end < n && uni::is_white_space(cps[run_end])) ++run_end;

        // \s*[\r\n]+ — greedy \s* backs off to the last CR/LF in the run, so
        // the piece is the run up to and including that newline.
        {
            std::size_t after_last_nl = 0;
            for (std::size_t k = run_end; k > i; --k) {
                if (is_newline(cps[k - 1])) { after_last_nl = k; break; }
            }
            if (after_last_nl != 0) {
                emit(i, after_last_nl);
                i = after_last_nl;
                continue;
            }
        }

        // \s+(?!\S) — at end of input the whole run; before a non-space the
        // run minus its last char (which then leads the next piece), unless
        // that leaves nothing, in which case \s+ takes the single char.
        if (run_end == n) {
            emit(i, run_end);
        } else if (run_end - i > 1) {
            emit(i, run_end - 1);
            i = run_end - 1;
            continue;
        } else {
            emit(i, run_end);  // \s+
        }
        i = run_end;
    }
    return pieces;
}

// Read the pre-tokenizer conventions out of a tokenizer.json: whether an NFC
// normalizer runs first, and how long a digit run the Split regex allows.
// Anything unrecognised keeps the Qwen2 defaults.
void read_pre_tokenizer_config(const j::Value& root, bool& nfc, int& digit_run_max) {
    if (const j::Value* norm = root.find("normalizer"); norm && norm->is_object()) {
        auto is_nfc = [](const j::Value& v) {
            const j::Value* t = v.find("type");
            return t && t->is_string() && t->as_string() == "NFC";
        };
        nfc = is_nfc(*norm);
        if (const j::Value* seq = norm->find("normalizers"); seq && seq->is_array()) {
            for (const auto& v : seq->as_array()) {
                if (v.is_object() && is_nfc(v)) nfc = true;
            }
        }
    } else if (norm && norm->is_null()) {
        nfc = false;
    }

    const j::Value* pre = root.find("pre_tokenizer");
    if (!pre || !pre->is_object()) return;
    auto scan_split = [&](const j::Value& v) {
        const j::Value* t = v.find("type");
        if (!t || !t->is_string() || t->as_string() != "Split") return;
        const j::Value* pat = v.find("pattern");
        if (!pat || !pat->is_object()) return;
        const j::Value* re = pat->find("Regex");
        if (!re || !re->is_string()) return;
        const auto& s = re->as_string();
        if (s.find("\\p{N}{1,3}") != std::string::npos || s.find("{1,3}") != std::string::npos) {
            digit_run_max = 3;
        }
    };
    scan_split(*pre);
    if (const j::Value* seq = pre->find("pretokenizers"); seq && seq->is_array()) {
        for (const auto& v : seq->as_array()) {
            if (v.is_object()) scan_split(v);
        }
    }
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
    // it, and register every one as an atomic special: HF extracts all added
    // tokens verbatim before the model runs, the special flag only governing
    // skip_special_tokens on decode — Qwen's <think> / <tool_call> are
    // "special": false yet still encode to their single id.
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
            t.specials_.add(s, id);
        }
    }

    for (const auto& [tok, id] : t.vocab_) {
        t.id_to_token_.emplace(id, tok);
    }

    // Normalizer + Split-regex conventions: Qwen2/Qwen3 files carry an NFC
    // normalizer and split digits singly; Llama-3's has no normalizer and
    // \p{N}{1,3}. Absent blocks keep the Qwen2 defaults.
    read_pre_tokenizer_config(root, t.normalize_nfc_, t.digit_run_max_);

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
    // BPE-encoding the spans between them: NFC first (HF matches specials on
    // the raw text and normalizes the segments between, which is the same
    // order), then the Unicode-property pre-tokenizer.
    bpe::encode_with_specials(
        text, specials_,
        [this](std::string_view span, std::vector<int32_t>& out) {
            std::string normalized;
            if (normalize_nfc_ && !uni::is_nfc(span)) {
                normalized = uni::nfc(span);
                span = normalized;
            }
            for (const auto p : pre_tokenize(span, digit_run_max_)) {
                encode_piece_(p, out);
            }
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
