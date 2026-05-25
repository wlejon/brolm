#include "brolm/whisper_tokenizer.h"

#include "brolm/detail/byte_level_bpe.h"
#include "brotensor/gguf.h"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

namespace brolm::whisper {

namespace bpe = brolm::detail::bpe;

// ─── Pre-tokenization (GPT-2 / Whisper) ────────────────────────────────────
//
// Whisper uses the same GPT-2 byte-level pre-tokenization as Qwen — same
// regex family, same leading-space convention. Kept inline here rather than
// shared with qwen_tokenizer.cpp because the two tokenizers don't otherwise
// have a reason to share a .cpp; the duplicated function is small and the
// alternative (yet another shared detail file) costs more than it saves.

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

// True iff `s` looks like a Whisper special token: "<|...|>" with a non-empty
// inner span and no embedded "|>" — covers <|endoftext|>, <|en|>, <|0.00|>,
// etc., and rejects ordinary vocab pieces that happen to start with "<".
bool looks_like_special(const std::string& s) {
    if (s.size() < 4) return false;
    if (s[0] != '<' || s[1] != '|') return false;
    if (s[s.size() - 2] != '|' || s[s.size() - 1] != '>') return false;
    return true;
}

// Parse the seconds value out of a "<|N.NN|>" timestamp token. Returns true
// on a clean match (and writes the parsed double); false otherwise.
bool parse_timestamp_seconds(const std::string& tok, double& out) {
    if (tok.size() < 6) return false;
    if (tok.compare(0, 2, "<|") != 0) return false;
    if (tok.compare(tok.size() - 2, 2, "|>") != 0) return false;

    const std::string body = tok.substr(2, tok.size() - 4);
    // Body must be digits-and-one-dot.
    std::size_t dot = body.find('.');
    if (dot == std::string::npos) return false;
    if (body.find('.', dot + 1) != std::string::npos) return false;
    for (char c : body) {
        if (c != '.' && (c < '0' || c > '9')) return false;
    }
    try {
        std::size_t consumed = 0;
        double v = std::stod(body, &consumed);
        if (consumed != body.size()) return false;
        out = v;
        return true;
    } catch (...) {
        return false;
    }
}

}  // namespace

// ─── Tokenizer driver ──────────────────────────────────────────────────────

Tokenizer Tokenizer::load(const std::string& vocab_json_path,
                          const std::string& merges_txt_path) {
    Tokenizer t;
    t.vocab_       = bpe::load_vocab_json(vocab_json_path);
    t.merge_ranks_ = bpe::load_merges_txt(merges_txt_path);
    bpe::build_byte_unicode_maps(t.byte_to_unicode_, t.unicode_to_byte_);

    for (const auto& [tok, id] : t.vocab_) {
        t.id_to_token_.emplace(id, tok);
    }

    // Auto-register every "<|...|>" entry as an atomic special, and pull out
    // the well-known ids while we're scanning. Timestamp tokens form a
    // contiguous id range; track the min/max here too.
    int ts_min = -1, ts_max = -1;
    for (const auto& [tok, id] : t.vocab_) {
        if (!looks_like_special(tok)) continue;
        t.special_tokens_.emplace(tok, id);
        t.special_ids_.emplace(id, tok);

        if      (tok == "<|endoftext|>")         t.endoftext_id_     = id;
        else if (tok == "<|startoftranscript|>") t.sot_id_           = id;
        else if (tok == "<|nospeech|>" ||
                 tok == "<|nocaptions|>")        t.no_speech_id_     = id;
        else if (tok == "<|notimestamps|>")      t.no_timestamps_id_ = id;
        else if (tok == "<|transcribe|>")        t.transcribe_id_    = id;
        else if (tok == "<|translate|>")         t.translate_id_     = id;
        else {
            double sec;
            if (parse_timestamp_seconds(tok, sec)) {
                if (ts_min < 0 || id < ts_min) ts_min = id;
                if (ts_max < 0 || id > ts_max) ts_max = id;
            }
        }
    }
    t.first_timestamp_id_ = ts_min;
    t.last_timestamp_id_  = ts_max;
    return t;
}

namespace {

namespace gg = ::brotensor::gguf;

const gg::Value& need_meta(const gg::File& f, const char* key) {
    const gg::Value* v = f.find_meta(key);
    if (!v) throw std::runtime_error(
        std::string("whisper::Tokenizer::from_gguf: missing metadata '") +
        key + "'");
    return *v;
}

}  // namespace

Tokenizer Tokenizer::from_gguf(const gg::File& f) {
    Tokenizer t;
    bpe::build_byte_unicode_maps(t.byte_to_unicode_, t.unicode_to_byte_);

    const auto& tokens_v = need_meta(f, "tokenizer.ggml.tokens");
    if (tokens_v.type != gg::ValueType::Array ||
        tokens_v.array_elem_type != gg::ValueType::String) {
        throw std::runtime_error(
            "whisper::Tokenizer::from_gguf: 'tokenizer.ggml.tokens' is not "
            "an array of strings");
    }
    const auto& tokens = tokens_v.array;
    t.vocab_.reserve(tokens.size());
    t.id_to_token_.reserve(tokens.size());
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        const std::int32_t id = static_cast<std::int32_t>(i);
        t.vocab_.emplace(tokens[i].str, id);
        t.id_to_token_.emplace(id, tokens[i].str);
    }

    const auto& merges_v = need_meta(f, "tokenizer.ggml.merges");
    if (merges_v.type != gg::ValueType::Array ||
        merges_v.array_elem_type != gg::ValueType::String) {
        throw std::runtime_error(
            "whisper::Tokenizer::from_gguf: 'tokenizer.ggml.merges' is not "
            "an array of strings");
    }
    const auto& merges = merges_v.array;
    t.merge_ranks_.reserve(merges.size());
    for (std::size_t i = 0; i < merges.size(); ++i) {
        const std::string& line = merges[i].str;
        const auto sp = line.find(' ');
        if (sp == std::string::npos) {
            throw std::runtime_error(
                "whisper::Tokenizer::from_gguf: malformed merges entry '" +
                line + "'");
        }
        std::string key = line.substr(0, sp) + '\x01' + line.substr(sp + 1);
        t.merge_ranks_.emplace(std::move(key), static_cast<std::int32_t>(i));
    }

    // Auto-register every "<|...|>" entry as a special and cache well-known
    // ids — same logic as load(), but iterating the freshly populated vocab.
    int ts_min = -1, ts_max = -1;
    for (const auto& [tok, id] : t.vocab_) {
        if (!looks_like_special(tok)) continue;
        t.special_tokens_.emplace(tok, id);
        t.special_ids_.emplace(id, tok);

        if      (tok == "<|endoftext|>")         t.endoftext_id_     = id;
        else if (tok == "<|startoftranscript|>") t.sot_id_           = id;
        else if (tok == "<|nospeech|>" ||
                 tok == "<|nocaptions|>")        t.no_speech_id_     = id;
        else if (tok == "<|notimestamps|>")      t.no_timestamps_id_ = id;
        else if (tok == "<|transcribe|>")        t.transcribe_id_    = id;
        else if (tok == "<|translate|>")         t.translate_id_     = id;
        else {
            double sec;
            if (parse_timestamp_seconds(tok, sec)) {
                if (ts_min < 0 || id < ts_min) ts_min = id;
                if (ts_max < 0 || id > ts_max) ts_max = id;
            }
        }
    }
    t.first_timestamp_id_ = ts_min;
    t.last_timestamp_id_  = ts_max;
    return t;
}

int Tokenizer::token_to_id(std::string_view tok) const {
    auto it = vocab_.find(std::string(tok));
    return it != vocab_.end() ? it->second : -1;
}

bool Tokenizer::is_timestamp(int32_t id) const {
    return first_timestamp_id_ >= 0 &&
           id >= first_timestamp_id_ && id <= last_timestamp_id_;
}

float Tokenizer::timestamp_seconds(int32_t id) const {
    // Whisper timestamp granularity is 0.02s (50 tokens per second).
    return 0.02f * static_cast<float>(id - first_timestamp_id_);
}

std::vector<int32_t> Tokenizer::build_prompt(std::string_view language,
                                             std::string_view task,
                                             bool with_timestamps) const {
    if (sot_id_ < 0) {
        throw std::runtime_error(
            "brolm::whisper::Tokenizer: build_prompt: vocab is missing "
            "<|startoftranscript|>");
    }
    std::string lang_tok = "<|";
    lang_tok.append(language);
    lang_tok += "|>";
    auto lang_it = vocab_.find(lang_tok);
    if (lang_it == vocab_.end()) {
        throw std::runtime_error(
            "brolm::whisper::Tokenizer: build_prompt: unknown language tag '" +
            lang_tok + "'");
    }

    int task_id = -1;
    if      (task == "transcribe") task_id = transcribe_id_;
    else if (task == "translate")  task_id = translate_id_;
    else {
        throw std::runtime_error(
            "brolm::whisper::Tokenizer: build_prompt: task must be "
            "'transcribe' or 'translate', got '" + std::string(task) + "'");
    }
    if (task_id < 0) {
        throw std::runtime_error(
            "brolm::whisper::Tokenizer: build_prompt: vocab is missing the "
            "task token for '" + std::string(task) + "'");
    }

    std::vector<int32_t> out;
    out.reserve(4);
    out.push_back(sot_id_);
    out.push_back(lang_it->second);
    out.push_back(task_id);
    if (!with_timestamps) {
        if (no_timestamps_id_ < 0) {
            throw std::runtime_error(
                "brolm::whisper::Tokenizer: build_prompt: vocab is missing "
                "<|notimestamps|>");
        }
        out.push_back(no_timestamps_id_);
    }
    return out;
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

    std::size_t i = 0;
    std::size_t span_start = 0;
    while (i < text.size()) {
        std::size_t best_len = 0;
        int32_t best_id = -1;
        for (const auto& [s, id] : special_tokens_) {
            if (s.size() > best_len &&
                i + s.size() <= text.size() &&
                std::memcmp(text.data() + i, s.data(), s.size()) == 0) {
                best_len = s.size();
                best_id = id;
            }
        }
        if (best_len > 0) {
            if (i > span_start) {
                std::string_view span = text.substr(span_start, i - span_start);
                for (const auto& p : pre_tokenize(span)) encode_piece_(p, ids);
            }
            ids.push_back(best_id);
            i += best_len;
            span_start = i;
        } else {
            ++i;
        }
    }
    if (text.size() > span_start) {
        std::string_view span = text.substr(span_start);
        for (const auto& p : pre_tokenize(span)) encode_piece_(p, ids);
    }

    if (add_special && endoftext_id_ >= 0) {
        ids.push_back(endoftext_id_);
    }
    return ids;
}

std::string Tokenizer::decode(const std::vector<int32_t>& ids,
                              bool skip_special) const {
    std::string out;
    std::string encoded;
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
        auto sp = special_ids_.find(id);
        if (sp != special_ids_.end()) {
            flush_encoded();
            if (!skip_special) out += sp->second;
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

}  // namespace brolm::whisper
