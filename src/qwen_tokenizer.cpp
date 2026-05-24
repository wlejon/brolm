#include "brolm/qwen_tokenizer.h"

#include "brolm/detail/byte_level_bpe.h"

#include <cstdint>
#include <cstring>
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
        t.special_tokens_.emplace(s, it->second);
        t.special_ids_.emplace(it->second, s);
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
    auto prev = special_tokens_.find(token);
    if (prev != special_tokens_.end()) {
        special_ids_.erase(prev->second);
        special_tokens_.erase(prev);
    }
    special_tokens_.emplace(token, id);
    special_ids_.emplace(id, token);
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
    // BPE-encoding the spans between them.
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
        auto sp = special_ids_.find(id);
        if (sp != special_ids_.end()) {
            flush_encoded();
            out += sp->second;
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
