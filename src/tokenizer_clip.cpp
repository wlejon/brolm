#include "brolm/tokenizer.h"

#include "brolm/detail/byte_level_bpe.h"

#include <cstdint>
#include <string>

namespace brolm::clip {

namespace bpe = brolm::detail::bpe;

// ─── BPE delegation ────────────────────────────────────────────────────────

std::vector<std::string> Tokenizer::bpe_(const std::string& token) const {
    // CLIP variant: append a "</w>" end-of-word marker to the last codepoint.
    return bpe::bpe_merge(token, merge_ranks_, /*append_end_of_word=*/true);
}

// ─── Pre-tokenization (CLIP-specific) ──────────────────────────────────────
//
// HF CLIP regex (simplified):
//   <|startoftext|> | <|endoftext|>
//   | 's | 't | 're | 've | 'm | 'll | 'd
//   | letters+ | digit | non-whitespace-non-letter-non-digit+
//
// ASCII-focused; non-ASCII bytes fall into the "punctuation run" category
// (still byte-encoded correctly, just not Unicode-chunked the way HF does it).
// Differs from GPT-2 in two ways: lowercase, and whitespace is dropped (not
// folded into the following pre-token).

namespace {

// Lowercase ASCII in place; non-ASCII bytes pass through.
std::string ascii_lower(std::string_view s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        unsigned char u = static_cast<unsigned char>(c);
        if (u >= 'A' && u <= 'Z') u = u - 'A' + 'a';
        out += static_cast<char>(u);
    }
    return out;
}

// Collapse runs of whitespace to a single space, strip leading/trailing.
// Mirrors HF's `_clean_text` for the SD1.5 path.
std::string collapse_ws(std::string_view s) {
    std::string out;
    out.reserve(s.size());
    bool prev_space = true;
    for (char c : s) {
        if (bpe::is_ascii_space(static_cast<unsigned char>(c))) {
            if (!prev_space) out += ' ';
            prev_space = true;
        } else {
            out += c;
            prev_space = false;
        }
    }
    if (!out.empty() && out.back() == ' ') out.pop_back();
    return out;
}

std::vector<std::string> pre_tokenize(std::string_view text) {
    std::vector<std::string> pieces;
    std::size_t i = 0;
    while (i < text.size()) {
        unsigned char c = static_cast<unsigned char>(text[i]);
        if (bpe::is_ascii_space(c)) { ++i; continue; }

        if (c == '\'') {
            if (bpe::starts_with(text, i, "'re")) { pieces.emplace_back("'re"); i += 3; continue; }
            if (bpe::starts_with(text, i, "'ve")) { pieces.emplace_back("'ve"); i += 3; continue; }
            if (bpe::starts_with(text, i, "'ll")) { pieces.emplace_back("'ll"); i += 3; continue; }
            if (bpe::starts_with(text, i, "'s"))  { pieces.emplace_back("'s");  i += 2; continue; }
            if (bpe::starts_with(text, i, "'t"))  { pieces.emplace_back("'t");  i += 2; continue; }
            if (bpe::starts_with(text, i, "'m"))  { pieces.emplace_back("'m");  i += 2; continue; }
            if (bpe::starts_with(text, i, "'d"))  { pieces.emplace_back("'d");  i += 2; continue; }
        }

        if (bpe::is_ascii_letter(c)) {
            std::size_t j = i;
            while (j < text.size() &&
                   bpe::is_ascii_letter(static_cast<unsigned char>(text[j]))) ++j;
            pieces.emplace_back(text.substr(i, j - i));
            i = j;
            continue;
        }

        if (bpe::is_ascii_digit(c)) {
            // HF emits one digit at a time.
            pieces.emplace_back(text.substr(i, 1));
            i += 1;
            continue;
        }

        // Punct run: non-whitespace, non-letter, non-digit bytes. Break on
        // contraction lookahead so "don't" splits cleanly into "don" / "'t".
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
        pieces.emplace_back(text.substr(i, j - i));
        i = j;
    }
    return pieces;
}

}  // namespace

// ─── Tokenizer driver ──────────────────────────────────────────────────────

Tokenizer Tokenizer::load(const std::string& vocab_json_path,
                          const std::string& merges_txt_path) {
    Tokenizer t;
    t.vocab_       = bpe::load_vocab_json(vocab_json_path);
    t.merge_ranks_ = bpe::load_merges_txt(merges_txt_path);
    bpe::build_byte_to_unicode(t.byte_to_unicode_);
    return t;
}

void Tokenizer::encode_piece_(std::string_view piece,
                              std::vector<int32_t>& out) const {
    bpe::encode_piece(piece, byte_to_unicode_, vocab_, merge_ranks_,
                      /*append_end_of_word=*/true, out);
}

std::vector<int32_t> Tokenizer::tokenize(std::string_view text) const {
    std::string cleaned = collapse_ws(ascii_lower(text));
    auto pieces = pre_tokenize(cleaned);
    std::vector<int32_t> ids;
    ids.reserve(pieces.size() * 2);
    for (const auto& p : pieces) encode_piece_(p, ids);
    return ids;
}

std::vector<int32_t> Tokenizer::encode(std::string_view text) const {
    auto ids = tokenize(text);

    std::vector<int32_t> out;
    out.reserve(max_length);
    out.push_back(bos_id);

    const std::size_t content_cap = static_cast<std::size_t>(max_length) - 2;
    bool truncated = ids.size() > content_cap;
    std::size_t n = truncated ? content_cap : ids.size();
    for (std::size_t i = 0; i < n; ++i) out.push_back(ids[i]);

    out.push_back(eos_id);
    while (out.size() < static_cast<std::size_t>(max_length)) out.push_back(pad_id);
    return out;
}

}  // namespace brolm::clip
