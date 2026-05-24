#pragma once

// Shared GPT-2 / CLIP-family byte-level BPE primitives.
//
// Two byte-level BPE tokenizers live in brolm — brolm::clip::Tokenizer
// (SD1.5's text branch) and brolm::qwen::Tokenizer (Qwen3 / GPT-2 family) —
// and a third (Whisper) is coming. They share the same algorithm family:
// byte-encode each pre-token into the GPT-2 "byte-level unicode" space, then
// run the standard greedy-lowest-rank BPE merge loop, then look the resulting
// pieces up in a vocab.
//
// What differs between consumers is pre-tokenization (CLIP lowercases + drops
// whitespace; GPT-2 preserves a leading space; Whisper adds a few more cases),
// special-token handling, decoding, and one BPE-loop detail (CLIP appends a
// "</w>" end-of-word marker to the last codepoint of the input; GPT-2 does
// not). This header pulls everything except those per-consumer pieces into
// one place so each tokenizer keeps only its actual quirks.

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace brolm::detail::bpe {

// ─── UTF-8 ─────────────────────────────────────────────────────────────────

// Append the UTF-8 encoding of unicode codepoint `cp` to `out`. Codepoints
// above U+10FFFF are encoded as 4 bytes with no clamping — the caller is
// expected to pass only valid codepoints (the byte-level mapping never
// produces values >= U+0200).
void encode_codepoint_utf8(uint32_t cp, std::string& out);

// Decode the UTF-8 codepoint starting at s[i] and advance i past it. Assumes
// the input is well-formed (which it is for byte-encoded BPE pieces).
uint32_t next_codepoint(const std::string& s, std::size_t& i);

// Split a (well-formed) UTF-8 string into its codepoints, each as its own
// UTF-8-encoded substring — the unit of the BPE merge loop.
std::vector<std::string> split_codepoints(const std::string& s);

// ─── GPT-2 byte↔unicode tables ─────────────────────────────────────────────

// Fill `out[b]` with the UTF-8 encoding of the GPT-2 byte-level unicode
// codepoint for byte `b`. Visible printable bytes map to themselves; the rest
// map to codepoints starting at U+0100 in the order they appear, so a space
// (0x20) becomes U+0120 ('Ġ').
void build_byte_to_unicode(std::string out[256]);

// Same as build_byte_to_unicode but also fills the inverse map (codepoint ->
// original byte), which decoders need to recover raw bytes from token strings.
void build_byte_unicode_maps(
    std::string out[256],
    std::unordered_map<uint32_t, unsigned char>& inverse);

// ─── File loaders ──────────────────────────────────────────────────────────

// Parse a Hugging Face vocab.json: a flat object `{"token": id, ...}`. Throws
// std::runtime_error with a "vocab.json: ..." message on I/O or parse failure.
// Strings handle the standard JSON escapes including \uXXXX (BMP only — the
// byte-level mapping has no codepoints outside the BMP).
std::unordered_map<std::string, int32_t>
load_vocab_json(const std::string& path);

// Parse a Hugging Face merges.txt: skips a leading "#version" comment, ignores
// blank lines, and reads one "a b" pair per line in priority order (line 0
// having the highest priority = lowest rank value). Stores keys as
// "a\x01b" where \x01 cannot appear in a byte-encoded token. Throws
// std::runtime_error on I/O or malformed-line failure.
std::unordered_map<std::string, int32_t>
load_merges_txt(const std::string& path);

// ─── BPE merge loop ────────────────────────────────────────────────────────

// Greedy lowest-rank BPE: split `token` into codepoints, then repeatedly
// merge the lowest-ranked adjacent pair until no merge applies. When
// `append_end_of_word` is true, "</w>" is glued onto the last codepoint
// before the loop (CLIP convention); when false, the codepoints are left
// alone (GPT-2/Qwen/Whisper convention).
std::vector<std::string> bpe_merge(
    const std::string& token,
    const std::unordered_map<std::string, int32_t>& merge_ranks,
    bool append_end_of_word);

// Byte-encode `piece` into GPT-2 unicode space, run bpe_merge, then look up
// each resulting unit in `vocab` and append the ids to `out`. Units that
// miss the vocab are silently dropped — with a complete byte-level vocab
// this never happens because every single-byte codepoint is itself in vocab.
void encode_piece(
    std::string_view piece,
    const std::string byte_to_unicode[256],
    const std::unordered_map<std::string, int32_t>& vocab,
    const std::unordered_map<std::string, int32_t>& merge_ranks,
    bool append_end_of_word,
    std::vector<int32_t>& out);

// ─── ASCII char-class helpers (pre-tokenizers reuse these) ─────────────────

inline bool is_ascii_letter(unsigned char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}
inline bool is_ascii_digit(unsigned char c) { return c >= '0' && c <= '9'; }
inline bool is_ascii_space(unsigned char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
}

// Compare s[i..i+len(lit)) against literal `lit`. Returns false if the
// remaining input is too short.
bool starts_with(std::string_view s, std::size_t i, const char* lit);

}  // namespace brolm::detail::bpe
