#pragma once

// Unicode property lookups and NFC normalization for the byte-level BPE
// pre-tokenizers.
//
// Hugging Face's GPT-2-family tokenizers (Qwen2/Qwen3, Llama-3) split text
// with a regex over Unicode properties — \p{L}, \p{N}, \s — after an NFC
// normalizer, so matching their ids on non-ASCII text needs the same
// classification and the same normalization. This header provides exactly
// that much of Unicode, table-driven from src/detail/unicode_tables.h (which
// scripts/gen_unicode_tables.py generates from CPython's unicodedata; the
// Unicode version is recorded in version()).
//
// Decoding is lenient: an invalid UTF-8 byte decodes to the surrogate-escape
// code point U+DC00 + byte, which every classifier treats as "other" and
// encode_utf8 turns back into the raw byte, so a tokenizer round-trips
// arbitrary bytes rather than throwing.

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace brolm::detail::unicode {

// Unicode database version the tables were generated from, e.g. "16.0.0".
const char* version();

// ─── UTF-8 ─────────────────────────────────────────────────────────────────

// Decode the code point at s[i] and advance i past it. A malformed sequence
// consumes one byte and yields U+DC00 + that byte.
uint32_t decode_utf8(std::string_view s, std::size_t& i);

// Append the UTF-8 encoding of `cp`; a surrogate-escape code point (U+DC80..
// U+DCFF) appends the raw byte it stands for.
void encode_utf8(uint32_t cp, std::string& out);

// ─── Properties ────────────────────────────────────────────────────────────

bool is_letter(uint32_t cp);       // general category L*  (\p{L})
bool is_number(uint32_t cp);       // general category N*  (\p{N})
bool is_white_space(uint32_t cp);  // White_Space property (\s)
uint8_t combining_class(uint32_t cp);  // Canonical_Combining_Class

// ─── NFC ───────────────────────────────────────────────────────────────────

// UAX #15 quick check: true when the string is certainly already in NFC.
// False means "No" or "Maybe" — call nfc() to find out. All-ASCII input is
// always true.
bool is_nfc(std::string_view s);

// Full NFC normalization (canonical decomposition, canonical ordering,
// canonical composition, Hangul algorithmic). Bytes that are not valid UTF-8
// pass through unchanged.
std::string nfc(std::string_view s);

// NFC over a decoded code-point sequence, in place.
void nfc(std::vector<uint32_t>& cps);

}  // namespace brolm::detail::unicode
