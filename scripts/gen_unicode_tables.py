#!/usr/bin/env python3
# Generates src/detail/unicode_tables.h — the Unicode property tables behind
# brolm::detail::unicode (include/brolm/detail/unicode.h): general categories
# L and N, the White_Space property, and the NFC data (canonical combining
# classes, full canonical decompositions, primary composites, NFC quick-check).
#
# The tables come from Python's own `unicodedata` module, so the Unicode
# version is whatever the running interpreter carries (printed into the header
# comment). Hugging Face `tokenizers` 0.22 (Oniguruma, unicode-normalization)
# and CPython 3.14 both carry Unicode 16.0; regenerate with a matching Python
# when tokenizers moves to a newer database.
#
# Offline, one-off: never run at build or run time. The output is checked in.
#
# Usage: python scripts/gen_unicode_tables.py [out-path]
from __future__ import annotations

import os
import sys
import unicodedata

MAX_CP = 0x110000

# Hangul syllables decompose algorithmically (UAX #15 §3.12); they are never
# put in the decomposition / composition tables.
S_BASE, S_COUNT = 0xAC00, 11172
L_BASE, V_BASE, T_BASE = 0x1100, 0x1161, 0x11A7
L_COUNT, V_COUNT, T_COUNT = 19, 21, 28


def is_hangul_syllable(cp: int) -> bool:
    return S_BASE <= cp < S_BASE + S_COUNT


def ranges(pred) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    lo = None
    for cp in range(MAX_CP):
        if pred(cp):
            if lo is None:
                lo = cp
        elif lo is not None:
            out.append((lo, cp - 1))
            lo = None
    if lo is not None:
        out.append((lo, MAX_CP - 1))
    return out


def value_ranges(fn) -> list[tuple[int, int, int]]:
    """Runs of consecutive code points sharing the same nonzero fn() value."""
    out: list[tuple[int, int, int]] = []
    lo = None
    cur = 0
    for cp in range(MAX_CP):
        v = fn(cp)
        if v != cur:
            if cur != 0:
                out.append((lo, cp - 1, cur))
            lo, cur = cp, v
    if cur != 0:
        out.append((lo, MAX_CP - 1, cur))
    return out


def category(cp: int) -> str:
    return unicodedata.category(chr(cp))


# The White_Space property (PropList.txt). unicodedata has no direct accessor;
# this is the standard list, stable since Unicode 6.3 (U+180E left it then).
WHITE_SPACE = [(0x09, 0x0D), (0x20, 0x20), (0x85, 0x85), (0xA0, 0xA0),
               (0x1680, 0x1680), (0x2000, 0x200A), (0x2028, 0x2029),
               (0x202F, 0x202F), (0x205F, 0x205F), (0x3000, 0x3000)]


def canonical_decomposition(cp: int) -> list[int] | None:
    """One-level canonical decomposition, or None (compat mappings excluded)."""
    if is_hangul_syllable(cp):
        return None
    d = unicodedata.decomposition(chr(cp))
    if not d or d.startswith("<"):
        return None
    return [int(x, 16) for x in d.split()]


def main() -> None:
    out_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "src", "detail",
        "unicode_tables.h")

    letters = ranges(lambda cp: category(cp)[0] == "L")
    numbers = ranges(lambda cp: category(cp)[0] == "N")
    ccc = value_ranges(lambda cp: unicodedata.combining(chr(cp)))

    # Full canonical decompositions (NFD of the single code point), for every
    # code point carrying a canonical mapping. Hangul is algorithmic.
    decomp: list[tuple[int, list[int]]] = []
    for cp in range(MAX_CP):
        if canonical_decomposition(cp) is None:
            continue
        nfd = [ord(c) for c in unicodedata.normalize("NFD", chr(cp))]
        decomp.append((cp, nfd))

    # Primary composites: a two-element canonical decomposition that NFC
    # recomposes into the same code point (this drops the composition
    # exclusions and the non-starter decompositions, which never recompose).
    compose: list[tuple[int, int, int]] = []
    primary = set()
    for cp in range(MAX_CP):
        d = canonical_decomposition(cp)
        if d is None or len(d) != 2:
            continue
        if unicodedata.normalize("NFC", chr(cp)) != chr(cp):
            continue
        compose.append((d[0], d[1], cp))
        primary.add(cp)
    compose.sort()

    # NFC quick check (UAX #15 §9): No = never in NFC output; Maybe = may
    # combine with a preceding character. Derived rather than read from
    # DerivedNormalizationProps.txt: No is a canonical mapping that does not
    # recompose to itself; Maybe is the second element of a primary composite
    # (plus the Hangul V/T jamo, whose composites are algorithmic).
    qc_no = set()
    for cp, _ in decomp:
        if unicodedata.normalize("NFC", chr(cp)) != chr(cp):
            qc_no.add(cp)
    qc_maybe = set(second for _, second, _ in compose)
    qc_maybe.update(range(V_BASE, V_BASE + V_COUNT))
    qc_maybe.update(range(T_BASE + 1, T_BASE + T_COUNT))
    no_ranges = ranges(lambda cp: cp in qc_no)
    maybe_ranges = ranges(lambda cp: cp in qc_maybe)

    pool: list[int] = []
    decomp_index: list[tuple[int, int, int]] = []
    for cp, seq in decomp:
        decomp_index.append((cp, len(pool), len(seq)))
        pool.extend(seq)
    assert len(pool) < 65536, "decomposition pool no longer fits uint16 offsets"
    assert max(len(seq) for _, seq in decomp) <= 255

    lines: list[str] = []
    w = lines.append
    w("// GENERATED by scripts/gen_unicode_tables.py — do not edit by hand.")
    w("//")
    w("// Unicode property tables for brolm::detail::unicode, produced from")
    w(f"// CPython {sys.version.split()[0]}'s unicodedata module, Unicode "
      f"{unicodedata.unidata_version}.")
    w("// Regenerate with a Python whose database matches the Hugging Face")
    w("// `tokenizers` build being matched (both carry Unicode 16.0 today).")
    w("#pragma once")
    w("")
    w("#include <cstdint>")
    w("")
    w("namespace brolm::detail::unicode::tables {")
    w("")
    w(f'inline constexpr const char* kUnicodeVersion = "{unicodedata.unidata_version}";')
    w("")
    w("struct Range { uint32_t lo, hi; };            // inclusive")
    w("struct CccRange { uint32_t lo, hi; uint8_t ccc; };")
    w("struct Decomp { uint32_t cp; uint16_t offset; uint8_t len; };")
    w("struct Compose { uint32_t first, second, composite; };")
    w("")

    def emit_ranges(name: str, rs, comment: str) -> None:
        w(f"// {comment} ({len(rs)} ranges)")
        w(f"inline constexpr Range {name}[] = {{")
        row = []
        for lo, hi in rs:
            row.append(f"{{0x{lo:04X},0x{hi:04X}}}")
            if len(row) == 6:
                w("    " + ",".join(row) + ",")
                row = []
        if row:
            w("    " + ",".join(row) + ",")
        w("};")
        w("")

    emit_ranges("kLetter", letters, "General category L (\\p{L})")
    emit_ranges("kNumber", numbers, "General category N (\\p{N})")
    emit_ranges("kWhiteSpace", WHITE_SPACE, "White_Space property (the regex crate's / Oniguruma's \\s)")
    emit_ranges("kNfcQcNo", no_ranges, "NFC_Quick_Check = No")
    emit_ranges("kNfcQcMaybe", maybe_ranges, "NFC_Quick_Check = Maybe")

    w(f"// Canonical_Combining_Class, nonzero runs only ({len(ccc)} ranges)")
    w("inline constexpr CccRange kCcc[] = {")
    row = []
    for lo, hi, v in ccc:
        row.append(f"{{0x{lo:04X},0x{hi:04X},{v}}}")
        if len(row) == 5:
            w("    " + ",".join(row) + ",")
            row = []
    if row:
        w("    " + ",".join(row) + ",")
    w("};")
    w("")

    w(f"// Full canonical decompositions (NFD), sorted by code point "
      f"({len(decomp_index)} entries); Hangul syllables are algorithmic")
    w("inline constexpr Decomp kDecomp[] = {")
    row = []
    for cp, off, n in decomp_index:
        row.append(f"{{0x{cp:04X},{off},{n}}}")
        if len(row) == 5:
            w("    " + ",".join(row) + ",")
            row = []
    if row:
        w("    " + ",".join(row) + ",")
    w("};")
    w("")
    w(f"inline constexpr uint32_t kDecompPool[] = {{  // {len(pool)} code points")
    row = []
    for cp in pool:
        row.append(f"0x{cp:04X}")
        if len(row) == 10:
            w("    " + ",".join(row) + ",")
            row = []
    if row:
        w("    " + ",".join(row) + ",")
    w("};")
    w("")

    w(f"// Primary composites, sorted by (first, second) ({len(compose)} pairs)")
    w("inline constexpr Compose kCompose[] = {")
    row = []
    for a, b, c in compose:
        row.append(f"{{0x{a:04X},0x{b:04X},0x{c:04X}}}")
        if len(row) == 4:
            w("    " + ",".join(row) + ",")
            row = []
    if row:
        w("    " + ",".join(row) + ",")
    w("};")
    w("")
    w("}  // namespace brolm::detail::unicode::tables")

    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {out_path}: Unicode {unicodedata.unidata_version}, "
          f"{len(letters)} L ranges, {len(numbers)} N ranges, {len(ccc)} ccc runs, "
          f"{len(decomp_index)} decompositions ({len(pool)} pool), "
          f"{len(compose)} composites, QC no/maybe {len(no_ranges)}/{len(maybe_ranges)}")


if __name__ == "__main__":
    main()
