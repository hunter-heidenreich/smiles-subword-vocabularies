"""Generate the base-glyph inventory table (appendix).

Enumerates the bare Smirk base (``SmirkAdapter.atomic()``, the $165$-token
OpenSMILES regex with no merges), groups every glyph by its OpenSMILES role, and
emits ``results/tables/base_glyphs.tex``. The grouping is curated here, but the
*membership* is read live from the tokenizer and asserted to partition the base
exactly, so the table cannot silently drift from the installed base.

Usage::

    uv run python results/build/table_base_glyphs.py
"""

from __future__ import annotations

import sys

import latex

from smiles_subword.paths import RESULTS_TABLES_DIR
from smiles_subword.tokenize.adapters.smirk import SmirkAdapter

OUT = RESULTS_TABLES_DIR / "base_glyphs.tex"

# Curated OpenSMILES-role groups. Element symbols are not listed exhaustively
# here: the aliphatic organic subset is pinned (it is the bare-writable set),
# the aromatic symbols are pinned, and every remaining element symbol falls into
# "other element symbols" automatically. Bonds/structure/charge/chirality/
# specials are pinned glyph-for-glyph. Coverage is asserted against the live base.
ORGANIC = ("B", "C", "N", "O", "P", "S", "F", "Cl", "Br", "I")
AROMATIC = ("b", "c", "n", "o", "p", "s", "se", "as")
BONDS = ("-", "=", "#", "$", ":", "/", "\\")
STRUCTURE = ("(", ")", "[", "]", ".", "%", *(str(d) for d in range(10)))
CHARGE_WILD = ("+", "*")
CHIRALITY = ("@", "@@", "@TH", "@AL", "@OH", "@SP", "@TB")
SPECIALS = ("[UNK]", "[BOS]", "[EOS]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")


def _base_vocab() -> set[str]:
    return set(SmirkAdapter.atomic().hf_tokenizer.get_vocab())


def main() -> int:
    vocab = _base_vocab()

    pinned = {
        *ORGANIC,
        *AROMATIC,
        *BONDS,
        *STRUCTURE,
        *CHARGE_WILD,
        *CHIRALITY,
        *SPECIALS,
    }
    missing = pinned - vocab
    if missing:
        raise AssertionError(f"pinned glyphs absent from the base: {sorted(missing)}")

    # Everything not pinned to a syntactic role is an element symbol; the bare
    # organic subset is split out from the bracket-only remainder.
    non_element = {*BONDS, *STRUCTURE, *CHARGE_WILD, *CHIRALITY, *SPECIALS, *AROMATIC}
    elements = vocab - non_element
    other_elements = tuple(sorted(elements - set(ORGANIC)))

    groups: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("Aliphatic organic-subset atoms", ORGANIC),
        ("Aromatic atoms", AROMATIC),
        ("Other element symbols (bracket atoms)", other_elements),
        ("Bonds", BONDS),
        ("Branches, rings, ring-closures", STRUCTURE),
        ("Charge and wildcard", CHARGE_WILD),
        ("Chirality", CHIRALITY),
        ("Special tokens", SPECIALS),
    )

    # The groups must partition the base exactly: no overlaps, full coverage.
    flat = [t for _, toks in groups for t in toks]
    if len(flat) != len(set(flat)):
        raise AssertionError("base-glyph groups overlap")
    if set(flat) != vocab:
        raise AssertionError(
            f"groups do not partition the base "
            f"(extra={sorted(set(flat) - vocab)}, missing={sorted(vocab - set(flat))})"
        )

    n_specials = len(SPECIALS)
    n_glyphs = len(vocab) - n_specials
    OUT.write_text(
        latex.render_base_glyphs(groups, n_glyphs=n_glyphs, n_specials=n_specials)
    )
    print(f"[base] wrote {OUT} ({len(vocab)} tokens: {n_glyphs} glyphs + {n_specials})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
