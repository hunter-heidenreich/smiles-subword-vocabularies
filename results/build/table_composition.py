"""Generate the substructure-composition table (appendix).

Classifies each corpus's shared / BPE-only / Unigram-LM-only multi-glyph pieces
(matched pairs at V=1024, NMB) by substructure type, to show that aromatic-ring
pieces are a BPE specialty the Unigram-LM arm almost never forms. Pieces are read
from the committed tokenizer artifacts via the Jaccard runner's
``glyph_tuple_map``. Emits the full appendix table ``results/tables/composition.tex``.

Usage::

    uv run python results/build/table_composition.py
"""

from __future__ import annotations

import sys
from collections import Counter
from typing import TYPE_CHECKING

import latex
from _pieces import multi_glyph_set

from smiles_subword.paths import RESULTS_TABLES_DIR

if TYPE_CHECKING:
    from smiles_subword.tokenize.measure.jaccard import GlyphTuple

VOCAB_SIZE = 1024
BOUNDARY = "nmb"
BOUNDARY_ABBR = "NMB"
OUT = RESULTS_TABLES_DIR / "composition.tex"

CORPORA = (("pubchem", "PubChem"), ("zinc22", "ZINC-22"), ("coconut", "COCONUT"))
AROMATIC = frozenset({"b", "c", "n", "o", "p", "s", "se", "as"})
UNSATURATION = frozenset({"=", "#", "$"})
# Class order matches the rendered columns: saturated C, unsaturated C,
# aromatic, heteroatom.
CLASSES = ("sat-C", "unsat-C", "aromatic", "heteroatom")


def _classify(piece: GlyphTuple) -> str:
    """Priority partition: aromatic > heteroatom > unsaturated-C > saturated-C.

    Bracket-internal pieces (any glyph with ``[``) do not occur under NMB.
    """
    glyphs = set(piece)
    if glyphs & AROMATIC:
        return "aromatic"
    if any(g[0].isupper() and g != "C" for g in piece):
        return "heteroatom"
    if glyphs & UNSATURATION:
        return "unsat-C"
    return "sat-C"


def _percentages(pieces: set[GlyphTuple]) -> tuple[int, tuple[int, int, int, int]]:
    counts = Counter(_classify(t) for t in pieces)
    n = len(pieces)
    pct = tuple(round(100 * counts[c] / n) for c in CLASSES)
    return n, pct  # type: ignore[return-value]


def main() -> int:
    blocks: list[tuple[str, list[tuple[str, int, tuple[int, int, int, int]]]]] = []
    for corpus, corpus_label in CORPORA:
        bpe = multi_glyph_set(corpus, VOCAB_SIZE, BOUNDARY, "bpe")
        ul = multi_glyph_set(corpus, VOCAB_SIZE, BOUNDARY, "unigram")
        rows: list[tuple[str, int, tuple[int, int, int, int]]] = []
        for bucket_label, pieces in (
            ("Shared", bpe & ul),
            ("BPE-only", bpe - ul),
            ("Unigram-LM-only", ul - bpe),
        ):
            n, pct = _percentages(pieces)
            rows.append((bucket_label, n, pct))
            by_class = dict(zip(CLASSES, pct, strict=True))
            print(f"[composition] {corpus} {bucket_label}: n={n} {by_class}")
        blocks.append((corpus_label, rows))

    OUT.write_text(
        latex.render_composition(
            blocks, vocab_size=VOCAB_SIZE, boundary_abbr=BOUNDARY_ABBR
        )
    )
    print(f"[composition] wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
