"""Generate the shared-core growth table (appendix).

For the PubChem matched pair under both boundary policies, traces the
BPE-cap-Unigram-LM shared multi-glyph set across $V \\in \\{256, 512, 1024,
2048\\}$. The core is strictly nested (verified at build time), so the table
lists each $V$'s *new* shared pieces as a layer. Pieces are read from the
committed tokenizer artifacts via the Jaccard runner's ``glyph_tuple_map``, so
the split matches the deposited overlap. Emits
``results/tables/shared_core_growth.tex``.

Usage::

    uv run python results/build/table_shared_core_growth.py
"""

from __future__ import annotations

import sys
from itertools import pairwise
from typing import TYPE_CHECKING

import latex
from _pieces import multi_glyph_set, ordered_surfaces

from smiles_subword.paths import RESULTS_TABLES_DIR

if TYPE_CHECKING:
    from smiles_subword.tokenize.measure.jaccard import GlyphTuple

CORPUS = "pubchem"
CORPUS_LABEL = "PubChem"
VOCAB_SIZES = (256, 512, 1024, 2048)
OUT = RESULTS_TABLES_DIR / "shared_core_growth.tex"

BOUNDARIES = (
    ("no-merge-brackets (NMB)", "nmb"),
    ("merge-brackets (MB)", "mb"),
)


def main() -> int:
    blocks: list[tuple[str, tuple[tuple[str, list[str]], ...]]] = []
    trend_parts: list[str] = []
    full_core: dict[str, set[GlyphTuple]] = {}
    for boundary_label, suffix in BOUNDARIES:
        shared_by_v: dict[int, set[GlyphTuple]] = {}
        sizes: list[int] = []
        jaccards: list[float] = []
        for v in VOCAB_SIZES:
            bpe = multi_glyph_set(CORPUS, v, suffix, "bpe")
            ul = multi_glyph_set(CORPUS, v, suffix, "unigram")
            shared = bpe & ul
            shared_by_v[v] = shared
            sizes.append(len(shared))
            jaccards.append(len(shared) / len(bpe | ul))

        # The core must be strictly nested across V (no piece ever leaves).
        for lo, hi in pairwise(VOCAB_SIZES):
            if not shared_by_v[lo] <= shared_by_v[hi]:
                lost = sorted("".join(t) for t in shared_by_v[lo] - shared_by_v[hi])
                raise AssertionError(
                    f"{suffix}: core not nested {lo}->{hi}; lost {lost}"
                )

        layers: list[tuple[str, list[str]]] = []
        prev: set[GlyphTuple] = set()
        for v in VOCAB_SIZES:
            new = shared_by_v[v] - prev
            label = (
                rf"Core, $V\le{v}$" if v == VOCAB_SIZES[0] else rf"New at $V{{=}}{v}$"
            )
            layers.append((label, ordered_surfaces(new)))
            prev = shared_by_v[v]

        trend_parts.append(
            r"$"
            + r" \to ".join(str(s) for s in sizes)
            + r"$ at overlap $J = "
            + r",\, ".join(f"{j:.3f}" for j in jaccards)
            + rf"$ ({suffix.upper()})"
        )
        full_core[suffix] = shared_by_v[VOCAB_SIZES[-1]]
        blocks.append((boundary_label, tuple(layers)))
        js = [round(j, 3) for j in jaccards]
        print(f"[shared-core] {suffix.upper()}: sizes={sizes} J={js}")

    trend_summary = "Cumulative core size grows " + " and ".join(trend_parts) + "."

    # Pieces shared (BPE-cap-UL) under *both* boundary policies: the
    # cross-boundary-robust core, greyed in both blocks.
    cross = full_core["nmb"] & full_core["mb"]
    cross_boundary = ["".join(t) for t in cross]
    print(
        f"[shared-core] cross-boundary core: {len(cross)} of "
        f"{len(full_core['nmb'])} (NMB) / {len(full_core['mb'])} (MB)"
    )

    OUT.write_text(
        latex.render_shared_core_growth(
            blocks,
            corpus_label=CORPUS_LABEL,
            trend_summary=trend_summary,
            cross_boundary=cross_boundary,
        )
    )
    print(f"[shared-core] wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
