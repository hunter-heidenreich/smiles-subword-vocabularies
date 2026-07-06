"""Generate the three-way learned-piece split table (appendix).

For the PubChem ``V=256`` matched pair under both boundary policies, lists every
learned multi-glyph piece (glyph count >= 2) each arm selects above the shared
base, partitioned by cross-algorithm membership into shared / BPE-only /
Unigram-LM-only. Pieces are read from the committed tokenizer artifacts via the
same ``glyph_tuple_map`` the Jaccard runner uses, so the split matches the
deposited overlap exactly. Emits ``results/tables/multiglyph_v256.tex``.

Usage::

    uv run python results/build/table_multiglyph_split.py
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import latex
from _corpora import CORPUS_LABEL
from _pieces import multi_glyph_set, ordered_surfaces

from smiles_subword.paths import RESULTS_TABLES_DIR

if TYPE_CHECKING:
    from smiles_subword.tokenize.measure.jaccard import GlyphTuple

CORPUS = "pubchem"
VOCAB_SIZE = 256
OUT = RESULTS_TABLES_DIR / "multiglyph_v256.tex"

# (boundary_label, artifact suffix)
BOUNDARIES = (
    ("no-merge-brackets (NMB)", "nmb"),
    ("merge-brackets (MB)", "mb"),
)


def main() -> int:
    blocks: list[tuple[str, float, tuple[tuple[str, list[str]], ...]]] = []
    per_boundary_pieces: dict[str, set[GlyphTuple]] = {}
    for boundary_label, suffix in BOUNDARIES:
        bpe = multi_glyph_set(CORPUS, VOCAB_SIZE, suffix, "bpe")
        ul = multi_glyph_set(CORPUS, VOCAB_SIZE, suffix, "unigram")
        per_boundary_pieces[suffix] = bpe | ul
        shared, bpe_only, ul_only = bpe & ul, bpe - ul, ul - bpe
        jaccard = len(shared) / len(bpe | ul)
        buckets = (
            ("Shared", ordered_surfaces(shared)),
            ("BPE-only", ordered_surfaces(bpe_only)),
            ("Unigram-LM-only", ordered_surfaces(ul_only)),
        )
        blocks.append((boundary_label, jaccard, buckets))
        print(
            f"[multiglyph] {suffix.upper()}: shared={len(shared)} "
            f"bpe_only={len(bpe_only)} ul_only={len(ul_only)} J={jaccard:.3f}"
        )

    # Pieces selected under *both* boundary policies are boundary-robust.
    nmb_pieces, mb_pieces = per_boundary_pieces["nmb"], per_boundary_pieces["mb"]
    robust = nmb_pieces & mb_pieces
    boundary_robust = ["".join(t) for t in robust]
    print(
        f"[multiglyph] boundary-robust: {len(robust)} of "
        f"{len(nmb_pieces)} (NMB) / {len(mb_pieces)} (MB)"
    )

    OUT.write_text(
        latex.render_multiglyph_split(
            blocks,
            corpus_label=CORPUS_LABEL[CORPUS],
            vocab_size=VOCAB_SIZE,
            boundary_robust=boundary_robust,
        )
    )
    print(f"[multiglyph] wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
