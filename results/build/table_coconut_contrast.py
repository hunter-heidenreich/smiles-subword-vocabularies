"""Generate the natural-products contrast table (appendix): COCONUT.

For the COCONUT matched pair under both boundary policies, tabulates per-arm
multi-glyph vocabulary size, the three-way cross-algorithm split, and the overlap
$J$ across $V \\in \\{256, 512, 1024\\}$. COCONUT's distinctive feature is that the
overlap is highest at small $V$ and falls as $V$ grows (the inverse of PubChem and
ZINC-22). Pieces are read from the committed tokenizer artifacts via the Jaccard
runner's ``glyph_tuple_map``. Emits ``results/tables/coconut_contrast.tex``.

Usage::

    uv run python results/build/table_coconut_contrast.py
"""

from __future__ import annotations

import sys

import latex
from _corpora import CORPUS_LABEL
from _pieces import multi_glyph_set

from smiles_subword.paths import RESULTS_TABLES_DIR

CORPUS = "coconut"
VOCAB_SIZES = (256, 512, 1024)
OUT = RESULTS_TABLES_DIR / "coconut_contrast.tex"

BOUNDARIES = (
    ("No-merge-brackets (NMB)", "nmb"),
    ("Merge-brackets (MB)", "mb"),
)

Row = tuple[int, int, int | None, int | None, int | None, int | None, float | None]


def main() -> int:
    blocks: list[tuple[str, list[Row]]] = []
    for boundary_label, suffix in BOUNDARIES:
        rows: list[Row] = []
        for v in VOCAB_SIZES:
            bpe = multi_glyph_set(CORPUS, v, suffix, "bpe")
            ul = multi_glyph_set(CORPUS, v, suffix, "unigram")
            shared = bpe & ul
            jaccard = len(shared) / len(bpe | ul)
            rows.append(
                (
                    v,
                    len(bpe),
                    len(ul),
                    len(shared),
                    len(bpe - ul),
                    len(ul - bpe),
                    jaccard,
                )
            )
            print(f"[coconut] {suffix.upper()} V={v}: {rows[-1]}")
        blocks.append((boundary_label, rows))

    OUT.write_text(
        latex.render_corpus_contrast(
            blocks, corpus_label=CORPUS_LABEL[CORPUS], label="tab:coconut-contrast"
        )
    )
    print(f"[coconut] wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
