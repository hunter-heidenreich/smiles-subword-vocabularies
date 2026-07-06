"""Generate the narrow-alphabet contrast table (appendix): ZINC-22.

For the ZINC-22 matched pair under both boundary policies, tabulates per-arm
multi-glyph vocabulary size, the three-way cross-algorithm split, and the overlap
$J$ across $V \\in \\{256, 512, 1024\\}$. ZINC-22's $V{=}2048$ Unigram arm is
embedding-tail-unsafe and untrained, so it forms no matched pair and is excluded
here (its single-arm ceiling is reported in \\S\\ref{sec:results}). Pieces are read
from the committed tokenizer artifacts via the Jaccard runner's
``glyph_tuple_map``. Emits ``results/tables/narrow_contrast.tex``.

Usage::

    uv run python results/build/table_narrow_contrast.py
"""

from __future__ import annotations

import sys

import latex
from _corpora import CORPUS_LABEL
from _pieces import multi_glyph_set_optional

from smiles_subword.paths import RESULTS_TABLES_DIR

CORPUS = "zinc22"
VOCAB_SIZES = (256, 512, 1024)
OUT = RESULTS_TABLES_DIR / "narrow_contrast.tex"

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
            bpe = multi_glyph_set_optional(CORPUS, v, suffix, "bpe")
            ul = multi_glyph_set_optional(CORPUS, v, suffix, "unigram")
            if bpe is None:
                raise AssertionError(f"missing BPE arm for {CORPUS} v{v} {suffix}")
            if ul is None:
                rows.append((v, len(bpe), None, None, None, None, None))
            else:
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
            print(f"[narrow] {suffix.upper()} V={v}: {rows[-1]}")
        blocks.append((boundary_label, rows))

    OUT.write_text(
        latex.render_corpus_contrast(
            blocks,
            corpus_label=CORPUS_LABEL[CORPUS],
            label="tab:narrow-contrast",
        )
    )
    print(f"[narrow] wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
