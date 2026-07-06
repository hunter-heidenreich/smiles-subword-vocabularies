"""Generate the arm-exclusive statistics table (appendix).

For the PubChem matched pair under both boundary policies, summarizes the
BPE-only and Unigram-LM-only multi-glyph sets across $V \\in \\{256, 512, 1024,
2048\\}$: how many exclusive pieces each arm holds and their glyph-length mean and
maximum. Pieces are read from the committed tokenizer artifacts via the Jaccard
runner's ``glyph_tuple_map``. Emits ``results/tables/arm_exclusive.tex``.

Usage::

    uv run python results/build/table_arm_exclusive.py
"""

from __future__ import annotations

import statistics as st
import sys

import latex
from _corpora import CORPUS_LABEL
from _pieces import multi_glyph_set

from smiles_subword.paths import RESULTS_TABLES_DIR

CORPUS = "pubchem"
VOCAB_SIZES = (256, 512, 1024, 2048)
OUT = RESULTS_TABLES_DIR / "arm_exclusive.tex"

BOUNDARIES = (
    ("No-merge-brackets (NMB)", "nmb"),
    ("Merge-brackets (MB)", "mb"),
)

Row = tuple[int, int, int, float, float, int, int]


def main() -> int:
    blocks: list[tuple[str, list[Row]]] = []
    for boundary_label, suffix in BOUNDARIES:
        rows: list[Row] = []
        for v in VOCAB_SIZES:
            bpe = multi_glyph_set(CORPUS, v, suffix, "bpe")
            ul = multi_glyph_set(CORPUS, v, suffix, "unigram")
            bpe_only, ul_only = bpe - ul, ul - bpe
            rows.append(
                (
                    v,
                    len(bpe_only),
                    len(ul_only),
                    st.mean(len(t) for t in bpe_only),
                    st.mean(len(t) for t in ul_only),
                    max(len(t) for t in bpe_only),
                    max(len(t) for t in ul_only),
                )
            )
            print(f"[arm-excl] {suffix.upper()} V={v}: {rows[-1]}")
        blocks.append((boundary_label, rows))

    OUT.write_text(
        latex.render_arm_exclusive(blocks, corpus_label=CORPUS_LABEL[CORPUS])
    )
    print(f"[arm-excl] wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
