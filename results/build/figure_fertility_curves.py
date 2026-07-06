"""Generate the fertility dumbbell figure for the paper body.

Reads the deposited per-condition fertility measurements
(`data/fertility_table.json`) and renders a compact dumbbell (paired-dot) chart
on a shared tokens-per-molecule axis: one row per corpus x boundary policy at the
headline `V=1024`, each row joining BPE (blue) to Unigram-LM (orange). The
connector length is the absolute fertility gap; the relative gap rel|df| is
annotated per row. Unigram-LM sits to the right of BPE in every row, on a single
shared scale so the gap is directly comparable across corpora.

The V-trend (the relative gap's rise with V) lives in the cross-V figure; the
exhaustive per-condition numbers with CIs live in the appendix table. Standalone;
run with:

    uv run python results/build/figure_fertility_curves.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import style
from _corpora import CORPUS_LABEL, CORPUS_ORDER, DUMBBELL_VOCAB, dumbbell_ys
from matplotlib.lines import Line2D

from smiles_subword.paths import RESULTS_FIGURES_DIR

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "fertility_table.json"

BND_ORDER = ["nmb", "mb"]
BND_LABEL = style.BOUNDARY_LABEL
ARM_COLOR = style.ARM_COLOR
ARM_LABEL = style.ARM_LABEL


def load_rows() -> list[dict]:
    records = json.loads(DATA.read_text())["matched"]
    index = {
        (r["corpus"], r["boundary"]): r
        for r in records
        if r.get("pair_status") == "matched"
        and r.get("extras_kind") is None
        and r["vocab_size"] == DUMBBELL_VOCAB
    }
    rows: list[dict] = []
    for corpus in CORPUS_ORDER:
        for i, boundary in enumerate(BND_ORDER):
            record = index.get((corpus, boundary))
            if record is None:
                continue
            rows.append(
                {
                    "label": f"{CORPUS_LABEL[corpus]}  {BND_LABEL[boundary]}",
                    "bpe": record["bpe_fertility"],
                    "ul": record["unigram_fertility"],
                    "rel": record["delta_fertility_relative"],
                    "new_group": i == 0,
                }
            )
    return rows


def main() -> int:
    rows = load_rows()
    style.apply_base_style()
    ys = dumbbell_ys(rows)

    fig, ax = plt.subplots(figsize=(6.5, 3.1))
    style.despine(ax, left=True)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=style.GRID_GRAY, lw=0.6, zorder=0)
    ax.tick_params(axis="y", length=0)

    for row, yv in zip(rows, ys, strict=True):
        ax.plot([row["bpe"], row["ul"]], [yv, yv], color="0.65", lw=1.6, zorder=2)
        ax.scatter([row["bpe"]], [yv], color=ARM_COLOR["bpe"], s=44, zorder=3)
        ax.scatter([row["ul"]], [yv], color=ARM_COLOR["unigram"], s=44, zorder=3)
        ax.annotate(
            f"{row['rel'] * 100:.0f}%",
            xy=(row["ul"], yv),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=7.5,
            color="0.4",
        )

    ax.set_yticks(ys)
    ax.set_yticklabels([row["label"] for row in rows])
    ax.set_ylim(min(ys) - 0.8, max(ys) + 0.8)
    ax.set_xlim(27, 80)
    ax.set_xlabel(r"tokens / molecule (held-out, $V{=}1024$)")

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=ARM_COLOR[a],
            markeredgecolor=ARM_COLOR[a],
            markersize=7,
            label=ARM_LABEL[a],
        )
        for a in ("bpe", "unigram")
    ]
    ax.legend(
        handles=handles,
        loc="lower right",
        frameon=False,
        fontsize=8,
        handletextpad=0.3,
        labelspacing=0.3,
    )
    fig.tight_layout()
    fig.savefig(
        RESULTS_FIGURES_DIR / "fertility_curves.pdf",
        bbox_inches="tight",
        metadata={"CreationDate": None},
    )
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
