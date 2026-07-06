"""Generate the token-distribution intrinsics figure for the paper body.

Reads the deposited per-condition distribution measurements
(`data/distribution_table.json`) and renders three side-by-side dumbbell panels
-- D (token-frequency imbalance), eta (normalized Shannon entropy), and Renyi
efficiency -- each with one row per corpus x boundary at V=1024, joining BPE
(blue) to Unigram-LM (orange) on that metric's own axis. The three panels show
the coherent within-family signature: BPE sits on the more-uniform side of every
panel (lower D, higher eta, higher Renyi) in every condition, while both arms sit
far from uniform. Standalone; run with:

    uv run python results/build/figure_distribution_intrinsics.py
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
from matplotlib.ticker import MaxNLocator

from smiles_subword.paths import RESULTS_FIGURES_DIR

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "distribution_table.json"

BND_ORDER = ["nmb", "mb"]
BND_LABEL = style.BOUNDARY_LABEL
ARM_COLOR = style.ARM_COLOR
ARM_LABEL = style.ARM_LABEL
# (key stem, panel title) -- BPE is on the more-uniform side of each.
METRICS = [
    ("d", "$D$ (imbalance)"),
    ("eta", r"$\eta$ (norm. entropy)"),
    ("renyi", "$R$ (Rényi eff.)"),
]


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
                    "record": record,
                    "new_group": i == 0,
                }
            )
    return rows


def main() -> int:
    rows = load_rows()
    style.apply_base_style()
    ys = dumbbell_ys(rows)

    fig, axes = plt.subplots(1, 3, figsize=(6.5, 3.6), sharey=True)
    for ax, (stem, title) in zip(axes, METRICS, strict=True):
        style.despine(ax, left=True)
        ax.set_axisbelow(True)
        ax.grid(axis="x", color=style.GRID_GRAY, lw=0.6, zorder=0)
        ax.tick_params(axis="y", length=0)
        for row, yv in zip(rows, ys, strict=True):
            rec = row["record"]
            bpe, ul = rec[f"bpe_{stem}"], rec[f"unigram_{stem}"]
            ax.plot([bpe, ul], [yv, yv], color="0.65", lw=1.5, zorder=2)
            ax.scatter([bpe], [yv], color=ARM_COLOR["bpe"], s=34, zorder=3)
            ax.scatter([ul], [yv], color=ARM_COLOR["unigram"], s=34, zorder=3)
        # Prune first/last ticks so adjacent panels' edge labels don't collide.
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
        ax.set_title(title, fontsize=9)

    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([row["label"] for row in rows])
    axes[0].set_ylim(min(ys) - 0.8, max(ys) + 0.8)

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
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=8,
        handletextpad=0.3,
        columnspacing=1.5,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1), w_pad=1.8)
    fig.savefig(
        RESULTS_FIGURES_DIR / "distribution_intrinsics.pdf",
        bbox_inches="tight",
        metadata={"CreationDate": None},
    )
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
