"""Generate the non-canonicity write-stability figure for the appendix.

Reads the deposited per-condition non-canonicity measurements
(`data/noncanon_table.json`) and renders four dumbbell panels -- one per rewrite
axis (OpenBabel, Randomized, Kekule, Explicit-H, in mild -> catastrophic order)
-- each with one row per corpus x boundary at V=1024, joining BPE (blue) to
Unigram-LM (orange) on a *shared* bag-instability axis. Because the four axes
share the same unit, one scale carries two readings at once: the magnitude
ordering (OpenBabel mildest on the left, Explicit-H catastrophic on the right)
and, within each panel, the arm gap and its Kekule sign flip -- Unigram-LM is the
more write-stable arm (its dot left of BPE's) under every axis but Kekule, where
it flips to the less stable (dot to the right). The full 22-condition numbers are
in Table~\ref{tab:results-noncanon}. Standalone; run with:

    uv run python results/build/figure_noncanon.py
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
DATA = HERE.parent / "data" / "noncanon_table.json"

BND_ORDER = ["nmb", "mb"]
BND_LABEL = style.BOUNDARY_LABEL
ARM_COLOR = style.ARM_COLOR
ARM_LABEL = style.ARM_LABEL
# (axis key, panel title) in mild -> catastrophic magnitude order; ringperm (the
# exact-invariant floor) is omitted, as in the table.
AXES = [
    ("obcanon", "OpenBabel"),
    ("random", "Randomized"),
    ("kekule", "Kekulé"),
    ("explicitH", "Explicit-H"),
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

    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.8), sharey=True, sharex=True)
    for ax, (axis, title) in zip(axes, AXES, strict=True):
        style.despine(ax, left=True)
        ax.set_axisbelow(True)
        ax.grid(axis="x", color=style.GRID_GRAY, lw=0.6, zorder=0)
        ax.tick_params(axis="y", length=0)
        for row, yv in zip(rows, ys, strict=True):
            rec = row["record"]
            bpe, ul = rec[f"bpe_bag_{axis}"], rec[f"ul_bag_{axis}"]
            ax.plot([bpe, ul], [yv, yv], color="0.65", lw=1.5, zorder=2)
            ax.scatter([bpe], [yv], color=ARM_COLOR["bpe"], s=34, zorder=3)
            ax.scatter([ul], [yv], color=ARM_COLOR["unigram"], s=34, zorder=3)
        ax.set_title(title, fontsize=9)
        ax.set_xlim(0.0, 0.9)
        ax.set_xticks([0.0, 0.3, 0.6, 0.9])

    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([row["label"] for row in rows])
    axes[0].set_ylim(min(ys) - 0.8, max(ys) + 0.8)
    fig.supxlabel(
        "bag-instability $b$ (fraction of token multiset changed on rewrite)",
        fontsize=8,
        y=0.05,
    )

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
        bbox_to_anchor=(0.5, -0.03),
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=1.4)
    fig.savefig(
        RESULTS_FIGURES_DIR / "noncanon_stability.pdf",
        bbox_inches="tight",
        metadata={"CreationDate": None},
    )
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
