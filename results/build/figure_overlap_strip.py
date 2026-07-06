"""Generate the vocabulary-overlap figures for the paper.

Reads the deposited per-condition Jaccard measurements
(`data/jaccard_table.json`) and renders two two-panel figures on a
shared log y-axis:

  * `overlap_strip.pdf`      -- main: frequency-weighted J_w | unweighted J
  * `overlap_strip_struct.pdf` -- appendix: structural variants J_w,struct | J_struct

Each panel plots all 22 matched conditions; x = target vocabulary size,
colour *and marker shape* = corpus (Okabe-Ito, colourblind-safe), marker fill =
boundary policy (filled = NMB, open = MB). Standalone; run with:

    uv run python results/build/figure_overlap_strip.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import style
from _corpora import CORPUS_LABEL, CORPUS_ORDER
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from smiles_subword.paths import RESULTS_FIGURES_DIR

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "jaccard_table.json"

CORPUS_COLOR = style.CORPUS_COLOR
CORPUS_DODGE = {"pubchem": -0.27, "zinc22": -0.09, "coconut": 0.09, "real_space": 0.27}
V_LEVELS = [256, 512, 1024, 2048]


def corpus_key(name: str) -> str:
    n = name.lower()
    for key in CORPUS_ORDER:
        if n.startswith(key):
            return key
    raise ValueError(f"unrecognised corpus: {name!r}")


def load_conditions() -> list[dict]:
    records = json.loads(DATA.read_text())["matched"]
    return [
        r
        for r in records
        if r.get("pair_status") == "matched" and r.get("extras_kind") is None
    ]


def draw(ax, rows, mkey):
    ax.set_axisbelow(True)
    ax.grid(axis="y", which="major", color="0.92", lw=0.6, zorder=0)
    for i in range(len(V_LEVELS)):  # faint alternating per-V bands
        if i % 2 == 1:
            ax.axvspan(i - 0.45, i + 0.45, color="0.965", zorder=0)
    for r in rows:
        ck = corpus_key(r["corpus"])
        x = V_LEVELS.index(r["vocab_size"]) + CORPUS_DODGE[ck]
        nmb = r["boundary"] == "nmb"
        # Marker *shape* carries corpus (matching style.CORPUS_MARKER and the
        # cross-V trend figure), so a shape means the same thing across figures;
        # marker *fill* carries the boundary policy (filled = NMB, open = MB).
        ax.scatter(
            [x],
            [r[mkey]],
            marker=style.CORPUS_MARKER[ck],
            s=27,
            facecolors=(CORPUS_COLOR[ck] if nmb else "white"),
            edgecolors=CORPUS_COLOR[ck],
            linewidths=1.0,
            zorder=3,
        )
    ax.set_yscale("log")
    # Ceiling sits just above the highest observed J (~0.16), so the data region
    # fills the panel rather than floating against an empty upper margin.
    ax.set_ylim(1.5e-3, 0.3)
    ax.set_yticks([0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.minorticks_off()
    ax.set_xticks(range(len(V_LEVELS)))
    ax.set_xticklabels([str(v) for v in V_LEVELS])
    ax.set_xlim(-0.65, 3.8)
    ax.set_xlabel("target vocabulary size $V$")


def make(lkey, rkey, ltitle, rtitle, out):
    rows = load_conditions()
    style.apply_base_style()
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(6.5, 2.9), sharey=True)
    style.despine(ax_l)
    style.despine(ax_r)
    draw(ax_l, rows, lkey)
    draw(ax_r, rows, rkey)
    ax_leg = ax_l  # legends on the weighted (left) panel
    ax_l.set_title(f"(a) {ltitle}", loc="left", fontsize=9.5)
    ax_r.set_title(f"(b) {rtitle}", loc="left", fontsize=9.5)
    ax_l.set_ylabel("Vocabulary overlap (Jaccard)")

    corpus_h = [
        Line2D(
            [0],
            [0],
            marker=style.CORPUS_MARKER[k],
            color="w",
            markerfacecolor=CORPUS_COLOR[k],
            markeredgecolor=CORPUS_COLOR[k],
            markersize=6,
            label=CORPUS_LABEL[k],
        )
        for k in CORPUS_ORDER
    ]
    # Boundary key: a single neutral shape distinguished only by fill, so it reads
    # as "fill = boundary" and never implies a corpus (which shape now carries).
    bound_h = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="0.45",
            markeredgecolor="0.45",
            markersize=6,
            label="NMB",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="white",
            markeredgecolor="0.45",
            markersize=6,
            label="MB",
        ),
    ]
    leg1 = ax_leg.legend(
        handles=corpus_h,
        title="Corpus",
        loc="upper left",
        frameon=False,
        fontsize=7,
        title_fontsize=7,
        handletextpad=0.2,
        labelspacing=0.25,
    )
    ax_leg.add_artist(leg1)
    ax_leg.legend(
        handles=bound_h,
        title="Boundary",
        loc="upper right",
        frameon=False,
        fontsize=7,
        title_fontsize=7,
        handletextpad=0.2,
        labelspacing=0.25,
    )
    fig.tight_layout()
    fig.savefig(
        RESULTS_FIGURES_DIR / out, bbox_inches="tight", metadata={"CreationDate": None}
    )
    plt.close(fig)


def main() -> int:
    # Frequency-weighted overlap leads (left); unweighted on the right.
    make(
        "weighted_jaccard",
        "jaccard",
        r"Frequency-weighted ($J_{\mathrm{w}}$)",
        r"Unweighted ($J$)",
        "overlap_strip.pdf",
    )
    make(
        "weighted_jaccard_struct",
        "jaccard_struct",
        r"Structural, weighted ($J_{\mathrm{w,struct}}$)",
        r"Structural ($J_{\mathrm{struct}}$)",
        "overlap_strip_struct.pdf",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
