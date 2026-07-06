"""Pairwise hyperparameter interaction surfaces.

Reads the deposited sensitivity report
(``data/sensitivity/sensitivity_report.json``) and renders the four pairwise
interactions as heatmaps of the cross-arm vocabulary overlap ``J``: (A)
sub-iterations × shrinking factor, (B) max piece length × V, (C) max piece
length × corpus typology, and (D) BPE merge frequency × Unigram max piece
length. Every cell annotates its ``J``; the whole grid staying near-disjoint is
the "robust under joint perturbation" gestalt.
Standalone; run with::

    uv run python results/build/figure_interaction_surfaces.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import style

from smiles_subword.paths import RESULTS_FIGURES_DIR

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "sensitivity" / "sensitivity_report.json"
OUT = RESULTS_FIGURES_DIR / "interaction_surfaces.pdf"

# Colorbar ceiling: a fixed bound just above the highest observed cell
# (J peaks at ~0.29 at the shortest-piece corner), so the color scale spans the
# data range rather than being keyed to any external reference.
J_VMAX = 0.3

_TYPOLOGY = {0.0: "PubChem", 1.0: "ZINC-22", 2.0: "COCONUT"}

# (interaction name, panel title, x label, y label, y tick formatter)
PANELS = [
    (
        "subiter_shrink",
        "(a) sub-iterations $\\times$ shrinking factor",
        "sub-iterations",
        "shrinking factor",
        lambda v: f"{v:g}",
    ),
    (
        "mpl_v",
        "(b) max piece length $\\times$ $V$",
        "max piece length",
        "$V$",
        lambda v: f"{int(v)}",
    ),
    (
        "mpl_typology",
        "(c) max piece length $\\times$ typology",
        "max piece length",
        "corpus",
        lambda v: _TYPOLOGY.get(v, f"{v:g}"),
    ),
    (
        "minfreq_mpl",
        "(d) BPE merge freq. $\\times$ max piece length",
        "BPE merge frequency",
        "max piece length",
        lambda v: f"{int(v)}",
    ),
]


def _text_color(cmap: object, t: float) -> str:
    """Black or white, whichever contrasts better with the cell's fill.

    ``t`` is the value normalised to [0, 1] over the colour scale. Uses relative
    luminance of the actual colormap colour rather than a hand-tuned value cutoff,
    so bright viridis cells (high ``J``) reliably get black text and dark cells
    white -- robust across the whole viridis ramp.
    """
    r, g, b, _ = cmap(max(0.0, min(1.0, t)))  # type: ignore[operator]
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "black" if luminance > 0.5 else "white"


def _grid(points: list[dict]) -> tuple[np.ndarray, list[float], list[float]]:
    xs = sorted({p["x"] for p in points})
    ys = sorted({p["y"] for p in points})
    grid = np.full((len(ys), len(xs)), np.nan)
    for p in points:
        grid[ys.index(p["y"])][xs.index(p["x"])] = p["jaccard"]
    return grid, xs, ys


def main() -> int:
    report = json.loads(DATA.read_text())

    style.apply_base_style()
    plt.rcParams.update({"font.size": 8})  # denser 2x2 heatmap grid than the base
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.2), layout="constrained")

    cmap = mpl.colormaps["viridis"]
    for ax, (name, title, xlabel, ylabel, yfmt) in zip(axes.flat, PANELS, strict=True):
        grid, xs, ys = _grid(report["interactions"][name]["points"])
        im = ax.imshow(
            grid, origin="lower", aspect="auto", cmap=cmap, vmin=0.0, vmax=J_VMAX
        )
        ax.set_xticks(range(len(xs)), [f"{x:g}" for x in xs])
        ax.set_yticks(range(len(ys)), [yfmt(y) for y in ys])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=7.5)
        for yi in range(len(ys)):
            for xi in range(len(xs)):
                v = grid[yi][xi]
                if not np.isnan(v):
                    ax.text(
                        xi,
                        yi,
                        f"{v:.02f}",
                        ha="center",
                        va="center",
                        color=_text_color(cmap, v / J_VMAX),
                        fontsize=6,
                    )
    # One shared colorbar for the whole grid: all four panels use the same
    # 0-J_VMAX viridis scale, so per-panel bars were four copies of one legend.
    cbar = fig.colorbar(im, ax=axes, fraction=0.046, pad=0.02, aspect=40)
    cbar.set_label("$J$")
    fig.savefig(OUT, bbox_inches="tight", metadata={"CreationDate": None})
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
