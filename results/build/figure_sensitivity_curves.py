"""Per-knob sensitivity response curves.

Reads the deposited sensitivity report
(``data/sensitivity/sensitivity_report.json``) and renders, for each swept
training hyperparameter, the two cross-arm contrasts: vocabulary
overlap ``J`` and the relative fertility gap ``rel|Δf|``. Both are read as effect
sizes, with no reference line. A knob leaves the BPE/Unigram contrast robust
when both curves hold their separation across its whole ladder.
Standalone; run with::

    uv run python results/build/figure_sensitivity_curves.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import style

from smiles_subword.paths import RESULTS_FIGURES_DIR

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "sensitivity" / "sensitivity_report.json"
OUT = RESULTS_FIGURES_DIR / "sensitivity_curves.pdf"

# Display order, axis label, and whether the rung ladder is geometric (log x).
KNOBS = [
    ("mpl", "Unigram max piece length", True),
    ("seed", "Unigram seed pool", True),
    ("subiter", "Unigram sub-iterations", False),
    ("shrink", "Unigram shrinking factor", False),
    ("minfreq", "BPE merge frequency", False),
]
J_COLOR = style.OVERLAP_COLOR
F_COLOR = style.GAP_COLOR


def _curve(
    report: dict, knob: str
) -> tuple[list[float], list[float], list[float], float]:
    pts = sorted(report["ladders"][knob]["points"], key=lambda p: p["x"])
    xs = [p["x"] for p in pts]
    js = [p["jaccard"] for p in pts]
    fs = [p["delta_fertility_relative"] for p in pts]
    default_x = next(p["x"] for p in pts if p["is_default"])
    return xs, js, fs, default_x


def main() -> int:
    report = json.loads(DATA.read_text())

    style.apply_base_style()
    plt.rcParams.update({"font.size": 8})  # denser 2xN grid than the base
    n = len(KNOBS)
    fig, axes = plt.subplots(2, n, figsize=(2.0 * n, 3.4), sharex="col", sharey="row")

    # J and rel|Δf| are the same quantity across all panels, so share one
    # y-scale per row (global max) rather than autoscaling each panel
    # independently, which would draw the rising and flat knobs at different
    # scales and make the row visually non-comparable.
    all_j, all_f = [], []
    for knob, _, _ in KNOBS:
        _, js, fs, _ = _curve(report, knob)
        all_j += js
        all_f += fs
    j_top, f_top = max(all_j) + 0.08, max(all_f) + 0.06

    for col, (knob, label, log_x) in enumerate(KNOBS):
        xs, js, fs, default_x = _curve(report, knob)
        ax_j, ax_f = axes[0][col], axes[1][col]
        style.despine(ax_j)
        style.despine(ax_f)

        ax_j.plot(xs, js, "o-", color=J_COLOR, ms=3, lw=1.2)
        ax_j.axvline(default_x, color="0.8", lw=0.8, zorder=0)
        ax_j.set_ylim(-0.03, j_top)
        ax_j.set_title(label, fontsize=7.5)

        ax_f.plot(xs, fs, "s-", color=F_COLOR, ms=3, lw=1.2)
        ax_f.axvline(default_x, color="0.8", lw=0.8, zorder=0)
        ax_f.set_ylim(-0.02, f_top)
        ax_f.set_xlabel(label.split(" ", 1)[1] if " " in label else label, fontsize=7)

        if log_x:
            ax_j.set_xscale("log")
            ax_f.set_xscale("log")
        if col == 0:
            ax_j.set_ylabel(r"$J$ (overlap)")
            ax_f.set_ylabel(r"$\mathrm{rel}|\Delta f|$")

    fig.tight_layout(pad=0.4)
    fig.savefig(OUT, bbox_inches="tight", metadata={"CreationDate": None})
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
