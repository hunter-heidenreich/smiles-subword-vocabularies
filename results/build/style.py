"""Shared figure style: one source of truth for typography and palette.

Colourblind-safe (Okabe-Ito) palette with two semantic axes that never share a
figure: corpus (colour + marker) and arm (BPE blue vs Unigram-LM vermillion).
PubChem uses the lighter sky blue, kept off the deeper BPE-blue, so corpus-blue
and arm-blue never read alike across figures. Spine choices vary by chart family,
so they are left to each figure via :func:`despine`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib as mpl

if TYPE_CHECKING:
    from matplotlib.axes import Axes

# --- Okabe-Ito colourblind-safe hues --------------------------------------- #
BLUE = "#0072B2"
SKYBLUE = "#56B4E9"
AMBER = "#E69F00"
GREEN = "#009E73"
PINK = "#CC79A7"
VERMILLION = "#D55E00"

# --- corpus axis (colour + marker; never shares a figure with the arm axis) - #
# Keyed by on-disk long names; display labels live in _corpora's CORPUS_LABEL.
# PubChem uses sky blue (not the arm axis's deeper BLUE) so corpus-blue and
# BPE-blue never read as the same colour across figures.
CORPUS_COLOR = {
    "pubchem": SKYBLUE,
    "zinc22": AMBER,
    "coconut": GREEN,
    "real_space": PINK,
}
CORPUS_MARKER = {"pubchem": "o", "zinc22": "s", "coconut": "^", "real_space": "D"}

# --- arm axis -------------------------------------------------------------- #
BPE_COLOR = BLUE
UNIGRAM_COLOR = VERMILLION
ARM_COLOR = {"bpe": BPE_COLOR, "unigram": UNIGRAM_COLOR}
ARM_LABEL = {"bpe": "BPE", "unigram": "Unigram-LM"}

# --- metric-pair figures (overlap vs fertility gap) ------------------------ #
# Off-arm teal/purple pair (arm palette is blue=BPE, vermillion=Unigram-LM) so a
# metric line is never mistaken for an arm in figures that also carry per-arm
# series. Colour is secondary; each metric is already keyed by its own y-axis.
OVERLAP_COLOR = "#44AA99"  # teal (Paul Tol muted)
GAP_COLOR = "#AA4499"  # purple (Paul Tol muted)

# --- diverging sign (NMB-MB interaction bars) ------------------------------ #
# Off-arm teal/purple pair so signed bars never read as an arm. Sign is carried
# by bar direction; colour only reinforces it.
SIGN_TEAL = "#44AA99"
SIGN_PURPLE = "#AA4499"
POS_COLOR = SIGN_TEAL
NEG_COLOR = SIGN_PURPLE

# --- boundary policy ------------------------------------------------------- #
BOUNDARY_STYLE = {"nmb": "-", "mb": "--"}
BOUNDARY_LABEL = {"nmb": "NMB", "mb": "MB"}

# --- shared accents -------------------------------------------------------- #
GRID_GRAY = "0.92"  # light gridlines
REF_GRAY = "0.5"  # reference / threshold rules

BASE_RCPARAMS = {
    "pdf.fonttype": 42,  # embed TrueType (portable, no Type-3)
    "font.family": "serif",
    "mathtext.fontset": "cm",  # Computer Modern math, matched to the serif text
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.6,
    "legend.frameon": False,
}


def apply_base_style() -> None:
    """Apply the shared regime-A rcParams. Call once before building a figure."""
    mpl.rcParams.update(BASE_RCPARAMS)


def despine(
    ax: Axes,
    *,
    top: bool = True,
    right: bool = True,
    left: bool = False,
    bottom: bool = False,
) -> None:
    """Hide the named spines (default top+right, the line-plot convention).

    ``left=True`` for strip/dumbbell charts with no y-axis rule; all-False for
    framed heatmaps.
    """
    for side, hide in (
        ("top", top),
        ("right", right),
        ("left", left),
        ("bottom", bottom),
    ):
        if hide:
            ax.spines[side].set_visible(False)
