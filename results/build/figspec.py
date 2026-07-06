"""Shape the per-cell measurement scalars into renderer-agnostic figure specs.

Pure data-shaping with no plotting backend: each
builder reads the per-cell cross-arm scalars (via :func:`extract.cross_axis_cells`,
joined across Jaccard / Fertility / Distribution) and returns frozen dataclasses
describing *what* to plot. The matplotlib rendering of these specs lives in
:mod:`render`.

Two figures, one builder each:

- :func:`cross_v_trend_spec` — the magnitude scalars (``J``, rel ``|Δf|``,
  ``|ΔD|``) vs ``V`` per ``(corpus, boundary)`` series.
- :func:`algo_boundary_interaction_spec` — the signed ``value_NMB − value_MB``
  interaction term per ``(corpus, V)`` per measurement.

One reference line is drawn on the trend panels: the measured ``|ΔD|`` noise
floor (``0.002``), Distribution's single source of truth. The overlap (``J``)
and fertility panels carry no reference line: both contrasts are read as
effect sizes, not against a threshold, so their panel thresholds are ``None``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import extract
from _corpora import CORPUS_LABEL, CORPUS_RANK

from smiles_subword.tokenize.measure.distribution import DELTA_D_NOISE_FLOOR

if TYPE_CHECKING:
    from collections.abc import Iterable

# Measurement keys; the per-cell attribute each maps to differs between the
# magnitude (trend) view and the signed cross-arm (interaction) view.
JACCARD = "jaccard"
FERTILITY = "fertility"
IMBALANCE_D = "imbalance_d"
MEASUREMENTS = (JACCARD, FERTILITY, IMBALANCE_D)

# Trend panels plot the magnitude scalar; interaction panels the signed gap.
_TREND_ATTR = {
    JACCARD: "jaccard",
    FERTILITY: "rel_fertility",
    IMBALANCE_D: "abs_delta_d",
}
_SIGNED_ATTR = {
    JACCARD: "jaccard",
    FERTILITY: "delta_fertility_signed",
    IMBALANCE_D: "delta_d_signed",
}

MEASUREMENT_LABEL = {
    JACCARD: "$J$",
    FERTILITY: r"$\mathrm{rel.}\,|\Delta f|$",
    IMBALANCE_D: r"$|\Delta D|$",
}

# The interaction figure plots the *signed* cross-arm gap per boundary
# (``value_nmb``/``value_mb`` are ``f_BPE - f_UL`` in tokens and ``D_BPE - D_UL``,
# not the magnitude scalars rel|Δf| / |ΔD|), so it needs its own axis labels naming
# those signed gaps. Reusing MEASUREMENT_LABEL here would mislabel the fertility
# panel (it is a token gap, not relative) and the imbalance panel (it is signed,
# so its bars read as the negative of |ΔD|).
INTERACTION_LABEL = {
    JACCARD: "$J$",
    FERTILITY: r"$(f_{\mathrm{BPE}}-f_{\mathrm{UL}})$",
    IMBALANCE_D: r"$(D_{\mathrm{BPE}}-D_{\mathrm{UL}})$",
}

# The overlap and fertility contrasts are read as effect sizes, not against a
# threshold, so their trend panels draw no reference line (``None``). Only the
# distribution (imbalance) panel keeps a line, the measured noise floor.
MEASUREMENT_THRESHOLD: dict[str, float | None] = {
    JACCARD: None,
    FERTILITY: None,
    IMBALANCE_D: DELTA_D_NOISE_FLOOR,
}

# In-panel label for the one reference line, so the dotted line names itself
# rather than relying on the caption.
MEASUREMENT_THRESHOLD_LABEL: dict[str, str | None] = {
    JACCARD: None,
    FERTILITY: None,
    IMBALANCE_D: "noise floor",
}


def _corpus(corpus: str) -> str:
    return CORPUS_LABEL.get(corpus, corpus)


# --------------------------------------------------------------------------- #
# Figure 1 — cross-V trends                                                   #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TrendSeries:
    """One ``(corpus, boundary)`` series of a magnitude scalar across ``V``."""

    corpus: str
    boundary: str
    xs: tuple[int, ...]
    ys: tuple[float, ...]


@dataclass(frozen=True)
class TrendPanel:
    """One panel: a measurement vs ``V`` across all series.

    ``threshold`` is the reference line to draw (the ``0.002`` noise floor on
    the imbalance panel) or ``None`` for the overlap and fertility panels, which
    are read as effect sizes and carry no line.
    ``threshold_label`` is its in-panel caption (``None`` when no line).
    """

    measurement: str
    label: str
    threshold: float | None
    threshold_label: str | None
    series: tuple[TrendSeries, ...]


@dataclass(frozen=True)
class TrendFigureSpec:
    """The cross-``V`` trend figure (one panel per measurement)."""

    panels: tuple[TrendPanel, ...]


def cross_v_trend_spec() -> TrendFigureSpec:
    """Build the cross-``V`` trend spec from the per-cell measurement scalars."""
    cells = extract.cross_axis_cells()
    # REAL-Space exists only at V=1024 (the anchor cell); a single point has no
    # cross-V trend, so it is excluded here (it appears in the overlap and
    # fertility figures instead) to keep this a trends-only figure.
    grouped = _group_by_series(c for c in cells if c.corpus != "real_space")
    panels: list[TrendPanel] = []
    for measurement in MEASUREMENTS:
        attr = _TREND_ATTR[measurement]
        series: list[TrendSeries] = []
        for (corpus, boundary), cs in grouped:
            series.append(
                TrendSeries(
                    corpus=corpus,
                    boundary=boundary,
                    xs=tuple(c.vocab_size for c in cs),
                    ys=tuple(float(getattr(c, attr)) for c in cs),
                )
            )
        panels.append(
            TrendPanel(
                measurement=measurement,
                label=MEASUREMENT_LABEL[measurement],
                threshold=MEASUREMENT_THRESHOLD[measurement],
                threshold_label=MEASUREMENT_THRESHOLD_LABEL[measurement],
                series=tuple(series),
            )
        )
    return TrendFigureSpec(panels=tuple(panels))


# --------------------------------------------------------------------------- #
# Figure 2 — algorithm×boundary interaction                                   #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class InteractionBar:
    """One signed interaction term (NMB − MB) at a ``(corpus, V)``."""

    label: str
    term: float
    sign: int
    corpus: str
    vocab_size: int


@dataclass(frozen=True)
class InteractionPanel:
    """One panel: a measurement's interaction terms across ``(corpus, V)``."""

    measurement: str
    label: str
    bars: tuple[InteractionBar, ...]


@dataclass(frozen=True)
class InteractionSpec:
    """The algorithm×boundary interaction figure (one panel per measurement)."""

    panels: tuple[InteractionPanel, ...]


def algo_boundary_interaction_spec() -> InteractionSpec:
    """Build the interaction spec from the per-cell signed cross-arm gaps."""
    by_cv = _group_by_cv(extract.cross_axis_cells())
    panels: list[InteractionPanel] = []
    for measurement in MEASUREMENTS:
        attr = _SIGNED_ATTR[measurement]
        bars: list[InteractionBar] = []
        for corpus, vocab_size, nmb, mb in by_cv:
            term = float(getattr(nmb, attr)) - float(getattr(mb, attr))
            bars.append(
                InteractionBar(
                    label=f"{_corpus(corpus)} {vocab_size}",
                    term=term,
                    sign=(term > 0) - (term < 0),
                    corpus=corpus,
                    vocab_size=vocab_size,
                )
            )
        panels.append(
            InteractionPanel(
                measurement=measurement,
                label=INTERACTION_LABEL[measurement],
                bars=tuple(bars),
            )
        )
    return InteractionSpec(panels=tuple(panels))


def _group_by_series(
    cells: Iterable[extract.CrossAxisCell],
) -> list[tuple[tuple[str, str], list[extract.CrossAxisCell]]]:
    """Group cells into ``(corpus, boundary)`` series for the trend figure.

    Series order is corpus typology then boundary; points within a series are
    ascending in ``V``.
    """
    groups: dict[tuple[str, str], list[extract.CrossAxisCell]] = {}
    for c in cells:
        groups.setdefault((c.corpus, c.boundary), []).append(c)
    ordered = sorted(
        groups.items(), key=lambda kv: (CORPUS_RANK.get(kv[0][0], 99), kv[0][1])
    )
    return [(key, sorted(cs, key=lambda c: c.vocab_size)) for key, cs in ordered]


def _group_by_cv(
    cells: Iterable[extract.CrossAxisCell],
) -> list[tuple[str, int, extract.CrossAxisCell, extract.CrossAxisCell]]:
    """Group cells by ``(corpus, V)`` where both boundary arms are present.

    Yields ``(corpus, vocab_size, nmb_cell, mb_cell)`` ordered by corpus
    typology then ``V`` — the algorithm×boundary interaction needs both arms.
    """
    by_cv: dict[tuple[str, int], dict[str, extract.CrossAxisCell]] = {}
    for c in cells:
        by_cv.setdefault((c.corpus, c.vocab_size), {})[c.boundary] = c
    out = [
        (corpus, vocab_size, bmap["nmb"], bmap["mb"])
        for (corpus, vocab_size), bmap in by_cv.items()
        if "nmb" in bmap and "mb" in bmap
    ]
    out.sort(key=lambda t: (CORPUS_RANK.get(t[0], 99), t[1]))
    return out


__all__ = [
    "MEASUREMENTS",
    "MEASUREMENT_LABEL",
    "InteractionBar",
    "InteractionPanel",
    "InteractionSpec",
    "TrendFigureSpec",
    "TrendPanel",
    "TrendSeries",
    "algo_boundary_interaction_spec",
    "cross_v_trend_spec",
]
