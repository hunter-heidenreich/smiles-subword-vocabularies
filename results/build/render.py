"""Matplotlib rendering of the two cross-axis figure specs.

The ``cross_v_trends`` and ``algo_boundary_interaction`` figures share this
non-trivial split-legend / signed-bar rendering, so it lives here as a helper
library; the thin ``figure_cross_v_trends.py`` and
``figure_algo_boundary_interaction.py`` drivers call :func:`render_trend` /
:func:`render_interaction`, mirroring how every other figure script is its own
standalone driver. All data shaping lives in :mod:`figspec`, which is
backend-free.

Rendering is deterministic — the headless ``Agg`` backend, fixed figure sizes,
and a suppressed PDF ``CreationDate`` — so regenerating a figure from unchanged
inputs produces stable bytes. matplotlib is provided by the ``results``
optional-dependency extra; CI installs it via ``uv sync --all-extras --dev``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import style
from _corpora import CORPUS_LABEL
from matplotlib.lines import Line2D

if TYPE_CHECKING:
    from pathlib import Path

    import figspec
    from matplotlib.figure import Figure

style.apply_base_style()

_SAVE_METADATA = {"CreationDate": None}

# Palette and display names come from the shared modules: corpus colour+marker
# and boundary linestyle from style, the corpus labels from _corpora (so the text
# and figure surfaces spell the four corpora the same).
_CORPUS_COLOR = style.CORPUS_COLOR
_CORPUS_MARKER = style.CORPUS_MARKER
_BOUNDARY_STYLE = style.BOUNDARY_STYLE
_BOUNDARY_LABEL = style.BOUNDARY_LABEL


def render_trend(spec: figspec.TrendFigureSpec, path: Path) -> Path:
    """Render the cross-``V`` trend figure (one panel per measurement).

    Encoding is split so the two distinctions stay legible: colour and marker
    both carry corpus (a redundant grayscale-safe channel), while linestyle
    alone carries the boundary policy. The legend is split to match — a corpus
    key and a boundary key — instead of one combinatorial pile. Every panel
    shares a zero baseline so a small trend in one is not visually exaggerated.
    """
    n = max(len(spec.panels), 1)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.8), squeeze=False)
    corpora: list[str] = []
    for ax, panel in zip(axes[0], spec.panels, strict=False):
        for series in panel.series:
            if series.corpus not in corpora:
                corpora.append(series.corpus)
            ax.plot(
                series.xs,
                series.ys,
                color=_CORPUS_COLOR.get(series.corpus, "#444444"),
                linestyle=_BOUNDARY_STYLE.get(series.boundary, ":"),
                marker=_CORPUS_MARKER.get(series.corpus, "o"),
                label="_nolegend_",
            )
        if panel.threshold is not None:
            ax.axhline(panel.threshold, color="0.4", linewidth=0.9, linestyle=":")
            if panel.threshold_label is not None:
                ax.text(
                    0.98,
                    panel.threshold,
                    panel.threshold_label,
                    transform=ax.get_yaxis_transform(),
                    ha="right",
                    va="bottom",
                    fontsize=7,
                    color="0.4",
                )
        ax.set_xscale("log", base=2)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.08)  # shared zero baseline + headroom
        ax.set_xlabel("vocabulary size $V$")
        ax.set_title(panel.label)
        style.despine(ax)
    # Plain integer V ticks (256, 512, ...) rather than the log2 axis's default
    # 2^n powers, so the size axis reads the same as every other figure's.
    all_vs = sorted(
        {x for panel in spec.panels for series in panel.series for x in series.xs}
    )
    for ax in axes[0]:
        ax.set_xticks(all_vs, [str(int(v)) for v in all_vs])
        ax.tick_params(axis="x", which="minor", length=0)
    axes[0][0].set_ylabel("value")

    corpus_handles = [
        Line2D(
            [0],
            [0],
            color=_CORPUS_COLOR.get(c, "#444444"),
            marker=_CORPUS_MARKER.get(c, "o"),
            linestyle="",
            markersize=7,
            label=CORPUS_LABEL.get(c, c),
        )
        for c in corpora
    ]
    boundary_handles = [
        Line2D(
            [0],
            [0],
            color="0.3",
            linestyle=_BOUNDARY_STYLE[b],
            label=_BOUNDARY_LABEL[b],
        )
        for b in ("nmb", "mb")
    ]
    fig.legend(
        handles=corpus_handles,
        title="corpus",
        loc="lower center",
        bbox_to_anchor=(0.34, 0.0),
        ncol=len(corpus_handles),
        fontsize="small",
        frameon=False,
    )
    fig.legend(
        handles=boundary_handles,
        title="boundary",
        loc="lower center",
        bbox_to_anchor=(0.72, 0.0),
        ncol=2,
        fontsize="small",
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    return _save(fig, path)


def render_interaction(spec: figspec.InteractionSpec, path: Path) -> Path:
    """Render the algorithm×boundary interaction (NMB − MB) vs ``V`` per corpus.

    Each panel is one contrast; within it the signed interaction term is drawn
    against ``V`` as one colour+marker line per corpus (the same cross-``V``
    palette the trend figure uses), with sign read off the zero baseline. The
    panels share the ``V`` axis, so the size labels are drawn once at the
    bottom rather than repeated, rotated, under all three.
    """
    n = max(len(spec.panels), 1)
    fig, axes = plt.subplots(n, 1, figsize=(6.8, 1.9 * n), sharex=True, squeeze=False)
    corpora: list[str] = []
    vocab_sizes: set[int] = set()
    for ax, panel in zip(axes[:, 0], spec.panels, strict=False):
        by_corpus: dict[str, list[figspec.InteractionBar]] = {}
        for bar in panel.bars:
            by_corpus.setdefault(bar.corpus, []).append(bar)
            vocab_sizes.add(bar.vocab_size)
        for corpus, cbars in by_corpus.items():
            if corpus not in corpora:
                corpora.append(corpus)
            pts = sorted(cbars, key=lambda b: b.vocab_size)
            ax.plot(
                [b.vocab_size for b in pts],
                [b.term for b in pts],
                color=_CORPUS_COLOR.get(corpus, "#444444"),
                marker=_CORPUS_MARKER.get(corpus, "o"),
                linestyle="-",
                label="_nolegend_",
            )
        ax.axhline(0.0, color="0.3", linewidth=0.8)
        # Symmetric-ish limits that always include the zero baseline, so a
        # near-zero panel is not autoscaled into looking large.
        terms = [bar.term for bar in panel.bars] + [0.0]
        lo, hi = min(terms), max(terms)
        pad = 0.12 * ((hi - lo) or 1.0)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xscale("log", base=2)
        ax.set_ylabel(f"$\\Delta$ {panel.label}")
        style.despine(ax)
    vs = sorted(vocab_sizes)
    axes[-1, 0].set_xticks(vs, [str(v) for v in vs])
    axes[-1, 0].tick_params(axis="x", which="minor", length=0)
    axes[-1, 0].set_xlabel("vocabulary size $V$")
    corpus_handles = [
        Line2D(
            [0],
            [0],
            color=_CORPUS_COLOR.get(c, "#444444"),
            marker=_CORPUS_MARKER.get(c, "o"),
            linestyle="",
            markersize=7,
            label=CORPUS_LABEL.get(c, c),
        )
        for c in corpora
    ]
    fig.legend(
        handles=corpus_handles,
        title="corpus",
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=len(corpus_handles),
        fontsize="small",
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return _save(fig, path)


def _save(fig: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf", metadata=_SAVE_METADATA)
    plt.close(fig)
    return path


__all__ = [
    "render_interaction",
    "render_trend",
]
