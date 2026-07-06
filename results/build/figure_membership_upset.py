"""Render the vocabulary-membership figure (BPE-only / shared / Unigram-LM-only).

Makes the near-disjointness of the learned multi-glyph vocabularies legible as a
normalized diverging bar per matched condition: BPE-only pieces extend left,
Unigram-LM-only right, and the shared core is a thin neutral spine straddling the
centre. Each bar is normalized to the combined (union) learned-piece set, so its
full width is 100% and the spine width is the unweighted overlap ``J``; the
symmetry of the two wings shows the near-equal exclusive sets, and their
asymmetry the Unigram saturation on narrow alphabets. The absolute union size is
annotated at the right. The three counts are derived exactly from the deposited
Jaccard and per-arm multi-glyph counts, given the overlap ``J`` and the per-arm
sizes ``|A|``, ``|B|``,

    shared = J * (|A| + |B|) / (1 + J),

with ``bpe_only = |A| - shared`` and ``ul_only = |B| - shared``; this recovers the
exact integer intersection (``J`` and the counts are exact).
Output lands at ``results/figures/membership_upset.pdf``. Reproducible: the headless
Agg backend and a suppressed PDF creation date make the bytes deterministic.

Usage::

    uv run python results/build/figure_membership_upset.py
"""

from __future__ import annotations

import sys

import matplotlib as mpl

mpl.use("Agg")

import extract
import matplotlib.pyplot as plt
import style
from _corpora import CORPUS_LABEL
from matplotlib.patches import Patch

from smiles_subword.paths import RESULTS_FIGURES_DIR

style.apply_base_style()

_SAVE_METADATA = {"CreationDate": None}

# Arm colours from the shared palette (BPE blue, Unigram-LM vermillion); the
# shared core is a neutral grey midpoint, so the diverging axis runs
# blue -> grey -> vermillion. Arm identity is also carried by side (BPE left,
# Unigram right), so the reading survives grayscale and colour-vision deficiency.
_BPE_COLOR = style.BPE_COLOR
_UL_COLOR = style.UNIGRAM_COLOR
_SHARED_COLOR = "#8A8A8A"


def _membership(j: float, n_bpe: int, n_ul: int) -> tuple[int, int, int]:
    """Exact (bpe_only, shared, ul_only) from the overlap and per-arm counts."""
    shared = round(j * (n_bpe + n_ul) / (1.0 + j))
    bpe_only = n_bpe - shared
    ul_only = n_ul - shared
    union = bpe_only + shared + ul_only
    # Guard: the derived split must reproduce the deposited Jaccard.
    if union > 0 and abs(shared / union - j) >= 1e-6:
        msg = f"derived shared {shared} inconsistent with J={j}"
        raise ValueError(msg)
    return bpe_only, shared, ul_only


def main() -> int:
    rows = [
        r
        for r in extract.jaccard_rows()
        if r.bpe_n_multi is not None and r.unigram_n_multi is not None
    ]
    # extract.jaccard_rows() is already ordered by corpus typology, V, boundary.
    data: list[tuple[str, int, str, int, int, int]] = []
    for r in rows:
        # Guaranteed non-None by the filter above; assert narrows it for the checker.
        assert r.bpe_n_multi is not None
        assert r.unigram_n_multi is not None
        b, s, u = _membership(r.jaccard, r.bpe_n_multi, r.unigram_n_multi)
        data.append((r.corpus, r.vocab_size, r.boundary, b, s, u))

    n = len(data)
    fig, ax = plt.subplots(figsize=(7.2, 0.34 * n + 1.0))

    groups: dict[str, list[int]] = {}
    for i, (corpus, _v, _bd, b, s, u) in enumerate(data):
        y = n - 1 - i  # first row on top
        groups.setdefault(corpus, []).append(y)
        total = b + s + u
        scale = 100.0 / total
        bw, sw, uw = b * scale, s * scale, u * scale
        # Shared spine centred on 0; the two exclusive wings flank it.
        ax.barh(
            y,
            sw,
            left=-sw / 2,
            color=_SHARED_COLOR,
            height=0.74,
            edgecolor="white",
            lw=0.6,
            zorder=3,
        )
        ax.barh(
            y,
            bw,
            left=-sw / 2 - bw,
            color=_BPE_COLOR,
            height=0.74,
            edgecolor="white",
            lw=0.6,
            zorder=3,
        )
        ax.barh(
            y,
            uw,
            left=sw / 2,
            color=_UL_COLOR,
            height=0.74,
            edgecolor="white",
            lw=0.6,
            zorder=3,
        )
        ax.text(118, y, f"{total:,}", va="center", ha="right", fontsize=6, color="0.45")

    ax.set_yticks([n - 1 - i for i in range(n)])
    ax.set_yticklabels(
        [rf"$V{{=}}{v}$ {bd.upper()}" for (_corpus, v, bd, *_counts) in data],
        fontsize=7,
    )
    # One bold corpus label per group in a left gutter, plus faint separators.
    for corpus, gys in groups.items():
        ax.text(
            -0.165,
            (max(gys) + min(gys)) / 2,
            CORPUS_LABEL.get(corpus, corpus),
            transform=ax.get_yaxis_transform(),
            rotation=90,
            va="center",
            ha="center",
            fontsize=8.5,
            fontweight="bold",
        )
        if min(gys) > 0:
            ax.axhline(min(gys) - 0.5, color="0.85", lw=0.6, zorder=1)
    ax.axvline(0, color="0.35", lw=0.9, zorder=4)
    ax.text(
        118,
        n - 0.15,
        "pieces",
        va="bottom",
        ha="right",
        fontsize=6,
        color="0.45",
        style="italic",
    )

    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlim(-100, 122)
    ax.set_xticks([-100, -50, 0, 50, 100])
    ax.set_xticklabels(["100", "50", "0", "50", "100"], fontsize=8)
    ax.set_xlabel(
        r"BPE-only $\leftarrow$   share of combined learned pieces (%)"
        r"   $\rightarrow$ Unigram-LM-only",
        fontsize=9,
    )
    style.despine(ax)
    ax.legend(
        handles=[
            Patch(color=_BPE_COLOR, label="BPE-only"),
            Patch(color=_SHARED_COLOR, label="shared core"),
            Patch(color=_UL_COLOR, label="Unigram-LM-only"),
        ],
        fontsize=8,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        frameon=False,
    )
    fig.subplots_adjust(left=0.20, right=0.97, top=0.95, bottom=0.07)

    RESULTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_FIGURES_DIR / "membership_upset.pdf"
    fig.savefig(out, format="pdf", metadata=_SAVE_METADATA)
    plt.close(fig)
    print(f"wrote {out} ({n} conditions)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
