"""Generate the arm-exclusive piece-length distribution figures.

Two figures, both read from the committed tokenizer artifacts via the package's
``glyph_tuple_map`` (same source as the appendix tables):

``piece_length_dist.pdf`` -- per corpus (PubChem, ZINC-22, COCONUT) at V=1024
  (NMB), the glyph-length distribution of the BPE-only vs Unigram-LM-only pieces.
  Unigram-LM is capped at 16 glyphs (``max_piece_length``); BPE is uncapped, with
  a corpus-dependent tail (to 48 on PubChem, 32 on COCONUT, 13 on ZINC-22).

``piece_length_cap.pdf`` -- PubChem at V=1024 (NMB), the Unigram-LM length
  distribution as ``max_piece_length`` is swept over {4,8,16,32,64,128}: the
  right edge of the distribution tracks the cap exactly, showing the wall is the
  hyperparameter, not an intrinsic limit.

Standalone; run with:

    uv run python results/build/figure_piece_length.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import style
from _pieces import multi_glyph_set
from matplotlib.colors import LinearSegmentedColormap

from smiles_subword.paths import RESULTS_FIGURES_DIR

HERE = Path(__file__).resolve().parent

VOCAB_SIZE = 1024
BOUNDARY = "nmb"
UL_CAP = 16  # Unigram-LM max_piece_length default
CORPORA = [("pubchem", "PubChem"), ("zinc22", "ZINC-22"), ("coconut", "COCONUT")]
CAP_LADDER = [4, 8, 16, 32, 64, 128]
BPE_COLOR = style.BPE_COLOR
UL_COLOR = style.UNIGRAM_COLOR


def _ul_lengths_for_cap(cap: int) -> list[int]:
    pieces = multi_glyph_set(
        "pubchem", VOCAB_SIZE, BOUNDARY, "unigram", suffix=f"ix_mpl_v_{cap}"
    )
    return [len(t) for t in pieces]


def _fractions(lengths: list[int], xs: range) -> list[float]:
    counts = Counter(lengths)
    n = len(lengths)
    return [counts.get(x, 0) / n for x in xs]


def make_exclusive_figure() -> None:
    """Per-corpus BPE-only vs Unigram-LM-only glyph-length distributions."""
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6), sharey=True)
    for ax, (corpus, label) in zip(axes, CORPORA, strict=True):
        style.despine(ax)
        bpe = multi_glyph_set(corpus, VOCAB_SIZE, BOUNDARY, "bpe")
        ul = multi_glyph_set(corpus, VOCAB_SIZE, BOUNDARY, "unigram")
        bpe_only = [len(t) for t in bpe - ul]
        ul_only = [len(t) for t in ul - bpe]
        xs = range(2, max(bpe_only + ul_only) + 1)
        ax.step(
            list(xs),
            _fractions(bpe_only, xs),
            where="mid",
            color=BPE_COLOR,
            lw=1.3,
            label="BPE-only",
        )
        ax.step(
            list(xs),
            _fractions(ul_only, xs),
            where="mid",
            color=UL_COLOR,
            lw=1.3,
            label="Unigram-LM-only",
        )
        ax.axvline(UL_CAP, color="0.5", ls=":", lw=0.9)
        ax.text(
            UL_CAP + 0.8,
            ax.get_ylim()[1] * 0.95,
            "UL cap (16)",
            fontsize=6.5,
            color="0.4",
            va="top",
        )
        ax.annotate(
            f"BPE max {max(bpe_only)}",
            xy=(0.97, 0.60),
            xycoords="axes fraction",
            ha="right",
            fontsize=6.5,
            color=BPE_COLOR,
        )
        ax.set_title(label, fontsize=9.5)
        ax.set_xlabel("piece length (glyphs)")
    axes[0].set_ylabel("fraction of arm-exclusive pieces")
    axes[0].legend(frameon=False, fontsize=7, handletextpad=0.4, loc="upper right")
    fig.tight_layout()
    fig.savefig(
        RESULTS_FIGURES_DIR / "piece_length_dist.pdf",
        bbox_inches="tight",
        metadata={"CreationDate": None},
    )
    plt.close(fig)


def make_cap_figure() -> None:
    """Unigram-LM length distribution as max_piece_length sweeps the ladder."""
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    style.despine(ax)
    # Every curve here is Unigram-LM; shade by cap with a single-hue
    # Unigram-vermillion ramp (light = small cap, dark = large) rather than a
    # blue-yellow map, whose blue end would read as BPE against the arm palette
    # (blue=BPE, vermillion=Unigram-LM; cf. style.py).
    cmap = LinearSegmentedColormap.from_list("ul_seq", ["#F6C9A6", UL_COLOR, "#5A2600"])
    n = len(CAP_LADDER)
    colors = [cmap(0.12 + 0.88 * i / (n - 1)) for i in range(n)]
    xmax = max(max(_ul_lengths_for_cap(c)) for c in CAP_LADDER)
    xs = range(2, xmax + 1)
    for cap, color in zip(CAP_LADDER, colors, strict=True):
        lengths = _ul_lengths_for_cap(cap)
        ax.step(
            list(xs),
            _fractions(lengths, xs),
            where="mid",
            color=color,
            lw=1.3,
            label=f"{cap}",
        )
        ax.axvline(cap, color=color, ls=":", lw=0.7, alpha=0.6)
    ax.set_yscale("log")
    ax.set_ylim(8e-4, 1.0)
    ax.set_xlabel("piece length (glyphs)")
    ax.set_ylabel("fraction of Unigram-LM multi-glyph pieces")
    ax.set_title("PubChem $V{=}1024$ (NMB)", fontsize=9.5)
    ax.legend(
        title="max piece length (cap)",
        frameon=False,
        fontsize=7,
        title_fontsize=7,
        ncol=2,
        handletextpad=0.4,
        labelspacing=0.25,
    )
    fig.tight_layout()
    fig.savefig(
        RESULTS_FIGURES_DIR / "piece_length_cap.pdf",
        bbox_inches="tight",
        metadata={"CreationDate": None},
    )
    plt.close(fig)


def main() -> int:
    style.apply_base_style()
    make_exclusive_figure()
    make_cap_figure()
    return 0


if __name__ == "__main__":
    sys.exit(main())
