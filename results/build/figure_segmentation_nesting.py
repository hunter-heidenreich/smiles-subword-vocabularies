"""Merged segmentation + nesting figure.

A multi-panel version of the nesting schematic that subsumes
``\\input{tables/segmentation_example}`` as well: one stacked panel per
molecule, each drawing both arms' token boxes on the shared OpenSMILES glyph
axis (Unigram-LM above, BPE below) with per-row token counts and compression
ratios annotated, so a single float carries membership (which pieces),
granularity (how many), and compatibility (how the cuts nest) at once.

Reuses the single-panel machinery: the molecule and tokenizer cell are pinned,
glyph alignment reuses the measurement's ``glyph_count_map`` / glyph-tuple maps,
and the headless Agg backend with a suppressed PDF creation date keeps the bytes
deterministic.

Usage::

    uv run python results/build/figure_segmentation_nesting.py
"""

from __future__ import annotations

import sys

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import style
from matplotlib.patches import FancyBboxPatch

from smiles_subword.config import cell_artifact_name
from smiles_subword.paths import RESULTS_FIGURES_DIR, tokenizer_artifact_dir
from smiles_subword.tokenize.adapters.smirk import SmirkAdapter
from smiles_subword.tokenize.adapters.smirk_unigram import UnigramSmirkAdapter
from smiles_subword.tokenize.measure.fertility.runner import glyph_count_map
from smiles_subword.tokenize.measure.jaccard.runner import glyph_tuple_map

style.apply_base_style()

_SAVE_METADATA = {"CreationDate": None}
_BPE_COLOR = style.BPE_COLOR
_UL_COLOR = style.UNIGRAM_COLOR

# Merge-brackets (MB) to match tables/segmentation_example: under MB the
# Unigram-LM arm forms its own learned pieces (ccccc, [C@@, [O-, [Na), so the
# figure shows UL-side membership rather than a trivially atomic top row.
CORPUS = "pubchem"
BPE_NAME = cell_artifact_name("bpe", 1024, "mb")
UL_NAME = cell_artifact_name("unigram", 1024, "mb")

# The three chemotypes from tables/segmentation_example, so the merged figure is
# a direct drop-in: drug-like, natural product (stereocentre + strong nesting),
# and a salt (coverage / charged brackets).
MOLECULES = [
    ("Drug-like (aspirin)", "CC(=O)Oc1ccccc1C(=O)O"),
    ("Natural product", "CC(=O)CCC[C@@H]1CC=CC(=O)O1"),
    ("Salt (sodium acetate)", "CC(=O)[O-].[Na+]"),
]


def _spans(glyph_lengths: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    pos = 0
    for length in glyph_lengths:
        spans.append((pos, pos + length))
        pos += length
    return spans


def _draw_boxes(
    ax: plt.Axes, spans: list[tuple[int, int]], y0: float, y1: float, color: str
) -> None:
    """Draw token boxes; multi-glyph (learned) pieces are filled, single-glyph
    base-passthrough tokens are left hollow, echoing the table's shading cue."""
    pad = 0.12
    for start, end in spans:
        learned = (end - start) > 1
        layers = (
            ((color, "white", 1.2), ("none", color, 1.4))
            if learned
            else (("none", color, 1.4),)
        )
        for face, edge, lw in layers:
            ax.add_patch(
                FancyBboxPatch(
                    (start - 0.5 + pad, y0),
                    (end - start) - 2 * pad,
                    y1 - y0,
                    boxstyle="round,pad=0.0,rounding_size=0.12",
                    facecolor=face,
                    edgecolor=edge,
                    alpha=0.30 if face != "none" else 1.0,
                    linewidth=lw,
                )
            )


def _segment(
    mol: str,
    bpe: SmirkAdapter,
    ul: UnigramSmirkAdapter,
    bpe_counts: dict[int, int],
    ul_tuples: dict[int, tuple[str, ...]],
) -> tuple[list[str], list[tuple[int, int]], list[tuple[int, int]], int, int]:
    bids = bpe.encode_batch([mol], add_special_tokens=False)[0]
    uids = ul.encode_batch([mol], add_special_tokens=False)[0]
    bpe_lengths = [bpe_counts.get(t, 1) for t in bids]
    ul_tok_tuples = [ul_tuples.get(t, ("?",)) for t in uids]
    glyphs = [g for tup in ul_tok_tuples for g in tup]
    ul_lengths = [len(t) for t in ul_tok_tuples]
    if sum(bpe_lengths) != len(glyphs):
        msg = f"BPE/UL glyph length mismatch on {mol}"
        raise ValueError(msg)
    return glyphs, _spans(bpe_lengths), _spans(ul_lengths), len(bids), len(uids)


def _draw_panel(
    ax: plt.Axes,
    label: str,
    mol: str,
    glyphs: list[str],
    bpe_spans: list[tuple[int, int]],
    ul_spans: list[tuple[int, int]],
    n_bpe: int,
    n_ul: int,
) -> None:
    n = len(glyphs)
    bpe_cuts = {e for _, e in bpe_spans[:-1]}
    ul_cuts = {e for _, e in ul_spans[:-1]}

    for k in sorted(bpe_cuts):
        ax.axvline(
            k - 0.5, color="#999999", linewidth=0.6, linestyle=(0, (2, 2)), zorder=0
        )

    _draw_boxes(ax, ul_spans, 0.55, 1.35, _UL_COLOR)
    _draw_boxes(ax, bpe_spans, -1.35, -0.55, _BPE_COLOR)
    for i, g in enumerate(glyphs):
        ax.text(i, 0.0, g, ha="center", va="center", fontsize=10, family="monospace")

    comp_ul = n / n_ul
    comp_bpe = n / n_bpe
    ax.text(
        -0.9,
        0.95,
        f"UL  {n_ul}\n{comp_ul:.2f}×",
        ha="right",
        va="center",
        fontsize=7.5,
        color=_UL_COLOR,
        fontweight="bold",
    )
    ax.text(
        -0.9,
        -0.95,
        f"BPE {n_bpe}\n{comp_bpe:.2f}×",
        ha="right",
        va="center",
        fontsize=7.5,
        color=_BPE_COLOR,
        fontweight="bold",
    )

    nest = len(ul_cuts - bpe_cuts)
    agree = len(ul_cuts & bpe_cuts)
    conflict = len(bpe_cuts - ul_cuts)
    ax.set_title(
        f"{label}:  {mol}    "
        f"(base {n}; agree-cut {agree}, nest {nest}, conflict {conflict})",
        fontsize=8.5,
        loc="left",
    )
    ax.set_xlim(-4.2, n - 0.3)
    ax.set_ylim(-1.7, 1.7)
    ax.axis("off")


def main() -> int:
    bdir = tokenizer_artifact_dir(CORPUS, BPE_NAME)
    udir = tokenizer_artifact_dir(CORPUS, UL_NAME)
    bpe = SmirkAdapter.load(bdir)
    ul = UnigramSmirkAdapter.load(udir)
    bpe_counts = glyph_count_map(bdir, "bpe")
    ul_tuples = glyph_tuple_map(udir, "unigram")

    segmented = [
        (lab, mol, *_segment(mol, bpe, ul, bpe_counts, ul_tuples))
        for lab, mol in MOLECULES
    ]
    max_n = max(len(s[2]) for s in segmented)

    fig, axes = plt.subplots(
        len(segmented), 1, figsize=(min(0.42 * max_n + 1.6, 9.0), 1.7 * len(segmented))
    )
    for ax, (lab, mol, glyphs, bpe_spans, ul_spans, n_bpe, n_ul) in zip(
        axes, segmented, strict=True
    ):
        _draw_panel(ax, lab, mol, glyphs, bpe_spans, ul_spans, n_bpe, n_ul)

    fig.tight_layout(h_pad=1.4)
    out = RESULTS_FIGURES_DIR / "segmentation_nesting.pdf"
    RESULTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="pdf", metadata=_SAVE_METADATA)
    plt.close(fig)
    print(f"wrote {out} ({len(segmented)} panels, max base glyphs={max_n})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
