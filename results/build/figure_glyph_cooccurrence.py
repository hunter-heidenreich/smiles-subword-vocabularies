"""Generate the glyph co-occurrence heatmap (appendix).

For PubChem $V{=}2048$, counts adjacent base-glyph pairs (bigrams) within every
learned multi-glyph piece, separately for Smirk-GPE (BPE) and Unigram-LM, under
both boundary policies (no-merge-brackets NMB, merge-brackets MB). Renders a
2x3 grid (rows {NMB, MB}, columns {BPE, Unigram-LM, log2(BPE/UL)}) with the
158 base glyphs ordered by OpenSMILES role class (matching the base-glyph
appendix table). The point is structural: under NMB no merge crosses ``[``/``]``,
so brackets and bracket-atom elements never co-occur (their region is empty by
construction); MB unlocks them. Emits ``results/figures/glyph_cooccurrence.pdf``.

Pinned to the committed $V{=}2048$ tokenizers, so re-running reproduces the
figure from the deposited artifacts. Needs the ``figures`` extra (matplotlib)
and reads the tokenizer artifacts (like the other piece-level appendix builders).

Usage::

    uv run python results/build/figure_glyph_cooccurrence.py
"""

from __future__ import annotations

import sys
from itertools import pairwise

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import style
from matplotlib.colors import LogNorm, TwoSlopeNorm

from smiles_subword.config import TokenizerAlgo, cell_artifact_name
from smiles_subword.paths import RESULTS_FIGURES_DIR, tokenizer_artifact_dir
from smiles_subword.tokenize.adapters.smirk import SmirkAdapter
from smiles_subword.tokenize.measure.jaccard.runner import glyph_tuple_map

CORPUS = "pubchem"
OUT = RESULTS_FIGURES_DIR / "glyph_cooccurrence.pdf"
V = 2048

# OpenSMILES role classes (mirrors results/build/table_base_glyphs.py); the rare
# bracket-atom element symbols fall into "Other elements" automatically.
ORGANIC = ("B", "C", "N", "O", "P", "S", "F", "Cl", "Br", "I")
AROMATIC = ("b", "c", "n", "o", "p", "s", "se", "as")
BONDS = ("-", "=", "#", "$", ":", "/", "\\")
STRUCTURE = ("(", ")", "[", "]", ".", "%", *(str(d) for d in range(10)))
CHARGE_WILD = ("+", "*")
CHIRALITY = ("@", "@@", "@TH", "@AL", "@OH", "@SP", "@TB")
SPECIALS = {"[UNK]", "[BOS]", "[EOS]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"}

style.apply_base_style()
_SAVE_METADATA = {"CreationDate": None}


def _glyph_order() -> tuple[list[str], list[tuple[str, int, int]]]:
    """Return the class-ordered glyph list and (class, start, end) spans."""
    vocab = set(SmirkAdapter.atomic().hf_tokenizer.get_vocab())
    pinned = {
        *ORGANIC,
        *AROMATIC,
        *BONDS,
        *STRUCTURE,
        *CHARGE_WILD,
        *CHIRALITY,
        *SPECIALS,
    }
    # Core backbone classes only. The ~108 rare bracket-atom element symbols
    # ("Other elements") are omitted: they merge sparsely and only under MB, so
    # including them leaves most of the matrix empty (their behaviour is in the
    # multi-glyph piece table). The boundary effect is still visible because the
    # bracket delimiters [ ] (in Structure) co-occur with core glyphs under MB.
    assert pinned - vocab == set(), "pinned glyphs absent from base"
    groups = (
        ("Organic", ORGANIC),
        ("Aromatic", AROMATIC),
        ("Bonds", BONDS),
        ("Charge", CHARGE_WILD),
        ("Chiral", CHIRALITY),
        ("Structure", STRUCTURE),
    )
    order, spans, c = [], [], 0
    for name, toks in groups:
        order.extend(toks)
        spans.append((name, c, c + len(toks)))
        c += len(toks)
    return order, spans


def _adjacency(
    dirname: str, arm: TokenizerAlgo, idx: dict[str, int], n: int
) -> np.ndarray:
    """Adjacent-glyph-pair counts over the cell's multi-glyph pieces."""
    m = np.zeros((n, n))
    for t in glyph_tuple_map(tokenizer_artifact_dir(CORPUS, dirname), arm).values():
        if len(t) < 2:
            continue
        for a, b in pairwise(t):
            if a in idx and b in idx:
                m[idx[a], idx[b]] += 1
    return m


def main() -> int:
    order, spans = _glyph_order()
    idx = {g: i for i, g in enumerate(order)}
    n = len(order)

    cells = {
        ("nmb", "BPE"): _adjacency(cell_artifact_name("bpe", V, "nmb"), "bpe", idx, n),
        ("nmb", "UL"): _adjacency(
            cell_artifact_name("unigram", V, "nmb"), "unigram", idx, n
        ),
        ("mb", "BPE"): _adjacency(cell_artifact_name("bpe", V, "mb"), "bpe", idx, n),
        ("mb", "UL"): _adjacency(
            cell_artifact_name("unigram", V, "mb"), "unigram", idx, n
        ),
    }

    # Restrict to core glyphs active in any cell so each axis is dense and every
    # glyph can carry its own tick label.
    union = sum(m.sum(0) + m.sum(1) for m in cells.values())
    active = [i for i in range(n) if union[i] > 0]
    aset = set(active)
    sub = {k: m[np.ix_(active, active)] for k, m in cells.items()}
    sublabels = [order[i] for i in active]
    # remap class-divider boundaries onto the active subset
    pos = {old: new for new, old in enumerate(active)}
    aspans = []
    for name, lo, hi in spans:
        present = [pos[i] for i in range(lo, hi) if i in aset]
        if present:
            aspans.append((name, min(present), max(present) + 1))

    # Empty cells (no adjacency in either arm) render as a light grey, distinct
    # from any in-scale colour -- in particular from the RdBu near-white that a
    # genuine log2 ratio of ~0 (equal counts) takes in the diverging panel, which
    # a white "bad" colour would be indistinguishable from.
    empty_grey = "0.82"
    seq = mpl.cm.magma_r.copy()
    seq.set_bad(empty_grey)
    # Diverging polarity follows the arm palette (blue=BPE, vermillion=UL): the
    # positive log2 end (BPE-heavier) is blue and the negative (UL-heavier) red,
    # so the ratio panel never inverts the arm hues used everywhere else.
    div = mpl.cm.RdBu.copy()
    div.set_bad(empty_grey)
    vmax = max(m.max() for m in sub.values())
    lognorm = LogNorm(vmin=1, vmax=vmax)
    dnorm = TwoSlopeNorm(vcenter=0, vmin=-6, vmax=6)

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 9.8), constrained_layout=True)
    seq_im = div_im = None
    for ri, bnd in enumerate(("nmb", "mb")):
        b, u = sub[(bnd, "BPE")], sub[(bnd, "UL")]
        diff = np.ma.masked_where((b == 0) & (u == 0), np.log2((b + 0.5) / (u + 0.5)))
        panels = (
            (np.ma.masked_where(b == 0, b), seq, lognorm, "Smirk-GPE (BPE)"),
            (np.ma.masked_where(u == 0, u), seq, lognorm, "Unigram-LM"),
            (diff, div, dnorm, r"$\log_2$(BPE / Unigram-LM)"),
        )
        for ci, (data, cmap, norm, label) in enumerate(panels):
            ax = axes[ri][ci]
            im = ax.imshow(data, cmap=cmap, norm=norm, interpolation="nearest")
            if ci < 2:
                seq_im = im
            else:
                div_im = im
            ax.set_title(f"{bnd.upper()} — {label}", fontsize=10, fontweight="bold")
            for _, lo, hi in aspans:
                for p in (lo, hi):
                    ax.axhline(p - 0.5, color="0.5", lw=0.6)
                    ax.axvline(p - 0.5, color="0.5", lw=0.6)
            ax.set_xticks(range(len(active)))
            ax.set_xticklabels(
                sublabels, fontsize=5, rotation=90, family="monospace", parse_math=False
            )
            ax.set_yticks(range(len(active)))
            ax.set_yticklabels(
                sublabels, fontsize=5, family="monospace", parse_math=False
            )
            if ci == 0:
                ax.set_ylabel("first glyph", fontsize=8)
            if ri == 1:
                ax.set_xlabel("second glyph", fontsize=8)

    fig.colorbar(
        seq_im,
        ax=axes[:, :2],
        orientation="horizontal",
        location="bottom",
        fraction=0.05,
        pad=0.06,
        aspect=50,
        label="adjacent-pair count (log)",
    )
    fig.colorbar(
        div_im,
        ax=axes[:, 2],
        orientation="horizontal",
        location="bottom",
        fraction=0.05,
        pad=0.06,
        aspect=25,
        label=r"$\log_2$(BPE / Unigram-LM)",
    )
    fig.suptitle(
        f"Base-glyph adjacency within learned multi-glyph pieces "
        f"(PubChem $V$={V}, core backbone glyphs)",
        fontsize=13,
        fontweight="bold",
    )
    fig.savefig(OUT, metadata=_SAVE_METADATA)
    plt.close(fig)
    print(f"[cooccurrence] wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
