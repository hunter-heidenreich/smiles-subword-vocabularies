"""Render the scale figure (Figure~\\ref{fig:scale}).

Panels on a shared log-V axis for PubChem (NMB), spanning the headline grid
plus the V=8192 convergence anchor, answering the two questions scale raises:

- **Left (the divergence does not close):** the frequency-weighted overlap
  ``J_w`` and the relative granularity gap ``rel|df|`` against V, stacked on
  separate y-axes because they are not commensurable (an overlap where low means
  diverged, a gap where high means diverged). Both run flat through V=8192, eight
  times the headline vocabulary, so larger vocabularies do not dissolve the
  contrast.
- **Right (and you could not go higher anyway):** the F95_100 learnability
  clearance for each arm against V, with the 0.95 bar. The knee shows the
  Unigram-LM arm crossing into the undertrained tail at V=2048 and BPE at
  V=8192, so the anchor already sits past where embeddings are learnable.

Reproducible: every value is read from the committed deposits (the per-condition
overlap/fertility tables and the F95 audit JSONs), the headless Agg backend and a
suppressed PDF creation date make the bytes deterministic.

Usage::

    uv run python results/build/figure_scale_anchor.py
"""

from __future__ import annotations

import json
import sys

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import style

from smiles_subword.paths import RESULTS_DATA_DIR, RESULTS_FIGURES_DIR

style.apply_base_style()

_SAVE_METADATA = {"CreationDate": None}
_BPE_COLOR = style.BPE_COLOR
_UL_COLOR = style.UNIGRAM_COLOR
_JW_COLOR = style.OVERLAP_COLOR
_FERT_COLOR = style.GAP_COLOR

# PubChem NMB is the only corpus whose Unigram arm reaches the full target at
# every studied V (so the size asymmetry never confounds the trend) and the one
# the anchor extends; the headline series is the grid plus the anchor.
_CORPUS = "pubchem"
_BOUNDARY = "nmb"
_SERIES_KINDS = {None, "large_v_anchor"}
# F95 deposits tag the grid points "headline", the V=2048 pair "sensitivity", and
# the anchor "extras_large_v_anchor"; the noise extras (subsample/size-matched/
# size-sweep) share a V with a grid cell and must be excluded so they do not
# clobber the series.
_F95_TIERS = {"headline", "sensitivity", "extras_large_v_anchor"}
_F95_BAR = 0.95


def _table_series(table_name: str, value_key: str) -> dict[int, float]:
    """Map V -> value for the PubChem-NMB grid + anchor row of a deposit table."""
    table = json.loads((RESULTS_DATA_DIR / f"{table_name}.json").read_text())
    out: dict[int, float] = {}
    for r in table["matched"]:
        if (
            r["corpus"] == _CORPUS
            and r["boundary"] == _BOUNDARY
            and r.get("extras_kind") in _SERIES_KINDS
        ):
            out[int(r["vocab_size"])] = float(r[value_key])
    return out


def _f95_series() -> dict[str, dict[int, float]]:
    """Map arm -> {V -> F95_100 clearance} from the PubChem-NMB F95 deposits."""
    out: dict[str, dict[int, float]] = {"bpe": {}, "unigram": {}}
    for path in sorted((RESULTS_DATA_DIR / "f95").glob("pubchem__*_nmb*.json")):
        d = json.loads(path.read_text())
        if d.get("boundary") != _BOUNDARY or d.get("tier") not in _F95_TIERS:
            continue
        out[d["arm"]][int(d["vocab_size"])] = float(d["headline_clearance"])
    return out


def _line(ax: plt.Axes, series: dict[int, float], **kw: object) -> None:
    vs = sorted(series)
    ax.plot(vs, [series[v] for v in vs], marker="o", **kw)  # type: ignore[arg-type]


def main() -> int:
    jw = _table_series("jaccard_table", "weighted_jaccard")
    fert = _table_series("fertility_table", "delta_fertility_relative")
    f95 = _f95_series()

    vs = sorted(jw)

    fig = plt.figure(figsize=(8.2, 3.5))
    gs = fig.add_gridspec(2, 2, wspace=0.32)
    ax_jw = fig.add_subplot(gs[0, 0])
    ax_fert = fig.add_subplot(gs[1, 0], sharex=ax_jw)
    ax_r = fig.add_subplot(gs[:, 1])

    # Left column: the two contrasts stacked on independent y-axes sharing V.
    # They are not commensurable, a similarity ($J_{\mathrm{w}}$, low = diverged)
    # against a difference (rel$|\Delta f|$, high = diverged), so each keeps its
    # own scale and direction rather than sharing a coined "contrast" axis. Both
    # run flat through the anchor.
    _line(ax_jw, jw, color=_JW_COLOR)
    ax_jw.set_ylim(0.0, 0.05)
    ax_jw.set_ylabel(r"overlap $J_{\mathrm{w}}$")
    ax_jw.set_title("The divergence does not close")

    _line(ax_fert, fert, color=_FERT_COLOR)
    ax_fert.set_ylim(0.0, 0.45)
    ax_fert.set_ylabel("granularity gap\nrel$|\\Delta f|$")
    ax_fert.set_xlabel("vocabulary size $V$")

    for ax in (ax_jw, ax_fert):
        ax.axvline(8192, color="#999999", lw=0.8, ls=(0, (2, 2)), zorder=0)
    ax_jw.annotate(
        "anchor ($8\\times$)",
        xy=(8192, 0.047),
        ha="right",
        va="top",
        fontsize=7.5,
        color="#555555",
    )

    # Right: F95 clearance knee, with the learnability bar.
    ax_r.axhspan(0.0, _F95_BAR, color="#000000", alpha=0.04, zorder=0)
    ax_r.axhline(_F95_BAR, color="#555555", lw=0.9, ls=(0, (4, 2)))
    ax_r.annotate(
        "$F_{95\\%,100}$ bar ($0.95$)",
        xy=(256, _F95_BAR),
        xytext=(256, 0.86),
        fontsize=7.5,
        color="#555555",
    )
    _line(ax_r, f95["bpe"], color=_BPE_COLOR, label="BPE")
    _line(ax_r, f95["unigram"], color=_UL_COLOR, label="Unigram-LM")
    ax_r.set_ylim(0.0, 1.05)
    ax_r.set_xlabel("vocabulary size $V$")
    ax_r.set_ylabel("$F_{95\\%,100}$ clearance")
    ax_r.set_title("The learnable regime ends")
    ax_r.legend(fontsize=8, loc="lower left", frameon=False)

    for ax in (ax_jw, ax_fert, ax_r):
        ax.set_xscale("log", base=2)
        ax.set_xticks(vs)
        ax.set_xticklabels([str(v) for v in vs], fontsize=8)
        style.despine(ax)
    # Shared V axis: only the bottom-left panel carries the tick labels.
    plt.setp(ax_jw.get_xticklabels(), visible=False)
    fig.tight_layout()

    RESULTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_FIGURES_DIR / "scale_anchor.pdf"
    fig.savefig(out, format="pdf", metadata=_SAVE_METADATA)
    plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
