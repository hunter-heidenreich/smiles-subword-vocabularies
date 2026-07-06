r"""Render the results tables as ``booktabs`` LaTeX.

Pure string-building over the :mod:`extract` rows,
plus a :func:`write_tables` orchestrator that deposits the ``.tex`` files under
``results/tables/`` and a manifest sidecar so a regenerated
table can detect an upstream measurement re-run. The headline tables are the
seven measurements, the cross-arm ΔF, and the three vocabulary Jaccards.

Coordinate columns render from ``corpus``/``vocab_size``/``boundary`` (not the
raw ``pair_key``) so no LaTeX ``_``-escaping is needed; missing/undefined values
render as ``---``. Numeric precision is three decimals throughout (matching the
deposited Markdown aggregators); signed deltas carry an explicit ``+``. Flag
glyphs follow the deposit conventions: ``$\dagger$`` embedding-tail-unsafe,
``$\circ$`` marks ``J`` near the reference-overlap scale (the
no-per-cell-error-bar caveat). The tables need only packages the manuscript
already loads (``booktabs``, ``amsmath``); wiring them into the manuscript is a
separate write-up step.
"""

from __future__ import annotations

import json
from itertools import pairwise
from typing import TYPE_CHECKING, Protocol, TypeVar

import extract
from _corpora import CORPUS_LABEL

from smiles_subword.paths import RESULTS_TABLES_DIR

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

RESULTS_MANIFEST = RESULTS_TABLES_DIR / "_results_manifest.json"

TABLE_FILES = {
    "seven_measurements": "seven_measurements.tex",
    "delta_f": "delta_f.tex",
    "three_jaccards": "three_jaccards.tex",
    "realized_vocab": "realized_vocab.tex",
    "fertility": "fertility_detail.tex",
    "nestedness": "nestedness_detail.tex",
    "closure": "closure_detail.tex",
    "fg_alignment": "fg_alignment_detail.tex",
    "noncanon": "noncanon_detail.tex",
    "distribution": "distribution_detail.tex",
    "absorption": "absorption_detail.tex",
    "deadzone_nsweep": "deadzone_nsweep.tex",
    "robustness_extras": "robustness_extras.tex",
}

_NA = "---"


# --------------------------------------------------------------------------- #
# Cell formatters                                                             #
# --------------------------------------------------------------------------- #


def _corpus(corpus: str) -> str:
    return CORPUS_LABEL.get(corpus, corpus)


def _f(value: float | None, decimals: int = 3) -> str:
    return _NA if value is None else f"{value:.{decimals}f}"


def _sf(value: float | None, decimals: int = 3) -> str:
    # Math mode so the leading sign is a true $+$/$-$, not an ASCII hyphen.
    return _NA if value is None else f"${value:+.{decimals}f}$"


def _pct(value: float | None, decimals: int = 1) -> str:
    return _NA if value is None else rf"{value * 100:.{decimals}f}\%"


def _count(value: int | None) -> str:
    # Thousands separators on counts (matches the prose); V stays bare elsewhere.
    return _NA if value is None else f"{value:,}"


def _table_float(
    *,
    caption: str,
    label: str,
    colspec: str,
    header: str,
    body: Sequence[str],
    note: str,
    fit_to_width: bool = False,
) -> str:
    """Render one booktabs table float.

    The ``note`` (column key, CI convention, units) is folded into the caption so
    every table is self-contained, the convention this paper uses elsewhere. Body
    type is uniform ``\\footnotesize``; ``fit_to_width`` wraps the tabular in a
    width-clamped ``\\resizebox`` and is reserved for the few tables too wide to
    fit otherwise (so type size stays constant across the rest).
    """
    full_caption = f"{caption} {note}" if note else caption
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        rf"  \caption{{{full_caption}}}",
        rf"  \label{{{label}}}",
        r"  \footnotesize",
    ]
    if fit_to_width:
        lines.append(
            r"  \resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{%"
        )
    lines += [
        rf"  \begin{{tabular}}{{@{{}}{colspec}@{{}}}}",
        r"    \toprule",
        f"    {header} \\\\",
        r"    \midrule",
    ]
    lines += [f"    {row}" for row in body]
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}%" if fit_to_width else r"  \end{tabular}",
    ]
    if fit_to_width:
        lines.append(r"  }")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


class _HasCorpus(Protocol):
    # Read-only member so frozen-dataclass rows (read-only fields) match.
    @property
    def corpus(self) -> str: ...


_RowT = TypeVar("_RowT", bound=_HasCorpus)


def _corpus_blocked_body(
    rows: Sequence[_RowT], row_fn: Callable[[_RowT, str], str]
) -> list[str]:
    """Build a per-condition table body grouped into per-corpus blocks.

    The corpus label prints once per block and the blocks are ruled apart with
    ``\\addlinespace``, so the cross-$V$/boundary trend within a corpus reads
    down a column. ``row_fn(row, label)`` renders one row using ``label`` as its
    corpus cell — the corpus name on a block's first row, empty on continuations.
    Rows must already be in per-corpus order (the enumeration is).
    """
    body: list[str] = []
    prev: str | None = None
    for r in rows:
        if prev is not None and r.corpus != prev:
            body.append(r"\addlinespace")
        label = _corpus(r.corpus) if r.corpus != prev else ""
        body.append(row_fn(r, label))
        prev = r.corpus
    return body


# --------------------------------------------------------------------------- #
# The table renderers                                                         #
# --------------------------------------------------------------------------- #


def render_seven_measurements(rows: Sequence[extract.MeasurementRow]) -> str:
    """Render the seven cross-arm measurement scalars per matched cell.

    Columns follow the manuscript's split: the three direct
    contrasts ($J$, rel$|\\Delta f|$, $|\\Delta D|$) grouped under one spanning
    header, the four mechanism diagnostics ($\\Delta c_{100}$, $\\Delta$abs,
    $\\Delta$scaf, $\\Delta H_g$) under another, with the corpus label printed
    once per block, matching the sibling per-condition tables.
    """
    header = (
        r"& & & \multicolumn{3}{c}{Direct contrasts} & "
        r"\multicolumn{4}{c}{Mechanism diagnostics} \\"
        "\n    "
        r"\cmidrule(lr){4-6}\cmidrule(lr){7-10}"
        "\n    "
        r"Corpus & $V$ & Bnd & $J$ & rel$|\Delta f|$ & $|\Delta D|$ & "
        r"$\Delta c_{100}$ & $\Delta$abs & $\Delta$scaf & $\Delta H_g$"
    )

    rel_max = max(r.rel_fertility for r in rows)

    def _row(r: extract.MeasurementRow, label: str) -> str:
        rel = _pct(r.rel_fertility)
        if r.rel_fertility == rel_max:
            rel = rf"\textbf{{{rel}}}"
        return (
            f"{label} & {r.vocab_size} & {r.boundary.upper()} & "
            f"{_f(r.jaccard)} & {rel} & {_f(r.abs_delta_d)} & "
            f"{_sf(r.delta_f)} & {_sf(r.delta_absorbed)} & "
            f"{_sf(r.delta_scaffold)} & {_sf(r.delta_entropy_per_glyph)} \\\\"
        )

    body = _corpus_blocked_body(rows, _row)
    note = (
        r"\emph{Direct contrasts:} $J$ vocabulary overlap (Jaccard), "
        r"rel$|\Delta f|$ relative fertility gap, $|\Delta D|$ token-imbalance "
        r"gap. \emph{Mechanism diagnostics:} $\Delta c_{100}$ dead-zone surplus "
        r"(cross-arm clearance gap at the $F_{95\%,100}$ bar), $\Delta$abs "
        r"whole-pretoken-absorption gap, $\Delta$scaf scaffold-fraction gap, "
        r"$\Delta H_g$ segmentation-entropy gap per glyph. Two-sided "
        r"$\Delta c_{100}$ and $\Delta$abs are signed BPE $-$ Unigram; one-sided "
        r"$\Delta$scaf (BPE-only) and $\Delta H_g$ (Unigram-only) print the "
        r"non-zero arm's magnitude. Bootstrap CIs for the held-out scalars are "
        r"in the Appendix~\ref{app:tables} detail tables; the exact-set $J$ and "
        r"$\Delta$scaf carry none. Largest rel$|\Delta f|$ in \textbf{bold}."
    )
    return _table_float(
        caption="Cross-algorithm measurements per condition (point estimates).",
        label="tab:results-seven",
        colspec="l r l r r r r r r r",
        header=header,
        body=body,
        note=note,
    )


def render_delta_f(rows: Sequence[extract.DeltaFRow]) -> str:
    """Render the Deadzone per-arm F95 clearance and cross-arm ΔF table.

    A spanning header groups the two per-arm clearances under $c_{100}$ with the
    derived $\\Delta c_{100}$ outside the span. The per-condition
    embedding-tail-unsafe flag rides the $V$ cell as a superscript $\\dagger$
    (the convention used elsewhere in this file), which frees the corpus column
    to print once per block, matching the sibling per-condition tables.
    """
    header = (
        r"& & & \multicolumn{2}{c}{$c_{100}$} & \\"
        "\n    "
        r"\cmidrule(lr){4-5}"
        "\n    "
        r"Corpus & $V$ & Bnd & BPE & UL & $\Delta c_{100}$"
    )

    def _row(r: extract.DeltaFRow, label: str) -> str:
        v = r.vocab_size
        vcell = rf"{v}$^{{\dagger}}$" if r.any_arm_unsafe else str(v)
        return (
            f"{label} & {vcell} & {r.boundary.upper()} & {_f(r.bpe_clearance)} & "
            f"{_f(r.unigram_clearance)} & {_sf(r.headline_delta_f)} \\\\"
        )

    body = _corpus_blocked_body(rows, _row)
    note = (
        r"$c_{100}^{\mathrm{BPE}}$, $c_{100}^{\mathrm{UL}}$ (columns BPE, UL; "
        r"UL $=$ Unigram-LM) are "
        r"the fraction of each arm's vocabulary clearing the $F_{95\%,100}$ "
        r"rare-token-tail bar; $\Delta c_{100}$ is their difference (BPE $-$ UL). "
        r"$\dagger$ corpus too small to certify the tail at this $V$."
    )
    return _table_float(
        caption="Dead-zone surplus: per-arm $F_{95\\%,100}$ clearance $c_{100}$ "
        "and its cross-arm difference $\\Delta c_{100}$.",
        label="tab:results-delta-f",
        colspec="l r l r r r",
        header=header,
        body=body,
        note=note,
    )


def render_three_jaccards(rows: Sequence[extract.JaccardRow]) -> str:
    """Render the four vocabulary-Jaccard quantities table (Jaccard).

    The four overlaps form a weighting (unweighted / frequency-weighted) $\\times$
    masking (all pieces / structural only) $2\\times2$, so a spanning header groups
    them by weighting. The weighted-Jaccard bootstrap CIs are $\\le 0.002$ in every
    condition, so they are stated in the caption rather than tabulated (their two
    ``[lo, hi]`` columns otherwise drive the table past ``\\linewidth``); they stay
    in the deposits. Rows print the corpus label once per block, with the blocks
    ruled apart, so the cross-$V$/boundary trend within a corpus reads down a
    column.
    """
    header = (
        r"& & & \multicolumn{2}{c}{Unweighted} & \multicolumn{2}{c}{Freq-weighted} \\"
        "\n    "
        r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}"
        "\n    "
        r"Corpus & $V$ & Bnd & $J$ & $J_{\mathrm{struct}}$ & "
        r"$J_{\mathrm{w}}$ & $J_{\mathrm{w,struct}}$"
    )

    j_extremes = {min(r.jaccard for r in rows), max(r.jaccard for r in rows)}

    def _row(r: extract.JaccardRow, label: str) -> str:
        jcell = _f(r.jaccard)
        if r.jaccard in j_extremes:
            jcell = rf"\textbf{{{jcell}}}"
        return (
            f"{label} & {r.vocab_size} & {r.boundary.upper()} & "
            f"{jcell} & {_f(r.jaccard_struct)} & "
            f"{_f(r.weighted_jaccard)} & {_f(r.weighted_jaccard_struct)} \\\\"
        )

    body = _corpus_blocked_body(rows, _row)
    note = (
        r"$J$ unweighted, $J_{\mathrm{struct}}$ structural-subword, "
        r"$J_{\mathrm{w}}$ frequency-weighted, $J_{\mathrm{w,struct}}$ "
        r"frequency-weighted over structural subwords only; all four are "
        r"vocabulary Jaccards on matched arm pairs. $J$ and $J_{\mathrm{struct}}$ "
        r"are exact set "
        r"quantities; the molecule-resampled bootstrap CIs on $J_{\mathrm{w}}$ "
        r"and $J_{\mathrm{w,struct}}$ span $\le 0.002$ in every condition and are "
        r"deposited in full. The smallest and largest $J$ are in \textbf{bold}."
    )
    return _table_float(
        caption="Cross-algorithm vocabulary overlap (Jaccard).",
        label="tab:results-jaccards",
        colspec="l r l r r r r",
        header=header,
        body=body,
        note=note,
    )


def render_fertility(rows: Sequence[extract.FertilityRow]) -> str:
    """Render the per-condition absolute fertility + compression table (Fertility)."""
    header = (
        r"& & & \multicolumn{2}{c}{$f$} & \multicolumn{2}{c}{$\tfrac{g}{t}$} & & \\"
        "\n    "
        r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}"
        "\n    "
        r"Corpus & $V$ & Bnd & BPE & UL & BPE & UL & $|\Delta f|$ & rel$|\Delta f|$"
    )

    def _row(r: extract.FertilityRow, label: str) -> str:
        fb = _mean_ci(r.bpe_fertility, r.bpe_fertility_ci, decimals=1)
        fu = _mean_ci(r.unigram_fertility, r.unigram_fertility_ci, decimals=1)
        gb = _mean_ci(r.bpe_glyphs_per_token, r.bpe_glyphs_per_token_ci, decimals=2)
        gu = _mean_ci(
            r.unigram_glyphs_per_token, r.unigram_glyphs_per_token_ci, decimals=2
        )
        return (
            f"{label} & {r.vocab_size} & {r.boundary.upper()} & "
            f"{fb} & {fu} & {gb} & {gu} & "
            f"{_f(abs(r.delta_fertility), 1)} & {_pct(r.rel_fertility)} \\\\"
        )

    body = _corpus_blocked_body(rows, _row)
    note = (
        r"Per matched pair. $f$ mean held-out tokens per "
        r"molecule (95\% molecule-resampled bootstrap CI in brackets); "
        r"$\tfrac{g}{t}$ glyphs per token (the compression ratio of "
        r"\S\ref{ssec:measurements}, higher $=$ coarser); $|\Delta f|$ the "
        r"absolute and rel$|\Delta f|$ the relative cross-arm fertility gap "
        r"(Eq.~\ref{eq:fertility}). Unigram-LM (UL) segments to more tokens at a "
        r"lower compression ratio than BPE in every condition."
    )
    return _table_float(
        caption="Per-condition absolute fertility and compression ratio.",
        label="tab:results-fertility",
        colspec="l r l r r r r r r",
        header=header,
        body=body,
        note=note,
        fit_to_width=True,
    )


def render_nestedness(rows: Sequence[extract.NestednessRow]) -> str:
    """Render the per-condition boundary-nestedness table (Nestedness)."""
    header = (
        r"& & & & & & & \multicolumn{3}{c}{Conflict $\mathrm{cut}^{c}$} \\"
        "\n    "
        r"\cmidrule(lr){8-10}"
        "\n    "
        r"Corpus & $V$ & Bnd & $J_{\partial}$ & conflict & nest & nested & "
        r"het & uns & sat"
    )

    def _row(r: extract.NestednessRow, label: str) -> str:
        return (
            f"{label} & {r.vocab_size} & {r.boundary.upper()} & "
            f"{_f(r.boundary_jaccard)} & {_f(r.conflict_rate, 4)} & "
            f"{_f(r.nest_rate)} & {_f(r.nested_molecule_fraction)} & "
            f"{_f(r.cut_rate_heteroatom)} & {_f(r.cut_rate_unsat_c)} & "
            f"{_f(r.cut_rate_sat_c)} \\\\"
        )

    body = _corpus_blocked_body(rows, _row)
    note = (
        r"Per matched pair (\S\ref{ssec:measurements}). Both "
        r"arms cut the same glyph stream, so their boundaries are subsets of the "
        r"same inter-glyph positions. $J_{\partial}$ the boundary Jaccard over "
        r"\emph{cut} positions (agree-cut over agree-cut $+$ nest $+$ conflict); "
        r"\emph{nest} the share of positions where Unigram-LM cuts and BPE merges "
        r"(the fertility gap of Table~\ref{tab:results-fertility} read "
        r"positionally) and \emph{conflict} the share where BPE cuts and "
        r"Unigram-LM merges (genuine crossing), both over all positions; "
        r"\emph{nested} the fraction of molecules with zero conflict, i.e.\ whose "
        r"BPE parse is a strict coarsening of Unigram-LM's. $\mathrm{cut}^{c}$ "
        r"localizes conflict: the fraction of multi-glyph Unigram-LM pieces of "
        r"class $c$ (heteroatom, unsaturated carbon, saturated carbon) that BPE "
        r"cuts through, with no entry for a class the corpus never emits. "
        r"Conflict is near zero in every condition."
    )
    return _table_float(
        caption="Per-condition cross-arm boundary nestedness.",
        label="tab:results-nestedness",
        colspec="l r l r r r r r r r",
        header=header,
        body=body,
        note=note,
    )


def render_closure(rows: Sequence[extract.ClosureRow]) -> str:
    """Render the per-condition compositional-closure table (Closure)."""
    header = (
        r"& & & \multicolumn{2}{c}{$c_{\mathrm{bin}}$} & & "
        r"\multicolumn{2}{c}{$c_{\mathrm{full}}$} \\"
        "\n    "
        r"\cmidrule(lr){4-5}\cmidrule(lr){7-8}"
        "\n    "
        r"Corpus & $V$ & Bnd & BPE & UL & $c_{\mathrm{orph}}^{\mathrm{UL}}$ & BPE & UL"
    )

    def _row(r: extract.ClosureRow, label: str) -> str:
        return (
            f"{label} & {r.vocab_size} & {r.boundary.upper()} & "
            f"{_f(r.bpe_c_bin)} & {_f(r.ul_c_bin)} & {_f(r.ul_c_orph)} & "
            f"{_f(r.bpe_c_full)} & {_f(r.ul_c_full)} \\\\"
        )

    body = _corpus_blocked_body(rows, _row)
    note = (
        r"Per matched pair (\S\ref{ssec:measurements}), read from the realized "
        r"vocabulary alone. "
        r"$c_{\mathrm{bin}}$ is the binary-split closure: the fraction of "
        r"multi-glyph pieces with some in-vocab split $p{=}a{\cdot}b$ (both parts "
        r"in the base-plus-multi vocabulary). It is BPE's merge-closure invariant "
        r"read off the realized set, so $c_{\mathrm{bin}}^{\mathrm{BPE}}{=}1$ "
        r"exactly (the correctness anchor) and the orphan rate "
        r"$c_{\mathrm{orph}}^{\mathrm{BPE}}{=}0$, the latter omitted for BPE. "
        r"$c_{\mathrm{orph}}^{\mathrm{UL}}$ is the Unigram-LM (UL) orphan rate: the "
        r"fraction of its length-$\ge 3$ pieces with \emph{no} proper "
        r"$\ge 2$-glyph sub-piece in vocabulary (pieces that share no building "
        r"block with the rest of the vocabulary). $c_{\mathrm{full}}$ is the "
        r"stronger full-substring closure (every $\ge 2$-glyph substring in "
        r"vocabulary), non-trivial for both arms. Closure is an exact-set "
        r"quantity and carries no CI. Unigram-LM is far less self-referential "
        r"than BPE in every condition: roughly half its pieces (as few as "
        r"$0.12$ on the narrowest alphabet) do not decompose into in-vocab parts."
    )
    return _table_float(
        caption="Per-condition within-arm compositional closure.",
        label="tab:results-closure",
        colspec="l r l r r r r r",
        header=header,
        body=body,
        note=note,
    )


def render_fg_alignment(rows: Sequence[extract.FgAlignmentRow]) -> str:
    """Render the per-condition functional-bond-locality table (FG-alignment)."""
    header = (
        r"& & & \multicolumn{2}{c}{$\ell$} & & "
        r"\multicolumn{2}{c}{$\ell_{\mathrm{C{=}O}}$} \\"
        "\n    "
        r"\cmidrule(lr){4-5}\cmidrule(lr){7-8}"
        "\n    "
        r"Corpus & $V$ & Bnd & BPE & UL & $\Delta\ell$ & BPE & UL"
    )

    def _row(r: extract.FgAlignmentRow, label: str) -> str:
        return (
            f"{label} & {r.vocab_size} & {r.boundary.upper()} & "
            f"{_f(r.bpe_locality)} & {_f(r.ul_locality)} & {_f(r.delta_locality)} & "
            f"{_f(r.bpe_carbonyl)} & {_f(r.ul_carbonyl)} \\\\"
        )

    body = _corpus_blocked_body(rows, _row)
    note = (
        r"Per matched pair (\S\ref{ssec:measurements}), read on the held-out "
        r"split. A "
        r"\emph{functional bond} is a multiply-bonded heteroatom, a non-carbon "
        r"atom joined by a double or triple bond (the $=$O of a carbonyl, the "
        r"$\#$N of a nitrile, the $=$N of an imine, the $=$O on sulfur, "
        r"phosphorus, or nitrogen, the "
        r"$=$S of a thiocarbonyl), the cores of the canonical functional groups, "
        r"read straight off the molecular graph. Locality $\ell$ is the fraction "
        r"of those bonds the arm keeps inside a single token (the heteroatom "
        r"sharing a token with its bond glyph); $\Delta\ell{=}\ell^{\mathrm{BPE}}{-}"
        r"\ell^{\mathrm{UL}}$. $\ell_{\mathrm{C{=}O}}$ isolates the carbonyl class. "
        r"BPE keeps nearly every functional bond local (the carbonyl essentially "
        r"always); Unigram-LM (UL) keeps almost none (the carbonyl column "
        r"$\ell^{\mathrm{UL}}_{\mathrm{C{=}O}}$ is $0.000$ to three places in "
        r"every condition), spending its unsaturation "
        r"budget on long homo-atomic carbon runs, not on binding the heteroatom "
        r"to its bond. 95\% molecule-resampled bootstrap CIs are in the deposited "
        r"per-condition records."
    )
    return _table_float(
        caption="Per-condition within-arm chemical functional-bond locality.",
        label="tab:results-fg-alignment",
        colspec="l r l r r r r r",
        header=header,
        body=body,
        note=note,
    )


def render_noncanon(rows: Sequence[extract.NoncanonRow]) -> str:
    """Render the per-condition non-canonicity-robustness table (Non-canonicity)."""
    header = (
        r"& & & \multicolumn{2}{c}{Randomized} & \multicolumn{2}{c}{Kekul\'e} & "
        r"\multicolumn{2}{c}{Explicit-H} & \multicolumn{2}{c}{OpenBabel} & "
        r"\multicolumn{2}{c}{Fertility gap} \\" + "\n"
        r"    \cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}"
        r"\cmidrule(lr){10-11}\cmidrule(lr){12-13}" + "\n"
        r"    Corpus & $V$ & Bnd & $b^{\mathrm{BPE}}$ & $b^{\mathrm{UL}}$ & "
        r"$b^{\mathrm{BPE}}$ & $b^{\mathrm{UL}}$ & "
        r"$b^{\mathrm{BPE}}$ & $b^{\mathrm{UL}}$ & "
        r"$b^{\mathrm{BPE}}$ & $b^{\mathrm{UL}}$ & $g_{\mathrm{c}}$ & $g_{\mathrm{r}}$"
    )

    def _row(r: extract.NoncanonRow, label: str) -> str:
        return (
            f"{label} & {r.vocab_size} & {r.boundary.upper()} & "
            f"{_f(r.bpe_random)} & {_f(r.ul_random)} & {_f(r.bpe_kekule)} & "
            f"{_f(r.ul_kekule)} & {_f(r.bpe_explicit_h)} & {_f(r.ul_explicit_h)} & "
            f"{_f(r.bpe_obcanon)} & {_f(r.ul_obcanon)} & "
            f"{_pct(r.gap_canon)} & {_pct(r.gap_rand)} \\\\"
        )

    body = _corpus_blocked_body(rows, _row)
    note = (
        r"Per matched pair (\S\ref{ssec:measurements}), on a seeded held-out "
        r"subsample; the paired columns $b^{\mathrm{BPE}}$ and $b^{\mathrm{UL}}$ are "
        r"the BPE and Unigram-LM (UL) arms. $b$ is the "
        r"\emph{bag-instability}: the fraction of an arm's token multiset that "
        r"changes when the molecule is rewritten (mean $1$ minus the multiset "
        r"Jaccard versus the canonical string), under four identity-preserving "
        r"rewrite axes: Randomized (RDKit restricted randomization, the "
        r"augmentation-realistic distribution of "
        r"\citet{aruspous2019randomized-smiles}), "
        r"Kekul\'e, Explicit-H (all-explicit-hydrogen, "
        r"AMORE's catastrophic axis \citep{ganeeva2025chemllm-robustness}), "
        r"and OpenBabel (the cross-toolkit swap "
        r"to OpenBabel's canonical SMILES, gated to identity-preservation by a "
        r"round-trip through RDKit). $g_{\mathrm{c}}$ and $g_{\mathrm{r}}$ are "
        r"the relative fertility gap $\mathrm{rel}|\Delta f|$ "
        r"(Eq.~\ref{eq:fertility}) on the canonical strings and on the "
        r"randomized orbit; their closeness shows the granularity gap survives "
        r"the orbit. A ring-digit-relabel floor (omitted) leaves the token count "
        r"exactly invariant. Figure~\ref{fig:noncanon} plots the per-axis "
        r"arm-stability pattern at $V{=}1024$; 95\% molecule-resampled bootstrap "
        r"CIs are in the deposited per-condition records."
    )
    return _table_float(
        caption="Per-condition within-arm robustness to non-canonical SMILES.",
        label="tab:results-noncanon",
        colspec="l r l r r r r r r r r r r",
        header=header,
        body=body,
        note=note,
        fit_to_width=True,
    )


def render_distribution(rows: Sequence[extract.DistributionRow]) -> str:
    """Render the per-condition token-distribution intrinsics table (Distribution)."""
    header = (
        r"& & & \multicolumn{2}{c}{$D$} & \multicolumn{2}{c}{$\eta$} & "
        r"\multicolumn{2}{c}{$R$} \\"
        "\n    "
        r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}"
        "\n    "
        r"Corpus & $V$ & Bnd & BPE & UL & BPE & UL & BPE & UL"
    )

    def _row(r: extract.DistributionRow, label: str) -> str:
        db = _mean_ci(r.bpe_d, r.bpe_d_ci, decimals=3)
        du = _mean_ci(r.unigram_d, r.unigram_d_ci, decimals=3)
        eb = _mean_ci(r.bpe_eta, r.bpe_eta_ci, decimals=3)
        eu = _mean_ci(r.unigram_eta, r.unigram_eta_ci, decimals=3)
        rb = _mean_ci(r.bpe_renyi, r.bpe_renyi_ci, decimals=3)
        ru = _mean_ci(r.unigram_renyi, r.unigram_renyi_ci, decimals=3)
        return (
            f"{label} & {r.vocab_size} & {r.boundary.upper()} & "
            f"{db} & {du} & {eb} & {eu} & {rb} & {ru} \\\\"
        )

    body = _corpus_blocked_body(rows, _row)
    note = (
        r"Within-family, per matched pair "
        r"(\S\ref{ssec:measurements}), each a held-out per-arm value with a 95\% "
        r"molecule-resampled bootstrap CI. $D$ token-frequency imbalance "
        r"(divergence from uniform, Eq.~\ref{eq:imbalance}; $0$ uniform, $1$ "
        r"maximally concentrated), $\eta$ normalized Shannon entropy, $R$ R\'enyi "
        r"efficiency at $\alpha{=}2.5$. BPE is more uniform than Unigram-LM (UL) "
        r"in every condition. The cross-arm gap $|\Delta D|$ is in "
        r"Table~\ref{tab:results-seven}."
    )
    return _table_float(
        caption="Per-condition token-distribution intrinsics "
        "(imbalance, entropy, R\\'enyi efficiency).",
        label="tab:results-distribution",
        colspec="l r l r r r r r r",
        header=header,
        body=body,
        note=note,
        fit_to_width=True,
    )


def render_absorption(rows: Sequence[extract.AbsorptionRow]) -> str:
    """Render the per-condition whole-pretoken absorption table (Absorption)."""
    header = (
        r"& & & \multicolumn{2}{c}{abs} & \\"
        "\n    "
        r"\cmidrule(lr){4-5}"
        "\n    "
        r"Corpus & $V$ & Bnd & BPE & UL & $\Delta$abs"
    )

    def _row(r: extract.AbsorptionRow, label: str) -> str:
        ab = _mean_ci(r.bpe_absorbed, r.bpe_absorbed_ci, decimals=3)
        au = _mean_ci(r.unigram_absorbed, r.unigram_absorbed_ci, decimals=3)
        return (
            f"{label} & {r.vocab_size} & {r.boundary.upper()} & "
            f"{ab} & {au} & {_sf(r.delta_absorbed)} \\\\"
        )

    body = _corpus_blocked_body(rows, _row)
    note = (
        r"Per matched pair: the held-out fraction of "
        r"pretokens emitted as a single token "
        r"\citep{reddy2025diminishing-tokenization}, per arm with a 95\% "
        r"molecule-resampled bootstrap CI, and the cross-arm gap "
        r"$\Delta$abs (BPE $-$ UL). BPE absorbs almost every pretoken whole; "
        r"Unigram-LM (UL) splits far more, the per-pretoken face of the fertility "
        r"gap (\S\ref{ssec:r-mechanism})."
    )
    return _table_float(
        caption="Per-condition whole-pretoken absorption.",
        label="tab:results-absorption",
        colspec="l r l r r r",
        header=header,
        body=body,
        note=note,
    )


def render_deadzone_nsweep(rows: Sequence[extract.DeadzoneNSweepRow]) -> str:
    """Render the per-arm rare-token clearance n-sweep table (Deadzone)."""
    ns = (50, 100, 200)
    span = " & ".join(rf"\multicolumn{{2}}{{c}}{{$c_{{{n}}}$}}" for n in ns)
    # Two body columns per n (BPE, UL), starting at column 4.
    rules = "".join(
        rf"\cmidrule(lr){{{4 + 2 * i}-{5 + 2 * i}}}" for i in range(len(ns))
    )
    subheads = " & ".join("BPE & UL" for _ in ns)
    header = (
        rf"& & & {span} \\"
        "\n    "
        f"{rules}"
        "\n    "
        rf"Corpus & $V$ & Bnd & {subheads}"
    )

    def _row(r: extract.DeadzoneNSweepRow, label: str) -> str:
        cells = " & ".join(
            f"{_f(r.bpe_c.get(n))} & {_f(r.unigram_c.get(n))}" for n in ns
        )
        vcell = (
            rf"{r.vocab_size}$^{{\dagger}}$" if r.any_arm_unsafe else str(r.vocab_size)
        )
        coords = f"{label} & {vcell} & {r.boundary.upper()}"
        return f"{coords} & {cells} \\\\"

    body = _corpus_blocked_body(rows, _row)
    note = (
        r"Following \citet{gowda2020optimal-vocab-nmt}, $c_n$ is the fraction of "
        r"an arm's "
        r"vocabulary firing at least $n$ times in the training corpus. The "
        r"learnability bar $F_{p,n}$ is $c_n \ge p$, so reading each $c_n$ against "
        r"$p \in \{0.90, 0.95, 0.99\}$ gives every $(p,n)$ bar outcome; $c_{100}$ at "
        r"$p{=}0.95$ is the headline $F_{95\%,100}$ (Table~\ref{tab:results-delta-f}). "
        r"$\dagger$ corpus too small to certify the "
        r"tail at this $V$ (as in Table~\ref{tab:results-delta-f})."
    )
    return _table_float(
        caption="Rare-token clearance $c_n$ across the firing-count sweep "
        "($n \\in \\{50,100,200\\}$, $p{=}0.95$).",
        label="tab:results-nsweep",
        colspec="l r l r r r r r r",
        header=header,
        body=body,
        note=note,
    )


def render_realized_vocab(rows: Sequence[extract.JaccardRow]) -> str:
    """Render the per-arm realized multi-glyph vocabulary table (Jaccard).

    A spanning header groups the two per-arm counts under
    $|\\mathcal{V}^{\\mathrm{multi}}|$ with the derived UL/BPE ratio standing
    outside the span; the corpus label prints once per block, ruled apart,
    matching the sibling Jaccard table.
    """
    header = (
        r"& & & \multicolumn{2}{c}{$|\mathcal{V}^{\mathrm{multi}}|$} & \\"
        "\n    "
        r"\cmidrule(lr){4-5}"
        "\n    "
        r"Corpus & $V$ & Bnd & BPE & UL & UL/BPE"
    )

    def _int(value: int | None) -> str:
        return _count(value)

    def _ratio(r: extract.JaccardRow) -> str:
        if (
            r.bpe_n_multi is not None
            and r.unigram_n_multi is not None
            and r.bpe_n_multi != 0
        ):
            return f"{r.unigram_n_multi / r.bpe_n_multi:.2f}"
        return "---"

    def _row(r: extract.JaccardRow, label: str) -> str:
        return (
            f"{label} & {r.vocab_size} & {r.boundary.upper()} & "
            f"{_int(r.bpe_n_multi)} & {_int(r.unigram_n_multi)} & {_ratio(r)} \\\\"
        )

    body = _corpus_blocked_body(rows, _row)
    note = (
        r"Per arm, the sets the Jaccards "
        r"compare, excluding the shared $165$-token base. BPE keeps the full "
        r"atomic base and fills the rest with merges to target; Unigram-LM (UL) "
        r"prunes to at or below target and may shed rarely-used base glyphs, "
        r"reallocating to multi-glyph pieces, so on diverse corpora it packs "
        r"comparably many (UL/BPE near $1$). On narrow "
        r"alphabets at larger $V$ its pruning runs short of high-likelihood "
        r"pieces and bottoms out well below target (UL/BPE $\ll 1$). The Jaccard "
        r"is a set ratio robust to this gap; its effect on the overlap ceiling "
        r"is discussed in \S\ref{ssec:r-jaccard}."
    )
    return _table_float(
        caption="Realized per-arm multi-glyph vocabulary per condition.",
        label="tab:results-realized-vocab",
        colspec="l r l r r r",
        header=header,
        body=body,
        note=note,
    )


_TEXTTT_ESCAPE = {
    "\\": r"\textbackslash{}",
    "{": r"\{",
    "}": r"\}",
    "$": r"\$",
    "&": r"\&",
    "#": r"\#",
    "%": r"\%",
    "_": r"\_",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def _texttt(s: str) -> str:
    """Escape a string for use inside ``\\texttt{...}``."""
    return "".join(_TEXTTT_ESCAPE.get(c, c) for c in s)


# The multi-glyph split and shared-core tables list every learned piece per class
# (80--190 surfaces); rendered in full they are 10--19 KB data dumps. Abridge long
# classes to a head and a tail so both the shortest and longest pieces survive
# (pieces are ordered by glyph count, so a head-only cut would hide the long
# heteroatom/homopolymer chains the prose draws on); the complete lists live in
# the data deposit.
_PIECE_EXCERPT_HEAD = 14
_PIECE_EXCERPT_TAIL = 4
_PIECE_EXCERPT_CAP = _PIECE_EXCERPT_HEAD + _PIECE_EXCERPT_TAIL


def _excerpt_pieces(surfaces: Sequence[str], render: Callable[[str], str]) -> str:
    """Render a comma-separated piece list, abridging classes past the cap.

    Lists no longer than :data:`_PIECE_EXCERPT_CAP` render in full. Longer lists
    show the first :data:`_PIECE_EXCERPT_HEAD` and last :data:`_PIECE_EXCERPT_TAIL`
    pieces with an inline ``(N more)`` ellipsis. ``render`` applies the per-table
    greying/dagger markup to each surface.
    """
    if len(surfaces) <= _PIECE_EXCERPT_CAP:
        return ", ".join(render(s) for s in surfaces)
    head = ", ".join(render(s) for s in surfaces[:_PIECE_EXCERPT_HEAD])
    tail = ", ".join(render(s) for s in surfaces[-_PIECE_EXCERPT_TAIL:])
    omitted = len(surfaces) - _PIECE_EXCERPT_CAP
    return rf"{head}, \textellipsis~({omitted} more), {tail}"


def render_base_glyphs(
    groups: Sequence[tuple[str, Sequence[str]]],
    *,
    n_glyphs: int,
    n_specials: int,
) -> str:
    """Render the full base-glyph inventory (appendix), grouped by OpenSMILES role.

    ``groups`` is an ordered sequence of ``(group_label, tokens)`` pairs that
    together partition the $165$-token base; ``n_glyphs`` / ``n_specials`` are the
    chemistry-grammatical and special-token counts reported in the caption.
    """
    total = n_glyphs + n_specials
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        rf"  \caption{{The complete Smirk base: the {total}-token OpenSMILES glyph "
        rf"alphabet ({n_glyphs} chemistry-grammatical glyphs and {n_specials} "
        r"special tokens), grouped by OpenSMILES role. Any OpenSMILES-conformant "
        r"string decomposes into "
        r"these glyphs with no \texttt{[UNK]}. Inside a bracket atom \texttt{+} and "
        r"\texttt{-} also denote charge, and \texttt{-} doubles as the single "
        r"bond.}",
        r"  \label{tab:base-glyphs}",
        r"  \small",
        r"  \begin{tabular}{@{}p{0.27\linewidth} p{0.66\linewidth}@{}}",
        r"    \toprule",
        r"    Group & Glyphs \\",
        r"    \midrule",
    ]
    for i, (label, toks) in enumerate(groups):
        if i:
            lines.append(r"    \addlinespace[3pt]")
        glyphs = ", ".join(rf"\texttt{{{_texttt(t)}}}" for t in toks)
        lines.append(rf"    {label} ({len(toks)}) & {glyphs} \\")
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def render_multiglyph_split(
    blocks: Sequence[tuple[str, float, Sequence[tuple[str, Sequence[str]]]]],
    *,
    corpus_label: str,
    vocab_size: int,
    boundary_robust: Sequence[str] = (),
) -> str:
    """Render the three-way learned-piece split as one joint table float.

    Each block is ``(boundary_label, jaccard, buckets)`` where ``buckets`` is an
    ordered sequence of ``(membership_label, surfaces)`` pairs (shared / BPE-only
    / Unigram-LM-only) and ``surfaces`` are pre-ordered glyph surfaces. Both
    boundary policies share one caption and one label (``tab:multiglyph-v256``),
    each rendered as a rule-separated block whose header carries its policy name
    and overlap $J$.

    ``boundary_robust`` is the set of surfaces selected under *both* boundary
    policies; these are rendered in grey so policy-invariant pieces stand apart
    from the policy-specific ones (shown plain).
    """
    robust = frozenset(boundary_robust)

    def _piece(surface: str) -> str:
        body = rf"\texttt{{{_texttt(surface)}}}"
        if surface in robust:
            body = rf"\bndtok{{{body}}}"
        return body

    caption = (
        rf"Learned multi-glyph pieces for the {corpus_label} "
        rf"$V{{=}}{vocab_size}$ matched pair, split three ways by cross-algorithm "
        r"membership (shared by both arms, selected only by BPE, and selected "
        r"only by Unigram-LM) under each boundary policy, with the per-policy "
        r"overlap $J$ in the block header. Each piece is its concatenated glyph "
        r"surface, ordered by glyph count then alphabetically; pieces also "
        r"selected under the other boundary policy (\emph{boundary-robust}: "
        r"selected under both) are \bndtok{grayed}, leaving the policy-specific "
        r"pieces plain. Classes "
        rf"larger than {_PIECE_EXCERPT_CAP} pieces are abridged to their first "
        rf"{_PIECE_EXCERPT_HEAD} and last {_PIECE_EXCERPT_TAIL} (the "
        r"parenthetical gives the full class size); complete lists are in the "
        r"data deposit."
    )
    # Grey marker for a piece selected under both boundary policies.
    lines = [
        r"\providecommand{\bndtok}[1]{\textcolor{black!50}{#1}}",
        r"\begin{table}[htbp]",
        r"  \centering",
        rf"  \caption{{{caption}}}",
        r"  \label{tab:multiglyph-v256}",
        r"  \footnotesize",
        r"  \begin{tabular}{@{}l p{0.78\linewidth}@{}}",
        r"    \toprule",
        r"    Membership & Pieces \\",
    ]
    for boundary_label, jaccard, buckets in blocks:
        lines.append(r"    \midrule")
        lines.append(
            rf"    \multicolumn{{2}}{{@{{}}l}}{{\emph{{{boundary_label}}}, "
            rf"overlap $J{{=}}{jaccard:.3f}$}} \\"
        )
        for i, (name, surfaces) in enumerate(buckets):
            gap = r"    \addlinespace[2pt]" if i == 0 else r"    \addlinespace[3pt]"
            lines.append(gap)
            pieces = _excerpt_pieces(surfaces, _piece)
            lines.append(rf"    {name} ({len(surfaces)}) & {pieces} \\")
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def render_shared_core_growth(
    blocks: Sequence[tuple[str, Sequence[tuple[str, Sequence[str]]]]],
    *,
    corpus_label: str,
    trend_summary: str,
    cross_boundary: Sequence[str] = (),
) -> str:
    """Render the nested BPE-cap-Unigram shared core as one joint table float.

    Each block is ``(boundary_label, layers)`` where ``layers`` is an ordered
    sequence of ``(layer_label, surfaces)`` pairs giving the pieces *newly*
    shared at each $V$ (the core is nested, so the cumulative core at a given
    $V$ is that layer plus all above). Both boundary policies share one caption
    and one label (``tab:shared-core-growth``), each a rule-separated block under
    a policy header; ``trend_summary`` states the per-policy cumulative sizes and
    overlaps in the caption.

    ``cross_boundary`` is the set of surfaces present in *both* boundaries' shared
    cores; these are greyed so the cross-boundary-robust core stands apart from
    the boundary-specific shared pieces (shown plain).
    """
    cross = frozenset(cross_boundary)

    def _piece(surface: str) -> str:
        body = rf"\texttt{{{_texttt(surface)}}}"
        return rf"\bndtok{{{body}}}" if surface in cross else body

    caption = (
        rf"Growth of the BPE\,$\cap$\,Unigram-LM shared core for the "
        rf"{corpus_label} matched pair under each boundary policy, as the target "
        r"vocabulary grows $V \in \{256, 512, 1024, 2048\}$. The core is "
        r"strictly nested (no piece ever leaves), so each row lists only "
        r"the pieces \emph{newly} shared at that $V$, and the cumulative core "
        rf"at a given $V$ is that row plus all above it. {trend_summary} Pieces "
        r"also in the other boundary's shared core (\emph{boundary-robust}: "
        r"shared under both) are \bndtok{grayed}, leaving the boundary-specific "
        r"shared pieces plain; "
        r"pieces are ordered by glyph count then alphabetically. Layers larger "
        rf"than {_PIECE_EXCERPT_CAP} pieces are abridged to their first "
        rf"{_PIECE_EXCERPT_HEAD} and last {_PIECE_EXCERPT_TAIL} (the "
        r"parenthetical gives the full layer size); complete lists are in the "
        r"data deposit."
    )
    # Grey marker for a piece shared under both boundary policies.
    lines = [
        r"\providecommand{\bndtok}[1]{\textcolor{black!50}{#1}}",
        r"\begin{table}[htbp]",
        r"  \centering",
        rf"  \caption{{{caption}}}",
        r"  \label{tab:shared-core-growth}",
        r"  \footnotesize",
        r"  \begin{tabular}{@{}l p{0.79\linewidth}@{}}",
        r"    \toprule",
        r"    Entered at & Newly shared pieces \\",
    ]
    for boundary_label, layers in blocks:
        lines.append(r"    \midrule")
        lines.append(
            rf"    \multicolumn{{2}}{{@{{}}l}}{{\emph{{{boundary_label}}}}} \\"
        )
        for i, (name, surfaces) in enumerate(layers):
            gap = r"    \addlinespace[2pt]" if i == 0 else r"    \addlinespace[3pt]"
            lines.append(gap)
            pieces = _excerpt_pieces(surfaces, _piece)
            lines.append(rf"    {name} ({len(surfaces)}) & {pieces} \\")
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def render_arm_exclusive(
    blocks: Sequence[
        tuple[str, Sequence[tuple[int, int, int, float, float, int, int]]]
    ],
    *,
    corpus_label: str,
) -> str:
    """Render arm-exclusive (BPE-only / Unigram-LM-only) statistics across $V$.

    Each block is ``(boundary_label, rows)`` where each row is
    ``(V, n_bpe_only, n_ul_only, mean_len_bpe, mean_len_ul, max_len_bpe,
    max_len_ul)`` (glyph-length means/maxes over the arm-exclusive multi-glyph
    pieces). Label is ``tab:arm-exclusive``.
    """
    caption = (
        rf"The arm-exclusive multi-glyph sets for the {corpus_label} matched pair "
        r"across $V$: counts and glyph-length statistics of the pieces each arm "
        r"selects but the other does not. The two arms select near-equal "
        r"\emph{numbers} of exclusive pieces at every $V$, and these dwarf the "
        r"shared core (Table~\ref{tab:shared-core-growth}), so the "
        r"near-disjointness persists at every "
        r"scale. The Unigram-LM (UL) exclusive pieces never exceed $16$ glyphs (its "
        r"\texttt{max\_piece\_length}, Table~\ref{tab:hyperparams}), whereas "
        r"BPE imposes no length cap, so its exclusive pieces run far longer "
        r"(degenerate periodic chains such as a $96$-glyph \texttt{OCCOCC}$\dots$ "
        r"at $V{=}2048$)."
    )
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        rf"  \caption{{{caption}}}",
        r"  \label{tab:arm-exclusive}",
        r"  \small",
        r"  \begin{tabular}{@{}r r r r r r r@{}}",
        r"    \toprule",
        r"    & \multicolumn{2}{c}{Exclusive pieces} "
        r"& \multicolumn{2}{c}{Mean glyph len.} "
        r"& \multicolumn{2}{c}{Max glyph len.} \\",
        r"    \cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r"    $V$ & BPE & UL & BPE & UL & BPE & UL \\",
        r"    \midrule",
    ]
    for i, (boundary_label, rows) in enumerate(blocks):
        if i:
            lines.append(r"    \addlinespace[2pt]")
        lines.append(
            rf"    \multicolumn{{7}}{{@{{}}l}}{{\emph{{{boundary_label}}}}} \\"
        )
        for v, n_bpe, n_ul, m_bpe, m_ul, x_bpe, x_ul in rows:
            lines.append(
                rf"    {v} & {_count(n_bpe)} & {_count(n_ul)} & {m_bpe:.1f} "
                rf"& {m_ul:.1f} & {x_bpe} & {x_ul} \\"
            )
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def render_corpus_contrast(
    blocks: Sequence[
        tuple[
            str,
            Sequence[
                tuple[
                    int,
                    int,
                    int | None,
                    int | None,
                    int | None,
                    int | None,
                    float | None,
                ]
            ],
        ]
    ],
    *,
    corpus_label: str,
    label: str,
    single_arm_note: bool = False,
) -> str:
    """Render a per-$V$ vocabulary-size / cross-arm-split contrast for one corpus.

    Each block is ``(boundary_label, rows)`` where each row is
    ``(V, n_bpe, n_ul, n_shared, n_bpe_only, n_ul_only, jaccard)``. A row whose
    ``n_ul`` is ``None`` is single-arm (the Unigram arm untrained at that $V$): its
    cross-arm columns render ``---`` and its $V$ carries a dagger. The caption is
    factual; the corpus-specific narrative lives in the surrounding prose. Set
    ``single_arm_note`` to append the dagger explanation when any row is single-arm.
    """

    def _i(x: int | None) -> str:
        return _count(x)

    def _jj(x: float | None) -> str:
        return _NA if x is None else f"{x:.3f}"

    caption = (
        rf"The {corpus_label} matched pair across $V$: each arm's multi-glyph "
        r"vocabulary size, the three-way cross-algorithm split "
        r"(shared / BPE-only / Unigram-LM-only), and the overlap $J$, under both "
        r"boundary policies (UL denotes Unigram-LM)."
    )
    if single_arm_note:
        caption += (
            r" $\dagger$~single-arm: the Unigram-LM arm is embedding-tail-unsafe "
            r"at this $V$ and was left untrained (\S\ref{sec:results}), so no "
            r"cross-arm split exists."
        )
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        rf"  \caption{{{caption}}}",
        rf"  \label{{{label}}}",
        r"  \small",
        r"  \begin{tabular}{@{}r r r r r r r@{}}",
        r"    \toprule",
        r"    & \multicolumn{2}{c}{Multi-glyph vocab.} "
        r"& \multicolumn{3}{c}{Cross-arm split} & \\",
        r"    \cmidrule(lr){2-3}\cmidrule(lr){4-6}",
        r"    $V$ & BPE & UL & Shared & BPE-only & UL-only & $J$ \\",
        r"    \midrule",
    ]
    for i, (boundary_label, rows) in enumerate(blocks):
        if i:
            lines.append(r"    \addlinespace[2pt]")
        lines.append(
            rf"    \multicolumn{{7}}{{@{{}}l}}{{\emph{{{boundary_label}}}}} \\"
        )
        for v, n_bpe, n_ul, n_shared, n_bpe_only, n_ul_only, jaccard in rows:
            vcell = rf"{v}$^{{\dagger}}$" if n_ul is None else str(v)
            lines.append(
                rf"    {vcell} & {_count(n_bpe)} & {_i(n_ul)} & {_i(n_shared)} & "
                rf"{_i(n_bpe_only)} & {_i(n_ul_only)} & {_jj(jaccard)} \\"
            )
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def render_composition(
    blocks: Sequence[tuple[str, Sequence[tuple[str, int, tuple[int, int, int, int]]]]],
    *,
    vocab_size: int,
    boundary_abbr: str,
) -> str:
    """Render the substructure composition of each cross-arm bucket, by corpus.

    Each block is ``(corpus_label, rows)`` where each row is
    ``(bucket_label, n, (pct_satC, pct_unsatC, pct_aromatic, pct_heteroatom))`` --
    integer percentages within that bucket. Label is ``tab:composition``.
    """
    caption = (
        r"Substructure composition of the shared, BPE-only, and Unigram-LM-only "
        rf"multi-glyph sets, by corpus (matched pairs at $V{{=}}{vocab_size}$, "
        rf"{boundary_abbr}). Each piece is classed from its glyphs by priority "
        r"aromatic atom $>$ aliphatic heteroatom $>$ unsaturated carbon $>$ "
        r"saturated carbon, and each row gives the percentage of that set's "
        r"pieces in each class. Aromatic-ring pieces are a BPE specialty the "
        r"Unigram-LM arm "
        r"almost never forms; bracket-internal pieces arise only under MB "
        r"(\S\ref{app:multiglyph}) and do not occur here."
    )
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        rf"  \caption{{{caption}}}",
        r"  \label{tab:composition}",
        r"  \small",
        r"  \begin{tabular}{@{}l r r r r r@{}}",
        r"    \toprule",
        r"    Set & $n$ & Sat.\,C & Unsat.\,C & Aromatic & Heteroatom \\",
        r"    \midrule",
    ]
    for i, (corpus_label, rows) in enumerate(blocks):
        if i:
            lines.append(r"    \addlinespace[2pt]")
        lines.append(rf"    \multicolumn{{6}}{{@{{}}l}}{{\emph{{{corpus_label}}}}} \\")
        for bucket_label, n, (p_sat, p_unsat, p_arom, p_het) in rows:
            lines.append(
                rf"    {bucket_label} & {n} & {p_sat}\% & {p_unsat}\% & "
                rf"{p_arom}\% & {p_het}\% \\"
            )
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def _joined(values: Sequence[float], decimals: int = 3) -> str:
    return " / ".join(f"{v:.{decimals}f}" for v in values) if values else _NA


def _redraw_rows(redraws: Sequence[extract.RedrawSpread]) -> list[str]:
    rows: list[str] = []
    for i, r in enumerate(redraws):
        lo, hi = min(r.clearances), max(r.clearances)
        value = f"{lo:.3f}" if lo == hi else f"{lo:.3f}--{hi:.3f}"
        # Print the probe label once; blank it on continuation rows so the
        # redraw rows read as one probe, matching the per-condition tables.
        label = "Subsample redraw" if i == 0 else ""
        rows.append(
            rf"{label} & {_corpus(r.corpus)} $V$=512 NMB "
            rf"($\times${len(r.clearances)}) & Unigram $c_{{100}}$ & {value} & "
            rf"spread {r.spread:.3f} \\"
        )
    if not rows:
        rows.append(
            rf"Subsample redraw & {_NA} & Unigram $c_{{100}}$ & {_NA} & {_NA} \\"
        )
    return rows


def render_robustness_extras(extras: extract.RobustnessExtras) -> str:
    """Render the compact robustness-extras summary (results subsection)."""
    header = r"Probe & Setting & Metric & Value & Reading"

    sweep_clearances = [p.unigram_clearance for p in extras.size_sweep]
    sweep_value = _joined(sweep_clearances)
    monotone = len(sweep_clearances) >= 2 and all(
        a < b for a, b in pairwise(sweep_clearances)
    )
    sweep_reading = r"$\uparrow$ with size" if monotone else _NA

    seed_inert = (
        extras.seed_cap_jaccard == 1.0 and extras.seed_cap_symmetric_difference == 0
    )
    if seed_inert:
        seed_reading = "inert"
    else:
        seed_reading = _NA if extras.seed_cap_jaccard is None else "shift"

    prune_value = _joined([p.jaccard for p in extras.prune])
    prune_setting = (
        "/".join(f"$V$={p.vocab_size}" for p in extras.prune) if extras.prune else _NA
    )

    merge_value = _count(extras.merge_exhaustion_realised_v)
    merge_reading = (
        r"natural ($<$ cap)"
        if extras.merge_exhaustion_natural
        and extras.merge_exhaustion_realised_v is not None
        and extras.merge_exhaustion_cap is not None
        and extras.merge_exhaustion_realised_v < extras.merge_exhaustion_cap
        else _NA
    )

    body = _redraw_rows(extras.redraws)
    body += [
        rf"Size sweep & PubChem 5/15/50M & Unigram $c_{{100}}$ & "
        rf"{sweep_value} & {sweep_reading} \\",
        rf"Seed cap & PubChem U $V$=1024 MB & multi-glyph $J$ (1e6/8e6) & "
        rf"{_f(extras.seed_cap_jaccard)} & {seed_reading} \\",
        rf"Prune schedule & PubChem U {prune_setting} MB & "
        rf"multi-glyph $J$ (0.75/0.9) & {prune_value} & schedule-sensitive \\",
        rf"Merge exhaustion & REAL-Space GPE, cap "
        rf"{_count(extras.merge_exhaustion_cap)} & realized $|\mathcal{{V}}|$ & "
        rf"{merge_value} & {merge_reading} \\",
    ]

    note = (
        r"Outside the headline grid. "
        r"Subsample redraw: Unigram $c_{100}$ across three independent draws "
        r"of the same corpus. Size sweep: the same cell trained at 5M / 15M / 50M "
        r"molecules. Seed cap and prune schedule: multi-glyph vocabulary Jaccard of "
        r"a one-armed hyperparameter probe against its baseline (1 = identical piece "
        r"set). Merge exhaustion: the realized vocabulary where \texttt{GpeTrainer} "
        r"terminated naturally below a $50{,}000$ cap."
    )
    return _table_float(
        caption="Robustness extras: subsample, size sweep, seed-cap, "
        "prune-schedule, merge-exhaustion.",
        label="tab:results-extras",
        colspec="l l l l l",
        header=header,
        body=body,
        note=note,
    )


def _mean_ci(
    mean: float | None, ci: tuple[float, float] | None, *, decimals: int
) -> str:
    """Format a mean with its bracketed CI in one cell: ``45.5 [45.1, 45.8]``."""
    if mean is None:
        return _NA
    m = f"{mean:.{decimals}f}"
    if ci is None:
        return m
    return f"{m} [{ci[0]:.{decimals}f}, {ci[1]:.{decimals}f}]"


# --------------------------------------------------------------------------- #
# Orchestration + freshness                                                   #
# --------------------------------------------------------------------------- #


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text)
    tmp_path.replace(path)


def is_tables_fresh(*, include_extras: bool = False) -> bool:
    """True iff every ``.tex`` exists and the manifest matches the upstream SHAs."""
    if not RESULTS_MANIFEST.is_file():
        return False
    if not all((RESULTS_TABLES_DIR / name).is_file() for name in TABLE_FILES.values()):
        return False
    try:
        manifest = json.loads(RESULTS_MANIFEST.read_text())
    except json.JSONDecodeError:
        return False
    return (
        manifest.get("include_extras") == include_extras
        and manifest.get("upstream_sha") == extract.upstream_sha_map()
        and manifest.get("audit_sha") == extract.audit_upstream_sha_map()
    )


def write_tables(*, include_extras: bool = False) -> list[Path]:
    """Render and deposit the ``.tex`` tables plus the manifest sidecar.

    The headline tables honour ``include_extras`` (grid-only by default). The
    ``robustness_extras`` table is its own deposit summarising the extras probes,
    so the headline tables stay clean regardless of the flag.
    """
    rendered = {
        "seven_measurements": render_seven_measurements(
            extract.measurement_rows(include_extras=include_extras)
        ),
        "delta_f": render_delta_f(extract.delta_f_rows(include_extras=include_extras)),
        "three_jaccards": render_three_jaccards(
            extract.jaccard_rows(include_extras=include_extras)
        ),
        "realized_vocab": render_realized_vocab(
            extract.jaccard_rows(include_extras=include_extras)
        ),
        "fertility": render_fertility(
            extract.fertility_rows(include_extras=include_extras)
        ),
        "nestedness": render_nestedness(
            extract.nestedness_rows(include_extras=include_extras)
        ),
        "closure": render_closure(extract.closure_rows(include_extras=include_extras)),
        "fg_alignment": render_fg_alignment(
            extract.fg_alignment_rows(include_extras=include_extras)
        ),
        "noncanon": render_noncanon(
            extract.noncanon_rows(include_extras=include_extras)
        ),
        "distribution": render_distribution(
            extract.distribution_rows(include_extras=include_extras)
        ),
        "absorption": render_absorption(
            extract.absorption_rows(include_extras=include_extras)
        ),
        "deadzone_nsweep": render_deadzone_nsweep(
            extract.deadzone_nsweep_rows(include_extras=include_extras)
        ),
        "robustness_extras": render_robustness_extras(extract.robustness_extras()),
    }
    written: list[Path] = []
    for key, text in rendered.items():
        path = RESULTS_TABLES_DIR / TABLE_FILES[key]
        _atomic_write_text(path, text)
        written.append(path)

    manifest = {
        "include_extras": include_extras,
        "artifacts": sorted(TABLE_FILES.values()),
        "upstream_sha": extract.upstream_sha_map(),
        "audit_sha": extract.audit_upstream_sha_map(),
    }
    _atomic_write_text(
        RESULTS_MANIFEST, json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return written


__all__ = [
    "RESULTS_MANIFEST",
    "TABLE_FILES",
    "is_tables_fresh",
    "render_absorption",
    "render_arm_exclusive",
    "render_base_glyphs",
    "render_closure",
    "render_composition",
    "render_corpus_contrast",
    "render_deadzone_nsweep",
    "render_delta_f",
    "render_distribution",
    "render_fertility",
    "render_fg_alignment",
    "render_multiglyph_split",
    "render_nestedness",
    "render_noncanon",
    "render_realized_vocab",
    "render_robustness_extras",
    "render_seven_measurements",
    "render_shared_core_growth",
    "render_three_jaccards",
    "write_tables",
]
