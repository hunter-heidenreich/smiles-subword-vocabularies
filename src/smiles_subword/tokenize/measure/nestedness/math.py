"""Cross-arm segmentation nestedness — a parse-geometry contrast.

Where, on a held-out molecule, do the two arms place their token boundaries
relative to each other? Both arms consume the identical Smirk glyph stream, so
each arm's boundaries are a subset of the same ``L-1`` inter-glyph positions
(the prefix sums of its per-token glyph counts). Comparing the two cut-sets
position-by-position gives a 2x2 over every inter-glyph position:

* **agree-cut**   — both arms cut here;
* **nest**        — Unigram-LM cuts, BPE merges (BPE coarser; ``nest - conflict``
  summed equals the absolute fertility gap);
* **conflict**    — BPE cuts, Unigram-LM merges (a *genuine* crossing
  disagreement, not mere depth);
* **agree-merge** — neither cuts.

A molecule is *nested* iff its conflict count is zero (BPE's segmentation is a
strict coarsening of Unigram-LM's). Headline scalars: the boundary Jaccard
``agree_cut / |B u U|`` (read on realized parses, unlike membership), the
fertility-orthogonal conflict rate, and the nested-molecule fraction.

The conflict residual is *localized*: each conflict falls strictly inside one
Unigram-LM token (kept atomic by UL, split by BPE). Classifying those
cut-through UL pieces by substructure (:func:`classify_piece`) shows whether
conflict concentrates in UL's exclusive heteroatom / unsaturated-carbon pieces
rather than the shared saturated-alkyl core.

UQ: 95% percentile bootstrap CIs, 1000 molecule resamples, on the three
headline scalars (per-class localization counts carry no CI); vectorized with
numpy for the ~10^6-molecule splits, seed derived from ``pair_key``.

Pure computation: per-token glyph counts / tuples in, per-molecule and
matched-pair records out; the dual held-out encode pass lives in :mod:`.runner`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from smiles_subword.tokenize.measure._bootstrap import (
    CI_LEVEL,
    N_BOOTSTRAP_RESAMPLES,
    bootstrap_seed,
    percentile_ci,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


Boundary = Literal["nmb", "mb"]
Arm = Literal["bpe", "unigram"]
UnpairedReason = Literal["conditional_negative_branch", "extras_single_arm_knob"]

GlyphTuple = tuple[str, ...]

CLASSES: tuple[str, ...] = ("sat-C", "unsat-C", "aromatic", "heteroatom", "bracket")
"""Substructure classes for conflict localization. The first four mirror
``results/build/table_composition`` (so the NMB reading agrees with the
composition table); ``bracket`` separates the bracket-internal pieces that only
arise under MB."""

_AROMATIC: frozenset[str] = frozenset({"b", "c", "n", "o", "p", "s", "se", "as"})
_UNSATURATION: frozenset[str] = frozenset({"=", "#", "$"})


def classify_piece(piece: GlyphTuple) -> str:
    """Priority-partition a glyph tuple into one of :data:`CLASSES`.

    Priority bracket > aromatic atom > aliphatic heteroatom > unsaturated
    carbon > saturated carbon. A piece is *bracket* iff it contains the
    ``[`` glyph (bracket-internal, MB only); otherwise it falls through the
    same ladder ``results/build/table_composition._classify`` uses, so under NMB
    (no bracket pieces) the two classifiers agree.
    """
    if "[" in piece:
        return "bracket"
    glyphs = set(piece)
    if glyphs & _AROMATIC:
        return "aromatic"
    if any(g[0].isupper() and g != "C" for g in piece):
        return "heteroatom"
    if glyphs & _UNSATURATION:
        return "unsat-C"
    return "sat-C"


def _cut_positions(glyph_counts: Sequence[int]) -> tuple[frozenset[int], int]:
    """Internal cut positions (prefix sums excl. the end) and total glyph length.

    ``glyph_counts`` is the per-token glyph count in segmentation order; the
    returned positions are the inter-glyph offsets the segmentation cuts at,
    i.e. all prefix sums but the final one (the end-of-string is not an
    internal position).
    """
    cuts: list[int] = []
    running = 0
    for count in glyph_counts:
        running += count
        cuts.append(running)
    return frozenset(cuts[:-1]), running


@dataclass(frozen=True)
class PerMoleculeNestedness:
    """Per-molecule boundary 2x2 plus per-class conflict localization.

    ``n_agree_cut + n_nest + n_conflict + n_agree_merge == n_positions``.
    ``emitted_by_class`` / ``cut_through_by_class`` are aligned to
    :data:`CLASSES`: emitted counts every multi-glyph Unigram-LM token, and
    cut-through counts those a BPE cut falls strictly inside.
    """

    n_positions: int
    n_agree_cut: int
    n_nest: int
    n_conflict: int
    n_agree_merge: int
    emitted_by_class: tuple[int, ...]
    cut_through_by_class: tuple[int, ...]

    @property
    def is_nested(self) -> bool:
        """True iff no conflict — BPE's parse is a strict coarsening of UL's."""
        return self.n_conflict == 0


def compare_molecule(
    bpe_glyph_counts: Sequence[int],
    ul_glyph_tuples: Sequence[GlyphTuple],
) -> PerMoleculeNestedness:
    """Compare one molecule's two segmentations into a per-molecule record.

    ``bpe_glyph_counts`` is the BPE arm's per-token glyph count; only counts
    are needed for its boundaries. ``ul_glyph_tuples`` is the Unigram-LM arm's
    per-token glyph tuple — its length gives the boundary, its glyphs the
    substructure class for conflict localization.

    Raises:
        ValueError: the two arms disagree on the total glyph length (they must
            share the glyph stream); the caller skips and counts such cases.
    """
    bpe_cuts, total_b = _cut_positions(bpe_glyph_counts)
    ul_counts = [len(t) for t in ul_glyph_tuples]
    ul_cuts, total_u = _cut_positions(ul_counts)
    if total_b != total_u:
        raise ValueError(f"glyph-length mismatch: bpe={total_b}, unigram={total_u}")

    n_positions = max(total_b - 1, 0)
    agree_cut = len(bpe_cuts & ul_cuts)
    nest = len(ul_cuts - bpe_cuts)
    conflict = len(bpe_cuts - ul_cuts)
    agree_merge = n_positions - len(bpe_cuts | ul_cuts)

    emitted = [0] * len(CLASSES)
    cut_through = [0] * len(CLASSES)
    start = 0
    for piece, length in zip(ul_glyph_tuples, ul_counts, strict=True):
        if length >= 2:
            cls_idx = CLASSES.index(classify_piece(piece))
            emitted[cls_idx] += 1
            internal = range(start + 1, start + length)
            if any(p in bpe_cuts for p in internal):
                cut_through[cls_idx] += 1
        start += length

    return PerMoleculeNestedness(
        n_positions=n_positions,
        n_agree_cut=agree_cut,
        n_nest=nest,
        n_conflict=conflict,
        n_agree_merge=agree_merge,
        emitted_by_class=tuple(emitted),
        cut_through_by_class=tuple(cut_through),
    )


@dataclass(frozen=True)
class MatchedPairNestedness:
    """Nestedness for one matched-arm ``(corpus, V, boundary)`` coordinate."""

    pair_key: str
    tier: str
    corpus: str
    vocab_size: int
    boundary: Boundary
    extras_kind: str | None
    extras_label: str | None
    bpe_cell_id: str
    unigram_cell_id: str
    bpe_training_corpus_sha: str
    unigram_training_corpus_sha: str
    eval_split_sha: str
    n_molecules: int
    n_length_mismatch: int
    n_positions: int
    n_agree_cut: int
    n_nest: int
    n_conflict: int
    n_agree_merge: int
    n_nested_molecules: int
    boundary_jaccard: float
    boundary_jaccard_ci: tuple[float, float]
    conflict_rate: float
    conflict_rate_ci: tuple[float, float]
    nest_rate: float
    agree_cut_rate: float
    agree_merge_rate: float
    conflict_share_of_disagreement: float
    nested_molecule_fraction: float
    nested_molecule_fraction_ci: tuple[float, float]
    emitted_by_class: tuple[int, ...]
    cut_through_by_class: tuple[int, ...]
    bootstrap_seed: int
    n_resamples: int

    @property
    def pair_status(self) -> Literal["matched"]:
        return "matched"

    def _class_map(self, counts: tuple[int, ...]) -> dict[str, int]:
        return dict(zip(CLASSES, counts, strict=True))

    def _cut_rate_by_class(self) -> dict[str, float | None]:
        out: dict[str, float | None] = {}
        for cls, emit, cut in zip(
            CLASSES, self.emitted_by_class, self.cut_through_by_class, strict=True
        ):
            out[cls] = (cut / emit) if emit else None
        return out

    def as_dict(self) -> dict[str, object]:
        return {
            "pair_key": self.pair_key,
            "pair_status": self.pair_status,
            "tier": self.tier,
            "corpus": self.corpus,
            "vocab_size": self.vocab_size,
            "boundary": self.boundary,
            "extras_kind": self.extras_kind,
            "extras_label": self.extras_label,
            "bpe_cell_id": self.bpe_cell_id,
            "unigram_cell_id": self.unigram_cell_id,
            "bpe_training_corpus_sha": self.bpe_training_corpus_sha,
            "unigram_training_corpus_sha": self.unigram_training_corpus_sha,
            "eval_split_sha": self.eval_split_sha,
            "n_molecules": self.n_molecules,
            "n_length_mismatch": self.n_length_mismatch,
            "n_positions": self.n_positions,
            "n_agree_cut": self.n_agree_cut,
            "n_nest": self.n_nest,
            "n_conflict": self.n_conflict,
            "n_agree_merge": self.n_agree_merge,
            "n_nested_molecules": self.n_nested_molecules,
            "boundary_jaccard": self.boundary_jaccard,
            "boundary_jaccard_ci": list(self.boundary_jaccard_ci),
            "conflict_rate": self.conflict_rate,
            "conflict_rate_ci": list(self.conflict_rate_ci),
            "nest_rate": self.nest_rate,
            "agree_cut_rate": self.agree_cut_rate,
            "agree_merge_rate": self.agree_merge_rate,
            "conflict_share_of_disagreement": self.conflict_share_of_disagreement,
            "nested_molecule_fraction": self.nested_molecule_fraction,
            "nested_molecule_fraction_ci": list(self.nested_molecule_fraction_ci),
            "emitted_by_class": self._class_map(self.emitted_by_class),
            "cut_through_by_class": self._class_map(self.cut_through_by_class),
            "cut_rate_by_class": self._cut_rate_by_class(),
            "bootstrap_seed": self.bootstrap_seed,
            "n_resamples": self.n_resamples,
            "missing_arm": None,
            "unpaired_reason": None,
        }


@dataclass(frozen=True)
class UnpairedNestedness:
    """Coordinate with only one arm — nestedness is undefined (it is cross-arm).

    Recorded for schema uniformity with the other measurements so the
    aggregator can list it; carries the present arm's cell and SHAs (so the
    freshness check works) but no metrics.
    """

    pair_key: str
    tier: str
    corpus: str
    vocab_size: int
    boundary: Boundary
    extras_kind: str | None
    extras_label: str | None
    present_arm: Arm
    present_cell_id: str
    present_training_corpus_sha: str
    eval_split_sha: str
    missing_arm: Arm
    unpaired_reason: UnpairedReason

    @property
    def pair_status(self) -> Literal["single_arm"]:
        return "single_arm"

    def as_dict(self) -> dict[str, object]:
        return {
            "pair_key": self.pair_key,
            "pair_status": self.pair_status,
            "tier": self.tier,
            "corpus": self.corpus,
            "vocab_size": self.vocab_size,
            "boundary": self.boundary,
            "extras_kind": self.extras_kind,
            "extras_label": self.extras_label,
            "present_arm": self.present_arm,
            "present_cell_id": self.present_cell_id,
            "present_training_corpus_sha": self.present_training_corpus_sha,
            "eval_split_sha": self.eval_split_sha,
            "missing_arm": self.missing_arm,
            "unpaired_reason": self.unpaired_reason,
        }


def _bootstrap_ratio_ci(
    numerators: np.ndarray,
    denominators: np.ndarray,
    *,
    seed: int,
    n_resamples: int = N_BOOTSTRAP_RESAMPLES,
    level: float = CI_LEVEL,
) -> tuple[float, float]:
    """Percentile bootstrap CI for ``Σ num / Σ denom`` over molecule resamples.

    Vectorized per resample (one numpy ``integers`` draw of size ``n`` per
    iteration) so peak memory stays ``O(n)`` rather than ``O(n_resamples * n)``
    — the held-out splits reach ~10^6 molecules. NaN resamples (an all-zero
    denominator) are dropped, then the surviving samples are reduced through the
    shared :func:`~smiles_subword.tokenize.measure._bootstrap.percentile_ci`
    estimator so nestedness uses the *same* percentile method as every
    other measurement. Returns ``(nan, nan)`` when every resampled denominator
    is zero.
    """
    n = int(numerators.shape[0])
    if n == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    samples = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        denom_sum = float(denominators[idx].sum())
        samples[i] = (
            float(numerators[idx].sum()) / denom_sum if denom_sum > 0 else np.nan
        )
    finite = samples[np.isfinite(samples)]
    return percentile_ci([float(x) for x in finite], level=level)


def compute_pair_nestedness(
    per_molecule: Sequence[PerMoleculeNestedness],
    *,
    pair_key: str,
    tier: str,
    corpus: str,
    vocab_size: int,
    boundary: Boundary,
    bpe_cell_id: str,
    unigram_cell_id: str,
    bpe_training_corpus_sha: str,
    unigram_training_corpus_sha: str,
    eval_split_sha: str,
    extras_kind: str | None = None,
    extras_label: str | None = None,
    n_length_mismatch: int = 0,
    seed: int | None = None,
    n_resamples: int = N_BOOTSTRAP_RESAMPLES,
) -> MatchedPairNestedness:
    """Aggregate per-molecule records into a matched-pair nestedness reading.

    ``boundary_jaccard`` is ``Σ agree_cut / Σ(agree_cut + nest + conflict)``;
    ``conflict_rate`` and ``nest_rate`` divide by total positions;
    ``nested_molecule_fraction`` is the share of molecules with zero conflict.
    Bootstrap CIs (molecule resample) accompany the three headline scalars.
    """
    n_mol = len(per_molecule)
    agree_cut = sum(pm.n_agree_cut for pm in per_molecule)
    nest = sum(pm.n_nest for pm in per_molecule)
    conflict = sum(pm.n_conflict for pm in per_molecule)
    agree_merge = sum(pm.n_agree_merge for pm in per_molecule)
    positions = agree_cut + nest + conflict + agree_merge
    union = agree_cut + nest + conflict
    n_nested = sum(1 for pm in per_molecule if pm.is_nested)
    disagreement = nest + conflict

    emitted = tuple(
        sum(pm.emitted_by_class[i] for pm in per_molecule) for i in range(len(CLASSES))
    )
    cut_through = tuple(
        sum(pm.cut_through_by_class[i] for pm in per_molecule)
        for i in range(len(CLASSES))
    )

    seed = seed if seed is not None else bootstrap_seed(pair_key)
    agree_cut_arr = np.fromiter(
        (pm.n_agree_cut for pm in per_molecule), dtype=np.int64, count=n_mol
    )
    union_arr = np.fromiter(
        (pm.n_agree_cut + pm.n_nest + pm.n_conflict for pm in per_molecule),
        dtype=np.int64,
        count=n_mol,
    )
    conflict_arr = np.fromiter(
        (pm.n_conflict for pm in per_molecule), dtype=np.int64, count=n_mol
    )
    positions_arr = np.fromiter(
        (pm.n_positions for pm in per_molecule), dtype=np.int64, count=n_mol
    )
    nested_arr = np.fromiter(
        (1 if pm.is_nested else 0 for pm in per_molecule), dtype=np.int64, count=n_mol
    )
    ones = np.ones(n_mol, dtype=np.int64)

    boundary_jaccard_ci = _bootstrap_ratio_ci(
        agree_cut_arr, union_arr, seed=seed, n_resamples=n_resamples
    )
    conflict_rate_ci = _bootstrap_ratio_ci(
        conflict_arr, positions_arr, seed=seed + 1, n_resamples=n_resamples
    )
    nested_fraction_ci = _bootstrap_ratio_ci(
        nested_arr, ones, seed=seed + 2, n_resamples=n_resamples
    )

    return MatchedPairNestedness(
        pair_key=pair_key,
        tier=tier,
        corpus=corpus,
        vocab_size=vocab_size,
        boundary=boundary,
        extras_kind=extras_kind,
        extras_label=extras_label,
        bpe_cell_id=bpe_cell_id,
        unigram_cell_id=unigram_cell_id,
        bpe_training_corpus_sha=bpe_training_corpus_sha,
        unigram_training_corpus_sha=unigram_training_corpus_sha,
        eval_split_sha=eval_split_sha,
        n_molecules=n_mol,
        n_length_mismatch=n_length_mismatch,
        n_positions=positions,
        n_agree_cut=agree_cut,
        n_nest=nest,
        n_conflict=conflict,
        n_agree_merge=agree_merge,
        n_nested_molecules=n_nested,
        boundary_jaccard=(agree_cut / union) if union else float("nan"),
        boundary_jaccard_ci=boundary_jaccard_ci,
        conflict_rate=(conflict / positions) if positions else float("nan"),
        conflict_rate_ci=conflict_rate_ci,
        nest_rate=(nest / positions) if positions else float("nan"),
        agree_cut_rate=(agree_cut / positions) if positions else float("nan"),
        agree_merge_rate=(agree_merge / positions) if positions else float("nan"),
        conflict_share_of_disagreement=(
            (conflict / disagreement) if disagreement else float("nan")
        ),
        nested_molecule_fraction=(n_nested / n_mol) if n_mol else float("nan"),
        nested_molecule_fraction_ci=nested_fraction_ci,
        emitted_by_class=emitted,
        cut_through_by_class=cut_through,
        bootstrap_seed=seed,
        n_resamples=n_resamples,
    )


def make_unpaired_nestedness(
    *,
    pair_key: str,
    tier: str,
    corpus: str,
    vocab_size: int,
    boundary: Boundary,
    extras_kind: str | None,
    extras_label: str | None,
    present_arm: Arm,
    present_cell_id: str,
    present_training_corpus_sha: str,
    eval_split_sha: str,
    missing_arm: Arm,
    unpaired_reason: UnpairedReason,
) -> UnpairedNestedness:
    """Build an :class:`UnpairedNestedness` for a structurally single-arm coord."""
    if missing_arm == present_arm:
        raise ValueError(
            f"missing_arm={missing_arm!r} cannot equal present_arm {present_arm!r}"
        )
    return UnpairedNestedness(
        pair_key=pair_key,
        tier=tier,
        corpus=corpus,
        vocab_size=vocab_size,
        boundary=boundary,
        extras_kind=extras_kind,
        extras_label=extras_label,
        present_arm=present_arm,
        present_cell_id=present_cell_id,
        present_training_corpus_sha=present_training_corpus_sha,
        eval_split_sha=eval_split_sha,
        missing_arm=missing_arm,
        unpaired_reason=unpaired_reason,
    )


__all__ = [
    "CI_LEVEL",
    "CLASSES",
    "N_BOOTSTRAP_RESAMPLES",
    "Arm",
    "Boundary",
    "GlyphTuple",
    "MatchedPairNestedness",
    "PerMoleculeNestedness",
    "UnpairedNestedness",
    "bootstrap_seed",
    "classify_piece",
    "compare_molecule",
    "compute_pair_nestedness",
    "make_unpaired_nestedness",
]
