"""Vocabulary overlap (Jaccard; membership contrast).

For each matched ``(V, corpus, boundary)`` (BPE, Unigram-LM) pair we compare
the two arms' **multi-glyph subword sets** (the single-glyph atomic base is
excluded — identical in both arms by construction) with four Jaccards:

* **``J``** — unweighted Jaccard over the multi-glyph subword sets. Exact, no
  CI. The headline membership input.
* **``J_w``** — frequency-weighted Jaccard. Each arm's weight ``w(x)`` is
  subword ``x``'s emitted-token count on the held-out split, normalized so an
  arm's weights sum to one; ``J_w = Σ min(w_BPE, w_UL) / Σ max(w_BPE, w_UL)``
  over the multi-glyph union. 95% percentile bootstrap CI over 1000
  held-out-molecule resamples.
* **``J_struct``** — unweighted Jaccard over the *structural* multi-glyph union
  only: a subword every training-corpus occurrence of which falls inside a
  bracketed-atom Layer-B chunk is *bracket-internal* and dropped. Exact, no CI.
  Diagnostic, not a headline input.
* **``J_w_struct``** — the fourth cell of the weighting×masking 2×2: ``J_w``
  over each arm's structural subwords only, weights renormalized per arm over
  the structural held-out mass; same bootstrap CI as ``J_w``. Separates a
  ``J_w < J`` head-disagreement into genuine high-frequency structural
  (cross-pretoken) disagreement (``J_w_struct`` stays low) versus an artifact of
  Unigram's high-frequency bracket-internal fragmentation (``J_w_struct`` rises
  toward ``J_struct``). Diagnostic, not a headline input.

The subword identity throughout is the **glyph-tuple** (exact glyph sequence),
shared across arms since both share the Smirk Layer-A alphabet. Pure
computation: takes the runner's per-arm :class:`ArmJaccardInputs` (multi-glyph
sets, the training-corpus structural split, the held-out emission data) and
emits the deposited per-arm and matched-pair records; the corpus passes live
in :mod:`.runner`.
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
    from collections.abc import Mapping

Boundary = Literal["nmb", "mb"]
Arm = Literal["bpe", "unigram"]
UnpairedReason = Literal["conditional_negative_branch", "extras_single_arm_knob"]
GlyphTuple = tuple[str, ...]


def jaccard(a: frozenset[GlyphTuple], b: frozenset[GlyphTuple]) -> float:
    """Unweighted Jaccard ``|a ∩ b| / |a ∪ b|``; ``nan`` when both are empty."""
    union = len(a | b)
    if union == 0:
        return float("nan")
    return len(a & b) / union


def normalized_weights(counts: Mapping[GlyphTuple, float]) -> dict[GlyphTuple, float]:
    """Normalize raw emitted-token counts so the weights sum to one."""
    total = float(sum(counts.values()))
    if total == 0.0:
        return {}
    return {k: v / total for k, v in counts.items()}


def weighted_jaccard(
    w_a: Mapping[GlyphTuple, float], w_b: Mapping[GlyphTuple, float]
) -> float:
    """``Σ min(w_a, w_b) / Σ max(w_a, w_b)`` over the key union (``J_w``).

    Inputs are per-arm normalized weights (see :func:`normalized_weights`); a
    subword absent from an arm has weight zero in that arm. Returns ``nan``
    when both weight maps are empty.
    """
    keys = set(w_a) | set(w_b)
    if not keys:
        return float("nan")
    num = sum(min(w_a.get(k, 0.0), w_b.get(k, 0.0)) for k in keys)
    den = sum(max(w_a.get(k, 0.0), w_b.get(k, 0.0)) for k in keys)
    return num / den if den else float("nan")


@dataclass(frozen=True)
class JwMoleculeData:
    """Per-arm held-out emission data driving the ``J_w`` bootstrap.

    Sparse coordinate form, one entry per (molecule, multi-glyph subword) with
    a multiplicity ``count``. ``local_tuples`` maps a local subword id to its
    glyph-tuple so the matched-pair join can remap both arms into one union id
    space. Aligned on the same held-out molecule order across arms, so a single
    resample of molecule indices drives both arms.
    """

    n_molecules: int
    mol_idx: np.ndarray
    sub_local: np.ndarray
    count: np.ndarray
    local_tuples: tuple[GlyphTuple, ...]

    @property
    def total_emitted(self) -> int:
        return int(self.count.sum()) if self.count.size else 0


@dataclass(frozen=True)
class ArmJaccardInputs:
    """Runner output for one arm — the inputs the pure math consumes.

    Not deposited; :meth:`to_record` projects it to the deposited
    :class:`ArmJaccard`. ``multi_subwords`` is the arm's full multi-glyph
    (glyph_count ≥ 2) vocabulary set; ``structural``/``bracket_internal``
    partition the *emitted* subset by training-corpus occurrence; ``unseen``
    are vocab multi-glyph pieces never emitted by any inventory chunk.
    """

    cell_id: str
    arm: Arm
    boundary: Boundary
    training_corpus_sha: str
    eval_split_sha: str
    multi_subwords: frozenset[GlyphTuple]
    structural_subwords: frozenset[GlyphTuple]
    bracket_internal_subwords: frozenset[GlyphTuple]
    unseen_subwords: frozenset[GlyphTuple]
    n_distinct_bracket_chunks: int
    n_distinct_nonbracket_chunks: int
    nonbracket_cap_bound: bool
    jw: JwMoleculeData
    bootstrap_seed: int

    def to_record(self, *, n_resamples: int = N_BOOTSTRAP_RESAMPLES) -> ArmJaccard:
        return ArmJaccard(
            cell_id=self.cell_id,
            arm=self.arm,
            boundary=self.boundary,
            n_multi_subwords=len(self.multi_subwords),
            n_structural=len(self.structural_subwords),
            n_bracket_internal=len(self.bracket_internal_subwords),
            n_unseen=len(self.unseen_subwords),
            n_held_out_molecules=self.jw.n_molecules,
            total_emitted_multi=self.jw.total_emitted,
            n_distinct_bracket_chunks=self.n_distinct_bracket_chunks,
            n_distinct_nonbracket_chunks=self.n_distinct_nonbracket_chunks,
            nonbracket_cap_bound=self.nonbracket_cap_bound,
            training_corpus_sha=self.training_corpus_sha,
            eval_split_sha=self.eval_split_sha,
            bootstrap_seed=self.bootstrap_seed,
            n_resamples=n_resamples,
        )


@dataclass(frozen=True)
class ArmJaccard:
    """Deposited per-arm Jaccard summary (counts only; the Jaccards are cross-arm)."""

    cell_id: str
    arm: Arm
    boundary: Boundary
    n_multi_subwords: int
    n_structural: int
    n_bracket_internal: int
    n_unseen: int
    n_held_out_molecules: int
    total_emitted_multi: int
    n_distinct_bracket_chunks: int
    n_distinct_nonbracket_chunks: int
    nonbracket_cap_bound: bool
    training_corpus_sha: str
    eval_split_sha: str
    bootstrap_seed: int
    n_resamples: int

    def as_dict(self) -> dict[str, object]:
        return {
            "cell_id": self.cell_id,
            "arm": self.arm,
            "boundary": self.boundary,
            "n_multi_subwords": self.n_multi_subwords,
            "n_structural": self.n_structural,
            "n_bracket_internal": self.n_bracket_internal,
            "n_unseen": self.n_unseen,
            "n_held_out_molecules": self.n_held_out_molecules,
            "total_emitted_multi": self.total_emitted_multi,
            "n_distinct_bracket_chunks": self.n_distinct_bracket_chunks,
            "n_distinct_nonbracket_chunks": self.n_distinct_nonbracket_chunks,
            "nonbracket_cap_bound": self.nonbracket_cap_bound,
            "training_corpus_sha": self.training_corpus_sha,
            "eval_split_sha": self.eval_split_sha,
            "bootstrap_seed": self.bootstrap_seed,
            "n_resamples": self.n_resamples,
        }


@dataclass(frozen=True)
class MatchedPairJaccard:
    """Jaccard for one matched-arm coordinate.

    ``jaccard`` (the headline input), ``jaccard_struct`` (the diagnostic),
    ``weighted_jaccard`` (+ CI), and ``weighted_jaccard_struct`` (+ CI, the
    structural-restricted weighting that completes the 2×2) compare the two
    arms; ``jaccard_minus_struct`` is the ``J − J_struct`` gap localizing whether
    a contrast lives in the structural merges or is masked by shared
    bracket-reassembly.
    """

    pair_key: str
    tier: str
    corpus: str
    vocab_size: int
    boundary: Boundary
    extras_kind: str | None
    extras_label: str | None
    bpe: ArmJaccard
    unigram: ArmJaccard
    jaccard: float
    jaccard_struct: float
    weighted_jaccard: float
    weighted_jaccard_ci: tuple[float, float]
    weighted_jaccard_struct: float
    weighted_jaccard_struct_ci: tuple[float, float]
    jaccard_minus_struct: float

    @property
    def pair_status(self) -> Literal["matched"]:
        return "matched"

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
            "bpe": self.bpe.as_dict(),
            "unigram": self.unigram.as_dict(),
            "jaccard": self.jaccard,
            "jaccard_struct": self.jaccard_struct,
            "weighted_jaccard": self.weighted_jaccard,
            "weighted_jaccard_ci": list(self.weighted_jaccard_ci),
            "weighted_jaccard_struct": self.weighted_jaccard_struct,
            "weighted_jaccard_struct_ci": list(self.weighted_jaccard_struct_ci),
            "jaccard_minus_struct": self.jaccard_minus_struct,
            "missing_arm": None,
            "unpaired_reason": None,
        }


@dataclass(frozen=True)
class UnpairedJaccard:
    """Jaccard for a single-arm coordinate; the cross-arm Jaccards are undefined."""

    pair_key: str
    tier: str
    corpus: str
    vocab_size: int
    boundary: Boundary
    extras_kind: str | None
    extras_label: str | None
    present_arm: ArmJaccard
    missing_arm: Arm
    unpaired_reason: UnpairedReason

    @property
    def pair_status(self) -> Literal["single_arm"]:
        return "single_arm"

    def as_dict(self) -> dict[str, object]:
        arm_blocks: dict[str, object] = {
            self.present_arm.arm: self.present_arm.as_dict(),
            self.missing_arm: None,
        }
        return {
            "pair_key": self.pair_key,
            "pair_status": self.pair_status,
            "tier": self.tier,
            "corpus": self.corpus,
            "vocab_size": self.vocab_size,
            "boundary": self.boundary,
            "extras_kind": self.extras_kind,
            "extras_label": self.extras_label,
            "bpe": arm_blocks["bpe"],
            "unigram": arm_blocks["unigram"],
            "jaccard": None,
            "jaccard_struct": None,
            "weighted_jaccard": None,
            "weighted_jaccard_ci": None,
            "weighted_jaccard_struct": None,
            "weighted_jaccard_struct_ci": None,
            "jaccard_minus_struct": None,
            "missing_arm": self.missing_arm,
            "unpaired_reason": self.unpaired_reason,
        }


def _emitted_counts(jw: JwMoleculeData) -> dict[GlyphTuple, float]:
    """Aggregate held-out emitted counts per glyph-tuple for one arm."""
    if jw.count.size == 0:
        return {}
    totals = np.bincount(jw.sub_local, weights=jw.count, minlength=len(jw.local_tuples))
    return {jw.local_tuples[i]: float(totals[i]) for i in range(len(jw.local_tuples))}


def _restrict_jw_entries(
    jw: JwMoleculeData, allowed: frozenset[GlyphTuple]
) -> JwMoleculeData:
    """Drop emission rows whose subword is not in ``allowed`` (structural mask).

    Preserves ``n_molecules`` and ``local_tuples`` so the resample-unit count and
    the union id space are unchanged; masked-out pieces simply carry zero
    held-out mass, which renormalizes both the point estimate and each bootstrap
    resample over the structural emission mass alone.
    """
    if jw.sub_local.size == 0:
        return jw
    keep = np.fromiter(
        (jw.local_tuples[lid] in allowed for lid in jw.sub_local.tolist()),
        dtype=bool,
        count=jw.sub_local.size,
    )
    return JwMoleculeData(
        n_molecules=jw.n_molecules,
        mol_idx=jw.mol_idx[keep],
        sub_local=jw.sub_local[keep],
        count=jw.count[keep],
        local_tuples=jw.local_tuples,
    )


def _bootstrap_weighted_jaccard(
    bpe: JwMoleculeData,
    unigram: JwMoleculeData,
    *,
    seed: int,
    n_resamples: int = N_BOOTSTRAP_RESAMPLES,
    level: float = CI_LEVEL,
) -> tuple[float, float]:
    """Percentile bootstrap CI for ``J_w`` over held-out molecule resamples.

    Both arms share one resample of molecule indices (they encode the same
    held-out molecule set in the same order). Each resample re-aggregates the
    per-arm emitted counts in a shared union id space via a weighted
    ``bincount``, renormalizes per arm, and recomputes ``Σ min / Σ max``.
    """
    if bpe.n_molecules != unigram.n_molecules:
        raise ValueError(
            "arms disagree on held-out molecule count: "
            f"bpe={bpe.n_molecules}, unigram={unigram.n_molecules}"
        )
    n_mol = bpe.n_molecules
    if n_mol == 0:
        return (float("nan"), float("nan"))

    union: dict[GlyphTuple, int] = {}
    for tup in (*bpe.local_tuples, *unigram.local_tuples):
        union.setdefault(tup, len(union))
    n_union = len(union)
    if n_union == 0:
        return (float("nan"), float("nan"))

    bpe_union = np.array(
        [union[bpe.local_tuples[i]] for i in bpe.sub_local], dtype=np.int64
    )
    ul_union = np.array(
        [union[unigram.local_tuples[i]] for i in unigram.sub_local], dtype=np.int64
    )

    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(n_resamples):
        picks = rng.integers(0, n_mol, n_mol)
        mult = np.bincount(picks, minlength=n_mol).astype(np.float64)
        agg_a = np.bincount(
            bpe_union, weights=bpe.count * mult[bpe.mol_idx], minlength=n_union
        )
        agg_b = np.bincount(
            ul_union, weights=unigram.count * mult[unigram.mol_idx], minlength=n_union
        )
        sum_a = agg_a.sum()
        sum_b = agg_b.sum()
        if sum_a == 0 or sum_b == 0:
            continue
        wa = agg_a / sum_a
        wb = agg_b / sum_b
        den = np.maximum(wa, wb).sum()
        samples.append(float(np.minimum(wa, wb).sum() / den) if den else float("nan"))
    finite = [s for s in samples if s == s]
    if not finite:
        return (float("nan"), float("nan"))
    return percentile_ci(finite, level=level)


def compute_matched_pair_jaccard(
    bpe: ArmJaccardInputs,
    unigram: ArmJaccardInputs,
    *,
    pair_key: str,
    tier: str,
    corpus: str,
    vocab_size: int,
    boundary: Boundary,
    extras_kind: str | None = None,
    extras_label: str | None = None,
    n_resamples: int = N_BOOTSTRAP_RESAMPLES,
) -> MatchedPairJaccard:
    """Join two arms into a matched-pair record, computing all three Jaccards.

    Raises:
        ValueError: arm-tag mismatch or boundary disagreement.
    """
    if bpe.arm != "bpe":
        raise ValueError(f"first argument must be the BPE arm; got arm={bpe.arm!r}")
    if unigram.arm != "unigram":
        raise ValueError(
            f"second argument must be the Unigram arm; got arm={unigram.arm!r}"
        )
    if bpe.boundary != unigram.boundary or bpe.boundary != boundary:
        raise ValueError(
            "arm boundaries must match the matched-pair boundary; got "
            f"bpe={bpe.boundary!r}, unigram={unigram.boundary!r}, pair={boundary!r}"
        )

    seed = bootstrap_seed(pair_key)
    j = jaccard(bpe.multi_subwords, unigram.multi_subwords)
    j_struct = jaccard(bpe.structural_subwords, unigram.structural_subwords)
    w_a = normalized_weights(_emitted_counts(bpe.jw))
    w_b = normalized_weights(_emitted_counts(unigram.jw))
    j_w = weighted_jaccard(w_a, w_b)
    j_w_ci = _bootstrap_weighted_jaccard(
        bpe.jw, unigram.jw, seed=seed, n_resamples=n_resamples
    )

    bpe_struct_jw = _restrict_jw_entries(bpe.jw, bpe.structural_subwords)
    ul_struct_jw = _restrict_jw_entries(unigram.jw, unigram.structural_subwords)
    j_w_struct = weighted_jaccard(
        normalized_weights(_emitted_counts(bpe_struct_jw)),
        normalized_weights(_emitted_counts(ul_struct_jw)),
    )
    j_w_struct_ci = _bootstrap_weighted_jaccard(
        bpe_struct_jw, ul_struct_jw, seed=seed, n_resamples=n_resamples
    )

    gap = j - j_struct if (j == j and j_struct == j_struct) else float("nan")
    return MatchedPairJaccard(
        pair_key=pair_key,
        tier=tier,
        corpus=corpus,
        vocab_size=vocab_size,
        boundary=boundary,
        extras_kind=extras_kind,
        extras_label=extras_label,
        bpe=bpe.to_record(n_resamples=n_resamples),
        unigram=unigram.to_record(n_resamples=n_resamples),
        jaccard=j,
        jaccard_struct=j_struct,
        weighted_jaccard=j_w,
        weighted_jaccard_ci=j_w_ci,
        weighted_jaccard_struct=j_w_struct,
        weighted_jaccard_struct_ci=j_w_struct_ci,
        jaccard_minus_struct=gap,
    )


def compute_unpaired_jaccard(
    arm_inputs: ArmJaccardInputs,
    *,
    pair_key: str,
    tier: str,
    corpus: str,
    vocab_size: int,
    boundary: Boundary,
    extras_kind: str | None,
    extras_label: str | None,
    missing_arm: Arm,
    unpaired_reason: UnpairedReason,
    n_resamples: int = N_BOOTSTRAP_RESAMPLES,
) -> UnpairedJaccard:
    """Wrap one present arm; the cross-arm Jaccards stay undefined."""
    if missing_arm == arm_inputs.arm:
        raise ValueError(
            f"missing_arm={missing_arm!r} cannot equal the present arm "
            f"{arm_inputs.arm!r}"
        )
    if arm_inputs.boundary != boundary:
        raise ValueError(
            f"arm boundary {arm_inputs.boundary!r} disagrees with pair {boundary!r}"
        )
    return UnpairedJaccard(
        pair_key=pair_key,
        tier=tier,
        corpus=corpus,
        vocab_size=vocab_size,
        boundary=boundary,
        extras_kind=extras_kind,
        extras_label=extras_label,
        present_arm=arm_inputs.to_record(n_resamples=n_resamples),
        missing_arm=missing_arm,
        unpaired_reason=unpaired_reason,
    )


__all__ = [
    "CI_LEVEL",
    "N_BOOTSTRAP_RESAMPLES",
    "Arm",
    "ArmJaccard",
    "ArmJaccardInputs",
    "Boundary",
    "GlyphTuple",
    "JwMoleculeData",
    "MatchedPairJaccard",
    "UnpairedJaccard",
    "bootstrap_seed",
    "compute_matched_pair_jaccard",
    "compute_unpaired_jaccard",
    "jaccard",
    "normalized_weights",
    "weighted_jaccard",
]
