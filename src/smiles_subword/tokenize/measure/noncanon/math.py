"""Non-canonicity robustness — tokenizer stability across the SMILES rewrite orbit.

Where every other measurement reads the tokenizers on canonical held-out strings,
this one rewrites each molecule into equivalent SMILES and asks whether each arm's
segmentation moves and whether BPE and Unigram-LM differ. Model-free, within-arm.

Five identity-preserving axes, four RDKit-internal rewrites in mild-to-catastrophic
order plus a cross-toolkit swap:

- **ringperm** — relabel ring-closure digits, same atom order (mild floor; token
  count is invariant by construction).
- **kekule** — aromatic notation to explicit alternating double bonds.
- **random** — RDKit's restricted (augmentation-realistic) randomized SMILES; K
  rewrites per molecule (Arus-Pous et al. 2019).
- **explicitH** — all hydrogens explicit, ``C`` to ``[CH4]`` (bracketed atoms are
  out-of-distribution / near-dead pieces).
- **obcanon** — the molecule's *OpenBabel* canonical SMILES, gated to
  identity-preservation by a round-trip through RDKit; a second canonicalizer's
  legitimate output. Where the two toolkits agree the molecule contributes a
  genuine zero.

Per arm, per axis we report two senses of movement, each with a 95%
molecule-resampled bootstrap CI (``_bootstrap``): relative fertility dispersion
``rel_dfert = mean |f_var - f_canon| / f_canon`` (token *count*) and bag-instability
``bag_instab = mean (1 - multiset Jaccard of token ids vs canonical)`` (the
*pieces*). The matched pair adds the per-axis cross-arm gap and the fertility-gap
survival ``gap = mean(UL tokens) / mean(BPE tokens)`` at canonical vs. randomized
orbit — whether the granularity gap is a canonical-notation artifact.

Within-arm: a single-arm coordinate carries a full reading. This module is the
pure aggregation; subsample, variant generation, and dual-encode live in
:mod:`.runner`.
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

# RDKit-internal axes in mild -> catastrophic order (cf. the AMORE model
# ordering), then the cross-toolkit canonical swap last (a distinct category).
AXES: tuple[str, ...] = ("ringperm", "kekule", "random", "explicitH", "obcanon")


@dataclass(frozen=True)
class PerMoleculeNoncanon:
    """One molecule's non-canonicity readings for one arm.

    ``canon_fert`` is the canonical token count; ``rand_fert_mean`` the mean
    token count over the random rewrites (for the gap-survival ratio).
    ``axis_dfert`` / ``axis_bag`` carry the molecule's mean relative fertility
    dispersion and mean bag-instability for each axis it has a variant for
    (ringperm needs a ring, kekule needs aromaticity; random / explicitH are
    always present).
    """

    canon_fert: int
    rand_fert_mean: float
    axis_dfert: dict[str, float]
    axis_bag: dict[str, float]


def _bootstrap_mean_ci(
    values: np.ndarray,
    rng: np.random.Generator,
    *,
    n_resamples: int,
    level: float,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of ``values`` (molecule resample)."""
    n = values.shape[0]
    if n == 0:
        return (float("nan"), float("nan"))
    samples = [
        float(np.dot(values, np.bincount(rng.integers(0, n, n), minlength=n)) / n)
        for _ in range(n_resamples)
    ]
    return percentile_ci(samples, level=level)


@dataclass(frozen=True)
class AxisReading:
    """Per-axis aggregate for one arm: count movement and piece movement."""

    n: int
    rel_dfert: float
    rel_dfert_ci: tuple[float, float]
    bag_instab: float
    bag_instab_ci: tuple[float, float]

    def as_dict(self) -> dict[str, object]:
        return {
            "n": self.n,
            "rel_dfert": self.rel_dfert,
            "rel_dfert_ci": list(self.rel_dfert_ci),
            "bag_instab": self.bag_instab,
            "bag_instab_ci": list(self.bag_instab_ci),
        }


@dataclass(frozen=True)
class ArmNoncanon:
    """Per-arm non-canonicity reading on one cell's (subsampled) held-out split."""

    cell_id: str
    arm: Arm
    boundary: Boundary
    n_molecules: int
    axes: dict[str, AxisReading]
    mean_canon_fert: float
    mean_rand_fert: float
    training_corpus_sha: str
    eval_split_sha: str
    bootstrap_seed: int
    n_resamples: int

    def as_block(self) -> dict[str, object]:
        return {
            "cell_id": self.cell_id,
            "arm": self.arm,
            "boundary": self.boundary,
            "n_molecules": self.n_molecules,
            "axes": {a: self.axes[a].as_dict() for a in AXES if a in self.axes},
            "mean_canon_fert": self.mean_canon_fert,
            "mean_rand_fert": self.mean_rand_fert,
            "training_corpus_sha": self.training_corpus_sha,
            "eval_split_sha": self.eval_split_sha,
            "bootstrap_seed": self.bootstrap_seed,
            "n_resamples": self.n_resamples,
        }


def compute_arm_noncanon(
    per_molecule: Sequence[PerMoleculeNoncanon],
    *,
    cell_id: str,
    arm: Arm,
    boundary: Boundary,
    training_corpus_sha: str,
    eval_split_sha: str,
    seed: int | None = None,
    n_resamples: int = N_BOOTSTRAP_RESAMPLES,
) -> ArmNoncanon:
    """Aggregate per-molecule readings into per-axis means with bootstrap CIs."""
    seed = seed if seed is not None else bootstrap_seed(cell_id)
    rng = np.random.default_rng(seed)

    def _mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else float("nan")

    axes: dict[str, AxisReading] = {}
    for axis in AXES:
        dfert = [pm.axis_dfert[axis] for pm in per_molecule if axis in pm.axis_dfert]
        bag = [pm.axis_bag[axis] for pm in per_molecule if axis in pm.axis_bag]
        if not bag:
            continue
        axes[axis] = AxisReading(
            n=len(bag),
            rel_dfert=_mean(dfert),
            rel_dfert_ci=_bootstrap_mean_ci(
                np.asarray(dfert, dtype=np.float64),
                rng,
                n_resamples=n_resamples,
                level=CI_LEVEL,
            ),
            bag_instab=_mean(bag),
            bag_instab_ci=_bootstrap_mean_ci(
                np.asarray(bag, dtype=np.float64),
                rng,
                n_resamples=n_resamples,
                level=CI_LEVEL,
            ),
        )

    return ArmNoncanon(
        cell_id=cell_id,
        arm=arm,
        boundary=boundary,
        n_molecules=len(per_molecule),
        axes=axes,
        mean_canon_fert=_mean([float(pm.canon_fert) for pm in per_molecule]),
        mean_rand_fert=_mean([pm.rand_fert_mean for pm in per_molecule]),
        training_corpus_sha=training_corpus_sha,
        eval_split_sha=eval_split_sha,
        bootstrap_seed=seed,
        n_resamples=n_resamples,
    )


def _ratio(num: float, den: float) -> float:
    return num / den if den else float("nan")


def _gap(a: float, b: float) -> float:
    """BPE - UL gap, propagating nan if either side is undefined."""
    if a != a or b != b:  # nan
        return float("nan")
    return a - b


@dataclass(frozen=True)
class MatchedPairNoncanon:
    """Non-canonicity for one matched ``(corpus, V, boundary)`` coordinate.

    ``delta_bag_instab[axis] = BPE - UL`` is positive when BPE's piece bag is the
    less stable of the two on that axis. ``gap_canon`` / ``gap_rand`` are
    ``mean(UL tokens) / mean(BPE tokens)`` on the canonical strings and on the
    randomized orbit; their closeness is the gap-survival test.
    """

    pair_key: str
    tier: str
    corpus: str
    vocab_size: int
    boundary: Boundary
    extras_kind: str | None
    extras_label: str | None
    bpe: ArmNoncanon
    unigram: ArmNoncanon
    delta_bag_instab: dict[str, float]
    delta_rel_dfert: dict[str, float]
    gap_canon: float
    gap_rand: float

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
            "bpe": self.bpe.as_block(),
            "unigram": self.unigram.as_block(),
            "delta_bag_instab": dict(self.delta_bag_instab),
            "delta_rel_dfert": dict(self.delta_rel_dfert),
            "gap_canon": self.gap_canon,
            "gap_rand": self.gap_rand,
            "missing_arm": None,
            "unpaired_reason": None,
        }


@dataclass(frozen=True)
class UnpairedNoncanon:
    """Coordinate with one trained arm; the cross-arm gap is undefined."""

    pair_key: str
    tier: str
    corpus: str
    vocab_size: int
    boundary: Boundary
    extras_kind: str | None
    extras_label: str | None
    present_arm: Arm
    missing_arm: Arm
    unpaired_reason: UnpairedReason
    present: ArmNoncanon

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
            "missing_arm": self.missing_arm,
            "unpaired_reason": self.unpaired_reason,
            self.present_arm: self.present.as_block(),
        }


def _axis_value(arm: ArmNoncanon, axis: str, attr: str) -> float:
    reading = arm.axes.get(axis)
    return getattr(reading, attr) if reading is not None else float("nan")


def compute_matched_pair_noncanon(
    bpe: ArmNoncanon,
    unigram: ArmNoncanon,
    *,
    pair_key: str,
    tier: str,
    corpus: str,
    vocab_size: int,
    boundary: Boundary,
    extras_kind: str | None = None,
    extras_label: str | None = None,
) -> MatchedPairNoncanon:
    """Join two :class:`ArmNoncanon` arms into a matched-pair record.

    Raises:
        ValueError: the arms are not one BPE + one Unigram, or disagree on the
            pair boundary.
    """
    if bpe.arm != "bpe":
        raise ValueError(f"first arg must be the BPE arm, got {bpe.arm!r}")
    if unigram.arm != "unigram":
        raise ValueError(f"second arg must be the Unigram arm, got {unigram.arm!r}")
    if bpe.boundary != unigram.boundary or bpe.boundary != boundary:
        raise ValueError("arm boundaries must match the pair boundary")
    delta_bag = {
        a: _gap(
            _axis_value(bpe, a, "bag_instab"), _axis_value(unigram, a, "bag_instab")
        )
        for a in AXES
    }
    delta_df = {
        a: _gap(_axis_value(bpe, a, "rel_dfert"), _axis_value(unigram, a, "rel_dfert"))
        for a in AXES
    }
    return MatchedPairNoncanon(
        pair_key=pair_key,
        tier=tier,
        corpus=corpus,
        vocab_size=vocab_size,
        boundary=boundary,
        extras_kind=extras_kind,
        extras_label=extras_label,
        bpe=bpe,
        unigram=unigram,
        delta_bag_instab=delta_bag,
        delta_rel_dfert=delta_df,
        gap_canon=_ratio(unigram.mean_canon_fert, bpe.mean_canon_fert),
        gap_rand=_ratio(unigram.mean_rand_fert, bpe.mean_rand_fert),
    )


def compute_unpaired_noncanon(
    present: ArmNoncanon,
    *,
    pair_key: str,
    tier: str,
    corpus: str,
    vocab_size: int,
    boundary: Boundary,
    extras_kind: str | None,
    extras_label: str | None,
    present_arm: Arm,
    missing_arm: Arm,
    unpaired_reason: UnpairedReason,
) -> UnpairedNoncanon:
    """Wrap a single present arm's reading for a structurally single-arm coord."""
    if missing_arm == present_arm:
        raise ValueError(
            f"missing_arm={missing_arm!r} cannot equal present_arm {present_arm!r}"
        )
    return UnpairedNoncanon(
        pair_key=pair_key,
        tier=tier,
        corpus=corpus,
        vocab_size=vocab_size,
        boundary=boundary,
        extras_kind=extras_kind,
        extras_label=extras_label,
        present_arm=present_arm,
        missing_arm=missing_arm,
        unpaired_reason=unpaired_reason,
        present=present,
    )


__all__ = [
    "AXES",
    "Arm",
    "ArmNoncanon",
    "AxisReading",
    "Boundary",
    "MatchedPairNoncanon",
    "PerMoleculeNoncanon",
    "UnpairedNoncanon",
    "bootstrap_seed",
    "compute_arm_noncanon",
    "compute_matched_pair_noncanon",
    "compute_unpaired_noncanon",
]
