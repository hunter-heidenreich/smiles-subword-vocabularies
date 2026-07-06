"""Functional-bond locality — chemical alignment of the realized segmentation.

When a chemically salient bond occurs in a held-out molecule, does the arm keep
it inside a single token or cut through it? The pure aggregation — per-molecule
``(n_bonds, n_local)`` counts in, per-arm and matched-pair records out; the
held-out dual-encode and RDKit bond classification live in :mod:`.runner`.

The salient bonds are the **multiply-bonded heteroatoms** — a non-carbon atom
joined by a double or triple bond: carbonyl ``=O``, nitrile ``#N``, imine
``=N``, sulfonyl/phosphoryl/nitro ``=O``, thiocarbonyl ``=S``. Cores of the
canonical functional groups, read straight off the molecular graph, so the
metric needs no curated per-group surface forms.

A bond is *local* for an arm iff no token boundary separates the heteroatom's
glyph from the adjacent ``=`` / ``#`` bond glyph. Locality of the
heteroatom-plus-bond (not the whole atom span) isolates the chemical question:
a branch-written carbonyl ``C(=O)`` puts the carbon in the backbone token while
``=O`` is (for BPE) its own unit, so a whole-span requirement would be
dominated by SMILES branch syntax, not the algorithm.

Per arm: overall locality fraction (95% molecule-resampled bootstrap CI) plus a
point breakdown by bond class. ``delta_locality = BPE - UL`` is positive when
BPE keeps functional bonds more local. Like closure, locality is *within-arm* —
a single-arm coordinate still carries a full reading.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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

# The functional bond classes surfaced in the table, in report order. Each is
# ``{partner}{op}{heteroatom}`` read off the graph; the runner emits these
# labels and folds every other multiply-bonded-heteroatom case into "other".
FUNCTIONAL_CLASSES: tuple[str, ...] = (
    "C=O",
    "C#N",
    "C=N",
    "S=O",
    "P=O",
    "N=O",
    "C=S",
    "other",
)


@dataclass(frozen=True)
class PerMoleculeFgLocality:
    """One molecule's functional-bond counts for one arm.

    ``n_bonds`` is the count of multiply-bonded heteroatoms whose bond glyph was
    locatable; ``n_local`` how many the arm kept token-local. ``class_bonds`` /
    ``class_local`` carry the same split per bond class (only classes the
    molecule contains appear).
    """

    n_bonds: int
    n_local: int
    class_bonds: dict[str, int] = field(default_factory=dict)
    class_local: dict[str, int] = field(default_factory=dict)


def _ratio(num: int, den: int) -> float:
    return num / den if den else float("nan")


def _bootstrap_locality_ci(
    n_local: np.ndarray,
    n_bonds: np.ndarray,
    *,
    seed: int,
    n_resamples: int = N_BOOTSTRAP_RESAMPLES,
    level: float = CI_LEVEL,
) -> tuple[float, float]:
    """Percentile bootstrap CI for ``Σn_local / Σn_bonds`` over molecules.

    The resample unit is the molecule (the shared molecule-resample bootstrap
    recipe in ``_bootstrap``); a resample whose total bond
    count is zero is skipped. Returns ``(nan, nan)`` for an empty split.
    Vectorized so the ~10^6-molecule held-out splits stay tractable.
    """
    n_mol = n_local.shape[0]
    if n_mol == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(n_resamples):
        mult = np.bincount(rng.integers(0, n_mol, n_mol), minlength=n_mol).astype(
            np.float64
        )
        bonds = float(np.dot(n_bonds, mult))
        if bonds > 0:
            samples.append(float(np.dot(n_local, mult)) / bonds)
    return percentile_ci(samples, level=level)


@dataclass(frozen=True)
class ArmFgAlignment:
    """Per-arm functional-bond locality reading on one cell's held-out split.

    ``locality`` is ``n_local / n_bonds`` with a 95% bootstrap CI;
    ``class_bonds`` / ``class_local`` give the per-class point breakdown (every
    :data:`FUNCTIONAL_CLASSES` key present, zero where the split has no such
    bond). ``cell_id`` + ``training_corpus_sha`` + ``eval_split_sha`` travel
    with the record for the deposit step's freshness check.
    """

    cell_id: str
    arm: Arm
    boundary: Boundary
    n_molecules: int
    n_bonds: int
    n_local: int
    locality_ci: tuple[float, float]
    class_bonds: dict[str, int]
    class_local: dict[str, int]
    training_corpus_sha: str
    eval_split_sha: str
    bootstrap_seed: int
    n_resamples: int

    @property
    def locality(self) -> float:
        """Overall functional-bond locality; nan when no functional bond fired."""
        return _ratio(self.n_local, self.n_bonds)

    def class_locality(self, label: str) -> float:
        """Locality within one bond class; nan when the class is absent."""
        return _ratio(self.class_local.get(label, 0), self.class_bonds.get(label, 0))

    def as_block(self) -> dict[str, object]:
        """Serialize the per-arm block embedded in the pair JSON."""
        return {
            "cell_id": self.cell_id,
            "arm": self.arm,
            "boundary": self.boundary,
            "n_molecules": self.n_molecules,
            "n_bonds": self.n_bonds,
            "n_local": self.n_local,
            "locality": self.locality,
            "locality_ci": list(self.locality_ci),
            "class_bonds": {k: self.class_bonds.get(k, 0) for k in FUNCTIONAL_CLASSES},
            "class_local": {k: self.class_local.get(k, 0) for k in FUNCTIONAL_CLASSES},
            "class_locality": {k: self.class_locality(k) for k in FUNCTIONAL_CLASSES},
            "training_corpus_sha": self.training_corpus_sha,
            "eval_split_sha": self.eval_split_sha,
            "bootstrap_seed": self.bootstrap_seed,
            "n_resamples": self.n_resamples,
        }


def compute_arm_fg_alignment(
    per_molecule: Sequence[PerMoleculeFgLocality],
    *,
    cell_id: str,
    arm: Arm,
    boundary: Boundary,
    training_corpus_sha: str,
    eval_split_sha: str,
    seed: int | None = None,
    n_resamples: int = N_BOOTSTRAP_RESAMPLES,
) -> ArmFgAlignment:
    """Aggregate per-molecule functional-bond counts into an arm record.

    The bootstrap seed is derived from ``cell_id`` (the shared per-cell recipe)
    unless given, so re-runs reproduce byte-identical CIs.
    """
    n_bonds = sum(pm.n_bonds for pm in per_molecule)
    n_local = sum(pm.n_local for pm in per_molecule)
    class_bonds: dict[str, int] = dict.fromkeys(FUNCTIONAL_CLASSES, 0)
    class_local: dict[str, int] = dict.fromkeys(FUNCTIONAL_CLASSES, 0)
    for pm in per_molecule:
        for label, count in pm.class_bonds.items():
            class_bonds[label] = class_bonds.get(label, 0) + count
        for label, count in pm.class_local.items():
            class_local[label] = class_local.get(label, 0) + count

    seed = seed if seed is not None else bootstrap_seed(cell_id)
    ci = _bootstrap_locality_ci(
        np.asarray([pm.n_local for pm in per_molecule], dtype=np.float64),
        np.asarray([pm.n_bonds for pm in per_molecule], dtype=np.float64),
        seed=seed,
        n_resamples=n_resamples,
    )
    return ArmFgAlignment(
        cell_id=cell_id,
        arm=arm,
        boundary=boundary,
        n_molecules=len(per_molecule),
        n_bonds=n_bonds,
        n_local=n_local,
        locality_ci=ci,
        class_bonds=class_bonds,
        class_local=class_local,
        training_corpus_sha=training_corpus_sha,
        eval_split_sha=eval_split_sha,
        bootstrap_seed=seed,
        n_resamples=n_resamples,
    )


def _gap(a: float, b: float) -> float:
    """BPE - UL gap, propagating nan if either side is undefined."""
    if a != a or b != b:  # nan
        return float("nan")
    return a - b


@dataclass(frozen=True)
class MatchedPairFgAlignment:
    """Functional-bond locality for one matched ``(corpus, V, boundary)`` coord.

    ``delta_locality = BPE - UL`` is positive when BPE keeps functional bonds
    more token-local; ``delta_locality_by_class`` is the same per bond class.
    """

    pair_key: str
    tier: str
    corpus: str
    vocab_size: int
    boundary: Boundary
    extras_kind: str | None
    extras_label: str | None
    bpe: ArmFgAlignment
    unigram: ArmFgAlignment
    delta_locality: float
    delta_locality_by_class: dict[str, float]

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
            "delta_locality": self.delta_locality,
            "delta_locality_by_class": dict(self.delta_locality_by_class),
            "missing_arm": None,
            "unpaired_reason": None,
        }


@dataclass(frozen=True)
class UnpairedFgAlignment:
    """Coordinate with only one trained arm; locality is within-arm so present.

    Only the cross-arm gap is undefined — the present arm carries a full reading.
    """

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
    present: ArmFgAlignment

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


def compute_matched_pair_fg_alignment(
    bpe: ArmFgAlignment,
    unigram: ArmFgAlignment,
    *,
    pair_key: str,
    tier: str,
    corpus: str,
    vocab_size: int,
    boundary: Boundary,
    extras_kind: str | None = None,
    extras_label: str | None = None,
) -> MatchedPairFgAlignment:
    """Join two :class:`ArmFgAlignment` arms into a matched-pair record.

    Raises:
        ValueError: the arms are not one BPE + one Unigram, or disagree on the
            pair boundary — both indicate a pairing bug upstream.
    """
    if bpe.arm != "bpe":
        raise ValueError(f"first arg must be the BPE arm, got {bpe.arm!r}")
    if unigram.arm != "unigram":
        raise ValueError(f"second arg must be the Unigram arm, got {unigram.arm!r}")
    if bpe.boundary != unigram.boundary or bpe.boundary != boundary:
        raise ValueError("arm boundaries must match the pair boundary")
    delta_by_class = {
        label: _gap(bpe.class_locality(label), unigram.class_locality(label))
        for label in FUNCTIONAL_CLASSES
    }
    return MatchedPairFgAlignment(
        pair_key=pair_key,
        tier=tier,
        corpus=corpus,
        vocab_size=vocab_size,
        boundary=boundary,
        extras_kind=extras_kind,
        extras_label=extras_label,
        bpe=bpe,
        unigram=unigram,
        delta_locality=_gap(bpe.locality, unigram.locality),
        delta_locality_by_class=delta_by_class,
    )


def compute_unpaired_fg_alignment(
    present: ArmFgAlignment,
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
) -> UnpairedFgAlignment:
    """Wrap a single present arm's locality for a structurally single-arm coord."""
    if missing_arm == present_arm:
        raise ValueError(
            f"missing_arm={missing_arm!r} cannot equal present_arm {present_arm!r}"
        )
    return UnpairedFgAlignment(
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
    "FUNCTIONAL_CLASSES",
    "Arm",
    "ArmFgAlignment",
    "Boundary",
    "MatchedPairFgAlignment",
    "PerMoleculeFgLocality",
    "UnpairedFgAlignment",
    "bootstrap_seed",
    "compute_arm_fg_alignment",
    "compute_matched_pair_fg_alignment",
    "compute_unpaired_fg_alignment",
]
