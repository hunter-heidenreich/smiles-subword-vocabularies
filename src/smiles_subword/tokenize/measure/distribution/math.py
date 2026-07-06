"""Token-frequency imbalance D (distribution contrast).

Per cell, on the held-out split, three per-arm readings of the token-frequency
distribution:

* **token-imbalance** ``D = ½ Σ|p_i − 1/|V||`` (Gowda 2020) — the headline
  statistic, reported as the cross-arm ``|ΔD|`` contrast;
* **normalized Shannon entropy** ``η`` and **Rényi efficiency** at ``α = 2.5``
  (Zouhar 2023) — within-family diagnostics, no cross-arm role.

All normalize by ``|V| = v_effective``, the **nominal target** vocabulary (165
base glyphs plus ``V − 165`` learned subwords, identical across a matched pair's
arms; special tokens occupy no ``V`` slot). Every base glyph is installed
whether or not a corpus exercises it, so ``η`` and Rényi are deflated by dead
glyphs and not cross-corpus comparable — hence the **live-token count** (distinct
tokens with nonzero held-out frequency) is recorded alongside ``|V|``. Holding
``|V|`` identical across arms makes a dead glyph contribute ``1/|V|`` to both
``D`` values and cancel exactly in ``ΔD``.

UQ: 95% percentile bootstrap CIs over 1000 molecule-resamples, seeded from
``cell_id``; the CIs play no role in the contrast (read off the ``|ΔD|`` point
estimate).

Pure computation: a sparse per-molecule token-count structure in, per-arm and
matched-pair records out. The held-out encode pass lives in :mod:`.runner`; the
three formulas in :mod:`smiles_subword.tokenize.intrinsics`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from smiles_subword.tokenize.intrinsics import (
    normalized_entropy,
    renyi_efficiency,
    token_imbalance,
)
from smiles_subword.tokenize.measure._bootstrap import (
    CI_LEVEL,
    N_BOOTSTRAP_RESAMPLES,
    bootstrap_seed,
    percentile_ci,
)

Boundary = Literal["nmb", "mb"]
Arm = Literal["bpe", "unigram"]
UnpairedReason = Literal["conditional_negative_branch", "extras_single_arm_knob"]

DELTA_D_NOISE_FLOOR = 0.002
"""The measured corpus-draw noise floor: the |ΔD| spread across our three
subsample redraws is ~0.0017, so a gap above ~0.002 is resolved from measurement
noise. The one measured threshold the paper applies — it asks only whether the
gap exceeds noise. Canonical home for this value."""


@dataclass(frozen=True)
class DistributionMoleculeData:
    """Per-arm held-out token-frequency data driving the Distribution bootstrap.

    Sparse coordinate form, one entry per (molecule, token id) with the
    per-molecule emission ``count``; molecules emitting nothing still advance the
    molecule index so the resample-unit count stays exact. ``local_token_ids``
    maps a local subword id back to its global token id. ``v_effective`` is the
    fixed normalizer ``|V|`` (nominal target vocabulary), so it does not move
    under resampling.
    """

    n_molecules: int
    mol_idx: np.ndarray
    sub_local: np.ndarray
    count: np.ndarray
    local_token_ids: tuple[int, ...]
    v_effective: int

    @property
    def n_live(self) -> int:
        return len(self.local_token_ids)

    @property
    def total_tokens(self) -> int:
        return int(self.count.sum()) if self.count.size else 0


@dataclass(frozen=True)
class ArmDistribution:
    """Distribution readings for one arm (BPE or Unigram-LM) on one cell.

    ``d``/``eta``/``renyi`` are point estimates; ``*_ci`` are 95% percentile
    bootstrap intervals over :data:`N_BOOTSTRAP_RESAMPLES` molecule-resamples.
    ``live_token_count`` (distinct tokens with nonzero held-out frequency) is
    reported alongside ``vocab_size``/``v_effective`` for the dead-glyph caveat.
    """

    cell_id: str
    arm: Arm
    boundary: Boundary
    n_molecules: int
    total_tokens: int
    vocab_size: int
    v_effective: int
    live_token_count: int
    d: float
    d_ci: tuple[float, float]
    eta: float
    eta_ci: tuple[float, float]
    renyi: float
    renyi_ci: tuple[float, float]
    training_corpus_sha: str
    eval_split_sha: str
    bootstrap_seed: int
    n_resamples: int

    def as_dict(self) -> dict[str, object]:
        return {
            "cell_id": self.cell_id,
            "arm": self.arm,
            "boundary": self.boundary,
            "n_molecules": self.n_molecules,
            "total_tokens": self.total_tokens,
            "vocab_size": self.vocab_size,
            "v_effective": self.v_effective,
            "live_token_count": self.live_token_count,
            "d": self.d,
            "d_ci": list(self.d_ci),
            "eta": self.eta,
            "eta_ci": list(self.eta_ci),
            "renyi": self.renyi,
            "renyi_ci": list(self.renyi_ci),
            "training_corpus_sha": self.training_corpus_sha,
            "eval_split_sha": self.eval_split_sha,
            "bootstrap_seed": self.bootstrap_seed,
            "n_resamples": self.n_resamples,
        }


@dataclass(frozen=True)
class MatchedPairDistribution:
    """Distribution for one matched-arm ``(corpus, V, boundary)`` coordinate.

    ``delta_d = bpe.d − unigram.d`` and its absolute value are the headline
    reads; ``delta_d_exceeds_threshold`` is the above-noise-floor bit;
    ``delta_eta``/``delta_renyi`` are diagnostic. Dead glyphs cancel in ``ΔD``
    only when both arms agree on ``v_effective``; ``v_effective_consistent``
    records that (it does not raise).
    """

    pair_key: str
    tier: str
    corpus: str
    vocab_size: int
    boundary: Boundary
    extras_kind: str | None
    extras_label: str | None
    bpe: ArmDistribution
    unigram: ArmDistribution
    delta_d: float
    abs_delta_d: float
    delta_d_exceeds_threshold: bool
    delta_eta: float
    delta_renyi: float
    v_effective_consistent: bool
    v_effective_delta: int

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
            "delta_d": self.delta_d,
            "abs_delta_d": self.abs_delta_d,
            "delta_d_exceeds_threshold": self.delta_d_exceeds_threshold,
            "delta_eta": self.delta_eta,
            "delta_renyi": self.delta_renyi,
            "v_effective_consistent": self.v_effective_consistent,
            "v_effective_delta": self.v_effective_delta,
            "missing_arm": None,
            "unpaired_reason": None,
        }


@dataclass(frozen=True)
class UnpairedDistribution:
    """Distribution for one single-arm coordinate; cross-arm ΔD undefined."""

    pair_key: str
    tier: str
    corpus: str
    vocab_size: int
    boundary: Boundary
    extras_kind: str | None
    extras_label: str | None
    present_arm: ArmDistribution
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
            "delta_d": None,
            "abs_delta_d": None,
            "delta_d_exceeds_threshold": None,
            "delta_eta": None,
            "delta_renyi": None,
            "v_effective_consistent": None,
            "v_effective_delta": None,
            "missing_arm": self.missing_arm,
            "unpaired_reason": self.unpaired_reason,
        }


def _aggregate_counts(data: DistributionMoleculeData) -> np.ndarray:
    """Aggregate held-out emitted counts per local token id for one arm."""
    if data.count.size == 0:
        return np.zeros(0, dtype=np.float64)
    return np.bincount(data.sub_local, weights=data.count, minlength=data.n_live)


def _bootstrap_intrinsics(
    data: DistributionMoleculeData,
    *,
    seed: int,
    n_resamples: int = N_BOOTSTRAP_RESAMPLES,
    level: float = CI_LEVEL,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Percentile bootstrap CIs for ``(D, η, Rényi)`` over molecule resamples.

    Each resample draws ``n_molecules`` molecules with replacement, re-aggregates
    the per-token counts via a weighted ``bincount`` (``v_effective`` stays
    fixed), and recomputes the three formulas. Returns ``(nan, nan)`` triples
    when the held-out split is empty.
    """
    nan = (float("nan"), float("nan"))
    n_mol = data.n_molecules
    if n_mol == 0 or data.count.size == 0:
        return nan, nan, nan

    n_live = data.n_live
    v = data.v_effective
    rng = np.random.default_rng(seed)
    d_samples: list[float] = []
    eta_samples: list[float] = []
    renyi_samples: list[float] = []
    for _ in range(n_resamples):
        picks = rng.integers(0, n_mol, n_mol)
        mult = np.bincount(picks, minlength=n_mol).astype(np.float64)
        agg = np.bincount(
            data.sub_local, weights=data.count * mult[data.mol_idx], minlength=n_live
        )
        total = float(agg.sum())
        if total <= 0:
            continue
        d_samples.append(token_imbalance(agg, total, v))
        eta_samples.append(normalized_entropy(agg, total, v))
        renyi_samples.append(renyi_efficiency(agg, total, v))
    return (
        percentile_ci(d_samples, level=level),
        percentile_ci(eta_samples, level=level),
        percentile_ci(renyi_samples, level=level),
    )


def compute_arm_distribution(
    data: DistributionMoleculeData,
    *,
    cell_id: str,
    arm: Arm,
    boundary: Boundary,
    vocab_size: int,
    training_corpus_sha: str,
    eval_split_sha: str,
    seed: int | None = None,
    n_resamples: int = N_BOOTSTRAP_RESAMPLES,
) -> ArmDistribution:
    """Aggregate the sparse held-out data into an :class:`ArmDistribution` record.

    Point estimates use the full distribution; CIs resample molecules with
    replacement. ``vocab_size`` is the realized ``adapter.vocab_size``, reported
    alongside the normalizer ``data.v_effective`` (nominal target ``|V|``).
    """
    v = data.v_effective
    agg = _aggregate_counts(data)
    total = float(agg.sum())
    d = token_imbalance(agg, total, v)
    eta = normalized_entropy(agg, total, v)
    renyi = renyi_efficiency(agg, total, v)
    live_token_count = int((agg > 0).sum())

    seed = seed if seed is not None else bootstrap_seed(cell_id)
    d_ci, eta_ci, renyi_ci = _bootstrap_intrinsics(
        data, seed=seed, n_resamples=n_resamples
    )
    return ArmDistribution(
        cell_id=cell_id,
        arm=arm,
        boundary=boundary,
        n_molecules=data.n_molecules,
        total_tokens=int(total),
        vocab_size=vocab_size,
        v_effective=v,
        live_token_count=live_token_count,
        d=d,
        d_ci=d_ci,
        eta=eta,
        eta_ci=eta_ci,
        renyi=renyi,
        renyi_ci=renyi_ci,
        training_corpus_sha=training_corpus_sha,
        eval_split_sha=eval_split_sha,
        bootstrap_seed=seed,
        n_resamples=n_resamples,
    )


def compute_matched_pair_distribution(
    bpe: ArmDistribution,
    unigram: ArmDistribution,
    *,
    pair_key: str,
    tier: str,
    corpus: str,
    vocab_size: int,
    boundary: Boundary,
    extras_kind: str | None = None,
    extras_label: str | None = None,
) -> MatchedPairDistribution:
    """Join two :class:`ArmDistribution` arms, computing the cross-arm ΔD reads.

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
    delta_d = bpe.d - unigram.d
    abs_delta_d = abs(delta_d)
    v_effective_delta = bpe.v_effective - unigram.v_effective
    return MatchedPairDistribution(
        pair_key=pair_key,
        tier=tier,
        corpus=corpus,
        vocab_size=vocab_size,
        boundary=boundary,
        extras_kind=extras_kind,
        extras_label=extras_label,
        bpe=bpe,
        unigram=unigram,
        delta_d=delta_d,
        abs_delta_d=abs_delta_d,
        delta_d_exceeds_threshold=abs_delta_d > DELTA_D_NOISE_FLOOR,
        delta_eta=bpe.eta - unigram.eta,
        delta_renyi=bpe.renyi - unigram.renyi,
        v_effective_consistent=v_effective_delta == 0,
        v_effective_delta=v_effective_delta,
    )


def compute_unpaired_distribution(
    arm_record: ArmDistribution,
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
) -> UnpairedDistribution:
    """Wrap one present arm as an :class:`UnpairedDistribution` record."""
    if missing_arm == arm_record.arm:
        raise ValueError(
            f"missing_arm={missing_arm!r} cannot equal the present arm "
            f"{arm_record.arm!r}"
        )
    if arm_record.boundary != boundary:
        raise ValueError(
            f"arm boundary {arm_record.boundary!r} disagrees with pair {boundary!r}"
        )
    return UnpairedDistribution(
        pair_key=pair_key,
        tier=tier,
        corpus=corpus,
        vocab_size=vocab_size,
        boundary=boundary,
        extras_kind=extras_kind,
        extras_label=extras_label,
        present_arm=arm_record,
        missing_arm=missing_arm,
        unpaired_reason=unpaired_reason,
    )


__all__ = [
    "CI_LEVEL",
    "DELTA_D_NOISE_FLOOR",
    "N_BOOTSTRAP_RESAMPLES",
    "Arm",
    "ArmDistribution",
    "Boundary",
    "DistributionMoleculeData",
    "MatchedPairDistribution",
    "UnpairedDistribution",
    "bootstrap_seed",
    "compute_arm_distribution",
    "compute_matched_pair_distribution",
    "compute_unpaired_distribution",
]
