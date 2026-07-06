"""Fertility on held-out (granularity contrast) — pure computation.

Per-molecule ``(n_tokens, n_glyphs)`` counts in, per-arm and matched-pair
records out; the glyph-count map and held-out encode pass live in
:mod:`.runner`. Per-arm statistics on the corpus's deterministic held-out
split:

* **fertility** — mean tokens per molecule (``Σ tokens / n_molecules``);
* **glyphs per token** — compression with the glyph as base unit,
  ``Σ glyphs / Σ tokens``, well-defined for both arms;
* **across-molecule variance** of tokens per molecule (point estimate).

95% percentile bootstrap CIs over 1000 molecule-resamples for fertility and
glyphs-per-token (variance is a point estimate); the seed is derived from
``cell_id`` for byte-identical re-runs. CIs play no role in the contrast — it
is read off the relative ``|Δfertility|`` on point estimates.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import pvariance
from typing import TYPE_CHECKING, Literal

from smiles_subword.tokenize.measure._bootstrap import (
    CI_LEVEL,
    N_BOOTSTRAP_RESAMPLES,
    bootstrap_ratio_ci,
    bootstrap_seed,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


Boundary = Literal["nmb", "mb"]
Arm = Literal["bpe", "unigram"]
UnpairedReason = Literal["conditional_negative_branch", "extras_single_arm_knob"]


@dataclass(frozen=True)
class PerMoleculeFertility:
    """Per-molecule token and glyph counts for one held-out molecule.

    ``n_glyphs`` is the molecule's base-glyph count under the cell's
    boundary — model-independent given the boundary, so it is identical
    across the two arms of a matched pair.
    """

    n_tokens: int
    n_glyphs: int


@dataclass(frozen=True)
class ArmFertility:
    """Fertility readings for one arm (BPE or Unigram-LM) on one cell.

    ``fertility_mean`` is ``Σ n_tokens / n_molecules``;
    ``glyphs_per_token_mean`` is ``Σ n_glyphs / Σ n_tokens``;
    ``tokens_per_molecule_variance`` is the population variance of the
    per-molecule token counts. Bootstrap CIs are 95% percentile intervals
    over :data:`N_BOOTSTRAP_RESAMPLES` molecule-resamples; the seed is
    recorded for reproducibility.
    """

    cell_id: str
    arm: Arm
    boundary: Boundary
    n_molecules: int
    total_tokens: int
    total_glyphs: int
    fertility_mean: float
    fertility_ci: tuple[float, float]
    glyphs_per_token_mean: float
    glyphs_per_token_ci: tuple[float, float]
    tokens_per_molecule_variance: float
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
            "total_glyphs": self.total_glyphs,
            "fertility_mean": self.fertility_mean,
            "fertility_ci": list(self.fertility_ci),
            "glyphs_per_token_mean": self.glyphs_per_token_mean,
            "glyphs_per_token_ci": list(self.glyphs_per_token_ci),
            "tokens_per_molecule_variance": self.tokens_per_molecule_variance,
            "training_corpus_sha": self.training_corpus_sha,
            "eval_split_sha": self.eval_split_sha,
            "bootstrap_seed": self.bootstrap_seed,
            "n_resamples": self.n_resamples,
        }


@dataclass(frozen=True)
class MatchedPairFertility:
    """Fertility for one matched-arm ``(corpus, V, boundary)`` coordinate.

    ``total_glyphs`` is model-independent at a fixed boundary, so the two
    arms must agree; ``total_glyphs_consistent`` records whether they do
    (``total_glyphs_delta = bpe − unigram`` aids diagnosis if they don't).
    ``delta_fertility_relative`` is ``|Δ| / f̄`` with ``f̄`` the two-arm
    mean — the headline statistic, recorded here as a convenience read.
    """

    pair_key: str
    tier: str
    corpus: str
    vocab_size: int
    boundary: Boundary
    extras_kind: str | None
    extras_label: str | None
    bpe: ArmFertility
    unigram: ArmFertility
    delta_fertility: float
    delta_fertility_relative: float
    delta_glyphs_per_token: float
    total_glyphs_consistent: bool
    total_glyphs_delta: int

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
            "delta_fertility": self.delta_fertility,
            "delta_fertility_relative": self.delta_fertility_relative,
            "delta_glyphs_per_token": self.delta_glyphs_per_token,
            "total_glyphs_consistent": self.total_glyphs_consistent,
            "total_glyphs_delta": self.total_glyphs_delta,
            "missing_arm": None,
            "unpaired_reason": None,
        }


@dataclass(frozen=True)
class UnpairedFertility:
    """Fertility for one structurally single-arm coordinate; cross-arm Δ undefined."""

    pair_key: str
    tier: str
    corpus: str
    vocab_size: int
    boundary: Boundary
    extras_kind: str | None
    extras_label: str | None
    present_arm: ArmFertility
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
            "delta_fertility": None,
            "delta_fertility_relative": None,
            "delta_glyphs_per_token": None,
            "total_glyphs_consistent": None,
            "total_glyphs_delta": None,
            "missing_arm": self.missing_arm,
            "unpaired_reason": self.unpaired_reason,
        }


def compute_arm_fertility(
    per_molecule: Sequence[PerMoleculeFertility],
    *,
    cell_id: str,
    arm: Arm,
    boundary: Boundary,
    training_corpus_sha: str,
    eval_split_sha: str,
    seed: int | None = None,
    n_resamples: int = N_BOOTSTRAP_RESAMPLES,
) -> ArmFertility:
    """Aggregate per-molecule counts into an :class:`ArmFertility` record.

    ``per_molecule`` is one :class:`PerMoleculeFertility` per held-out molecule (in
    any order). Bootstrap CIs are computed over molecule resamples with
    replacement; fertility uses unit denominators (mean), glyphs-per-token
    the per-molecule token counts (ratio).
    """
    n_molecules = len(per_molecule)
    total_tokens = sum(pm.n_tokens for pm in per_molecule)
    total_glyphs = sum(pm.n_glyphs for pm in per_molecule)
    tok_counts = [pm.n_tokens for pm in per_molecule]
    glyph_counts = [pm.n_glyphs for pm in per_molecule]

    fertility_mean = (total_tokens / n_molecules) if n_molecules else float("nan")
    glyphs_per_token_mean = (
        (total_glyphs / total_tokens) if total_tokens else float("nan")
    )
    variance = float(pvariance(tok_counts)) if n_molecules else float("nan")

    seed = seed if seed is not None else bootstrap_seed(cell_id)
    fertility_ci = bootstrap_ratio_ci(
        tok_counts, [1] * n_molecules, seed=seed, n_resamples=n_resamples
    )
    glyphs_per_token_ci = bootstrap_ratio_ci(
        glyph_counts, tok_counts, seed=seed + 1, n_resamples=n_resamples
    )
    return ArmFertility(
        cell_id=cell_id,
        arm=arm,
        boundary=boundary,
        n_molecules=n_molecules,
        total_tokens=total_tokens,
        total_glyphs=total_glyphs,
        fertility_mean=fertility_mean,
        fertility_ci=fertility_ci,
        glyphs_per_token_mean=glyphs_per_token_mean,
        glyphs_per_token_ci=glyphs_per_token_ci,
        tokens_per_molecule_variance=variance,
        training_corpus_sha=training_corpus_sha,
        eval_split_sha=eval_split_sha,
        bootstrap_seed=seed,
        n_resamples=n_resamples,
    )


def relative_fertility_gap(f_bpe: float, f_unigram: float) -> float:
    """Relative cross-arm fertility gap ``|f_bpe - f_unigram| / mean``.

    ``nan`` when the two-arm mean is zero. The canonical home for the contrast
    formula the supplementary sensitivity and OOD analyses also report.
    """
    mean = 0.5 * (f_bpe + f_unigram)
    return abs(f_bpe - f_unigram) / mean if mean else float("nan")


def compute_matched_pair_fertility(
    bpe: ArmFertility,
    unigram: ArmFertility,
    *,
    pair_key: str,
    tier: str,
    corpus: str,
    vocab_size: int,
    boundary: Boundary,
    extras_kind: str | None = None,
    extras_label: str | None = None,
) -> MatchedPairFertility:
    """Join two :class:`ArmFertility` arms into a matched-pair record.

    ``delta_fertility = bpe.fertility_mean − unigram.fertility_mean``;
    ``delta_fertility_relative = |Δ| / f̄`` with ``f̄`` the two-arm mean
    (headline statistic). ``total_glyphs`` must agree across arms (it is
    model-independent at a fixed boundary); ``total_glyphs_consistent``
    flags any disagreement rather than raising, so a sweep is not aborted.

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
    delta_fertility = bpe.fertility_mean - unigram.fertility_mean
    delta_fertility_relative = relative_fertility_gap(
        bpe.fertility_mean, unigram.fertility_mean
    )
    delta_glyphs_per_token = bpe.glyphs_per_token_mean - unigram.glyphs_per_token_mean
    total_glyphs_delta = bpe.total_glyphs - unigram.total_glyphs
    return MatchedPairFertility(
        pair_key=pair_key,
        tier=tier,
        corpus=corpus,
        vocab_size=vocab_size,
        boundary=boundary,
        extras_kind=extras_kind,
        extras_label=extras_label,
        bpe=bpe,
        unigram=unigram,
        delta_fertility=delta_fertility,
        delta_fertility_relative=delta_fertility_relative,
        delta_glyphs_per_token=delta_glyphs_per_token,
        total_glyphs_consistent=total_glyphs_delta == 0,
        total_glyphs_delta=total_glyphs_delta,
    )


def compute_unpaired_fertility(
    arm_record: ArmFertility,
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
) -> UnpairedFertility:
    """Wrap one present arm as an :class:`UnpairedFertility` record."""
    if missing_arm == arm_record.arm:
        raise ValueError(
            f"missing_arm={missing_arm!r} cannot equal the present arm "
            f"{arm_record.arm!r}"
        )
    if arm_record.boundary != boundary:
        raise ValueError(
            f"arm boundary {arm_record.boundary!r} disagrees with pair {boundary!r}"
        )
    return UnpairedFertility(
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
    "N_BOOTSTRAP_RESAMPLES",
    "ArmFertility",
    "Boundary",
    "MatchedPairFertility",
    "PerMoleculeFertility",
    "UnpairedFertility",
    "bootstrap_seed",
    "compute_arm_fertility",
    "compute_matched_pair_fertility",
    "compute_unpaired_fertility",
    "relative_fertility_gap",
]
