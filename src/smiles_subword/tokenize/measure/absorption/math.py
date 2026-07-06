"""Whole-pretoken absorption (mechanism diagnostic).

The reference pretoken unit is the Layer-B chunk. Per cell, on the held-out
split, report the fraction of Layer-B chunks whose glyph span equals exactly one
token (*absorbed*). Under MB a token may span chunk boundaries, so also report
the *cross-chunk* fraction — chunks strictly inside a larger token — keeping MB
and NMB comparable; under NMB merges cannot cross a chunk, so it is undefined
(None).

UQ: 95% percentile bootstrap CI, 1000 molecule-resamples; seed derived from
cell_id so re-runs reproduce byte-identical CIs.

Pure computation: per-molecule chunk / token offset arrays in, per-arm and
matched-pair records out. The held-out encode pass lives in :mod:`.runner`.
"""

from __future__ import annotations

from dataclasses import dataclass
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
class PerMoleculeAbsorption:
    """Per-molecule chunk-and-absorption counts.

    ``n_cross_chunk`` is ``None`` under NMB (merges cannot cross a Layer-B
    boundary); an int ≥ 0 under MB.
    """

    n_chunks: int
    n_absorbed: int
    n_cross_chunk: int | None


@dataclass(frozen=True)
class ArmAbsorption:
    """Absorption readings for one arm (BPE or Unigram-LM) on one cell.

    ``absorbed_fraction = Σ n_absorbed / Σ n_chunks`` across the held-out split;
    ``cross_chunk_fraction`` only under MB (else None). CIs are 95% percentile
    intervals over :data:`N_BOOTSTRAP_RESAMPLES` molecule-resamples.
    """

    cell_id: str
    arm: Arm
    boundary: Boundary
    n_molecules: int
    n_chunks: int
    n_absorbed: int
    n_cross_chunk_total: int | None
    absorbed_fraction: float
    absorbed_ci: tuple[float, float]
    cross_chunk_fraction: float | None
    cross_chunk_ci: tuple[float, float] | None
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
            "n_chunks": self.n_chunks,
            "n_absorbed": self.n_absorbed,
            "n_cross_chunk_total": self.n_cross_chunk_total,
            "absorbed_fraction": self.absorbed_fraction,
            "absorbed_ci": list(self.absorbed_ci),
            "cross_chunk_fraction": self.cross_chunk_fraction,
            "cross_chunk_ci": (
                list(self.cross_chunk_ci) if self.cross_chunk_ci is not None else None
            ),
            "training_corpus_sha": self.training_corpus_sha,
            "eval_split_sha": self.eval_split_sha,
            "bootstrap_seed": self.bootstrap_seed,
            "n_resamples": self.n_resamples,
        }


@dataclass(frozen=True)
class MatchedPairAbsorption:
    """Absorption for one matched-arm ``(corpus, V, boundary)`` coordinate."""

    pair_key: str
    tier: str
    corpus: str
    vocab_size: int
    boundary: Boundary
    extras_kind: str | None
    extras_label: str | None
    bpe: ArmAbsorption
    unigram: ArmAbsorption
    delta_absorbed: float
    delta_cross_chunk: float | None

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
            "delta_absorbed": self.delta_absorbed,
            "delta_cross_chunk": self.delta_cross_chunk,
            "missing_arm": None,
            "unpaired_reason": None,
        }


@dataclass(frozen=True)
class UnpairedAbsorption:
    """Absorption for one structurally single-arm coordinate; cross-arm Δ undefined."""

    pair_key: str
    tier: str
    corpus: str
    vocab_size: int
    boundary: Boundary
    extras_kind: str | None
    extras_label: str | None
    present_arm: ArmAbsorption
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
            "delta_absorbed": None,
            "delta_cross_chunk": None,
            "missing_arm": self.missing_arm,
            "unpaired_reason": self.unpaired_reason,
        }


def classify_chunks(
    chunks: Sequence[tuple[str, tuple[int, int]]],
    token_offsets: Sequence[tuple[int, int]],
    *,
    boundary: Boundary,
) -> PerMoleculeAbsorption:
    """Classify Layer-B chunks against token offsets for one molecule.

    A chunk is *absorbed* iff some token has identical offsets ``(start, end)``.
    Under MB it is *cross-chunk* iff some token's span strictly contains it;
    under NMB cross-chunk is undefined (``None``). Empty-span chunks are skipped
    (defensive — the chunker never emits them).
    """
    token_set: frozenset[tuple[int, int]] = frozenset(
        (int(s), int(e)) for s, e in token_offsets
    )
    n_chunks = 0
    n_absorbed = 0
    n_cross_chunk = 0 if boundary == "mb" else None
    for _chunk, span in chunks:
        c_start, c_end = int(span[0]), int(span[1])
        if c_end <= c_start:
            continue
        n_chunks += 1
        if (c_start, c_end) in token_set:
            n_absorbed += 1
            continue
        if boundary == "mb":
            for t_start, t_end in token_offsets:
                t_start_i = int(t_start)
                t_end_i = int(t_end)
                if (
                    t_start_i <= c_start
                    and c_end <= t_end_i
                    and (t_end_i - t_start_i) > (c_end - c_start)
                ):
                    assert n_cross_chunk is not None
                    n_cross_chunk += 1
                    break
    return PerMoleculeAbsorption(
        n_chunks=n_chunks, n_absorbed=n_absorbed, n_cross_chunk=n_cross_chunk
    )


def compute_arm_absorption(
    per_molecule: Sequence[PerMoleculeAbsorption],
    *,
    cell_id: str,
    arm: Arm,
    boundary: Boundary,
    training_corpus_sha: str,
    eval_split_sha: str,
    seed: int | None = None,
    n_resamples: int = N_BOOTSTRAP_RESAMPLES,
) -> ArmAbsorption:
    """Aggregate per-molecule counts into an :class:`ArmAbsorption` record.

    ``per_molecule`` is :func:`classify_chunks` output for every held-out
    molecule (any order); CIs resample molecules with replacement.

    Raises:
        ValueError: an entry's cross-chunk presence does not match ``boundary``
            (NMB ⇒ all None; MB ⇒ all int).
    """
    n_chunks = sum(pm.n_chunks for pm in per_molecule)
    n_absorbed = sum(pm.n_absorbed for pm in per_molecule)
    if boundary == "mb":
        if any(pm.n_cross_chunk is None for pm in per_molecule):
            raise ValueError("MB cells require integer n_cross_chunk per molecule")
        cross_total: int | None = sum(
            pm.n_cross_chunk for pm in per_molecule if pm.n_cross_chunk is not None
        )
    else:
        if any(pm.n_cross_chunk is not None for pm in per_molecule):
            raise ValueError("NMB cells must have n_cross_chunk=None per molecule")
        cross_total = None
    absorbed_fraction = (n_absorbed / n_chunks) if n_chunks > 0 else float("nan")
    cross_fraction: float | None
    if boundary == "mb" and n_chunks > 0:
        assert cross_total is not None
        cross_fraction = cross_total / n_chunks
    elif boundary == "mb":
        cross_fraction = float("nan")
    else:
        cross_fraction = None
    seed = seed if seed is not None else bootstrap_seed(cell_id)
    absorbed_nums = [pm.n_absorbed for pm in per_molecule]
    denoms = [pm.n_chunks for pm in per_molecule]
    absorbed_ci = bootstrap_ratio_ci(
        absorbed_nums, denoms, seed=seed, n_resamples=n_resamples
    )
    cross_ci: tuple[float, float] | None
    if boundary == "mb":
        cross_nums = [int(pm.n_cross_chunk or 0) for pm in per_molecule]
        cross_ci = bootstrap_ratio_ci(
            cross_nums, denoms, seed=seed + 1, n_resamples=n_resamples
        )
    else:
        cross_ci = None
    return ArmAbsorption(
        cell_id=cell_id,
        arm=arm,
        boundary=boundary,
        n_molecules=len(per_molecule),
        n_chunks=n_chunks,
        n_absorbed=n_absorbed,
        n_cross_chunk_total=cross_total,
        absorbed_fraction=absorbed_fraction,
        absorbed_ci=absorbed_ci,
        cross_chunk_fraction=cross_fraction,
        cross_chunk_ci=cross_ci,
        training_corpus_sha=training_corpus_sha,
        eval_split_sha=eval_split_sha,
        bootstrap_seed=seed,
        n_resamples=n_resamples,
    )


def compute_matched_pair_absorption(
    bpe: ArmAbsorption,
    unigram: ArmAbsorption,
    *,
    pair_key: str,
    tier: str,
    corpus: str,
    vocab_size: int,
    boundary: Boundary,
    extras_kind: str | None = None,
    extras_label: str | None = None,
) -> MatchedPairAbsorption:
    """Join two :class:`ArmAbsorption` arms into a matched-pair record.

    ``delta_absorbed = bpe.absorbed_fraction − unigram.absorbed_fraction``
    (positive ⇒ BPE absorbs the higher fraction); under MB
    ``delta_cross_chunk`` likewise, else ``None``.

    Raises:
        ValueError: arm-tag mismatch or boundary disagreement between arms.
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
    delta_absorbed = bpe.absorbed_fraction - unigram.absorbed_fraction
    delta_cross: float | None
    if boundary == "mb":
        assert bpe.cross_chunk_fraction is not None
        assert unigram.cross_chunk_fraction is not None
        delta_cross = bpe.cross_chunk_fraction - unigram.cross_chunk_fraction
    else:
        delta_cross = None
    return MatchedPairAbsorption(
        pair_key=pair_key,
        tier=tier,
        corpus=corpus,
        vocab_size=vocab_size,
        boundary=boundary,
        extras_kind=extras_kind,
        extras_label=extras_label,
        bpe=bpe,
        unigram=unigram,
        delta_absorbed=delta_absorbed,
        delta_cross_chunk=delta_cross,
    )


def compute_unpaired_absorption(
    arm_record: ArmAbsorption,
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
) -> UnpairedAbsorption:
    """Wrap one present arm as an :class:`UnpairedAbsorption` record."""
    if missing_arm == arm_record.arm:
        raise ValueError(
            f"missing_arm={missing_arm!r} cannot equal the present arm "
            f"{arm_record.arm!r}"
        )
    if arm_record.boundary != boundary:
        raise ValueError(
            f"arm boundary {arm_record.boundary!r} disagrees with pair {boundary!r}"
        )
    return UnpairedAbsorption(
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
    "ArmAbsorption",
    "Boundary",
    "MatchedPairAbsorption",
    "PerMoleculeAbsorption",
    "UnpairedAbsorption",
    "bootstrap_seed",
    "classify_chunks",
    "compute_arm_absorption",
    "compute_matched_pair_absorption",
    "compute_unpaired_absorption",
]
