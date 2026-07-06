"""Unigram segmentation entropy (mechanism diagnostic).

The fitted Unigram-LM induces, per held-out molecule, a probability distribution
over that molecule's segmentations into vocabulary pieces (within Layer-B chunk
boundaries, identically to the encoder). Segmentation is the Shannon entropy of
that distribution, computed exactly by a forward dynamic program over the
per-molecule segmentation lattice — no sampling. Per Unigram cell we report the
held-out mean both raw (nats/molecule) and normalized per glyph, with 95%
bootstrap CIs.

Unigram-only by construction: stock BPE produces a unique segmentation per
string (zero entropy). BPE arms emit a ``verified_by_construction=True`` zero
record so the matched-pair schema stays symmetric — the inverted mirror of
Scaffold (BPE-only scaffolding, Unigram zero by construction). It is a
one-directional corroborating signal bounding the representativeness of the
Viterbi-argmax single-segmentation measurements, not a headline statistic.

The distribution factorizes over Layer-B chunks (``split_structure=True`` ⇒
tokens never span chunks), so a molecule's entropy is the sum of its chunks'
entropies and its per-glyph value is total entropy / total glyphs.

This module is the pure math: a chunk's glyph sequence + the piece-score map in,
per-arm and matched-pair records out. The held-out encode pass and glyph-sequence
reconstruction live in :mod:`.runner`.
"""

from __future__ import annotations

import math
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
    from collections.abc import Mapping, Sequence

Boundary = Literal["nmb", "mb"]
Arm = Literal["bpe", "unigram"]
UnpairedReason = Literal["conditional_negative_branch", "extras_single_arm_knob"]

GlyphTuple = tuple[str, ...]


def chunk_segmentation_entropy(
    glyphs: GlyphTuple,
    piece_scores: Mapping[GlyphTuple, float],
    *,
    max_piece_len: int | None = None,
) -> float:
    """Exact Shannon entropy (nats) of one chunk's segmentation distribution.

    A *segmentation* of the glyph sequence ``glyphs`` is a partition into
    contiguous pieces, each a key of ``piece_scores`` (the Unigram log-prob of
    that piece). The induced distribution is ``P(seg) ∝ exp(Σ piece scores)``;
    this returns its Shannon entropy via a numerically stable forward DP over
    the segmentation lattice — at each position the posterior over the incoming
    edge and the standard entropy recursion
    ``H[j] = Σ_i q_ij·(H[i] − log q_ij)`` with
    ``log q_ij = logZ[i] + s_{i→j} − logZ[j]``.

    The all-singletons path always exists (the base glyphs are installed in
    every vocabulary), so the lattice is connected. A single feasible
    segmentation yields exactly ``0.0``. ``max_piece_len`` bounds the inner
    span scan; when ``None`` it is the longest key in ``piece_scores``.

    Args:
        glyphs: the chunk's Layer-A glyph sequence.
        piece_scores: ``{glyph_tuple: log_prob}`` over the vocabulary pieces.
        max_piece_len: longest piece length to consider; defaults to the
            longest key in ``piece_scores``.

    Returns:
        Segmentation entropy in nats; ``0.0`` for an empty chunk.
    """
    n = len(glyphs)
    if n == 0:
        return 0.0
    if max_piece_len is None:
        max_piece_len = max((len(k) for k in piece_scores), default=1)

    neg_inf = float("-inf")
    log_z = [neg_inf] * (n + 1)
    entropy = [0.0] * (n + 1)
    log_z[0] = 0.0

    for j in range(1, n + 1):
        edges: list[tuple[float, int]] = []
        i_lo = max(0, j - max_piece_len)
        for i in range(i_lo, j):
            if log_z[i] == neg_inf:
                continue
            score = piece_scores.get(glyphs[i:j])
            if score is None:
                continue
            edges.append((log_z[i] + score, i))
        if not edges:
            continue
        peak = max(log_w for log_w, _ in edges)
        log_z[j] = peak + math.log(
            math.fsum(math.exp(log_w - peak) for log_w, _ in edges)
        )
        entropy[j] = math.fsum(
            math.exp(log_w - log_z[j]) * (entropy[i] - (log_w - log_z[j]))
            for log_w, i in edges
        )

    return entropy[n] if log_z[n] != neg_inf else 0.0


@dataclass(frozen=True)
class PerMoleculeSegmentation:
    """Per-molecule segmentation entropy and glyph count for one molecule.

    ``entropy_nats`` is the molecule's exact segmentation entropy (sum over
    its Layer-B chunks); ``n_glyphs`` is its base-glyph count under the cell's
    boundary, the per-glyph normalizer.
    """

    entropy_nats: float
    n_glyphs: int


@dataclass(frozen=True)
class ArmSegmentation:
    """Segmentation readings for one arm on one cell.

    Unigram arms carry the mean segmentation entropy per molecule and per
    glyph with 95% bootstrap CIs. BPE arms emit a structural zero with
    ``verified_by_construction=True`` (unique segmentation ⇒ zero entropy),
    leaving the encode/DP unrun; their ``n_molecules`` / ``total_glyphs`` are
    ``0`` (unmeasured) and ``eval_split_sha`` is ``None``.
    """

    cell_id: str
    arm: Arm
    boundary: Boundary
    n_molecules: int
    total_glyphs: int
    total_entropy_nats: float
    entropy_per_molecule_mean: float
    entropy_per_molecule_ci: tuple[float, float]
    entropy_per_glyph: float
    entropy_per_glyph_ci: tuple[float, float]
    verified_by_construction: bool
    training_corpus_sha: str
    eval_split_sha: str | None
    bootstrap_seed: int
    n_resamples: int

    def as_dict(self) -> dict[str, object]:
        return {
            "cell_id": self.cell_id,
            "arm": self.arm,
            "boundary": self.boundary,
            "n_molecules": self.n_molecules,
            "total_glyphs": self.total_glyphs,
            "total_entropy_nats": self.total_entropy_nats,
            "entropy_per_molecule_mean": self.entropy_per_molecule_mean,
            "entropy_per_molecule_ci": list(self.entropy_per_molecule_ci),
            "entropy_per_glyph": self.entropy_per_glyph,
            "entropy_per_glyph_ci": list(self.entropy_per_glyph_ci),
            "verified_by_construction": self.verified_by_construction,
            "training_corpus_sha": self.training_corpus_sha,
            "eval_split_sha": self.eval_split_sha,
            "bootstrap_seed": self.bootstrap_seed,
            "n_resamples": self.n_resamples,
        }


@dataclass(frozen=True)
class MatchedPairSegmentation:
    """Segmentation for one matched-arm ``(corpus, V, boundary)`` coordinate.

    ``delta_entropy_per_molecule = unigram − bpe`` and
    ``delta_entropy_per_glyph`` likewise; since BPE is zero by construction
    these equal the Unigram readings. They are a one-directional corroborating
    signal, not a headline statistic.
    """

    pair_key: str
    tier: str
    corpus: str
    vocab_size: int
    boundary: Boundary
    extras_kind: str | None
    extras_label: str | None
    bpe: ArmSegmentation
    unigram: ArmSegmentation
    delta_entropy_per_molecule: float
    delta_entropy_per_glyph: float

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
            "delta_entropy_per_molecule": self.delta_entropy_per_molecule,
            "delta_entropy_per_glyph": self.delta_entropy_per_glyph,
            "missing_arm": None,
            "unpaired_reason": None,
        }


@dataclass(frozen=True)
class UnpairedSegmentation:
    """Segmentation for one single-arm coordinate; cross-arm Δ undefined."""

    pair_key: str
    tier: str
    corpus: str
    vocab_size: int
    boundary: Boundary
    extras_kind: str | None
    extras_label: str | None
    present_arm: ArmSegmentation
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
            "delta_entropy_per_molecule": None,
            "delta_entropy_per_glyph": None,
            "missing_arm": self.missing_arm,
            "unpaired_reason": self.unpaired_reason,
        }


def _bootstrap_entropy_cis(
    entropies: np.ndarray,
    glyph_counts: np.ndarray,
    *,
    seed: int,
    n_resamples: int = N_BOOTSTRAP_RESAMPLES,
    level: float = CI_LEVEL,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Percentile bootstrap CIs for the per-molecule and per-glyph means.

    Each resample draws ``n_molecules`` molecules with replacement (one shared
    resample stream, so both statistics share the same draws — the Distribution recipe);
    the per-molecule mean is ``Σ entropy / n_molecules`` and the per-glyph value
    is ``Σ entropy / Σ glyphs``. Returns ``((nan, nan), (nan, nan))`` when the
    held-out split is empty. Vectorized over numpy for the 1M-molecule corpora.
    """
    nan = (float("nan"), float("nan"))
    n_mol = entropies.shape[0]
    if n_mol == 0:
        return nan, nan

    rng = np.random.default_rng(seed)
    per_molecule_samples: list[float] = []
    per_glyph_samples: list[float] = []
    for _ in range(n_resamples):
        mult = np.bincount(rng.integers(0, n_mol, n_mol), minlength=n_mol).astype(
            np.float64
        )
        entropy_sum = float(np.dot(entropies, mult))
        glyph_sum = float(np.dot(glyph_counts, mult))
        per_molecule_samples.append(entropy_sum / n_mol)
        if glyph_sum > 0:
            per_glyph_samples.append(entropy_sum / glyph_sum)
    return (
        percentile_ci(per_molecule_samples, level=level),
        percentile_ci(per_glyph_samples, level=level),
    )


def compute_arm_segmentation(
    per_molecule: Sequence[PerMoleculeSegmentation],
    *,
    cell_id: str,
    arm: Arm,
    boundary: Boundary,
    training_corpus_sha: str,
    eval_split_sha: str,
    seed: int | None = None,
    n_resamples: int = N_BOOTSTRAP_RESAMPLES,
) -> ArmSegmentation:
    """Aggregate per-molecule entropies into an :class:`ArmSegmentation` record.

    The encoded path — one :class:`PerMoleculeSegmentation` per held-out molecule.
    ``entropy_per_molecule_mean`` is ``Σ entropy / n_molecules``;
    ``entropy_per_glyph`` is ``Σ entropy / Σ glyphs`` (cross-corpus
    normalizer). Bootstrap CIs are 95% percentile over molecule resamples; the
    seed is derived from ``cell_id`` so re-runs reproduce byte-identical CIs.
    """
    n_molecules = len(per_molecule)
    entropies = np.asarray([pm.entropy_nats for pm in per_molecule], dtype=np.float64)
    glyph_counts = np.asarray([pm.n_glyphs for pm in per_molecule], dtype=np.float64)
    total_entropy = float(entropies.sum()) if n_molecules else 0.0
    total_glyphs = sum(pm.n_glyphs for pm in per_molecule)

    mean_per_molecule = (total_entropy / n_molecules) if n_molecules else float("nan")
    per_glyph = (total_entropy / total_glyphs) if total_glyphs else float("nan")

    seed = seed if seed is not None else bootstrap_seed(cell_id)
    per_molecule_ci, per_glyph_ci = _bootstrap_entropy_cis(
        entropies, glyph_counts, seed=seed, n_resamples=n_resamples
    )
    return ArmSegmentation(
        cell_id=cell_id,
        arm=arm,
        boundary=boundary,
        n_molecules=n_molecules,
        total_glyphs=total_glyphs,
        total_entropy_nats=total_entropy,
        entropy_per_molecule_mean=mean_per_molecule,
        entropy_per_molecule_ci=per_molecule_ci,
        entropy_per_glyph=per_glyph,
        entropy_per_glyph_ci=per_glyph_ci,
        verified_by_construction=False,
        training_corpus_sha=training_corpus_sha,
        eval_split_sha=eval_split_sha,
        bootstrap_seed=seed,
        n_resamples=n_resamples,
    )


def compute_bpe_arm_segmentation(
    *,
    cell_id: str,
    boundary: Boundary,
    training_corpus_sha: str,
) -> ArmSegmentation:
    """BPE cells produce a unique segmentation; emit a zero record by construction.

    Stock GPE/BPE is a deterministic merge encoder (one segmentation per string),
    so entropy is zero; satisfied structurally with ``entropy=0`` and
    ``verified_by_construction=True``, leaving the held-out encode and DP unrun.
    The mirror of Scaffold's ``compute_unigram_arm_scaffold``.
    """
    return ArmSegmentation(
        cell_id=cell_id,
        arm="bpe",
        boundary=boundary,
        n_molecules=0,
        total_glyphs=0,
        total_entropy_nats=0.0,
        entropy_per_molecule_mean=0.0,
        entropy_per_molecule_ci=(0.0, 0.0),
        entropy_per_glyph=0.0,
        entropy_per_glyph_ci=(0.0, 0.0),
        verified_by_construction=True,
        training_corpus_sha=training_corpus_sha,
        eval_split_sha=None,
        bootstrap_seed=0,
        n_resamples=0,
    )


def compute_matched_pair_segmentation(
    bpe: ArmSegmentation,
    unigram: ArmSegmentation,
    *,
    pair_key: str,
    tier: str,
    corpus: str,
    vocab_size: int,
    boundary: Boundary,
    extras_kind: str | None = None,
    extras_label: str | None = None,
) -> MatchedPairSegmentation:
    """Join the two :class:`ArmSegmentation` arms into a matched-pair record.

    ``delta_entropy_per_molecule = unigram − bpe`` (and per-glyph likewise);
    with the BPE arm zero by construction these equal the Unigram readings.

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
    return MatchedPairSegmentation(
        pair_key=pair_key,
        tier=tier,
        corpus=corpus,
        vocab_size=vocab_size,
        boundary=boundary,
        extras_kind=extras_kind,
        extras_label=extras_label,
        bpe=bpe,
        unigram=unigram,
        delta_entropy_per_molecule=unigram.entropy_per_molecule_mean
        - bpe.entropy_per_molecule_mean,
        delta_entropy_per_glyph=unigram.entropy_per_glyph - bpe.entropy_per_glyph,
    )


def compute_unpaired_segmentation(
    arm_record: ArmSegmentation,
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
) -> UnpairedSegmentation:
    """Wrap one present arm as an :class:`UnpairedSegmentation` record."""
    if missing_arm == arm_record.arm:
        raise ValueError(
            f"missing_arm={missing_arm!r} cannot equal the present arm "
            f"{arm_record.arm!r}"
        )
    if arm_record.boundary != boundary:
        raise ValueError(
            f"arm boundary {arm_record.boundary!r} disagrees with pair {boundary!r}"
        )
    return UnpairedSegmentation(
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
    "Arm",
    "ArmSegmentation",
    "Boundary",
    "GlyphTuple",
    "MatchedPairSegmentation",
    "PerMoleculeSegmentation",
    "UnpairedSegmentation",
    "bootstrap_seed",
    "chunk_segmentation_entropy",
    "compute_arm_segmentation",
    "compute_bpe_arm_segmentation",
    "compute_matched_pair_segmentation",
    "compute_unpaired_segmentation",
]
