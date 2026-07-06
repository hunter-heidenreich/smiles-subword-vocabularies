"""Dead-zone surplus ``ΔF_{p,n}`` (mechanism diagnostic).

Reports the Gowda & May 2020 embedding-learnability fraction ``F_{p,n}`` per
cell (already on disk from the F95 confirmations) and the cross-arm difference
``ΔF_{p,n} = F^BPE_{p,n} − F^UL_{p,n}`` at matched ``(V, corpus, boundary)``,
headlined at ``ΔF_{0.95,100}``. Sign convention: positive ``delta_f`` ⇒ BPE
clears the bar at a higher fraction than Unigram-LM (Bostrom 2020 Fig 2b analog).

Pure computation — data classes and the join: an F95 JSON payload in, a slice
out (no tokenizer loading, no F95 recomputation). Single-arm coordinates (the
ZINC-22 BPE ``V=2048`` conditional branch and the four single-arm-knob extras)
emit :class:`UnpairedDeadzone`. The JSON exposes ``headline_delta_f`` +
``delta_fp`` as the cross-arm ΔF reads.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from smiles_subword.tokenize.audit.f95 import F95_GRID, HEADLINE_N, HEADLINE_P

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True)
class ArmF95Slice:
    """The slice of one F95 JSON payload that Deadzone reads.

    ``clearance_by_n`` carries the F fraction per fixed ``n``. ``F_{p,n}`` is
    ``p``-independent in computation (``_evaluate_fp`` counts ``freq ≥ n_min``;
    p only sets the bar tested), so :meth:`f_at` looks ``n`` up regardless of ``p``.
    """

    cell_id: str
    arm: Literal["bpe", "unigram"]
    clearance_by_n: dict[int, float]
    headline_clearance: float
    embedding_tail_unsafe: bool
    training_corpus_sha: str
    v_observed: int
    n_non_atomic: int

    @classmethod
    def from_f95_payload(cls, payload: Mapping[str, Any]) -> ArmF95Slice:
        """Project an on-disk F95 JSON dict to the slice Deadzone consumes.

        ``payload`` values are typed ``Any`` (``json``'s return type); the
        runtime ``int``/``float``/``bool``/``str`` conversions validate.
        """
        arm = payload["arm"]
        if arm not in ("bpe", "unigram"):
            raise ValueError(f"arm must be 'bpe' or 'unigram'; got {arm!r}")
        raw_by_n = payload["clearance_by_n"]
        if not isinstance(raw_by_n, dict):
            raise TypeError("F95 payload 'clearance_by_n' must be a dict")
        clearance_by_n = {int(k): float(v) for k, v in raw_by_n.items()}
        return cls(
            cell_id=str(payload["cell_id"]),
            arm=arm,
            clearance_by_n=clearance_by_n,
            headline_clearance=float(payload["headline_clearance"]),
            embedding_tail_unsafe=bool(payload["embedding_tail_unsafe"]),
            training_corpus_sha=str(payload["training_corpus_sha"]),
            v_observed=int(payload["v_observed"]),
            n_non_atomic=int(payload["n_non_atomic"]),
        )

    def f_at(self, p: float, n: int) -> float:
        """Return ``F_{p,n}`` for this arm.

        Raises:
            KeyError: ``n`` is not in :data:`F95_GRID`'s firing thresholds.
        """
        del p  # F_{p,n} is p-independent in computation; p sets the bar, not the value.
        return self.clearance_by_n[n]

    def as_dict(self) -> dict[str, object]:
        """JSON-ready payload mirroring the F95 slice fields."""
        return {
            "cell_id": self.cell_id,
            "arm": self.arm,
            "clearance_by_n": {str(n): c for n, c in self.clearance_by_n.items()},
            "headline_clearance": self.headline_clearance,
            "embedding_tail_unsafe": self.embedding_tail_unsafe,
            "training_corpus_sha": self.training_corpus_sha,
            "v_observed": self.v_observed,
            "n_non_atomic": self.n_non_atomic,
        }


@dataclass(frozen=True)
class DeltaFp:
    """One ``(p, n)`` cross-arm ``ΔF`` entry.

    ``delta_f = f_bpe − f_unigram``; positive ⇒ BPE has the higher clearance
    fraction (fewer dead-zone tokens). ``is_headline`` is True iff
    ``(p, n) == (HEADLINE_P, HEADLINE_N)``.
    """

    p: float
    n: int
    f_bpe: float
    f_unigram: float
    delta_f: float
    is_headline: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "p": self.p,
            "n": self.n,
            "f_bpe": self.f_bpe,
            "f_unigram": self.f_unigram,
            "delta_f": self.delta_f,
            "is_headline": self.is_headline,
        }


@dataclass(frozen=True)
class MatchedPairDeadzone:
    """``ΔF`` for one matched-arm coordinate."""

    pair_key: str
    tier: str
    corpus: str
    vocab_size: int
    boundary: str
    extras_kind: str | None
    extras_label: str | None
    bpe: ArmF95Slice
    unigram: ArmF95Slice
    delta_fp: list[DeltaFp]
    headline_delta_f: float
    any_arm_unsafe: bool
    both_arms_unsafe: bool

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
            "delta_fp": [d.as_dict() for d in self.delta_fp],
            "headline_delta_f": self.headline_delta_f,
            "any_arm_unsafe": self.any_arm_unsafe,
            "both_arms_unsafe": self.both_arms_unsafe,
            "missing_arm": None,
            "unpaired_reason": None,
        }


@dataclass(frozen=True)
class UnpairedDeadzone:
    """``F`` for one structurally single-arm coordinate; ΔF undefined.

    Emitted for ZINC-22 BPE ``V=2048`` (Unigram untrained) and the four
    single-arm-knob extras (seed-cap, two prune-schedule, REAL-Space
    merge-exhaustion). The present arm's F slice is preserved so aggregation can
    still report ``F`` alongside the undefined ΔF.
    """

    pair_key: str
    tier: str
    corpus: str
    vocab_size: int
    boundary: str
    extras_kind: str | None
    extras_label: str | None
    present_arm: ArmF95Slice
    missing_arm: Literal["bpe", "unigram"]
    unpaired_reason: Literal["conditional_negative_branch", "extras_single_arm_knob"]
    delta_fp: list[DeltaFp] = field(default_factory=list)

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
            "delta_fp": None,
            "headline_delta_f": None,
            "any_arm_unsafe": self.present_arm.embedding_tail_unsafe,
            "both_arms_unsafe": False,
            "missing_arm": self.missing_arm,
            "unpaired_reason": self.unpaired_reason,
        }


def compute_matched_pair_deadzone(
    bpe: ArmF95Slice,
    unigram: ArmF95Slice,
    *,
    pair_key: str,
    tier: str,
    corpus: str,
    vocab_size: int,
    boundary: str,
    extras_kind: str | None = None,
    extras_label: str | None = None,
) -> MatchedPairDeadzone:
    """Build the per-(p,n) ΔF grid and headline from two F95 slices.

    Raises:
        ValueError: ``bpe.arm`` is not ``"bpe"`` or ``unigram.arm`` is not
            ``"unigram"`` — keeps the sign convention unambiguous.
    """
    if bpe.arm != "bpe":
        raise ValueError(f"first argument must be the BPE arm; got arm={bpe.arm!r}")
    if unigram.arm != "unigram":
        raise ValueError(
            f"second argument must be the Unigram arm; got arm={unigram.arm!r}"
        )
    delta_fp = [
        DeltaFp(
            p=p,
            n=n,
            f_bpe=bpe.f_at(p, n),
            f_unigram=unigram.f_at(p, n),
            delta_f=bpe.f_at(p, n) - unigram.f_at(p, n),
            is_headline=(p == HEADLINE_P and n == HEADLINE_N),
        )
        for p, n in F95_GRID
    ]
    headline = next(d for d in delta_fp if d.is_headline)
    return MatchedPairDeadzone(
        pair_key=pair_key,
        tier=tier,
        corpus=corpus,
        vocab_size=vocab_size,
        boundary=boundary,
        extras_kind=extras_kind,
        extras_label=extras_label,
        bpe=bpe,
        unigram=unigram,
        delta_fp=delta_fp,
        headline_delta_f=headline.delta_f,
        any_arm_unsafe=bpe.embedding_tail_unsafe or unigram.embedding_tail_unsafe,
        both_arms_unsafe=bpe.embedding_tail_unsafe and unigram.embedding_tail_unsafe,
    )


def compute_unpaired_deadzone(
    arm_slice: ArmF95Slice,
    *,
    pair_key: str,
    tier: str,
    corpus: str,
    vocab_size: int,
    boundary: str,
    extras_kind: str | None,
    extras_label: str | None,
    missing_arm: Literal["bpe", "unigram"],
    unpaired_reason: Literal["conditional_negative_branch", "extras_single_arm_knob"],
) -> UnpairedDeadzone:
    """Wrap one present arm's F slice as an :class:`UnpairedDeadzone` record.

    Raises:
        ValueError: ``missing_arm`` equals ``arm_slice.arm`` — both arms
            present means it is not a single-arm record.
    """
    if missing_arm == arm_slice.arm:
        raise ValueError(
            f"missing_arm={missing_arm!r} cannot equal present arm "
            f"arm_slice.arm={arm_slice.arm!r}"
        )
    return UnpairedDeadzone(
        pair_key=pair_key,
        tier=tier,
        corpus=corpus,
        vocab_size=vocab_size,
        boundary=boundary,
        extras_kind=extras_kind,
        extras_label=extras_label,
        present_arm=arm_slice,
        missing_arm=missing_arm,
        unpaired_reason=unpaired_reason,
    )


__all__ = [
    "ArmF95Slice",
    "DeltaFp",
    "MatchedPairDeadzone",
    "UnpairedDeadzone",
    "compute_matched_pair_deadzone",
    "compute_unpaired_deadzone",
]
