"""F_{p,n} embedding-learnability confirmation.

The Gowda & May 2020 embedding-learnability metric over a trained tokenizer's
full training corpus: the fraction of the *learned* vocabulary firing at least
``n`` times. A cell whose ``F_{0.95,100}`` confirmation fails is flagged
``embedding-tail-unsafe`` and must never be silently pooled into a cross-corpus
comparison. Arm-agnostic; the Gowda core is :func:`_evaluate_fp`.

The metric runs over the **non-atomic** vocabulary — every token whose string is
neither a single chemical glyph nor a special. The glyph alphabet is supplied by
the caller as ``atomic_tokens``: for a Smirk-GPE artifact it is exactly
``hf_tokenizer.get_vocab()`` (the WordLevel base — Smirk-GPE surfaces only the
atomic base, not derived merges), and identical across both arms for a given
corpus by construction. Resolving it the same way for both arms is what makes
the cross-arm ``ΔF`` (Dead-zone surplus) apples-to-apples; the ~165 atomic
glyphs are excluded as they are shared by both vocabularies.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING

from smiles_subword.tokenize._batched import iter_encoded_batches
from smiles_subword.tokenize.base import collect_special_ids

if TYPE_CHECKING:
    from collections.abc import Iterable

    from smiles_subword.tokenize.base import Tokenizer

_PERCENTILES: tuple[float, ...] = (0.90, 0.95, 0.99)
_MIN_FIRINGS: tuple[int, ...] = (50, 100, 200)

F95_GRID: tuple[tuple[float, int], ...] = tuple(
    (p, n) for p in _PERCENTILES for n in _MIN_FIRINGS
)
"""The fixed ``(p, n)`` grid — 3 percentiles × 3 firing thresholds."""

HEADLINE_P: float = 0.95
HEADLINE_N: int = 100
"""``F_{0.95,100}`` — the Gowda & May 2020 embedding-learnability bar."""


@dataclass(frozen=True)
class FpThreshold:
    """Outcome of one ``F_p >= n_min`` query against a rank-frequency sequence."""

    p: float
    n_min: int
    crossed: bool
    crossed_at_rank: int | None
    n_merges_clearing: int

    def as_dict(self) -> dict[str, float | int | bool | None]:
        return asdict(self)


def _count_clearing(rank_freq: list[tuple[int, int]], *, n_min: int) -> int:
    """Count ``rank_freq`` entries firing ``>= n_min`` times (independent of ``p``)."""
    return sum(1 for _, freq in rank_freq if freq >= n_min)


def _evaluate_fp(
    rank_freq: list[tuple[int, int]], *, p: float, n_min: int
) -> FpThreshold:
    """Fraction of ``rank_freq`` entries firing ``>= n_min`` times vs. the ``p`` bar.

    ``rank_freq`` is ``[(rank, frequency), ...]`` in ascending rank order;
    ``crossed`` is whether the clearing fraction meets ``p``, and
    ``crossed_at_rank`` is the first rank that falls below ``n_min`` when it
    does not (a diagnostic only).
    """
    if not 0.0 < p <= 1.0:
        raise ValueError(f"p must be in (0, 1]; got {p!r}")

    n = len(rank_freq)
    if n == 0:
        return FpThreshold(
            p=p, n_min=n_min, crossed=False, crossed_at_rank=None, n_merges_clearing=0
        )

    n_clearing = _count_clearing(rank_freq, n_min=n_min)
    fraction_clearing = n_clearing / n
    crossed = fraction_clearing >= p

    crossed_at_rank: int | None
    if crossed:
        crossed_at_rank = None
    else:
        crossed_at_rank = next(
            (rank for rank, freq in rank_freq if freq < n_min),
            None,
        )

    return FpThreshold(
        p=p,
        n_min=n_min,
        crossed=crossed,
        crossed_at_rank=crossed_at_rank,
        n_merges_clearing=n_clearing,
    )


@dataclass(frozen=True)
class F95Result:
    """Per-cell ``F_{p,n}`` confirmation outcome.

    ``clearance_by_n`` maps each firing threshold ``n`` to the fraction of the
    non-atomic vocabulary that fires at least ``n`` times; the percentile ``p``
    is the bar that fraction is tested against, not a second input to it.
    ``headline_clearance`` is the ``n=100`` fraction and ``embedding_tail_unsafe``
    is its failure against the ``p=0.95`` bar.

    ``FpThreshold.crossed_at_rank`` in ``fp_thresholds`` indexes the id-ordered
    non-atomic vocabulary, not a merge-introduction order — a diagnostic only;
    the clearance fraction, the sole learnability input, is order-independent.
    """

    arm: str
    v_observed: int
    n_non_atomic: int
    n_corpus_tokens: int
    n_corpus_molecules: int
    fp_thresholds: list[FpThreshold]
    clearance_by_n: dict[int, float]
    headline_clearance: float
    embedding_tail_unsafe: bool
    training_counts_by_id: dict[int, int] = field(default_factory=dict)
    """Per-token-id training-corpus firing counts for the non-atomic vocabulary;
    the training rank-frequency input to the vocabulary characterization."""

    def as_dict(self) -> dict[str, object]:
        """JSON-ready payload; ``clearance_by_n`` keys are stringified."""
        return {
            "arm": self.arm,
            "v_observed": self.v_observed,
            "n_non_atomic": self.n_non_atomic,
            "n_corpus_tokens": self.n_corpus_tokens,
            "n_corpus_molecules": self.n_corpus_molecules,
            "fp_thresholds": [t.as_dict() for t in self.fp_thresholds],
            "clearance_by_n": {str(n): c for n, c in self.clearance_by_n.items()},
            "headline_clearance": self.headline_clearance,
            "embedding_tail_unsafe": self.embedding_tail_unsafe,
            "training_counts_by_id": {
                str(tid): c for tid, c in self.training_counts_by_id.items()
            },
        }


def _non_atomic_token_ids(tok: Tokenizer, atomic_tokens: frozenset[str]) -> list[int]:
    """Return the learned-subword ids — not glyphs, not specials.

    A token is atomic iff its string is in ``atomic_tokens`` (the corpus glyph
    alphabet). Ids are returned in ascending order.
    """
    specials = collect_special_ids(tok)
    ids: list[int] = []
    for tid in range(len(tok)):
        if tid in specials:
            continue
        token = tok.id_to_token(tid)
        if token and token not in atomic_tokens:
            ids.append(tid)
    return ids


def compute_f95_from_encoded(
    tok: Tokenizer,
    train_encoded: Iterable[list[int]],
    *,
    arm: str,
    atomic_tokens: frozenset[str],
    fp_thresholds: tuple[tuple[float, int], ...] = F95_GRID,
) -> F95Result:
    """Compute ``F_{p,n}`` from a pre-encoded training corpus.

    Args:
        tok: the trained tokenizer being confirmed.
        train_encoded: one token-id list per training-corpus molecule.
        arm: ``"bpe"`` or ``"unigram"`` — recorded on the result.
        atomic_tokens: the corpus glyph alphabet; tokens in it are excluded
            from the metric (see the module docstring).
        fp_thresholds: ``(p, n)`` pairs to evaluate. Defaults to
            :data:`F95_GRID`; ``headline_clearance`` /
            ``embedding_tail_unsafe`` always use ``(0.95, 100)`` regardless.

    Raises:
        ValueError: ``arm`` is not a known arm, the corpus is empty, or the
            tokenizer has no non-atomic vocabulary to confirm.
    """
    if arm not in ("bpe", "unigram"):
        raise ValueError(f"arm must be 'bpe' or 'unigram'; got {arm!r}")

    counts: Counter[int] = Counter()
    n_molecules = 0
    for ids in train_encoded:
        n_molecules += 1
        counts.update(ids)
    if n_molecules == 0:
        raise ValueError("corpus produced zero molecules")

    non_atomic = _non_atomic_token_ids(tok, atomic_tokens)
    if not non_atomic:
        raise ValueError("tokenizer has no non-atomic vocabulary to confirm")

    rank_freq = [(rank, int(counts.get(tid, 0))) for rank, tid in enumerate(non_atomic)]
    n_total = len(non_atomic)

    threshold_results = [
        _evaluate_fp(rank_freq, p=p, n_min=n) for p, n in fp_thresholds
    ]
    # ``n_merges_clearing`` counts entries firing >= n_min, independent of ``p``;
    # compute the per-n clearance fraction from the count directly so the dummy
    # ``p`` and a redundant headline pass drop out.
    clearance_by_n = {
        n: _count_clearing(rank_freq, n_min=n) / n_total
        for n in sorted({n for _, n in fp_thresholds})
    }
    headline_clearance = clearance_by_n.get(
        HEADLINE_N, _count_clearing(rank_freq, n_min=HEADLINE_N) / n_total
    )

    return F95Result(
        arm=arm,
        v_observed=len(tok),
        n_non_atomic=n_total,
        n_corpus_tokens=int(counts.total()),
        n_corpus_molecules=n_molecules,
        fp_thresholds=threshold_results,
        clearance_by_n=clearance_by_n,
        headline_clearance=headline_clearance,
        embedding_tail_unsafe=headline_clearance < HEADLINE_P,
        training_counts_by_id={tid: int(counts.get(tid, 0)) for tid in non_atomic},
    )


def compute_f95(
    tok: Tokenizer,
    train_smiles: Iterable[str],
    *,
    arm: str,
    atomic_tokens: frozenset[str],
    fp_thresholds: tuple[tuple[float, int], ...] = F95_GRID,
) -> F95Result:
    """Compute ``F_{p,n}`` for one tokenizer over its full training corpus.

    Thin wrapper over :func:`compute_f95_from_encoded` that streams
    ``train_smiles`` through the batched encode path. Specials are excluded.
    """
    return compute_f95_from_encoded(
        tok,
        iter_encoded_batches(tok, train_smiles, add_special_tokens=False),
        arm=arm,
        atomic_tokens=atomic_tokens,
        fp_thresholds=fp_thresholds,
    )


__all__ = [
    "F95_GRID",
    "HEADLINE_N",
    "HEADLINE_P",
    "F95Result",
    "FpThreshold",
    "compute_f95",
    "compute_f95_from_encoded",
]
