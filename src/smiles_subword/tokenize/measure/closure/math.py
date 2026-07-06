"""Compositional closure — a within-arm self-structure contrast.

Unlike the cross-arm contrasts, this asks how internally self-referential each
arm's learned piece set is. BPE merges two in-vocab pieces, so every BPE piece
is the concatenation of two in-vocab pieces — its vocabulary is *merge-closed*
by construction; Unigram-LM prunes a seed pool under no such constraint, so
whether its pieces decompose is empirical. We measure from the realized
vocabulary alone (never a merge history), so the metrics apply identically to
both arms and BPE's value is a measured anchor.

Let ``V`` be the full realized vocabulary as glyph tuples (length-1 base glyphs
plus every multi-glyph piece) and ``M = {p in V : len(p) >= 2}`` the multi-glyph
pieces (the set Jaccard compares). The base is complete, so metrics range over
splits/sub-pieces of length >= 2:

* **binary-split closure** ``c_bin`` — fraction of ``M`` with some split
  ``p = a.b``, both ``a, b`` in ``V``. BPE's merge-closure invariant, so
  ``c_bin == 1`` for BPE exactly (the correctness anchor); UL's is empirical.
* **orphan rate** ``c_orph`` — fraction of length->=3 pieces with *no* proper
  contiguous sub-piece of length >= 2 in ``V`` (equivalently ``M``): shares no
  multi-glyph building block. Length-2 pieces are excluded (no proper >= 2
  sub-piece). A subset of ``1 - c_bin``.
* **full-substring closure** ``c_full`` — fraction of length->=3 pieces with
  *every* proper substring (length 2..len-1) in ``V``. Strictly stronger than
  ``c_bin`` and non-trivial even for BPE (its merge tree guarantees one split,
  not all substrings), so it differentiates both arms.

Pure computation: a realized vocabulary in, per-arm :class:`ArmClosure` out,
joined to :class:`MatchedPairClosure` (cross-arm BPE - UL gaps). A single-arm
coordinate still has a well-defined value, so :func:`compute_unpaired_closure`
carries the present arm's reading. Exact-set quantity, no bootstrap CI. The
per-cell vocab read lives in :mod:`.runner`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Boundary = Literal["nmb", "mb"]
Arm = Literal["bpe", "unigram"]
UnpairedReason = Literal["conditional_negative_branch", "extras_single_arm_knob"]

GlyphTuple = tuple[str, ...]

MIN_ORPHAN_LEN = 3
"""Orphan / full-substring metrics range over pieces of length >= 3; a length-2
piece has no proper sub-piece of length >= 2, so it is excluded as undefined
rather than counted as a (trivial) orphan."""


def proper_substrings(piece: GlyphTuple, *, min_len: int = 2) -> list[GlyphTuple]:
    """Proper contiguous sub-tuples of ``piece``, length in ``[min_len, len-1]``.

    Excludes the whole tuple (length ``len(piece)``) and any sub-tuple shorter
    than ``min_len``. For a length < ``min_len + 1`` piece the list is empty.
    """
    n = len(piece)
    return [
        piece[start : start + length]
        for length in range(min_len, n)
        for start in range(n - length + 1)
    ]


def binary_split_closed(piece: GlyphTuple, vocab: frozenset[GlyphTuple]) -> bool:
    """True if some split ``piece = a.b`` has both parts in ``vocab``.

    ``vocab`` is the full realized vocabulary (base glyphs + multi-glyph
    pieces). This is BPE's merge-closure invariant; a length-1 piece has no
    split and returns ``False`` (callers range over multi-glyph pieces only).
    """
    return any(piece[:i] in vocab and piece[i:] in vocab for i in range(1, len(piece)))


def is_orphan(piece: GlyphTuple, multi: frozenset[GlyphTuple]) -> bool:
    """True if no proper >= 2-glyph sub-piece of ``piece`` is in ``multi``.

    Defined for ``len(piece) >= MIN_ORPHAN_LEN``; the caller restricts to that
    range. ``multi`` is the multi-glyph subset of the vocabulary (length-1
    glyphs are always present and so never informative here).
    """
    return not any(s in multi for s in proper_substrings(piece, min_len=2))


def full_substring_closed(piece: GlyphTuple, vocab: frozenset[GlyphTuple]) -> bool:
    """True if every proper contiguous substring (length 2..len-1) is in ``vocab``.

    Vacuously true for ``len(piece) < MIN_ORPHAN_LEN`` (no such substring); the
    caller restricts the denominator to ``len >= MIN_ORPHAN_LEN``.
    """
    return all(s in vocab for s in proper_substrings(piece, min_len=2))


@dataclass(frozen=True)
class ArmClosure:
    """Per-arm compositional-closure reading on one cell's realized vocabulary.

    Counts are exact (no CI). ``n_multi = |M|`` (Jaccard-comparable multi-glyph
    set); ``n_ge3`` the length->= :data:`MIN_ORPHAN_LEN` subset the orphan and
    full-substring rates range over. ``n_bracket_nonclosed`` counts non-closed
    bracket-glyph pieces (raw count). ``cell_id`` / ``training_corpus_sha`` travel
    with the record for the deposit-step freshness check.
    """

    cell_id: str
    arm: Arm
    boundary: Boundary
    vocab_size: int
    training_corpus_sha: str
    n_multi: int
    n_ge3: int
    n_bin_closed: int
    n_orphan: int
    n_full_closed: int
    n_bracket_nonclosed: int

    @property
    def c_bin(self) -> float:
        """Binary-split closure fraction over ``M``; nan when ``M`` is empty."""
        return self.n_bin_closed / self.n_multi if self.n_multi else float("nan")

    @property
    def c_orph(self) -> float:
        """Orphan rate over the length-``>= MIN_ORPHAN_LEN`` pieces; nan when none."""
        return self.n_orphan / self.n_ge3 if self.n_ge3 else float("nan")

    @property
    def c_full(self) -> float:
        """Full-substring closure over the length-``>= MIN_ORPHAN_LEN`` pieces."""
        return self.n_full_closed / self.n_ge3 if self.n_ge3 else float("nan")

    def as_block(self) -> dict[str, object]:
        """Serialize the per-arm block embedded in the pair JSON."""
        return {
            "cell_id": self.cell_id,
            "arm": self.arm,
            "boundary": self.boundary,
            "vocab_size": self.vocab_size,
            "training_corpus_sha": self.training_corpus_sha,
            "n_multi": self.n_multi,
            "n_ge3": self.n_ge3,
            "n_bin_closed": self.n_bin_closed,
            "n_orphan": self.n_orphan,
            "n_full_closed": self.n_full_closed,
            "n_bracket_nonclosed": self.n_bracket_nonclosed,
            "c_bin": self.c_bin,
            "c_orph": self.c_orph,
            "c_full": self.c_full,
        }


def compute_arm_closure(
    vocab_tuples: list[GlyphTuple],
    *,
    cell_id: str,
    arm: Arm,
    boundary: Boundary,
    vocab_size: int,
    training_corpus_sha: str,
) -> ArmClosure:
    """Compute one arm's closure counts from its realized vocabulary tuples.

    ``vocab_tuples`` is the full ``{glyph_tuple}`` realized vocabulary (base +
    multi), e.g. ``glyph_tuple_map(...).values()``. A piece is *bracket* (for
    the non-closed stratification) iff it touches a ``[`` or ``]`` delimiter.
    """
    vocab = frozenset(vocab_tuples)
    multi = frozenset(t for t in vocab if len(t) >= 2)
    ge3 = [p for p in multi if len(p) >= MIN_ORPHAN_LEN]

    n_bin_closed = sum(binary_split_closed(p, vocab) for p in multi)
    n_orphan = sum(is_orphan(p, multi) for p in ge3)
    n_full_closed = sum(full_substring_closed(p, vocab) for p in ge3)
    n_bracket_nonclosed = sum(
        1
        for p in multi
        if not binary_split_closed(p, vocab) and any("[" in g or "]" in g for g in p)
    )

    return ArmClosure(
        cell_id=cell_id,
        arm=arm,
        boundary=boundary,
        vocab_size=vocab_size,
        training_corpus_sha=training_corpus_sha,
        n_multi=len(multi),
        n_ge3=len(ge3),
        n_bin_closed=n_bin_closed,
        n_orphan=n_orphan,
        n_full_closed=n_full_closed,
        n_bracket_nonclosed=n_bracket_nonclosed,
    )


@dataclass(frozen=True)
class MatchedPairClosure:
    """Closure for one matched-arm ``(corpus, V, boundary)`` coordinate.

    The cross-arm gaps ``delta_*`` are ``BPE - UL`` and are positive when BPE is
    the more closed arm (``delta_c_bin`` is ``1 - UL.c_bin``, since BPE is 1).
    """

    pair_key: str
    tier: str
    corpus: str
    vocab_size: int
    boundary: Boundary
    extras_kind: str | None
    extras_label: str | None
    bpe: ArmClosure
    unigram: ArmClosure
    delta_c_bin: float
    delta_c_orph: float
    delta_c_full: float

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
            "delta_c_bin": self.delta_c_bin,
            "delta_c_orph": self.delta_c_orph,
            "delta_c_full": self.delta_c_full,
            "missing_arm": None,
            "unpaired_reason": None,
        }


@dataclass(frozen=True)
class UnpairedClosure:
    """Coordinate with only one trained arm.

    Unlike the cross-arm contrasts, closure is within-arm, so the present arm
    still carries a full reading; only the cross-arm gaps are undefined.
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
    present: ArmClosure

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


def _gap(a: float, b: float) -> float:
    """BPE - UL gap, propagating nan if either side is undefined."""
    if a != a or b != b:  # nan
        return float("nan")
    return a - b


def compute_matched_pair_closure(
    bpe: ArmClosure,
    unigram: ArmClosure,
    *,
    pair_key: str,
    tier: str,
    corpus: str,
    vocab_size: int,
    boundary: Boundary,
    extras_kind: str | None = None,
    extras_label: str | None = None,
) -> MatchedPairClosure:
    """Join two :class:`ArmClosure` arms into a matched-pair record.

    Raises:
        ValueError: the arms are not one BPE + one Unigram, or disagree on
            the pair boundary — both indicate a pairing bug upstream.
    """
    if bpe.arm != "bpe":
        raise ValueError(f"first arg must be the BPE arm, got {bpe.arm!r}")
    if unigram.arm != "unigram":
        raise ValueError(f"second arg must be the Unigram arm, got {unigram.arm!r}")
    if bpe.boundary != unigram.boundary or bpe.boundary != boundary:
        raise ValueError("arm boundaries must match the pair boundary")
    return MatchedPairClosure(
        pair_key=pair_key,
        tier=tier,
        corpus=corpus,
        vocab_size=vocab_size,
        boundary=boundary,
        extras_kind=extras_kind,
        extras_label=extras_label,
        bpe=bpe,
        unigram=unigram,
        delta_c_bin=_gap(bpe.c_bin, unigram.c_bin),
        delta_c_orph=_gap(bpe.c_orph, unigram.c_orph),
        delta_c_full=_gap(bpe.c_full, unigram.c_full),
    )


def compute_unpaired_closure(
    present: ArmClosure,
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
) -> UnpairedClosure:
    """Wrap a single present arm's closure for a structurally single-arm coord."""
    if missing_arm == present_arm:
        raise ValueError(
            f"missing_arm={missing_arm!r} cannot equal present_arm {present_arm!r}"
        )
    return UnpairedClosure(
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
    "MIN_ORPHAN_LEN",
    "Arm",
    "ArmClosure",
    "Boundary",
    "GlyphTuple",
    "MatchedPairClosure",
    "UnpairedClosure",
    "binary_split_closed",
    "compute_arm_closure",
    "compute_matched_pair_closure",
    "compute_unpaired_closure",
    "full_substring_closed",
    "is_orphan",
    "proper_substrings",
]
