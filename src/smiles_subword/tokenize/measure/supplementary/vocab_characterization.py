"""Per-pair vocabulary characterization (supplementary diagnostics).

Decomposes a matched ``(BPE, Unigram-LM)`` pair's multi-glyph subword sets into
the three populations behind the Jaccard (``shared``, ``bpe_only``,
``unigram_only``) and summarizes each arm's piece-length distribution and
rank-frequency (Zipf) curve — the qualitative complement to the ``J`` / ``D`` /
``eta`` / Renyi scalars.

Every function here is pure over the glyph-tuple sets and count maps Jaccard
(held-out emission counts) and the F95 pass (training-corpus counts) already
produce; no new training or encoding pass. Both the training and held-out
rank-frequency curves are reported because the attrition between them is the
dead-zone made visible: a piece frequent in training but absent on held-out is an
over-fit / dead piece.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from statistics import median
from typing import TYPE_CHECKING

from smiles_subword._io import atomic_write_json
from smiles_subword.paths import RESULTS_DATA_DIR
from smiles_subword.tokenize.measure.jaccard import GlyphTuple

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

VOCAB_CHAR_DIR = RESULTS_DATA_DIR / "vocab_characterization"


def surface(piece: GlyphTuple) -> str:
    """Readable surface of a glyph tuple --- ``("C","C")`` becomes ``CC``."""
    return "".join(piece)


@dataclass(frozen=True)
class MembershipPartition:
    """The three multi-glyph populations behind a matched pair's Jaccard."""

    shared: frozenset[GlyphTuple]
    bpe_only: frozenset[GlyphTuple]
    unigram_only: frozenset[GlyphTuple]

    @property
    def jaccard(self) -> float:
        """``|shared| / |union|`` --- reproduces the Jaccard ``J`` for this pair."""
        union = len(self.shared) + len(self.bpe_only) + len(self.unigram_only)
        return len(self.shared) / union if union else float("nan")


def partition(
    bpe: frozenset[GlyphTuple], unigram: frozenset[GlyphTuple]
) -> MembershipPartition:
    """Split two multi-glyph subword sets into shared / BPE-only / Unigram-only.

    This is the set-algebra behind
    :func:`smiles_subword.tokenize.measure.jaccard.jaccard`; the returned
    partition's ``jaccard`` property equals that ``J`` on the same inputs.
    """
    return MembershipPartition(
        shared=frozenset(bpe & unigram),
        bpe_only=frozenset(bpe - unigram),
        unigram_only=frozenset(unigram - bpe),
    )


@dataclass(frozen=True)
class LengthProfile:
    """Glyph-length distribution of one arm's multi-glyph subword set."""

    histogram: dict[int, int]
    mean: float
    median: float
    max_length: int

    @property
    def n_pieces(self) -> int:
        return sum(self.histogram.values())


def length_profile(pieces: frozenset[GlyphTuple]) -> LengthProfile:
    """Summarize the piece-length distribution: histogram + mean / median / max."""
    lengths = [len(p) for p in pieces]
    histogram = dict(sorted(Counter(lengths).items()))
    if not lengths:
        return LengthProfile(histogram={}, mean=0.0, median=0.0, max_length=0)
    return LengthProfile(
        histogram=histogram,
        mean=sum(lengths) / len(lengths),
        median=float(median(lengths)),
        max_length=max(lengths),
    )


def rank_frequency(
    counts: Mapping[GlyphTuple, int],
) -> list[tuple[int, GlyphTuple, int]]:
    """Rank pieces by descending frequency (ties broken by glyph tuple).

    Returns ``(rank, piece, frequency)`` with rank starting at 1. Zero-count
    pieces keep their place in the tail, so the curve shows the dead tail
    explicitly rather than dropping it.
    """
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [(rank, piece, freq) for rank, (piece, freq) in enumerate(ordered, start=1)]


def _rank_freq_payload(
    counts: Mapping[GlyphTuple, int] | None,
) -> list[dict[str, object]] | None:
    if counts is None:
        return None
    return [
        {"rank": rank, "piece": surface(piece), "freq": freq}
        for rank, piece, freq in rank_frequency(counts)
    ]


def characterize_pair(
    pair_key: str,
    bpe_multi: frozenset[GlyphTuple],
    unigram_multi: frozenset[GlyphTuple],
    *,
    bpe_train_counts: Mapping[GlyphTuple, int] | None = None,
    unigram_train_counts: Mapping[GlyphTuple, int] | None = None,
    bpe_holdout_counts: Mapping[GlyphTuple, int] | None = None,
    unigram_holdout_counts: Mapping[GlyphTuple, int] | None = None,
) -> dict[str, object]:
    """Assemble the JSON-serializable vocabulary characterization for one pair.

    ``*_multi`` are the arms' multi-glyph subword sets (Jaccard's ``multi_subwords``).
    The optional ``*_counts`` maps drive the rank-frequency curves; pass the F95
    training counts and/or the Jaccard held-out emission counts. Omitted curves are
    serialized as ``null``.
    """
    part = partition(bpe_multi, unigram_multi)
    return {
        "pair_key": pair_key,
        "jaccard": part.jaccard,
        "partition": {
            "shared": sorted(surface(p) for p in part.shared),
            "bpe_only": sorted(surface(p) for p in part.bpe_only),
            "unigram_only": sorted(surface(p) for p in part.unigram_only),
        },
        "length_profile": {
            "bpe": _length_payload(length_profile(bpe_multi)),
            "unigram": _length_payload(length_profile(unigram_multi)),
        },
        "rank_frequency": {
            "bpe_train": _rank_freq_payload(bpe_train_counts),
            "unigram_train": _rank_freq_payload(unigram_train_counts),
            "bpe_holdout": _rank_freq_payload(bpe_holdout_counts),
            "unigram_holdout": _rank_freq_payload(unigram_holdout_counts),
        },
    }


def _length_payload(profile: LengthProfile) -> dict[str, object]:
    return {
        "histogram": {str(k): v for k, v in profile.histogram.items()},
        "mean": profile.mean,
        "median": profile.median,
        "max_length": profile.max_length,
        "n_pieces": profile.n_pieces,
    }


def vocab_char_path(pair_key: str) -> Path:
    """Return the per-pair vocabulary-characterization JSON path."""
    return VOCAB_CHAR_DIR / f"{pair_key}.json"


def write_vocab_characterization(payload: dict[str, object]) -> Path:
    """Deposit a characterization payload as per-pair JSON; return its path."""
    path = vocab_char_path(str(payload["pair_key"]))
    atomic_write_json(path, payload)
    return path
