"""Marginal cross-arm Jaccard across vocabulary-size steps.

Quantifies whether the pieces each arm *adds* between consecutive ``V`` are
themselves near-disjoint (the mechanism behind cross-arm overlap rising with
``V``). For one ``(corpus, boundary)`` and a step ``v_lower -> v_upper`` the
freshly-added multi-glyph pieces of an arm are

    fresh_arm = multi(arm, v_upper) \\ multi(arm, v_lower),

and the marginal cross-arm Jaccard is ``jaccard(fresh_bpe, fresh_unigram)``. The
multi-glyph set follows the glyph-tuple convention of :mod:`jaccard` (single-glyph
base excluded — identical in both arms), reusing its :func:`jaccard`.

This module is the pure computation; tokenizer-artifact loading and the JSON
deposit live in ``scripts/measure/compute_marginal_jaccard.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from smiles_subword.paths import RESULTS_DATA_DIR
from smiles_subword.tokenize.measure.jaccard import jaccard

if TYPE_CHECKING:
    from smiles_subword.tokenize.measure.jaccard import GlyphTuple

MARGINAL_JACCARD_SCHEMA_VERSION = 1

MARGINAL_JACCARD_TABLE = RESULTS_DATA_DIR / "marginal_jaccard_table.json"
"""Aggregate deposit: one record per ``(corpus, boundary)`` V-step."""


def fresh_pieces(
    lower: frozenset[GlyphTuple], upper: frozenset[GlyphTuple]
) -> frozenset[GlyphTuple]:
    """Multi-glyph pieces present at the larger ``V`` but not the smaller one.

    Set difference, so it is correct whether or not the arm's vocabularies nest
    across ``V`` (BPE nests by construction; Unigram-LM need not).
    """
    return upper - lower


@dataclass(frozen=True)
class MarginalStep:
    """Marginal cross-arm overlap for one ``(corpus, boundary)`` V-step."""

    corpus: str
    boundary: str
    v_lower: int
    v_upper: int
    n_fresh_bpe: int
    n_fresh_unigram: int
    n_fresh_shared: int
    marginal_jaccard: float

    def as_dict(self) -> dict[str, object]:
        return {
            "corpus": self.corpus,
            "boundary": self.boundary,
            "v_lower": self.v_lower,
            "v_upper": self.v_upper,
            "n_fresh_bpe": self.n_fresh_bpe,
            "n_fresh_unigram": self.n_fresh_unigram,
            "n_fresh_shared": self.n_fresh_shared,
            "marginal_jaccard": self.marginal_jaccard,
        }


def build_step(
    *,
    corpus: str,
    boundary: str,
    v_lower: int,
    v_upper: int,
    bpe_lower: frozenset[GlyphTuple],
    bpe_upper: frozenset[GlyphTuple],
    unigram_lower: frozenset[GlyphTuple],
    unigram_upper: frozenset[GlyphTuple],
) -> MarginalStep:
    """Compute the marginal cross-arm Jaccard for one V-step (pure)."""
    fresh_bpe = fresh_pieces(bpe_lower, bpe_upper)
    fresh_unigram = fresh_pieces(unigram_lower, unigram_upper)
    return MarginalStep(
        corpus=corpus,
        boundary=boundary,
        v_lower=v_lower,
        v_upper=v_upper,
        n_fresh_bpe=len(fresh_bpe),
        n_fresh_unigram=len(fresh_unigram),
        n_fresh_shared=len(fresh_bpe & fresh_unigram),
        marginal_jaccard=jaccard(fresh_bpe, fresh_unigram),
    )
