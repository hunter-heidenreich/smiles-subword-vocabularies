"""Shared piece reader for the artifact-derived builders.

Every table/figure that inspects *learned* pieces (glyph count >= 2) reads them
the same way: locate a trained cell's artifact directory, map token ids to glyph
tuples via the package's ``glyph_tuple_map``, and keep the multi-glyph ones.
This module is that one reader, so the builders agree on the convention and on
the on-disk layout (via ``cell_artifact_name`` + ``tokenizer_artifact_dir``)
rather than each re-deriving the ``smirk_…`` directory name.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from smiles_subword.config import cell_artifact_name
from smiles_subword.paths import tokenizer_artifact_dir
from smiles_subword.tokenize.measure.jaccard.runner import glyph_tuple_map

if TYPE_CHECKING:
    from pathlib import Path

    from smiles_subword.config import TokenizerAlgo
    from smiles_subword.tokenize.measure.jaccard import GlyphTuple


def multi_glyph_pieces(artifact_dir: Path, arm: TokenizerAlgo) -> set[GlyphTuple]:
    """The multi-glyph (>= 2 glyphs) learned pieces in a built tokenizer dir."""
    return {t for t in glyph_tuple_map(artifact_dir, arm).values() if len(t) >= 2}


def multi_glyph_set(
    corpus: str,
    vocab_size: int,
    boundary: str,
    arm: TokenizerAlgo,
    *,
    suffix: str | None = None,
) -> set[GlyphTuple]:
    """Multi-glyph pieces for a trained grid/extras cell, addressed by its axes."""
    name = cell_artifact_name(arm, vocab_size, boundary, suffix=suffix)
    return multi_glyph_pieces(tokenizer_artifact_dir(corpus, name), arm)


def multi_glyph_set_optional(
    corpus: str, vocab_size: int, boundary: str, arm: TokenizerAlgo
) -> set[GlyphTuple] | None:
    """As :func:`multi_glyph_set`, but ``None`` when the cell was never trained."""
    name = cell_artifact_name(arm, vocab_size, boundary)
    artifact_dir = tokenizer_artifact_dir(corpus, name)
    return multi_glyph_pieces(artifact_dir, arm) if artifact_dir.exists() else None


def ordered_surfaces(pieces: set[GlyphTuple]) -> list[str]:
    """Piece surfaces ordered by glyph count, then alphabetically."""
    return ["".join(t) for t in sorted(pieces, key=lambda t: (len(t), "".join(t)))]
