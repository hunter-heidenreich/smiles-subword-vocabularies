"""Per-cell glyph maps: a token id's Layer-A glyph sequence (and its length).

The glyph-count map is the per-id tuple *length*, so it is derived from the
tuple map rather than walked a second time. A leaf module (stdlib only) so any
runner can import it without a cycle.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from pathlib import Path

GlyphTuple = tuple[str, ...]
Arm = Literal["bpe", "unigram"]

__all__ = [
    "build_bpe_glyph_tuples",
    "build_unigram_glyph_tuples",
    "glyph_count_map",
    "glyph_tuple_map",
]


def build_bpe_glyph_tuples(tokenizer_json: Path) -> dict[int, GlyphTuple]:
    """Return ``{token_id: glyph_tuple}`` for a Smirk-GPE ``tokenizer.json``.

    ``model.vocab`` is the base alphabet, ids ``0 .. base_size-1`` (each a
    1-glyph surface, multi-char atoms like ``Cl`` included); merge ``k`` mints id
    ``base_size + k`` = its two operands' tuples concatenated (operands may be
    earlier merge ids). Base glyphs never appear as a merge result, so this
    self-adapts to the actual base size.
    """
    model = json.loads(tokenizer_json.read_text())["model"]
    base: dict[str, int] = model["vocab"]
    base_size = len(base)
    tuples: dict[int, GlyphTuple] = {tid: (surf,) for surf, tid in base.items()}
    for k, (a, b) in enumerate(model.get("merges", [])):
        tuples[base_size + k] = tuples[int(a)] + tuples[int(b)]
    return tuples


def build_unigram_glyph_tuples(tokenizer_json: Path) -> dict[int, GlyphTuple]:
    """Return ``{token_id: glyph_tuple}`` for a Unigram ``tokenizer.json``.

    ``model.vocab`` is an ordered list indexed by token id; each entry's
    ``glyphs`` is the piece's glyph sequence.
    """
    model = json.loads(tokenizer_json.read_text())["model"]
    if model.get("type") != "Unigram":
        raise ValueError(f"{tokenizer_json}: model type is not Unigram")
    return {tid: tuple(entry["glyphs"]) for tid, entry in enumerate(model["vocab"])}


def glyph_tuple_map(artifact_dir: Path, arm: Arm) -> dict[int, GlyphTuple]:
    """Return ``{token_id: glyph_tuple}`` for the cell's tokenizer, by ``arm``."""
    tokenizer_json = artifact_dir / "tokenizer.json"
    if arm == "bpe":
        return build_bpe_glyph_tuples(tokenizer_json)
    return build_unigram_glyph_tuples(tokenizer_json)


def glyph_count_map(artifact_dir: Path, arm: Arm) -> dict[int, int]:
    """Return ``{token_id: glyph_count}`` for the cell's tokenizer.

    Read off :func:`glyph_tuple_map` as tuple length — the merge walk runs once.
    """
    return {tid: len(t) for tid, t in glyph_tuple_map(artifact_dir, arm).items()}
