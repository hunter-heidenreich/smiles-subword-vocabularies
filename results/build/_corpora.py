"""Shared corpus presentation constants (backend-free).

The headline-corpus display labels live here, not in :mod:`style`, because the
table renderers (``latex``, ``figspec``, the ``table_*`` drivers) must stay free
of matplotlib — ``style`` imports it. Both the text and figure surfaces import
this map so the four corpora are spelled the same everywhere.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

CORPUS_LABEL = {
    "pubchem": "PubChem",
    "zinc22": "ZINC-22",
    "coconut": "COCONUT",
    "real_space": "REAL-Space",
}
"""Map each headline corpus's on-disk name to its display label."""

CORPUS_ORDER: tuple[str, ...] = tuple(CORPUS_LABEL)
"""Headline corpora in canonical display order (the keys of :data:`CORPUS_LABEL`)."""

CORPUS_RANK: dict[str, int] = {name: i for i, name in enumerate(CORPUS_ORDER)}
"""Sort rank per headline corpus; consumers fall back to a large rank for OOD."""

DUMBBELL_VOCAB = 1024
"""The single vocab size the dumbbell figures (fertility, distribution) render at."""


def dumbbell_ys(rows: Sequence[Mapping[str, object]]) -> list[float]:
    """Top-to-bottom y positions for dumbbell rows, gapped between corpus groups.

    Each row carries a truthy ``"new_group"`` flag on the first row of a corpus
    group; groups get an extra 0.6 gap above the 1.0 per-row step. Shared by the
    fertility and distribution-intrinsics dumbbell charts so their geometry stays
    identical.
    """
    ys: list[float] = []
    y = 0.0
    for i, row in enumerate(rows):
        if i and row["new_group"]:
            y -= 0.6
        y -= 1.0
        ys.append(y)
    return ys
