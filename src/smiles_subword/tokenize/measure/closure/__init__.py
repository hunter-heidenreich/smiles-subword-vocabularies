"""closure measurement.

Pure math in :mod:`.math` (re-exported here), per-cell build in
:mod:`.runner`, and JSON deposit in :mod:`.io`.
"""

from __future__ import annotations

from smiles_subword.tokenize.measure.closure.math import (
    MIN_ORPHAN_LEN,
    Arm,
    ArmClosure,
    Boundary,
    GlyphTuple,
    MatchedPairClosure,
    UnpairedClosure,
    binary_split_closed,
    compute_arm_closure,
    compute_matched_pair_closure,
    compute_unpaired_closure,
    full_substring_closed,
    is_orphan,
    proper_substrings,
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
