"""fg_alignment measurement.

Pure math in :mod:`.math` (re-exported here), per-cell build in
:mod:`.runner`, and JSON deposit in :mod:`.io`.
"""

from __future__ import annotations

from smiles_subword.tokenize.measure.fg_alignment.math import (
    FUNCTIONAL_CLASSES,
    Arm,
    ArmFgAlignment,
    Boundary,
    MatchedPairFgAlignment,
    PerMoleculeFgLocality,
    UnpairedFgAlignment,
    bootstrap_seed,
    compute_arm_fg_alignment,
    compute_matched_pair_fg_alignment,
    compute_unpaired_fg_alignment,
)

__all__ = [
    "FUNCTIONAL_CLASSES",
    "Arm",
    "ArmFgAlignment",
    "Boundary",
    "MatchedPairFgAlignment",
    "PerMoleculeFgLocality",
    "UnpairedFgAlignment",
    "bootstrap_seed",
    "compute_arm_fg_alignment",
    "compute_matched_pair_fg_alignment",
    "compute_unpaired_fg_alignment",
]
