"""absorption measurement.

Pure math in :mod:`.math` (re-exported here), per-cell build in
:mod:`.runner`, and JSON deposit in :mod:`.io`.
"""

from __future__ import annotations

from smiles_subword.tokenize.measure.absorption.math import (
    CI_LEVEL,
    N_BOOTSTRAP_RESAMPLES,
    ArmAbsorption,
    Boundary,
    MatchedPairAbsorption,
    PerMoleculeAbsorption,
    UnpairedAbsorption,
    bootstrap_seed,
    classify_chunks,
    compute_arm_absorption,
    compute_matched_pair_absorption,
    compute_unpaired_absorption,
)

__all__ = [
    "CI_LEVEL",
    "N_BOOTSTRAP_RESAMPLES",
    "ArmAbsorption",
    "Boundary",
    "MatchedPairAbsorption",
    "PerMoleculeAbsorption",
    "UnpairedAbsorption",
    "bootstrap_seed",
    "classify_chunks",
    "compute_arm_absorption",
    "compute_matched_pair_absorption",
    "compute_unpaired_absorption",
]
