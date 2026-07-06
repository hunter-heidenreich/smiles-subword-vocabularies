"""segmentation measurement.

Pure math in :mod:`.math` (re-exported here), per-cell build in
:mod:`.runner`, and JSON deposit in :mod:`.io`.
"""

from __future__ import annotations

from smiles_subword.tokenize.measure.segmentation.math import (
    CI_LEVEL,
    N_BOOTSTRAP_RESAMPLES,
    Arm,
    ArmSegmentation,
    Boundary,
    GlyphTuple,
    MatchedPairSegmentation,
    PerMoleculeSegmentation,
    UnpairedSegmentation,
    bootstrap_seed,
    chunk_segmentation_entropy,
    compute_arm_segmentation,
    compute_bpe_arm_segmentation,
    compute_matched_pair_segmentation,
    compute_unpaired_segmentation,
)

__all__ = [
    "CI_LEVEL",
    "N_BOOTSTRAP_RESAMPLES",
    "Arm",
    "ArmSegmentation",
    "Boundary",
    "GlyphTuple",
    "MatchedPairSegmentation",
    "PerMoleculeSegmentation",
    "UnpairedSegmentation",
    "bootstrap_seed",
    "chunk_segmentation_entropy",
    "compute_arm_segmentation",
    "compute_bpe_arm_segmentation",
    "compute_matched_pair_segmentation",
    "compute_unpaired_segmentation",
]
