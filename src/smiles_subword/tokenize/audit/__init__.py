"""Audit machinery for the subword-vocabulary study.

Implements the verification pilots the study relies on: train-twice determinism
(:mod:`~smiles_subword.tokenize.audit.determinism`), the F95 learnability floor
(:mod:`~smiles_subword.tokenize.audit.f95`), and scaffold retraining.
"""

from __future__ import annotations

from smiles_subword.tokenize.audit.determinism import (
    ArtifactDigest,
    DeterminismResult,
    compare_artifacts,
    digest_artifact,
)
from smiles_subword.tokenize.audit.f95 import F95Result, FpThreshold, compute_f95

__all__ = [
    "ArtifactDigest",
    "DeterminismResult",
    "F95Result",
    "FpThreshold",
    "compare_artifacts",
    "compute_f95",
    "digest_artifact",
]
