"""Cross-corpus generalization: the train x eval transfer matrix.

The package re-exports the pure-math core (:mod:`.math`); the per-cell
build (:mod:`.runner`) and the out-of-distribution extension (:mod:`.ood`) are
imported from their submodules.
"""

from __future__ import annotations

from smiles_subword.tokenize.measure.supplementary.transfer.math import (
    CI_LEVEL,
    N_BOOTSTRAP_RESAMPLES,
    TRANSFER_DIR,
    Arm,
    Boundary,
    PerMoleculeTransfer,
    TransferRecord,
    compute_transfer_record,
)

__all__ = [
    "CI_LEVEL",
    "N_BOOTSTRAP_RESAMPLES",
    "TRANSFER_DIR",
    "Arm",
    "Boundary",
    "PerMoleculeTransfer",
    "TransferRecord",
    "compute_transfer_record",
]
