"""Hyperparameter-sensitivity curve aggregation.

Pure aggregation in :mod:`.math` (re-exported here), the per-cell measurement
bridge in :mod:`.runner`, and the JSON deposit in :mod:`.io`.
"""

from __future__ import annotations

from smiles_subword.tokenize.measure.supplementary.sensitivity.math import (
    CellMeasured,
    ContrastPoint,
    InteractionGrid,
    LadderCurve,
    LadderSpec,
    SensitivityReport,
    build_interaction_minfreq_mpl,
    build_interaction_mpl_typology,
    build_interaction_mpl_v,
    build_interaction_subiter_shrink,
    build_ladder,
    build_report,
    contrast_point,
)

__all__ = [
    "CellMeasured",
    "ContrastPoint",
    "InteractionGrid",
    "LadderCurve",
    "LadderSpec",
    "SensitivityReport",
    "build_interaction_minfreq_mpl",
    "build_interaction_mpl_typology",
    "build_interaction_mpl_v",
    "build_interaction_subiter_shrink",
    "build_ladder",
    "build_report",
    "contrast_point",
]
