"""JSON deposit for the sensitivity report.

Writes / reads the single ``sensitivity_report.json`` figure payload assembled
by :func:`smiles_subword.tokenize.measure.supplementary.sensitivity.math.build_report`.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from smiles_subword._io import atomic_write_json
from smiles_subword.paths import RESULTS_DATA_DIR

if TYPE_CHECKING:
    from pathlib import Path

    from smiles_subword.tokenize.measure.supplementary.sensitivity.math import (
        SensitivityReport,
    )

__all__ = [
    "SENSITIVITY_DIR",
    "read_report",
    "report_path",
    "write_report",
]

SENSITIVITY_DIR = RESULTS_DATA_DIR / "sensitivity"


def report_path() -> Path:
    return SENSITIVITY_DIR / "sensitivity_report.json"


def write_report(report: SensitivityReport) -> Path:
    path = report_path()
    atomic_write_json(path, report.as_dict())
    return path


def read_report() -> dict[str, object]:
    return json.loads(report_path().read_text())
