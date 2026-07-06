"""Rollup audit deposition for the scaffold-log supplementary retrain sweep.

Each :func:`retrain_with_scaffold_log` call appends a per-cell record to a single
rollup JSON + human-readable MD under ``results/data/audits/``, listing every
scaffold-retrained cell, its ``scaffold_log_sha``, and the byte-identity outcome.

Idempotent: re-running a cell that already passed updates its row in place.
Mismatches halt the sweep, so the rollup never reaches the ``failed`` state in
production — the caller raises before writing.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from smiles_subword._io import atomic_write_json, atomic_write_text
from smiles_subword.paths import RESULTS_DATA_DIR

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from smiles_subword.tokenize.audit.scaffold_retrain import ScaffoldRetrainResult

SCHEMA_VERSION = 1

AUDIT_DIR = RESULTS_DATA_DIR / "audits"
AUDIT_JSON = AUDIT_DIR / "scaffold_byte_identity_audit.json"
AUDIT_MD = AUDIT_DIR / "scaffold_byte_identity_audit.md"


def _read_existing() -> dict[str, Any]:
    if not AUDIT_JSON.is_file():
        return {"schema_version": SCHEMA_VERSION, "cells": []}
    try:
        return json.loads(AUDIT_JSON.read_text())
    except json.JSONDecodeError:
        return {"schema_version": SCHEMA_VERSION, "cells": []}


def _format_md(rollup: dict[str, Any]) -> str:
    cells = rollup.get("cells", [])
    assert isinstance(cells, list)
    n_ok = sum(1 for c in cells if c.get("status") == "ok")
    n_already = sum(1 for c in cells if c.get("status") == "already_done")
    n_skipped = sum(1 for c in cells if c.get("status") == "skipped")
    n_failed = sum(1 for c in cells if c.get("status") == "failed")
    lines = [
        "# Scaffold byte-identity audit (scaffold)",
        "",
        f"- cells listed: **{len(cells)}**",
        f"- ok (retrained + byte-identical): **{n_ok}**",
        f"- already_done (idempotent skip): **{n_already}**",
        f"- skipped (non-BPE or dry-run): **{n_skipped}**",
        f"- failed (byte-identity mismatch): **{n_failed}**",
        "",
        "| cell_id | status | scaffold_log_sha | reason |",
        "|---|---|---|---|",
    ]
    sorted_cells = sorted(cells, key=lambda c: str(c.get("cell_id", "")))
    for cell in sorted_cells:
        sha = cell.get("scaffold_log_sha")
        sha_short = (sha[:12] + "…") if isinstance(sha, str) and len(sha) > 12 else "—"
        reason = cell.get("reason") or "—"
        lines.append(
            f"| {cell.get('cell_id')} | {cell.get('status')} | {sha_short} | {reason} |"
        )
    return "\n".join(lines) + "\n"


def record_results(results: Iterable[ScaffoldRetrainResult]) -> tuple[Path, Path]:
    """Upsert every :class:`ScaffoldRetrainResult` into the rollup audit JSON + MD.

    Returns the audit JSON + MD paths. Mismatches are recorded with
    ``status='failed'`` only if the caller explicitly hands them
    through; the production retrain path raises on byte-identity
    violation before this function is called, so the rollup carries no
    silent failures.
    """
    rollup = _read_existing()
    cells: list[dict[str, Any]] = list(rollup.get("cells", []))
    by_id = {c.get("cell_id"): i for i, c in enumerate(cells)}
    for result in results:
        row = {
            "cell_id": result.cell_id,
            "status": result.status,
            "scaffold_log_sha": result.scaffold_log_sha,
            "reason": result.reason,
        }
        if result.cell_id in by_id:
            cells[by_id[result.cell_id]] = row
        else:
            cells.append(row)
            by_id[result.cell_id] = len(cells) - 1
    rollup = {"schema_version": SCHEMA_VERSION, "cells": cells}
    atomic_write_json(AUDIT_JSON, rollup)
    atomic_write_text(AUDIT_MD, _format_md(rollup))
    return AUDIT_JSON, AUDIT_MD


__all__ = [
    "AUDIT_DIR",
    "AUDIT_JSON",
    "AUDIT_MD",
    "SCHEMA_VERSION",
    "record_results",
]
