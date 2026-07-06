"""Deposition and resumability for the F_{p,n} confirmation.

Per-cell ``F_{p,n}`` results are deposited one JSON per grid cell under
``results/data/f95/<cell_id>.json``. A result is stale once its corpus is
reprocessed, so :func:`is_f95_done` checks the recorded ``training_corpus_sha``
against the corpus on disk; :func:`build_f95_table` tolerates a partial grid.

The deposit/skip-if-fresh/table-join skeleton is the shared
:mod:`smiles_subword.tokenize.audit._celldeposit` engine; this module supplies
the F95-specific row projector, Markdown formatter, and ``flagged`` roll-up via
a :class:`CellDepositSpec` rebuilt from the path globals on each call (so the
test suite's path monkeypatching stays transparent).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from smiles_subword.paths import RESULTS_DATA_DIR
from smiles_subword.tokenize.audit import _celldeposit

if TYPE_CHECKING:
    from pathlib import Path

    from smiles_subword.tokenize.audit._celldeposit import (
        AuditableCell,
        CellDepositSpec,
    )
    from smiles_subword.tokenize.audit.f95 import F95Result

SCHEMA_VERSION = 1

F95_DATA_DIR = RESULTS_DATA_DIR
F95_CELL_DIR = F95_DATA_DIR / "f95"
F95_TABLE_JSON = F95_DATA_DIR / "f95_table.json"
F95_TABLE_MD = F95_DATA_DIR / "f95_table.md"


def _spec() -> CellDepositSpec:
    """Rebuild the deposit spec from the current path globals (test-patchable)."""
    return _celldeposit.CellDepositSpec(
        schema_version=SCHEMA_VERSION,
        cell_dir=F95_CELL_DIR,
        table_json=F95_TABLE_JSON,
        table_md=F95_TABLE_MD,
        row_projector=_table_row,
        format_md=_format_md_table,
        table_rollups=_table_rollups,
    )


def f95_json_path(cell_id: str) -> Path:
    """Return the per-cell F_{p,n} JSON path for ``cell_id``."""
    return _celldeposit.cell_json_path(F95_CELL_DIR, cell_id)


def write_f95_json(
    cell: AuditableCell, result: F95Result, *, training_corpus_sha: str
) -> Path:
    """Deposit ``result`` for ``cell`` as a per-cell JSON; return its path.

    The payload carries the grid coordinates alongside the measurement so the
    cross-arm ΔF pairing join needs no manifest lookup.
    """
    return _celldeposit.write_cell_json(
        _spec(), cell, result.as_dict(), training_corpus_sha=training_corpus_sha
    )


def read_f95_json(cell: AuditableCell) -> dict[str, object] | None:
    """Return the deposited F_{p,n} payload for ``cell``, or None if absent."""
    return _celldeposit.read_cell_json(_spec(), cell.cell_id)


def is_f95_done(cell: AuditableCell, *, training_corpus_sha: str) -> bool:
    """True iff a deposited result exists for ``cell`` against the current corpus."""
    return _celldeposit.is_cell_done(
        _spec(), cell.cell_id, training_corpus_sha=training_corpus_sha
    )


def _table_row(payload: dict[str, object]) -> dict[str, object]:
    clearance = payload.get("clearance_by_n", {})
    assert isinstance(clearance, dict)
    return {
        "cell_id": payload["cell_id"],
        "arm": payload["arm"],
        "algo": payload["algo"],
        "vocab_size": payload["vocab_size"],
        "corpus": payload["corpus"],
        "boundary": payload["boundary"],
        "tier": payload["tier"],
        "v_observed": payload["v_observed"],
        "n_non_atomic": payload["n_non_atomic"],
        "clearance_by_n": clearance,
        "headline_clearance": payload["headline_clearance"],
        "embedding_tail_unsafe": payload["embedding_tail_unsafe"],
    }


def _table_rollups(rows: list[dict[str, object]]) -> dict[str, object]:
    return {"flagged": [r["cell_id"] for r in rows if r["embedding_tail_unsafe"]]}


def _format_md_table(rows: list[dict[str, object]], pending: list[str]) -> str:
    header = (
        "| cell | arm | V_obs | non-atomic "
        "| F@50 | F@100 | F@200 | embedding tail |\n"
        "|---|---|--:|--:|--:|--:|--:|---|"
    )
    lines = [header]
    for row in rows:
        clearance = row["clearance_by_n"]
        assert isinstance(clearance, dict)
        flag = "**UNSAFE**" if row["embedding_tail_unsafe"] else "ok"
        lines.append(
            f"| {row['cell_id']} | {row['arm']} | {row['v_observed']} "
            f"| {row['n_non_atomic']} "
            f"| {float(clearance.get('50', 0.0)):.4f} "
            f"| {float(clearance.get('100', 0.0)):.4f} "
            f"| {float(clearance.get('200', 0.0)):.4f} | {flag} |"
        )
    body = "\n".join(lines)
    if pending:
        body += "\n\npending (" + str(len(pending)) + " cells not yet confirmed):\n"
        body += "\n".join(f"- {cid}" for cid in pending)
    return body + "\n"


def build_f95_table() -> tuple[Path, Path]:
    """Aggregate every deposited per-cell JSON into the F_{p,n} table.

    Writes ``f95_table.json`` (per-cell rows + the flagged / pending lists) and
    ``f95_table.md`` (the human-readable table). Cells with no deposited JSON
    are listed as pending, so the table is meaningful on a partial grid.
    """
    return _celldeposit.build_cell_table(_spec())


__all__ = [
    "F95_CELL_DIR",
    "F95_TABLE_JSON",
    "F95_TABLE_MD",
    "SCHEMA_VERSION",
    "build_f95_table",
    "f95_json_path",
    "is_f95_done",
    "read_f95_json",
    "write_f95_json",
]
