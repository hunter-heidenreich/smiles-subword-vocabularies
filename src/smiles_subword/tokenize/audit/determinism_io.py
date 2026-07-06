"""Deposition and resumability for the determinism verification.

Per-cell determinism results are deposited one per grid cell under
``results/data/determinism/<cell_id>.json``, alongside the F95 JSONs. A result
is stale once its corpus is reprocessed, so :func:`is_determinism_done` checks
the recorded ``training_corpus_sha`` against the corpus on disk;
:func:`build_determinism_table` tolerates a partial grid.

The table separates two failure classes: ``flagged`` is every cell that failed
its determinism assertion; ``unexpected`` is the subset whose failure is *not*
expected — a non-empty ``unexpected`` list forces investigation, whereas the
known, expected Unigram ``V=1024 NMB`` jitter lands in ``flagged`` only.

The deposit/skip-if-fresh/table-join skeleton is the shared
:mod:`smiles_subword.tokenize.audit._celldeposit` engine; this module supplies
the determinism-specific row projector, Markdown formatter, and
``flagged``/``unexpected`` roll-ups via a :class:`CellDepositSpec` rebuilt from
the path globals on each call.
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
    from smiles_subword.tokenize.audit.determinism import DeterminismResult

SCHEMA_VERSION = 1

DETERMINISM_DATA_DIR = RESULTS_DATA_DIR
DETERMINISM_CELL_DIR = DETERMINISM_DATA_DIR / "determinism"
DETERMINISM_TABLE_JSON = DETERMINISM_DATA_DIR / "determinism_table.json"
DETERMINISM_TABLE_MD = DETERMINISM_DATA_DIR / "determinism_table.md"


def _spec() -> CellDepositSpec:
    """Rebuild the deposit spec from the current path globals (test-patchable)."""
    return _celldeposit.CellDepositSpec(
        schema_version=SCHEMA_VERSION,
        cell_dir=DETERMINISM_CELL_DIR,
        table_json=DETERMINISM_TABLE_JSON,
        table_md=DETERMINISM_TABLE_MD,
        row_projector=_table_row,
        format_md=_format_md_table,
        table_rollups=_table_rollups,
    )


def determinism_json_path(cell_id: str) -> Path:
    """Return the per-cell determinism JSON path for ``cell_id``."""
    return _celldeposit.cell_json_path(DETERMINISM_CELL_DIR, cell_id)


def write_determinism_json(
    cell: AuditableCell,
    result: DeterminismResult,
    *,
    training_corpus_sha: str,
    expected_failure: bool,
) -> Path:
    """Deposit ``result`` for ``cell`` as a per-cell JSON; return its path.

    ``expected_failure`` records whether a failure of this cell's assertion is
    expected (the Unigram ``V=1024 NMB`` set) — stored in
    the payload so :func:`build_determinism_table` can mechanically separate
    "flagged as expected" from "must investigate".
    """
    return _celldeposit.write_cell_json(
        _spec(),
        cell,
        result.as_dict(),
        training_corpus_sha=training_corpus_sha,
        extra={"expected_failure": expected_failure},
    )


def read_determinism_json(cell: AuditableCell) -> dict[str, object] | None:
    """Return the deposited determinism payload for ``cell``, or None if absent."""
    return _celldeposit.read_cell_json(_spec(), cell.cell_id)


def is_determinism_done(cell: AuditableCell, *, training_corpus_sha: str) -> bool:
    """True iff a deposited result exists for ``cell`` against the current corpus."""
    return _celldeposit.is_cell_done(
        _spec(), cell.cell_id, training_corpus_sha=training_corpus_sha
    )


def _table_row(payload: dict[str, object]) -> dict[str, object]:
    return {
        "cell_id": payload["cell_id"],
        "arm": payload["arm"],
        "algo": payload["algo"],
        "vocab_size": payload["vocab_size"],
        "corpus": payload["corpus"],
        "boundary": payload["boundary"],
        "tier": payload["tier"],
        "deterministic": payload["deterministic"],
        "mismatch_kind": payload["mismatch_kind"],
        "rerun_spread": payload["rerun_spread"],
        "expected_failure": payload["expected_failure"],
    }


def _table_rollups(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "flagged": [r["cell_id"] for r in rows if not r["deterministic"]],
        "unexpected": [
            r["cell_id"]
            for r in rows
            if not r["deterministic"] and not r["expected_failure"]
        ],
    }


def _format_md_table(rows: list[dict[str, object]], pending: list[str]) -> str:
    header = (
        "| cell | arm | deterministic | mismatch | rerun spread | expected |\n"
        "|---|---|---|---|--:|---|"
    )
    lines = [header]
    for row in rows:
        ok = "yes" if row["deterministic"] else "**NO**"
        mismatch = row["mismatch_kind"] or "—"
        expected = "expected" if row["expected_failure"] else "—"
        lines.append(
            f"| {row['cell_id']} | {row['arm']} | {ok} "
            f"| {mismatch} | {row['rerun_spread']} | {expected} |"
        )
    body = "\n".join(lines)
    if pending:
        body += "\n\npending (" + str(len(pending)) + " cells not yet verified):\n"
        body += "\n".join(f"- {cid}" for cid in pending)
    return body + "\n"


def build_determinism_table() -> tuple[Path, Path]:
    """Aggregate every deposited per-cell JSON into the determinism table.

    Writes ``determinism_table.json`` (per-cell rows + the flagged / unexpected
    / pending lists) and ``determinism_table.md`` (the human-readable table).
    Cells with no deposited JSON are listed as pending, so the table is
    meaningful on a partial grid.
    """
    return _celldeposit.build_cell_table(_spec())


__all__ = [
    "DETERMINISM_CELL_DIR",
    "DETERMINISM_TABLE_JSON",
    "DETERMINISM_TABLE_MD",
    "SCHEMA_VERSION",
    "build_determinism_table",
    "determinism_json_path",
    "is_determinism_done",
    "read_determinism_json",
    "write_determinism_json",
]
