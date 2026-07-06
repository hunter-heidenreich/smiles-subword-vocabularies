"""Shared per-cell deposit + grid-table engine for the audit pilots.

The F95 and determinism pilots deposit one JSON per grid/extras cell keyed by
``cell_id``, gate freshness on ``training_corpus_sha``, and aggregate the grid
into a ``{json,md}`` table — this is that skeleton. Each topic's ``*_io`` module
keeps its own path globals (so the test suite's path monkeypatching stays
transparent) and supplies the topic-specific pieces — row projector, Markdown
formatter, extra per-payload fields, table roll-ups — through a
:class:`CellDepositSpec` rebuilt from those globals on each call.

Single-cell (one arm, no cross-arm pairing) and keyed by the grid manifest;
the per-measurement ``measure/_deposit`` engine is the *paired*-cell analog.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from smiles_subword._io import atomic_write_json, atomic_write_text, read_json_or_none
from smiles_subword.tokenize.grid import load_grid_manifest

if TYPE_CHECKING:
    from smiles_subword.tokenize.extras import ExtrasCell
    from smiles_subword.tokenize.grid import GridCell

    AuditableCell = GridCell | ExtrasCell


@dataclass(frozen=True)
class CellDepositSpec:
    """The topic-specific pieces the shared engine routes around.

    ``row_projector`` flattens a deposited payload into a table row;
    ``format_md`` renders the matched rows + pending list to Markdown; and
    ``table_rollups`` returns the extra top-level table keys (e.g. ``flagged``,
    ``unexpected``) computed from the rows.
    """

    schema_version: int
    cell_dir: Path
    table_json: Path
    table_md: Path
    row_projector: Callable[[dict[str, object]], dict[str, object]]
    format_md: Callable[[list[dict[str, object]], list[str]], str]
    table_rollups: Callable[[list[dict[str, object]]], dict[str, object]]


def cell_json_path(cell_dir: Path, cell_id: str) -> Path:
    """Return the per-cell JSON path for ``cell_id`` under ``cell_dir``."""
    return cell_dir / f"{cell_id}.json"


def _coordinate_fields(cell: AuditableCell) -> dict[str, object]:
    return {
        "cell_id": cell.cell_id,
        "algo": cell.algo,
        "vocab_size": cell.vocab_size,
        "corpus": cell.corpus,
        "boundary": cell.boundary,
        "tier": cell.tier,
    }


def write_cell_json(
    spec: CellDepositSpec,
    cell: AuditableCell,
    result_dict: dict[str, object],
    *,
    training_corpus_sha: str,
    extra: dict[str, object] | None = None,
) -> Path:
    """Deposit one cell's result as a per-cell JSON; return its path.

    The payload carries the grid coordinates + ``training_corpus_sha`` + any
    topic-specific ``extra`` fields alongside the measurement, so the
    downstream join needs no manifest lookup.
    """
    payload: dict[str, object] = {
        "schema_version": spec.schema_version,
        **_coordinate_fields(cell),
        "training_corpus_sha": training_corpus_sha,
        **(extra or {}),
        **result_dict,
    }
    path = cell_json_path(spec.cell_dir, cell.cell_id)
    atomic_write_json(path, payload)
    return path


def read_cell_json(spec: CellDepositSpec, cell_id: str) -> dict[str, object] | None:
    """Return the deposited payload for ``cell_id``, or None if absent/corrupt."""
    return read_json_or_none(cell_json_path(spec.cell_dir, cell_id))


def is_cell_done(
    spec: CellDepositSpec, cell_id: str, *, training_corpus_sha: str
) -> bool:
    """True iff a deposited result exists for ``cell_id`` against the current corpus.

    A missing, unparseable, or stale (``training_corpus_sha`` no longer matches
    the corpus on disk) deposit counts as not done — the pilot must re-run.
    """
    payload = read_cell_json(spec, cell_id)
    if payload is None:
        return False
    return payload.get("training_corpus_sha") == training_corpus_sha


def build_cell_table(spec: CellDepositSpec) -> tuple[Path, Path]:
    """Aggregate every deposited per-cell JSON over the grid into the table.

    Walks the committed grid manifest; cells with no deposit are listed as
    ``pending`` so the table is meaningful on a partial grid. Writes the JSON
    (rows + roll-ups + pending) and the Markdown atomically.
    """
    cells = load_grid_manifest()
    rows: list[dict[str, object]] = []
    pending: list[str] = []
    for cell in cells:
        payload = read_cell_json(spec, cell.cell_id)
        if payload is None:
            pending.append(cell.cell_id)
            continue
        rows.append(spec.row_projector(payload))
    rows.sort(key=lambda r: (r["corpus"], r["algo"], r["vocab_size"], r["boundary"]))

    table_json: dict[str, object] = {
        "schema_version": spec.schema_version,
        "n_cells": len(cells),
        "n_present": len(rows),
        **spec.table_rollups(rows),
        "pending": sorted(pending),
        "cells": rows,
    }
    atomic_write_json(spec.table_json, table_json)
    atomic_write_text(spec.table_md, spec.format_md(rows, sorted(pending)))

    return spec.table_json, spec.table_md


__all__ = [
    "CellDepositSpec",
    "build_cell_table",
    "cell_json_path",
    "is_cell_done",
    "read_cell_json",
    "write_cell_json",
]
