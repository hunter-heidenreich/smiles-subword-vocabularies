"""Deposition + aggregator for scaffold-token records.

Per-pair JSONs land under ``results/data/scaffold/``, one file per ``pair_key``.
The deposit dispatch and table join are the shared engine in
:mod:`smiles_subword.tokenize.measure._deposit`; this module supplies the
Scaffold-specific record builders, row projection, Markdown columns, and
freshness key.

:func:`is_scaffold_done` checks both ``training_corpus_sha`` (each cell's
``meta.yaml``) and ``scaffold_log_sha`` (recomputed from the sidecar log bytes)
so a re-run of either upstream input invalidates the record. Scaffold is a
training-time measurement, so there is no held-out eval-split SHA in the
freshness key — that is the one place it departs from the per-arm template.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from smiles_subword.paths import RESULTS_DATA_DIR
from smiles_subword.tokenize.measure import _deposit
from smiles_subword.tokenize.measure._cellmeta import (
    cell_training_sha_fresh,
    resolve_cell_meta,
)
from smiles_subword.tokenize.measure._pairing import (
    MatchedPair,
    UnpairedCell,
    pair_all_cells,
)
from smiles_subword.tokenize.measure._tables import fmt_md
from smiles_subword.tokenize.measure.scaffold.math import (
    Arm,
    ArmScaffold,
    MatchedPairScaffold,
    UnpairedScaffold,
    compute_matched_pair_scaffold,
    compute_unpaired_scaffold,
)
from smiles_subword.tokenize.measure.scaffold.runner import (
    ScaffoldLogMissingError,
    read_scaffold_log_sha,
    run_arm_scaffold,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

SCHEMA_VERSION = 1

SCAFFOLD_DATA_DIR = RESULTS_DATA_DIR
SCAFFOLD_CELL_DIR = SCAFFOLD_DATA_DIR / "scaffold"
SCAFFOLD_TABLE_JSON = SCAFFOLD_DATA_DIR / "scaffold_table.json"
SCAFFOLD_TABLE_MD = SCAFFOLD_DATA_DIR / "scaffold_table.md"


def _arm_block_is_fresh(arm_block: object) -> bool:
    if not isinstance(arm_block, dict):
        return False
    cell = cell_training_sha_fresh(
        arm_block.get("cell_id"), arm_block.get("training_corpus_sha")
    )
    if cell is None:
        return False
    corpus, name = cell
    return read_scaffold_log_sha(corpus, name) == arm_block.get("scaffold_log_sha")


def _arm_from_cell(cell_id: str, arm: Arm) -> ArmScaffold | str:
    fields = resolve_cell_meta(cell_id)
    if isinstance(fields, str):
        return fields
    try:
        return run_arm_scaffold(
            cell_id=cell_id,
            corpus=fields.corpus,
            name=fields.name,
            arm=arm,
            boundary=fields.boundary,
        )
    except ScaffoldLogMissingError as exc:
        return str(exc)
    except FileNotFoundError as exc:
        return str(exc)


def _matched_pair_record(pair: MatchedPair) -> MatchedPairScaffold | str:
    bpe_arm = _arm_from_cell(pair.bpe_cell_id, "bpe")
    if isinstance(bpe_arm, str):
        return bpe_arm
    unigram_arm = _arm_from_cell(pair.unigram_cell_id, "unigram")
    if isinstance(unigram_arm, str):
        return unigram_arm
    if bpe_arm.boundary != unigram_arm.boundary:
        return (
            f"arm boundary mismatch for {pair.key.slug}: "
            f"bpe={bpe_arm.boundary!r}, unigram={unigram_arm.boundary!r}"
        )
    return compute_matched_pair_scaffold(
        bpe_arm,
        unigram_arm,
        pair_key=pair.key.slug,
        tier=pair.tier,
        corpus=pair.key.corpus,
        vocab_size=pair.key.vocab_size,
        boundary=bpe_arm.boundary,
        extras_kind=pair.key.extras_kind,
        extras_label=pair.key.extras_label,
    )


def _unpaired_record(unpaired: UnpairedCell) -> UnpairedScaffold | str:
    arm = _arm_from_cell(unpaired.cell_id, unpaired.arm)
    if isinstance(arm, str):
        return arm
    missing_arm: Arm = "unigram" if unpaired.arm == "bpe" else "bpe"
    return compute_unpaired_scaffold(
        arm,
        pair_key=unpaired.key.slug,
        tier=unpaired.tier,
        corpus=unpaired.key.corpus,
        vocab_size=unpaired.key.vocab_size,
        boundary=arm.boundary,
        extras_kind=unpaired.key.extras_kind,
        extras_label=unpaired.key.extras_label,
        missing_arm=missing_arm,
        unpaired_reason=unpaired.reason,
    )


def _table_row_matched(payload: dict[str, Any]) -> dict[str, Any]:
    bpe = payload.get("bpe")
    unigram = payload.get("unigram")
    assert isinstance(bpe, dict)
    assert isinstance(unigram, dict)
    return {
        "pair_key": payload["pair_key"],
        "pair_status": payload["pair_status"],
        "tier": payload["tier"],
        "corpus": payload["corpus"],
        "vocab_size": payload["vocab_size"],
        "boundary": payload["boundary"],
        "extras_kind": payload["extras_kind"],
        "extras_label": payload["extras_label"],
        "bpe_cell_id": bpe["cell_id"],
        "unigram_cell_id": unigram["cell_id"],
        "bpe_scaffold_count": bpe["scaffold_count"],
        "bpe_scaffold_fraction_of_v": bpe["scaffold_fraction_of_v"],
        "bpe_surface_form_breakdown": bpe["surface_form_breakdown"],
        "bpe_threshold": bpe.get("threshold"),
        "unigram_scaffold_count": unigram["scaffold_count"],
        "unigram_scaffold_fraction_of_v": unigram["scaffold_fraction_of_v"],
        "delta_scaffold_fraction": payload["delta_scaffold_fraction"],
    }


def _table_row_unpaired(payload: dict[str, Any]) -> dict[str, Any]:
    present_arm_key = "bpe" if payload.get("bpe") is not None else "unigram"
    present = payload[present_arm_key]
    assert isinstance(present, dict)
    return {
        "pair_key": payload["pair_key"],
        "pair_status": payload["pair_status"],
        "tier": payload["tier"],
        "corpus": payload["corpus"],
        "vocab_size": payload["vocab_size"],
        "boundary": payload["boundary"],
        "extras_kind": payload["extras_kind"],
        "extras_label": payload["extras_label"],
        "present_arm": present_arm_key,
        "present_cell_id": present["cell_id"],
        "present_scaffold_count": present["scaffold_count"],
        "present_scaffold_fraction_of_v": present["scaffold_fraction_of_v"],
        "present_threshold": present.get("threshold"),
        "missing_arm": payload["missing_arm"],
        "unpaired_reason": payload["unpaired_reason"],
    }


def _fmt_breakdown(breakdown: object) -> str:
    if not isinstance(breakdown, dict):
        return "—"
    parts = [
        f"{k}={breakdown.get(k, 0)}"
        for k in ("bracket_internal", "structural", "atomic")
    ]
    return " ".join(parts)


def _format_md_table(
    matched_rows: list[dict[str, Any]],
    unpaired_rows: list[dict[str, Any]],
    pending: list[str],
) -> str:
    header_cols = (
        "| pair | tier | V | corpus | bnd | n^BPE | frac^BPE "
        "| n^UL | frac^UL | Δfrac | breakdown^BPE |"
    )
    lines: list[str] = [
        "## Matched pairs (Δscaffold-fraction reportable)",
        "",
        header_cols,
        "|---|---|--:|---|---|--:|--:|--:|--:|--:|---|",
    ]
    lines.extend(
        f"| {row['pair_key']} | {row['tier']} | {row['vocab_size']} "
        f"| {row['corpus']} | {row['boundary']} "
        f"| {row['bpe_scaffold_count']} "
        f"| {fmt_md(row['bpe_scaffold_fraction_of_v'])} "
        f"| {row['unigram_scaffold_count']} "
        f"| {fmt_md(row['unigram_scaffold_fraction_of_v'])} "
        f"| {fmt_md(row['delta_scaffold_fraction'], spec='+.4f')} "
        f"| {_fmt_breakdown(row['bpe_surface_form_breakdown'])} |"
        for row in matched_rows
    )
    if unpaired_rows:
        lines.extend(["", "## Single-arm coordinates (Δ undefined)", ""])
        lines.append("| pair | tier | V | corpus | bnd | arm | n | frac | reason |")
        lines.append("|---|---|--:|---|---|---|--:|--:|---|")
        lines.extend(
            f"| {row['pair_key']} | {row['tier']} | {row['vocab_size']} "
            f"| {row['corpus']} | {row['boundary']} | {row['present_arm']} "
            f"| {row['present_scaffold_count']} "
            f"| {fmt_md(row['present_scaffold_fraction_of_v'])} "
            f"| {row['unpaired_reason']} |"
            for row in unpaired_rows
        )
    if pending:
        lines.extend(
            ["", f"## Pending ({len(pending)} pair_keys without Scaffold records)", ""]
        )
        lines.extend(f"- {pk}" for pk in pending)
    return "\n".join(lines) + "\n"


def _spec() -> _deposit.DepositSpec:
    return _deposit.DepositSpec(
        schema_version=SCHEMA_VERSION,
        cell_dir=SCAFFOLD_CELL_DIR,
        table_json=SCAFFOLD_TABLE_JSON,
        table_md=SCAFFOLD_TABLE_MD,
        pair_provider=pair_all_cells,
        build_matched=_matched_pair_record,
        build_unpaired=_unpaired_record,
        row_matched=_table_row_matched,
        row_unpaired=_table_row_unpaired,
        format_md=_format_md_table,
        is_done=is_scaffold_done,
    )


def scaffold_json_path(pair_key: str) -> Path:
    """Return the per-pair Scaffold JSON path for ``pair_key``."""
    return _deposit.json_path(SCAFFOLD_CELL_DIR, pair_key)


def write_scaffold_json(record: MatchedPairScaffold | UnpairedScaffold) -> Path:
    """Deposit ``record`` as a per-pair JSON; return its path."""
    return _deposit.write_json(SCAFFOLD_CELL_DIR, SCHEMA_VERSION, record)


def read_scaffold_json(pair_key: str) -> dict[str, object] | None:
    """Return the deposited Scaffold payload for ``pair_key``, or None if absent."""
    return _deposit.read_json(SCAFFOLD_CELL_DIR, pair_key)


def is_scaffold_done(pair_key: str) -> bool:
    """True iff a deposited Scaffold record exists whose upstream SHAs still match."""
    return _deposit.nested_is_done(read_scaffold_json(pair_key), _arm_block_is_fresh)


def deposit_pair(pair: MatchedPair) -> tuple[Path | None, str | None]:
    """Compute and deposit ``pair``; return ``(path, None)`` or ``(None, reason)``."""
    return _deposit.deposit_pair(_spec(), pair)


def deposit_unpaired(unpaired: UnpairedCell) -> tuple[Path | None, str | None]:
    """Compute and deposit ``unpaired``; return ``(path, None)`` or pending reason."""
    return _deposit.deposit_unpaired(_spec(), unpaired)


def build_scaffold_table() -> tuple[Path, Path]:
    """Aggregate every deposited Scaffold JSON into ``scaffold_table.{json,md}``."""
    return _deposit.build_table(_spec())


def deposit_all(
    matched_pairs: Sequence[MatchedPair] | None = None,
    unpaired_cells: Sequence[UnpairedCell] | None = None,
    *,
    rebuild: bool = False,
    only_pair_keys: frozenset[str] | None = None,
) -> tuple[list[str], list[tuple[str, str]]]:
    """Deposit every Scaffold record across the committed grid + extras."""
    return _deposit.deposit_all(
        _spec(),
        matched_pairs,
        unpaired_cells,
        rebuild=rebuild,
        only_pair_keys=only_pair_keys,
    )


__all__ = [
    "SCAFFOLD_CELL_DIR",
    "SCAFFOLD_DATA_DIR",
    "SCAFFOLD_TABLE_JSON",
    "SCAFFOLD_TABLE_MD",
    "SCHEMA_VERSION",
    "build_scaffold_table",
    "deposit_all",
    "deposit_pair",
    "deposit_unpaired",
    "is_scaffold_done",
    "read_scaffold_json",
    "scaffold_json_path",
    "write_scaffold_json",
]
