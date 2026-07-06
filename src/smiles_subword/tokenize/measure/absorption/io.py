"""Deposition + aggregator for absorption records.

Per-pair JSONs under ``results/data/absorption/``, one per ``pair_key``. The
deposit dispatch and table join are the shared :mod:`..._deposit` engine; this
module supplies the record builders, row projection, and Markdown columns.

:func:`is_absorption_done` gates resume on both ``training_corpus_sha`` (cell
``meta.yaml``) and ``eval_split_sha`` (held-out split MANIFEST), so a re-run of
either invalidates the record. :func:`build_absorption_table` is a pure on-disk
join, tolerant of missing pairs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from smiles_subword.paths import RESULTS_DATA_DIR
from smiles_subword.tokenize.measure import _deposit
from smiles_subword.tokenize.measure._cellmeta import resolve_cell_meta
from smiles_subword.tokenize.measure._cells import load_cell_adapter
from smiles_subword.tokenize.measure._pairing import (
    MatchedPair,
    UnpairedCell,
    pair_all_cells,
)
from smiles_subword.tokenize.measure._tables import fmt_md
from smiles_subword.tokenize.measure.absorption.math import (
    Arm,
    ArmAbsorption,
    MatchedPairAbsorption,
    UnpairedAbsorption,
    compute_matched_pair_absorption,
    compute_unpaired_absorption,
)
from smiles_subword.tokenize.measure.absorption.runner import run_arm_absorption

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

SCHEMA_VERSION = 1

ABSORPTION_DATA_DIR = RESULTS_DATA_DIR
ABSORPTION_CELL_DIR = ABSORPTION_DATA_DIR / "absorption"
ABSORPTION_TABLE_JSON = ABSORPTION_DATA_DIR / "absorption_table.json"
ABSORPTION_TABLE_MD = ABSORPTION_DATA_DIR / "absorption_table.md"


def _arm_from_cell(cell_id: str, arm: Arm) -> ArmAbsorption | str:
    fields = resolve_cell_meta(cell_id)
    if isinstance(fields, str):
        return fields
    try:
        adapter = load_cell_adapter(fields.corpus, fields.name)
    except FileNotFoundError as exc:
        return str(exc)
    return run_arm_absorption(
        adapter,
        cell_id=cell_id,
        corpus=fields.corpus,
        arm=arm,
        boundary=fields.boundary,
        training_corpus_sha=fields.training_corpus_sha,
    )


def _matched_pair_record(pair: MatchedPair) -> MatchedPairAbsorption | str:
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
    return compute_matched_pair_absorption(
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


def _unpaired_record(unpaired: UnpairedCell) -> UnpairedAbsorption | str:
    arm = _arm_from_cell(unpaired.cell_id, unpaired.arm)
    if isinstance(arm, str):
        return arm
    missing_arm: Arm = "unigram" if unpaired.arm == "bpe" else "bpe"
    return compute_unpaired_absorption(
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
        "bpe_absorbed_fraction": bpe["absorbed_fraction"],
        "bpe_absorbed_ci_lo": bpe["absorbed_ci"][0],
        "bpe_absorbed_ci_hi": bpe["absorbed_ci"][1],
        "unigram_absorbed_fraction": unigram["absorbed_fraction"],
        "unigram_absorbed_ci_lo": unigram["absorbed_ci"][0],
        "unigram_absorbed_ci_hi": unigram["absorbed_ci"][1],
        "delta_absorbed": payload["delta_absorbed"],
        "bpe_cross_chunk_fraction": bpe.get("cross_chunk_fraction"),
        "unigram_cross_chunk_fraction": unigram.get("cross_chunk_fraction"),
        "delta_cross_chunk": payload["delta_cross_chunk"],
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
        "present_absorbed_fraction": present["absorbed_fraction"],
        "present_cross_chunk_fraction": present.get("cross_chunk_fraction"),
        "missing_arm": payload["missing_arm"],
        "unpaired_reason": payload["unpaired_reason"],
    }


def _format_md_table(
    matched_rows: list[dict[str, Any]],
    unpaired_rows: list[dict[str, Any]],
    pending: list[str],
) -> str:
    header_cols = "| pair | tier | V | corpus | bnd | abs^BPE | abs^UL | Δabs"
    header_cols += " | xchk^BPE | xchk^UL | Δxchk |"
    lines: list[str] = [
        "## Matched pairs (Δabsorbed reportable)",
        "",
        header_cols,
        "|---|---|--:|---|---|--:|--:|--:|--:|--:|--:|",
    ]
    lines.extend(
        f"| {row['pair_key']} | {row['tier']} | {row['vocab_size']} "
        f"| {row['corpus']} | {row['boundary']} "
        f"| {fmt_md(row['bpe_absorbed_fraction'])} "
        f"| {fmt_md(row['unigram_absorbed_fraction'])} "
        f"| {fmt_md(row['delta_absorbed'], spec='+.4f')} "
        f"| {fmt_md(row['bpe_cross_chunk_fraction'])} "
        f"| {fmt_md(row['unigram_cross_chunk_fraction'])} "
        f"| {fmt_md(row['delta_cross_chunk'], spec='+.4f')} |"
        for row in matched_rows
    )
    if unpaired_rows:
        lines.extend(["", "## Single-arm coordinates (Δ undefined)", ""])
        lines.append("| pair | tier | V | corpus | bnd | arm | abs | xchk | reason |")
        lines.append("|---|---|--:|---|---|---|--:|--:|---|")
        lines.extend(
            f"| {row['pair_key']} | {row['tier']} | {row['vocab_size']} "
            f"| {row['corpus']} | {row['boundary']} | {row['present_arm']} "
            f"| {fmt_md(row['present_absorbed_fraction'])} "
            f"| {fmt_md(row['present_cross_chunk_fraction'])} "
            f"| {row['unpaired_reason']} |"
            for row in unpaired_rows
        )
    if pending:
        lines.extend(
            [
                "",
                f"## Pending ({len(pending)} pair_keys without Absorption records)",
                "",
            ]
        )
        lines.extend(f"- {pk}" for pk in pending)
    return "\n".join(lines) + "\n"


def _spec() -> _deposit.DepositSpec:
    return _deposit.DepositSpec(
        schema_version=SCHEMA_VERSION,
        cell_dir=ABSORPTION_CELL_DIR,
        table_json=ABSORPTION_TABLE_JSON,
        table_md=ABSORPTION_TABLE_MD,
        pair_provider=pair_all_cells,
        build_matched=_matched_pair_record,
        build_unpaired=_unpaired_record,
        row_matched=_table_row_matched,
        row_unpaired=_table_row_unpaired,
        format_md=_format_md_table,
        is_done=is_absorption_done,
    )


def absorption_json_path(pair_key: str) -> Path:
    """Return the per-pair Absorption JSON path for ``pair_key``."""
    return _deposit.json_path(ABSORPTION_CELL_DIR, pair_key)


def write_absorption_json(record: MatchedPairAbsorption | UnpairedAbsorption) -> Path:
    """Deposit ``record`` as a per-pair JSON; return its path."""
    return _deposit.write_json(ABSORPTION_CELL_DIR, SCHEMA_VERSION, record)


def read_absorption_json(pair_key: str) -> dict[str, object] | None:
    """Return the deposited Absorption payload for ``pair_key``, or None if absent."""
    return _deposit.read_json(ABSORPTION_CELL_DIR, pair_key)


def is_absorption_done(pair_key: str) -> bool:
    """True iff a deposited Absorption record exists whose upstream SHAs still match."""
    return _deposit.nested_is_done(
        read_absorption_json(pair_key), _deposit.standard_arm_block_fresh
    )


def deposit_pair(pair: MatchedPair) -> tuple[Path | None, str | None]:
    """Compute and deposit ``pair``; return ``(path, None)`` or ``(None, reason)``."""
    return _deposit.deposit_pair(_spec(), pair)


def deposit_unpaired(unpaired: UnpairedCell) -> tuple[Path | None, str | None]:
    """Compute and deposit ``unpaired``; return ``(path, None)`` or pending reason."""
    return _deposit.deposit_unpaired(_spec(), unpaired)


def build_absorption_table() -> tuple[Path, Path]:
    """Aggregate every deposited Absorption JSON into ``absorption_table.{json,md}``."""
    return _deposit.build_table(_spec())


def deposit_all(
    matched_pairs: Sequence[MatchedPair] | None = None,
    unpaired_cells: Sequence[UnpairedCell] | None = None,
    *,
    rebuild: bool = False,
    only_pair_keys: frozenset[str] | None = None,
) -> tuple[list[str], list[tuple[str, str]]]:
    """Deposit every Absorption record across the committed grid + extras."""
    return _deposit.deposit_all(
        _spec(),
        matched_pairs,
        unpaired_cells,
        rebuild=rebuild,
        only_pair_keys=only_pair_keys,
    )


__all__ = [
    "ABSORPTION_CELL_DIR",
    "ABSORPTION_DATA_DIR",
    "ABSORPTION_TABLE_JSON",
    "ABSORPTION_TABLE_MD",
    "SCHEMA_VERSION",
    "absorption_json_path",
    "build_absorption_table",
    "deposit_all",
    "deposit_pair",
    "deposit_unpaired",
    "is_absorption_done",
    "read_absorption_json",
    "write_absorption_json",
]
