"""Deposition + aggregator for dead-zone records.

Per-pair JSONs under ``results/data/deadzone/``, one per ``pair_key``. The
deposit dispatch and table join are the shared :mod:`..._deposit` engine; this
module supplies the record builders (which join already-deposited F95 records
rather than rebuilding from tokenizers), row projection, and Markdown columns.

:func:`is_deadzone_done` re-reads each referenced F95 JSON, so a re-run of an
F95 confirmation (new ``training_corpus_sha``) invalidates the record.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from smiles_subword._io import read_json_or_none
from smiles_subword.paths import RESULTS_DATA_DIR
from smiles_subword.tokenize.audit import f95_io
from smiles_subword.tokenize.measure import _deposit
from smiles_subword.tokenize.measure._pairing import (
    MatchedPair,
    UnpairedCell,
    pair_all_cells,
)
from smiles_subword.tokenize.measure.deadzone.math import (
    ArmF95Slice,
    MatchedPairDeadzone,
    UnpairedDeadzone,
    compute_matched_pair_deadzone,
    compute_unpaired_deadzone,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

SCHEMA_VERSION = 1

DEADZONE_DATA_DIR = RESULTS_DATA_DIR
DEADZONE_CELL_DIR = DEADZONE_DATA_DIR / "deadzone"
DEADZONE_TABLE_JSON = DEADZONE_DATA_DIR / "deadzone_table.json"
DEADZONE_TABLE_MD = DEADZONE_DATA_DIR / "deadzone_table.md"


def _read_f95_payload_by_id(cell_id: str) -> dict[str, object] | None:
    return read_json_or_none(f95_io.f95_json_path(cell_id))


def _f95_block_fresh(block: object) -> bool:
    """True iff an arm block's referenced F95 JSON still carries its SHA.

    Stale if the F95 JSON is missing or has been re-run since deposition
    (``training_corpus_sha`` changed).
    """
    if not isinstance(block, dict):
        return False
    cell_id = block.get("cell_id")
    deposited_sha = block.get("training_corpus_sha")
    if not isinstance(cell_id, str) or not isinstance(deposited_sha, str):
        return False
    f95_payload = _read_f95_payload_by_id(cell_id)
    if f95_payload is None:
        return False
    return f95_payload.get("training_corpus_sha") == deposited_sha


def _matched_pair_record(pair: MatchedPair) -> MatchedPairDeadzone | str:
    """Build the Deadzone record for ``pair`` or return a pending-reason string."""
    bpe_payload = _read_f95_payload_by_id(pair.bpe_cell_id)
    ul_payload = _read_f95_payload_by_id(pair.unigram_cell_id)
    missing: list[str] = []
    if bpe_payload is None:
        missing.append(pair.bpe_cell_id)
    if ul_payload is None:
        missing.append(pair.unigram_cell_id)
    if missing:
        return f"missing F95 JSON for {', '.join(missing)}"
    assert bpe_payload is not None
    assert ul_payload is not None
    bpe_slice = ArmF95Slice.from_f95_payload(bpe_payload)
    ul_slice = ArmF95Slice.from_f95_payload(ul_payload)
    return compute_matched_pair_deadzone(
        bpe_slice,
        ul_slice,
        pair_key=pair.key.slug,
        tier=pair.tier,
        corpus=pair.key.corpus,
        vocab_size=pair.key.vocab_size,
        boundary=pair.key.boundary,
        extras_kind=pair.key.extras_kind,
        extras_label=pair.key.extras_label,
    )


def _unpaired_record(unpaired: UnpairedCell) -> UnpairedDeadzone | str:
    """Build the Deadzone record for ``unpaired`` or return a pending-reason string."""
    payload = _read_f95_payload_by_id(unpaired.cell_id)
    if payload is None:
        return f"missing F95 JSON for {unpaired.cell_id}"
    arm_slice = ArmF95Slice.from_f95_payload(payload)
    missing_arm = "unigram" if unpaired.arm == "bpe" else "bpe"
    return compute_unpaired_deadzone(
        arm_slice,
        pair_key=unpaired.key.slug,
        tier=unpaired.tier,
        corpus=unpaired.key.corpus,
        vocab_size=unpaired.key.vocab_size,
        boundary=unpaired.key.boundary,
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
        "bpe_headline_clearance": bpe["headline_clearance"],
        "unigram_headline_clearance": unigram["headline_clearance"],
        "headline_delta_f": payload["headline_delta_f"],
        "any_arm_unsafe": payload["any_arm_unsafe"],
        "both_arms_unsafe": payload["both_arms_unsafe"],
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
        "present_headline_clearance": present["headline_clearance"],
        "missing_arm": payload["missing_arm"],
        "unpaired_reason": payload["unpaired_reason"],
        "any_arm_unsafe": payload["any_arm_unsafe"],
    }


def _format_md_table(
    matched_rows: list[dict[str, Any]],
    unpaired_rows: list[dict[str, Any]],
    pending: list[str],
) -> str:
    lines: list[str] = ["## Matched pairs (ΔF reportable)", ""]
    lines.append("| pair | tier | V | corpus | bnd | F^BPE | F^UL | ΔF | flags |")
    lines.append("|---|---|--:|---|---|--:|--:|--:|---|")
    for row in matched_rows:
        flags: list[str] = []
        if row["both_arms_unsafe"]:
            flags.append("BOTH-UNSAFE")
        elif row["any_arm_unsafe"]:
            flags.append("one-unsafe")
        flag_text = ", ".join(flags) if flags else "ok"
        lines.append(
            f"| {row['pair_key']} | {row['tier']} | {row['vocab_size']} "
            f"| {row['corpus']} | {row['boundary']} "
            f"| {float(row['bpe_headline_clearance']):.4f} "
            f"| {float(row['unigram_headline_clearance']):.4f} "
            f"| {float(row['headline_delta_f']):+.4f} | {flag_text} |"
        )
    if unpaired_rows:
        lines.extend(["", "## Single-arm coordinates (ΔF undefined)", ""])
        lines.append("| pair | tier | V | corpus | bnd | arm | F | reason |")
        lines.append("|---|---|--:|---|---|---|--:|---|")
        lines.extend(
            f"| {row['pair_key']} | {row['tier']} | {row['vocab_size']} "
            f"| {row['corpus']} | {row['boundary']} | {row['present_arm']} "
            f"| {float(row['present_headline_clearance']):.4f} "
            f"| {row['unpaired_reason']} |"
            for row in unpaired_rows
        )
    if pending:
        lines.extend(
            ["", f"## Pending ({len(pending)} pair_keys without F95 inputs)", ""]
        )
        lines.extend(f"- {pk}" for pk in pending)
    return "\n".join(lines) + "\n"


def _table_extra(
    matched_rows: list[dict[str, Any]], unpaired_rows: list[dict[str, Any]]
) -> dict[str, object]:
    del unpaired_rows
    return {
        "flagged_pairs": [r["pair_key"] for r in matched_rows if r["any_arm_unsafe"]],
    }


def _spec() -> _deposit.DepositSpec:
    return _deposit.DepositSpec(
        schema_version=SCHEMA_VERSION,
        cell_dir=DEADZONE_CELL_DIR,
        table_json=DEADZONE_TABLE_JSON,
        table_md=DEADZONE_TABLE_MD,
        pair_provider=pair_all_cells,
        build_matched=_matched_pair_record,
        build_unpaired=_unpaired_record,
        row_matched=_table_row_matched,
        row_unpaired=_table_row_unpaired,
        format_md=_format_md_table,
        is_done=is_deadzone_done,
        table_extra=_table_extra,
    )


def deadzone_json_path(pair_key: str) -> Path:
    """Return the per-pair Deadzone JSON path for ``pair_key``."""
    return _deposit.json_path(DEADZONE_CELL_DIR, pair_key)


def write_deadzone_json(record: MatchedPairDeadzone | UnpairedDeadzone) -> Path:
    """Deposit ``record`` as a per-pair JSON; return its path."""
    return _deposit.write_json(DEADZONE_CELL_DIR, SCHEMA_VERSION, record)


def read_deadzone_json(pair_key: str) -> dict[str, object] | None:
    """Return the deposited Deadzone payload for ``pair_key``, or None if absent."""
    return _deposit.read_json(DEADZONE_CELL_DIR, pair_key)


def is_deadzone_done(pair_key: str) -> bool:
    """True iff a deposited Deadzone record exists whose referenced F95 SHAs match."""
    return _deposit.nested_is_done(read_deadzone_json(pair_key), _f95_block_fresh)


def deposit_pair(pair: MatchedPair) -> tuple[Path | None, str | None]:
    """Compute and deposit ``pair``; return ``(path, None)`` or ``(None, reason)``."""
    return _deposit.deposit_pair(_spec(), pair)


def deposit_unpaired(unpaired: UnpairedCell) -> tuple[Path | None, str | None]:
    """Compute and deposit ``unpaired``; return ``(path, None)`` or pending reason."""
    return _deposit.deposit_unpaired(_spec(), unpaired)


def build_deadzone_table() -> tuple[Path, Path]:
    """Aggregate every deposited Deadzone JSON into ``deadzone_table.{json,md}``.

    Walks the committed grid + extras via :func:`pair_all_cells`.
    """
    return _deposit.build_table(_spec())


def deposit_all(
    matched_pairs: Sequence[MatchedPair] | None = None,
    unpaired_cells: Sequence[UnpairedCell] | None = None,
    *,
    rebuild: bool = False,
    only_pair_keys: frozenset[str] | None = None,
) -> tuple[list[str], list[tuple[str, str]]]:
    """Deposit every Deadzone record across the committed grid + extras.

    ``pending_pairs`` carries ``(pair_key, reason)`` for pairs whose F95 inputs
    are not yet on disk (lenient policy — the caller decides whether to raise).
    """
    return _deposit.deposit_all(
        _spec(),
        matched_pairs,
        unpaired_cells,
        rebuild=rebuild,
        only_pair_keys=only_pair_keys,
    )


__all__ = [
    "DEADZONE_CELL_DIR",
    "DEADZONE_DATA_DIR",
    "DEADZONE_TABLE_JSON",
    "DEADZONE_TABLE_MD",
    "SCHEMA_VERSION",
    "build_deadzone_table",
    "deadzone_json_path",
    "deposit_all",
    "deposit_pair",
    "deposit_unpaired",
    "is_deadzone_done",
    "read_deadzone_json",
    "write_deadzone_json",
]
