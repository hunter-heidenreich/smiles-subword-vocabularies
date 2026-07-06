"""Deposition + aggregator for non-canonicity records.

Per-pair JSONs land under ``results/data/noncanon/``, one file per ``pair_key``.
The deposit dispatch and table join are the shared engine in
:mod:`smiles_subword.tokenize.measure._deposit`; this module supplies the record
builders, row projection, and Markdown columns.

Non-canonicity is held-out-evaluated (a seeded subsample), so
:func:`is_noncanon_done` validates each present arm block against both its cell's
``training_corpus_sha`` and the corpus's ``eval_split_sha``; being within-arm, a
single-arm coordinate deposits a real reading.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from smiles_subword.paths import RESULTS_DATA_DIR
from smiles_subword.tokenize.measure import _deposit
from smiles_subword.tokenize.measure._cellmeta import arm_info
from smiles_subword.tokenize.measure._cells import load_cell_adapter
from smiles_subword.tokenize.measure._pairing import (
    MatchedPair,
    UnpairedCell,
    pair_all_cells,
)
from smiles_subword.tokenize.measure._tables import fmt_md
from smiles_subword.tokenize.measure.noncanon.math import (
    AXES,
    Arm,
    Boundary,
    MatchedPairNoncanon,
    UnpairedNoncanon,
    compute_unpaired_noncanon,
)
from smiles_subword.tokenize.measure.noncanon.runner import (
    run_pair_noncanon,
    run_single_arm_noncanon,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

SCHEMA_VERSION = 1

NONCANON_DATA_DIR = RESULTS_DATA_DIR
NONCANON_CELL_DIR = NONCANON_DATA_DIR / "noncanon"
NONCANON_TABLE_JSON = NONCANON_DATA_DIR / "noncanon_table.json"
NONCANON_TABLE_MD = NONCANON_DATA_DIR / "noncanon_table.md"


def _matched_pair_record(pair: MatchedPair) -> MatchedPairNoncanon | str:
    bpe_info = arm_info(pair.bpe_cell_id)
    if isinstance(bpe_info, str):
        return bpe_info
    ul_info = arm_info(pair.unigram_cell_id)
    if isinstance(ul_info, str):
        return ul_info
    boundary: Boundary = "mb" if pair.key.boundary == "mb" else "nmb"
    try:
        bpe_adapter = load_cell_adapter(pair.key.corpus, bpe_info.name)
        ul_adapter = load_cell_adapter(pair.key.corpus, ul_info.name)
    except FileNotFoundError as exc:
        return str(exc)
    return run_pair_noncanon(
        bpe_adapter,
        ul_adapter,
        pair_key=pair.key.slug,
        tier=pair.tier,
        corpus=pair.key.corpus,
        vocab_size=pair.key.vocab_size,
        boundary=boundary,
        bpe_cell_id=pair.bpe_cell_id,
        unigram_cell_id=pair.unigram_cell_id,
        bpe_training_corpus_sha=bpe_info.training_corpus_sha,
        unigram_training_corpus_sha=ul_info.training_corpus_sha,
        extras_kind=pair.key.extras_kind,
        extras_label=pair.key.extras_label,
    )


def _unpaired_record(unpaired: UnpairedCell) -> UnpairedNoncanon | str:
    info = arm_info(unpaired.cell_id)
    if isinstance(info, str):
        return info
    boundary: Boundary = "mb" if unpaired.key.boundary == "mb" else "nmb"
    present_arm: Arm = "bpe" if unpaired.arm == "bpe" else "unigram"
    missing_arm: Arm = "unigram" if present_arm == "bpe" else "bpe"
    try:
        adapter = load_cell_adapter(unpaired.key.corpus, info.name)
    except FileNotFoundError as exc:
        return str(exc)
    present = run_single_arm_noncanon(
        adapter,
        arm=present_arm,
        corpus=unpaired.key.corpus,
        cell_id=unpaired.cell_id,
        boundary=boundary,
        training_corpus_sha=info.training_corpus_sha,
    )
    return compute_unpaired_noncanon(
        present,
        pair_key=unpaired.key.slug,
        tier=unpaired.tier,
        corpus=unpaired.key.corpus,
        vocab_size=unpaired.key.vocab_size,
        boundary=boundary,
        extras_kind=unpaired.key.extras_kind,
        extras_label=unpaired.key.extras_label,
        present_arm=present_arm,
        missing_arm=missing_arm,
        unpaired_reason=unpaired.reason,
    )


def _block(payload: dict[str, Any], arm: str) -> dict[str, Any]:
    block = payload.get(arm)
    return block if isinstance(block, dict) else {}


def _bag(block: dict[str, Any], axis: str) -> float | None:
    axes = block.get("axes")
    if not isinstance(axes, dict) or axis not in axes:
        return None
    return axes[axis].get("bag_instab")


def _table_row_matched(payload: dict[str, Any]) -> dict[str, Any]:
    bpe = _block(payload, "bpe")
    ul = _block(payload, "unigram")
    row: dict[str, Any] = {
        "pair_key": payload["pair_key"],
        "pair_status": payload["pair_status"],
        "tier": payload["tier"],
        "extras_kind": payload.get("extras_kind"),
        "extras_label": payload.get("extras_label"),
        "corpus": payload["corpus"],
        "vocab_size": payload["vocab_size"],
        "boundary": payload["boundary"],
        "gap_canon": payload["gap_canon"],
        "gap_rand": payload["gap_rand"],
        "n_molecules": bpe.get("n_molecules"),
    }
    for axis in AXES:
        row[f"bpe_bag_{axis}"] = _bag(bpe, axis)
        row[f"ul_bag_{axis}"] = _bag(ul, axis)
    return row


def _table_row_unpaired(payload: dict[str, Any]) -> dict[str, Any]:
    present_arm = payload["present_arm"]
    present = _block(payload, present_arm)
    row: dict[str, Any] = {
        "pair_key": payload["pair_key"],
        "pair_status": payload["pair_status"],
        "tier": payload["tier"],
        "extras_kind": payload.get("extras_kind"),
        "extras_label": payload.get("extras_label"),
        "corpus": payload["corpus"],
        "vocab_size": payload["vocab_size"],
        "boundary": payload["boundary"],
        "present_arm": present_arm,
        "missing_arm": payload["missing_arm"],
        "unpaired_reason": payload["unpaired_reason"],
    }
    for axis in AXES:
        row[f"present_bag_{axis}"] = _bag(present, axis)
    return row


def _format_md_table(
    matched_rows: list[dict[str, Any]],
    unpaired_rows: list[dict[str, Any]],
    pending: list[str],
) -> str:
    lines: list[str] = [
        "## Matched pairs (non-canonicity bag-instability)",
        "",
        "| pair | tier | V | corpus | bnd | rand^B | rand^U | kek^B | kek^U "
        "| eH^B | eH^U | ob^B | ob^U | gap_c | gap_r |",
        "|---|---|--:|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    lines.extend(
        f"| {row['pair_key']} | {row['tier']} | {row['vocab_size']} "
        f"| {row['corpus']} | {row['boundary']} "
        f"| {fmt_md(row['bpe_bag_random'])} | {fmt_md(row['ul_bag_random'])} "
        f"| {fmt_md(row['bpe_bag_kekule'])} | {fmt_md(row['ul_bag_kekule'])} "
        f"| {fmt_md(row['bpe_bag_explicitH'])} | {fmt_md(row['ul_bag_explicitH'])} "
        f"| {fmt_md(row['bpe_bag_obcanon'])} | {fmt_md(row['ul_bag_obcanon'])} "
        f"| {fmt_md(row['gap_canon'])} | {fmt_md(row['gap_rand'])} |"
        for row in matched_rows
    )
    if unpaired_rows:
        lines.extend(["", "## Single-arm coordinates (present arm only)", ""])
        lines.append("| pair | tier | V | corpus | bnd | present | rand | eH |")
        lines.append("|---|---|--:|---|---|---|--:|--:|")
        lines.extend(
            f"| {row['pair_key']} | {row['tier']} | {row['vocab_size']} "
            f"| {row['corpus']} | {row['boundary']} | {row['present_arm']} "
            f"| {fmt_md(row['present_bag_random'])} "
            f"| {fmt_md(row['present_bag_explicitH'])} |"
            for row in unpaired_rows
        )
    if pending:
        lines.extend(["", f"## Pending ({len(pending)} pair_keys without records)", ""])
        lines.extend(f"- {pk}" for pk in pending)
    return "\n".join(lines) + "\n"


def _spec() -> _deposit.DepositSpec:
    return _deposit.DepositSpec(
        schema_version=SCHEMA_VERSION,
        cell_dir=NONCANON_CELL_DIR,
        table_json=NONCANON_TABLE_JSON,
        table_md=NONCANON_TABLE_MD,
        pair_provider=pair_all_cells,
        build_matched=_matched_pair_record,
        build_unpaired=_unpaired_record,
        row_matched=_table_row_matched,
        row_unpaired=_table_row_unpaired,
        format_md=_format_md_table,
        is_done=is_noncanon_done,
    )


def noncanon_json_path(pair_key: str) -> Path:
    """Return the per-pair non-canonicity JSON path for ``pair_key``."""
    return _deposit.json_path(NONCANON_CELL_DIR, pair_key)


def write_noncanon_json(record: MatchedPairNoncanon | UnpairedNoncanon) -> Path:
    """Deposit ``record`` as a per-pair JSON; return its path."""
    return _deposit.write_json(NONCANON_CELL_DIR, SCHEMA_VERSION, record)


def read_noncanon_json(pair_key: str) -> dict[str, object] | None:
    """Return the deposited non-canonicity payload for ``pair_key``, or None."""
    return _deposit.read_json(NONCANON_CELL_DIR, pair_key)


def is_noncanon_done(pair_key: str) -> bool:
    """True iff a deposited record exists whose per-arm SHAs still match.

    Within-arm and single-arm-real, so this does not use the plain nested-block
    loop: a single-arm coordinate must carry its present arm block, and a matched
    record must carry both (a missing block is stale, not skipped).
    """
    return _deposit.within_arm_is_done(
        read_noncanon_json(pair_key), _deposit.standard_arm_block_fresh
    )


def deposit_pair(pair: MatchedPair) -> tuple[Path | None, str | None]:
    """Compute and deposit ``pair``; return ``(path, None)`` or ``(None, reason)``."""
    return _deposit.deposit_pair(_spec(), pair)


def deposit_unpaired(unpaired: UnpairedCell) -> tuple[Path | None, str | None]:
    """Compute and deposit ``unpaired``; return ``(path, None)`` or pending reason."""
    return _deposit.deposit_unpaired(_spec(), unpaired)


def build_noncanon_table() -> tuple[Path, Path]:
    """Aggregate every deposited JSON into ``noncanon_table.{json,md}``."""
    return _deposit.build_table(_spec())


def deposit_all(
    matched_pairs: Sequence[MatchedPair] | None = None,
    unpaired_cells: Sequence[UnpairedCell] | None = None,
    *,
    rebuild: bool = False,
    only_pair_keys: frozenset[str] | None = None,
) -> tuple[list[str], list[tuple[str, str]]]:
    """Deposit every non-canonicity record across the committed grid + extras."""
    return _deposit.deposit_all(
        _spec(),
        matched_pairs,
        unpaired_cells,
        rebuild=rebuild,
        only_pair_keys=only_pair_keys,
    )


__all__ = [
    "NONCANON_CELL_DIR",
    "NONCANON_DATA_DIR",
    "NONCANON_TABLE_JSON",
    "NONCANON_TABLE_MD",
    "SCHEMA_VERSION",
    "build_noncanon_table",
    "deposit_all",
    "deposit_pair",
    "deposit_unpaired",
    "is_noncanon_done",
    "noncanon_json_path",
    "read_noncanon_json",
    "write_noncanon_json",
]
