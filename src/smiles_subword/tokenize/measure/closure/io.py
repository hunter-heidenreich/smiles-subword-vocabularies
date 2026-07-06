"""Deposition + aggregator for compositional-closure records.

Per-pair JSONs under ``results/data/closure/``, one per ``pair_key``. The
deposit dispatch and table join are the shared :mod:`..._deposit` engine; this
module supplies the record builders, freshness predicate, row projection, and
Markdown columns.

Vocabulary-only, so :func:`is_closure_done` keys invalidation on
``training_corpus_sha`` alone (each arm's ``meta.yaml``); no ``eval_split_sha``.
Within-arm, so a single-arm coordinate deposits a real reading for its present
arm. :func:`build_closure_table` is a pure on-disk join, tolerant of missing
pairs.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from smiles_subword.paths import RESULTS_DATA_DIR
from smiles_subword.tokenize.measure import _deposit
from smiles_subword.tokenize.measure._cellmeta import (
    arm_info,
    cell_training_sha_fresh,
)
from smiles_subword.tokenize.measure._pairing import (
    MatchedPair,
    UnpairedCell,
    pair_all_cells,
)
from smiles_subword.tokenize.measure._tables import fmt_md
from smiles_subword.tokenize.measure.closure.math import (
    Arm,
    ArmClosure,
    Boundary,
    MatchedPairClosure,
    UnpairedClosure,
    compute_matched_pair_closure,
    compute_unpaired_closure,
)
from smiles_subword.tokenize.measure.closure.runner import run_arm_closure

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

SCHEMA_VERSION = 1

CLOSURE_DATA_DIR = RESULTS_DATA_DIR
CLOSURE_CELL_DIR = CLOSURE_DATA_DIR / "closure"
CLOSURE_TABLE_JSON = CLOSURE_DATA_DIR / "closure_table.json"
CLOSURE_TABLE_MD = CLOSURE_DATA_DIR / "closure_table.md"


def _arm_block_is_fresh(arm_block: object) -> bool:
    if not isinstance(arm_block, dict):
        return False
    return (
        cell_training_sha_fresh(
            arm_block.get("cell_id"), arm_block.get("training_corpus_sha")
        )
        is not None
    )


def is_closure_done(pair_key: str) -> bool:
    """True iff a deposited record exists whose per-arm corpus SHAs still match.

    Vocabulary-only: fresh iff every present arm block's ``training_corpus_sha``
    still equals its cell's ``meta.yaml`` (no held-out split to validate).
    """
    return _deposit.within_arm_is_done(read_closure_json(pair_key), _arm_block_is_fresh)


def _run_arm(
    cell_id: str, arm: Arm, boundary: Boundary, vocab_size: int
) -> ArmClosure | str:
    info = arm_info(cell_id)
    if isinstance(info, str):
        return info
    corpus, _, _name = cell_id.partition("__")
    try:
        return run_arm_closure(
            cell_id=cell_id,
            corpus=corpus,
            name=info.name,
            arm=arm,
            boundary=boundary,
            vocab_size=vocab_size,
            training_corpus_sha=info.training_corpus_sha,
        )
    except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError) as exc:
        return f"{cell_id}: {exc}"


def _matched_pair_record(pair: MatchedPair) -> MatchedPairClosure | str:
    boundary: Boundary = "mb" if pair.key.boundary == "mb" else "nmb"
    bpe = _run_arm(pair.bpe_cell_id, "bpe", boundary, pair.key.vocab_size)
    if isinstance(bpe, str):
        return bpe
    unigram = _run_arm(pair.unigram_cell_id, "unigram", boundary, pair.key.vocab_size)
    if isinstance(unigram, str):
        return unigram
    return compute_matched_pair_closure(
        bpe,
        unigram,
        pair_key=pair.key.slug,
        tier=pair.tier,
        corpus=pair.key.corpus,
        vocab_size=pair.key.vocab_size,
        boundary=boundary,
        extras_kind=pair.key.extras_kind,
        extras_label=pair.key.extras_label,
    )


def _unpaired_record(unpaired: UnpairedCell) -> UnpairedClosure | str:
    boundary: Boundary = "mb" if unpaired.key.boundary == "mb" else "nmb"
    present_arm: Arm = "bpe" if unpaired.arm == "bpe" else "unigram"
    missing_arm: Arm = "unigram" if present_arm == "bpe" else "bpe"
    present = _run_arm(unpaired.cell_id, present_arm, boundary, unpaired.key.vocab_size)
    if isinstance(present, str):
        return present
    return compute_unpaired_closure(
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


def _table_row_matched(payload: dict[str, Any]) -> dict[str, Any]:
    bpe = _block(payload, "bpe")
    ul = _block(payload, "unigram")
    return {
        "pair_key": payload["pair_key"],
        "pair_status": payload["pair_status"],
        "tier": payload["tier"],
        "extras_kind": payload.get("extras_kind"),
        "extras_label": payload.get("extras_label"),
        "corpus": payload["corpus"],
        "vocab_size": payload["vocab_size"],
        "boundary": payload["boundary"],
        "bpe_c_bin": bpe.get("c_bin"),
        "bpe_c_orph": bpe.get("c_orph"),
        "bpe_c_full": bpe.get("c_full"),
        "bpe_n_multi": bpe.get("n_multi"),
        "ul_c_bin": ul.get("c_bin"),
        "ul_c_orph": ul.get("c_orph"),
        "ul_c_full": ul.get("c_full"),
        "ul_n_multi": ul.get("n_multi"),
        "delta_c_bin": payload["delta_c_bin"],
        "delta_c_orph": payload["delta_c_orph"],
        "delta_c_full": payload["delta_c_full"],
    }


def _table_row_unpaired(payload: dict[str, Any]) -> dict[str, Any]:
    present_arm = payload["present_arm"]
    present = _block(payload, present_arm)
    return {
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
        "present_c_bin": present.get("c_bin"),
        "present_c_orph": present.get("c_orph"),
        "present_c_full": present.get("c_full"),
    }


def _format_md_table(
    matched_rows: list[dict[str, Any]],
    unpaired_rows: list[dict[str, Any]],
    pending: list[str],
) -> str:
    lines: list[str] = [
        "## Matched pairs (compositional closure)",
        "",
        "| pair | tier | V | corpus | bnd | c_bin^UL | Δc_bin | c_orph^UL "
        "| c_full^BPE | c_full^UL |",
        "|---|---|--:|---|---|--:|--:|--:|--:|--:|",
    ]
    lines.extend(
        f"| {row['pair_key']} | {row['tier']} | {row['vocab_size']} "
        f"| {row['corpus']} | {row['boundary']} "
        f"| {fmt_md(row['ul_c_bin'])} | {fmt_md(row['delta_c_bin'])} "
        f"| {fmt_md(row['ul_c_orph'])} | {fmt_md(row['bpe_c_full'])} "
        f"| {fmt_md(row['ul_c_full'])} |"
        for row in matched_rows
    )
    if unpaired_rows:
        lines.extend(["", "## Single-arm coordinates (present arm only)", ""])
        lines.append("| pair | tier | V | corpus | bnd | present | c_bin | c_orph |")
        lines.append("|---|---|--:|---|---|---|--:|--:|")
        lines.extend(
            f"| {row['pair_key']} | {row['tier']} | {row['vocab_size']} "
            f"| {row['corpus']} | {row['boundary']} | {row['present_arm']} "
            f"| {fmt_md(row['present_c_bin'])} | {fmt_md(row['present_c_orph'])} |"
            for row in unpaired_rows
        )
    if pending:
        lines.extend(["", f"## Pending ({len(pending)} pair_keys without records)", ""])
        lines.extend(f"- {pk}" for pk in pending)
    return "\n".join(lines) + "\n"


def _spec() -> _deposit.DepositSpec:
    return _deposit.DepositSpec(
        schema_version=SCHEMA_VERSION,
        cell_dir=CLOSURE_CELL_DIR,
        table_json=CLOSURE_TABLE_JSON,
        table_md=CLOSURE_TABLE_MD,
        pair_provider=pair_all_cells,
        build_matched=_matched_pair_record,
        build_unpaired=_unpaired_record,
        row_matched=_table_row_matched,
        row_unpaired=_table_row_unpaired,
        format_md=_format_md_table,
        is_done=is_closure_done,
    )


def closure_json_path(pair_key: str) -> Path:
    """Return the per-pair Closure JSON path for ``pair_key``."""
    return _deposit.json_path(CLOSURE_CELL_DIR, pair_key)


def write_closure_json(record: MatchedPairClosure | UnpairedClosure) -> Path:
    """Deposit ``record`` as a per-pair JSON; return its path."""
    return _deposit.write_json(CLOSURE_CELL_DIR, SCHEMA_VERSION, record)


def read_closure_json(pair_key: str) -> dict[str, object] | None:
    """Return the deposited Closure payload for ``pair_key``, or None if absent."""
    return _deposit.read_json(CLOSURE_CELL_DIR, pair_key)


def deposit_pair(pair: MatchedPair) -> tuple[Path | None, str | None]:
    """Compute and deposit ``pair``; return ``(path, None)`` or ``(None, reason)``."""
    return _deposit.deposit_pair(_spec(), pair)


def deposit_unpaired(unpaired: UnpairedCell) -> tuple[Path | None, str | None]:
    """Compute and deposit ``unpaired``; return ``(path, None)`` or pending reason."""
    return _deposit.deposit_unpaired(_spec(), unpaired)


def build_closure_table() -> tuple[Path, Path]:
    """Aggregate every deposited Closure JSON into ``closure_table.{json,md}``."""
    return _deposit.build_table(_spec())


def deposit_all(
    matched_pairs: Sequence[MatchedPair] | None = None,
    unpaired_cells: Sequence[UnpairedCell] | None = None,
    *,
    rebuild: bool = False,
    only_pair_keys: frozenset[str] | None = None,
) -> tuple[list[str], list[tuple[str, str]]]:
    """Deposit every Closure record across the committed grid + extras."""
    return _deposit.deposit_all(
        _spec(),
        matched_pairs,
        unpaired_cells,
        rebuild=rebuild,
        only_pair_keys=only_pair_keys,
    )


__all__ = [
    "CLOSURE_CELL_DIR",
    "CLOSURE_DATA_DIR",
    "CLOSURE_TABLE_JSON",
    "CLOSURE_TABLE_MD",
    "SCHEMA_VERSION",
    "build_closure_table",
    "closure_json_path",
    "deposit_all",
    "deposit_pair",
    "deposit_unpaired",
    "is_closure_done",
    "read_closure_json",
    "write_closure_json",
]
