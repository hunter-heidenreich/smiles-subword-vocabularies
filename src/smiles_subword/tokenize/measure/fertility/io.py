"""Deposition + aggregator for fertility records.

Per-pair JSONs land under ``results/data/fertility/``, one file per ``pair_key``.
The deposit dispatch and table join are the shared engine in
:mod:`smiles_subword.tokenize.measure._deposit`; this module supplies the
Fertility-specific record builders, row projection, and Markdown columns.

:func:`is_fertility_done` gates resume on both ``training_corpus_sha`` (each
cell's ``meta.yaml``) and ``eval_split_sha`` (the corpus's held-out test-split
MANIFEST). The aggregator :func:`build_fertility_table` is a pure on-disk join
that tolerates missing pairs and surfaces any matched pair whose two arms
disagree on ``total_glyphs`` (the model-independent invariant).
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
from smiles_subword.tokenize.measure.fertility.math import (
    Arm,
    ArmFertility,
    MatchedPairFertility,
    UnpairedFertility,
    compute_matched_pair_fertility,
    compute_unpaired_fertility,
)
from smiles_subword.tokenize.measure.fertility.runner import run_arm_fertility

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

SCHEMA_VERSION = 1

FERTILITY_DATA_DIR = RESULTS_DATA_DIR
FERTILITY_CELL_DIR = FERTILITY_DATA_DIR / "fertility"
FERTILITY_TABLE_JSON = FERTILITY_DATA_DIR / "fertility_table.json"
FERTILITY_TABLE_MD = FERTILITY_DATA_DIR / "fertility_table.md"


def _arm_from_cell(cell_id: str, arm: Arm) -> ArmFertility | str:
    fields = resolve_cell_meta(cell_id)
    if isinstance(fields, str):
        return fields
    try:
        adapter = load_cell_adapter(fields.corpus, fields.name)
    except FileNotFoundError as exc:
        return str(exc)
    return run_arm_fertility(
        adapter,
        cell_id=cell_id,
        corpus=fields.corpus,
        name=fields.name,
        arm=arm,
        boundary=fields.boundary,
        training_corpus_sha=fields.training_corpus_sha,
    )


def _matched_pair_record(pair: MatchedPair) -> MatchedPairFertility | str:
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
    return compute_matched_pair_fertility(
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


def _unpaired_record(unpaired: UnpairedCell) -> UnpairedFertility | str:
    arm = _arm_from_cell(unpaired.cell_id, unpaired.arm)
    if isinstance(arm, str):
        return arm
    missing_arm: Arm = "unigram" if unpaired.arm == "bpe" else "bpe"
    return compute_unpaired_fertility(
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
        "bpe_fertility": bpe["fertility_mean"],
        "bpe_fertility_ci_lo": bpe["fertility_ci"][0],
        "bpe_fertility_ci_hi": bpe["fertility_ci"][1],
        "unigram_fertility": unigram["fertility_mean"],
        "unigram_fertility_ci_lo": unigram["fertility_ci"][0],
        "unigram_fertility_ci_hi": unigram["fertility_ci"][1],
        "delta_fertility": payload["delta_fertility"],
        "delta_fertility_relative": payload["delta_fertility_relative"],
        "bpe_glyphs_per_token": bpe["glyphs_per_token_mean"],
        "bpe_glyphs_per_token_ci_lo": bpe["glyphs_per_token_ci"][0],
        "bpe_glyphs_per_token_ci_hi": bpe["glyphs_per_token_ci"][1],
        "unigram_glyphs_per_token": unigram["glyphs_per_token_mean"],
        "unigram_glyphs_per_token_ci_lo": unigram["glyphs_per_token_ci"][0],
        "unigram_glyphs_per_token_ci_hi": unigram["glyphs_per_token_ci"][1],
        "delta_glyphs_per_token": payload["delta_glyphs_per_token"],
        "total_glyphs_consistent": payload["total_glyphs_consistent"],
        "total_glyphs_delta": payload["total_glyphs_delta"],
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
        "present_fertility": present["fertility_mean"],
        "present_glyphs_per_token": present["glyphs_per_token_mean"],
        "missing_arm": payload["missing_arm"],
        "unpaired_reason": payload["unpaired_reason"],
    }


def _format_md_table(
    matched_rows: list[dict[str, Any]],
    unpaired_rows: list[dict[str, Any]],
    pending: list[str],
) -> str:
    header = "| pair | tier | V | corpus | bnd | f^BPE | f^UL | Δf | relΔf"
    header += " | g/t^BPE | g/t^UL | Δg/t | glyph✓ |"
    lines: list[str] = [
        "## Matched pairs (Δfertility reportable)",
        "",
        header,
        "|---|---|--:|---|---|--:|--:|--:|--:|--:|--:|--:|:--:|",
    ]
    lines.extend(
        f"| {row['pair_key']} | {row['tier']} | {row['vocab_size']} "
        f"| {row['corpus']} | {row['boundary']} "
        f"| {fmt_md(row['bpe_fertility'])} | {fmt_md(row['unigram_fertility'])} "
        f"| {fmt_md(row['delta_fertility'], spec='+.4f')} "
        f"| {fmt_md(row['delta_fertility_relative'])} "
        f"| {fmt_md(row['bpe_glyphs_per_token'])} "
        f"| {fmt_md(row['unigram_glyphs_per_token'])} "
        f"| {fmt_md(row['delta_glyphs_per_token'], spec='+.4f')} "
        f"| {'✓' if row['total_glyphs_consistent'] else '✗'} |"
        for row in matched_rows
    )
    if unpaired_rows:
        lines.extend(["", "## Single-arm coordinates (Δ undefined)", ""])
        lines.append("| pair | tier | V | corpus | bnd | arm | f | g/t | reason |")
        lines.append("|---|---|--:|---|---|---|--:|--:|---|")
        lines.extend(
            f"| {row['pair_key']} | {row['tier']} | {row['vocab_size']} "
            f"| {row['corpus']} | {row['boundary']} | {row['present_arm']} "
            f"| {fmt_md(row['present_fertility'])} "
            f"| {fmt_md(row['present_glyphs_per_token'])} "
            f"| {row['unpaired_reason']} |"
            for row in unpaired_rows
        )
    if pending:
        lines.extend(
            ["", f"## Pending ({len(pending)} pair_keys without Fertility records)", ""]
        )
        lines.extend(f"- {pk}" for pk in pending)
    return "\n".join(lines) + "\n"


def _table_extra(
    matched_rows: list[dict[str, Any]], unpaired_rows: list[dict[str, Any]]
) -> dict[str, object]:
    del unpaired_rows
    return {
        "glyph_invariant_violations": [
            r["pair_key"] for r in matched_rows if not r["total_glyphs_consistent"]
        ]
    }


def _spec() -> _deposit.DepositSpec:
    return _deposit.DepositSpec(
        schema_version=SCHEMA_VERSION,
        cell_dir=FERTILITY_CELL_DIR,
        table_json=FERTILITY_TABLE_JSON,
        table_md=FERTILITY_TABLE_MD,
        pair_provider=lambda: pair_all_cells(include_large_v_anchor=True),
        build_matched=_matched_pair_record,
        build_unpaired=_unpaired_record,
        row_matched=_table_row_matched,
        row_unpaired=_table_row_unpaired,
        format_md=_format_md_table,
        is_done=is_fertility_done,
        table_extra=_table_extra,
    )


def fertility_json_path(pair_key: str) -> Path:
    """Return the per-pair Fertility JSON path for ``pair_key``."""
    return _deposit.json_path(FERTILITY_CELL_DIR, pair_key)


def write_fertility_json(record: MatchedPairFertility | UnpairedFertility) -> Path:
    """Deposit ``record`` as a per-pair JSON; return its path."""
    return _deposit.write_json(FERTILITY_CELL_DIR, SCHEMA_VERSION, record)


def read_fertility_json(pair_key: str) -> dict[str, object] | None:
    """Return the deposited Fertility payload for ``pair_key``, or None if absent."""
    return _deposit.read_json(FERTILITY_CELL_DIR, pair_key)


def is_fertility_done(pair_key: str) -> bool:
    """True iff a deposited Fertility record exists whose upstream SHAs still match."""
    return _deposit.nested_is_done(
        read_fertility_json(pair_key), _deposit.standard_arm_block_fresh
    )


def deposit_pair(pair: MatchedPair) -> tuple[Path | None, str | None]:
    """Compute and deposit ``pair``; return ``(path, None)`` or ``(None, reason)``."""
    return _deposit.deposit_pair(_spec(), pair)


def deposit_unpaired(unpaired: UnpairedCell) -> tuple[Path | None, str | None]:
    """Compute and deposit ``unpaired``; return ``(path, None)`` or pending reason."""
    return _deposit.deposit_unpaired(_spec(), unpaired)


def build_fertility_table() -> tuple[Path, Path]:
    """Aggregate every deposited Fertility JSON into ``fertility_table.{json,md}``."""
    return _deposit.build_table(_spec())


def deposit_all(
    matched_pairs: Sequence[MatchedPair] | None = None,
    unpaired_cells: Sequence[UnpairedCell] | None = None,
    *,
    rebuild: bool = False,
    only_pair_keys: frozenset[str] | None = None,
) -> tuple[list[str], list[tuple[str, str]]]:
    """Deposit every Fertility record across the committed grid + extras."""
    return _deposit.deposit_all(
        _spec(),
        matched_pairs,
        unpaired_cells,
        rebuild=rebuild,
        only_pair_keys=only_pair_keys,
    )


__all__ = [
    "FERTILITY_CELL_DIR",
    "FERTILITY_DATA_DIR",
    "FERTILITY_TABLE_JSON",
    "FERTILITY_TABLE_MD",
    "SCHEMA_VERSION",
    "build_fertility_table",
    "deposit_all",
    "deposit_pair",
    "deposit_unpaired",
    "fertility_json_path",
    "is_fertility_done",
    "read_fertility_json",
    "write_fertility_json",
]
