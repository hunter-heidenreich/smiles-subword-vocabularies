"""Deposition + aggregator for functional-bond-locality records.

Per-pair JSONs land under ``results/data/fg_alignment/``, one file per
``pair_key``. The deposit dispatch and table join are the shared engine in
:mod:`smiles_subword.tokenize.measure._deposit`; this module supplies the
FG-alignment-specific record builders, row projection, and Markdown columns.

Locality is *held-out-evaluated* and *within-arm* (like Nestedness and
Closure): a single-arm coordinate deposits a real reading for its present arm,
so :func:`is_fg_alignment_done` uses its own within-arm branch rather than the
plain :func:`_deposit.nested_is_done`. :func:`build_fg_alignment_table` is a
pure on-disk join over the deposited JSONs, tolerant of missing pairs.
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
from smiles_subword.tokenize.measure.fg_alignment.math import (
    Arm,
    Boundary,
    MatchedPairFgAlignment,
    UnpairedFgAlignment,
    compute_unpaired_fg_alignment,
)
from smiles_subword.tokenize.measure.fg_alignment.runner import (
    run_pair_fg_alignment,
    run_single_arm_fg_alignment,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

SCHEMA_VERSION = 1

FG_ALIGNMENT_DATA_DIR = RESULTS_DATA_DIR
FG_ALIGNMENT_CELL_DIR = FG_ALIGNMENT_DATA_DIR / "fg_alignment"
FG_ALIGNMENT_TABLE_JSON = FG_ALIGNMENT_DATA_DIR / "fg_alignment_table.json"
FG_ALIGNMENT_TABLE_MD = FG_ALIGNMENT_DATA_DIR / "fg_alignment_table.md"


def _matched_pair_record(pair: MatchedPair) -> MatchedPairFgAlignment | str:
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
    return run_pair_fg_alignment(
        bpe_adapter,
        ul_adapter,
        pair_key=pair.key.slug,
        tier=pair.tier,
        corpus=pair.key.corpus,
        vocab_size=pair.key.vocab_size,
        boundary=boundary,
        bpe_cell_id=pair.bpe_cell_id,
        unigram_cell_id=pair.unigram_cell_id,
        bpe_name=bpe_info.name,
        unigram_name=ul_info.name,
        bpe_training_corpus_sha=bpe_info.training_corpus_sha,
        unigram_training_corpus_sha=ul_info.training_corpus_sha,
        extras_kind=pair.key.extras_kind,
        extras_label=pair.key.extras_label,
    )


def _unpaired_record(unpaired: UnpairedCell) -> UnpairedFgAlignment | str:
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
    present = run_single_arm_fg_alignment(
        adapter,
        arm=present_arm,
        corpus=unpaired.key.corpus,
        name=info.name,
        cell_id=unpaired.cell_id,
        boundary=boundary,
        training_corpus_sha=info.training_corpus_sha,
    )
    return compute_unpaired_fg_alignment(
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


def _class_loc(block: dict[str, Any], label: str) -> float | None:
    cl = block.get("class_locality")
    if not isinstance(cl, dict):
        return None
    return cl.get(label)


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
        "bpe_locality": bpe.get("locality"),
        "ul_locality": ul.get("locality"),
        "delta_locality": payload["delta_locality"],
        "bpe_carbonyl": _class_loc(bpe, "C=O"),
        "ul_carbonyl": _class_loc(ul, "C=O"),
        "bpe_nitrile": _class_loc(bpe, "C#N"),
        "ul_nitrile": _class_loc(ul, "C#N"),
        "n_bonds": bpe.get("n_bonds"),
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
        "present_locality": present.get("locality"),
        "present_carbonyl": _class_loc(present, "C=O"),
    }


def _format_md_table(
    matched_rows: list[dict[str, Any]],
    unpaired_rows: list[dict[str, Any]],
    pending: list[str],
) -> str:
    lines: list[str] = [
        "## Matched pairs (functional-bond locality)",
        "",
        "| pair | tier | V | corpus | bnd | loc^B | loc^U | Δloc | C=O^B | C=O^U |",
        "|---|---|--:|---|---|--:|--:|--:|--:|--:|",
    ]
    lines.extend(
        f"| {row['pair_key']} | {row['tier']} | {row['vocab_size']} "
        f"| {row['corpus']} | {row['boundary']} "
        f"| {fmt_md(row['bpe_locality'])} | {fmt_md(row['ul_locality'])} "
        f"| {fmt_md(row['delta_locality'])} "
        f"| {fmt_md(row['bpe_carbonyl'])} | {fmt_md(row['ul_carbonyl'])} |"
        for row in matched_rows
    )
    if unpaired_rows:
        lines.extend(["", "## Single-arm coordinates (present arm only)", ""])
        lines.append("| pair | tier | V | corpus | bnd | present | loc | C=O |")
        lines.append("|---|---|--:|---|---|---|--:|--:|")
        lines.extend(
            f"| {row['pair_key']} | {row['tier']} | {row['vocab_size']} "
            f"| {row['corpus']} | {row['boundary']} | {row['present_arm']} "
            f"| {fmt_md(row['present_locality'])} | {fmt_md(row['present_carbonyl'])} |"
            for row in unpaired_rows
        )
    if pending:
        lines.extend(["", f"## Pending ({len(pending)} pair_keys without records)", ""])
        lines.extend(f"- {pk}" for pk in pending)
    return "\n".join(lines) + "\n"


def _spec() -> _deposit.DepositSpec:
    return _deposit.DepositSpec(
        schema_version=SCHEMA_VERSION,
        cell_dir=FG_ALIGNMENT_CELL_DIR,
        table_json=FG_ALIGNMENT_TABLE_JSON,
        table_md=FG_ALIGNMENT_TABLE_MD,
        pair_provider=pair_all_cells,
        build_matched=_matched_pair_record,
        build_unpaired=_unpaired_record,
        row_matched=_table_row_matched,
        row_unpaired=_table_row_unpaired,
        format_md=_format_md_table,
        is_done=is_fg_alignment_done,
    )


def fg_alignment_json_path(pair_key: str) -> Path:
    """Return the per-pair FG-alignment JSON path for ``pair_key``."""
    return _deposit.json_path(FG_ALIGNMENT_CELL_DIR, pair_key)


def write_fg_alignment_json(
    record: MatchedPairFgAlignment | UnpairedFgAlignment,
) -> Path:
    """Deposit ``record`` as a per-pair JSON; return its path."""
    return _deposit.write_json(FG_ALIGNMENT_CELL_DIR, SCHEMA_VERSION, record)


def read_fg_alignment_json(pair_key: str) -> dict[str, object] | None:
    """Return the deposited FG-alignment payload for ``pair_key``, or None."""
    return _deposit.read_json(FG_ALIGNMENT_CELL_DIR, pair_key)


def is_fg_alignment_done(pair_key: str) -> bool:
    """True iff a deposited record exists whose per-arm SHAs still match.

    Each present arm block must agree with its cell's ``training_corpus_sha`` and
    the corpus's held-out ``eval_split_sha``; a reprocessed corpus or re-drawn
    split drifts a SHA and triggers a re-deposit on the next sweep. A single-arm
    coordinate gates on its present arm alone (it deposits a real within-arm
    reading), so this does not use the plain :func:`_deposit.nested_is_done`.
    """
    return _deposit.within_arm_is_done(
        read_fg_alignment_json(pair_key), _deposit.standard_arm_block_fresh
    )


def deposit_pair(pair: MatchedPair) -> tuple[Path | None, str | None]:
    """Compute and deposit ``pair``; return ``(path, None)`` or ``(None, reason)``."""
    return _deposit.deposit_pair(_spec(), pair)


def deposit_unpaired(unpaired: UnpairedCell) -> tuple[Path | None, str | None]:
    """Compute and deposit ``unpaired``; return ``(path, None)`` or pending reason."""
    return _deposit.deposit_unpaired(_spec(), unpaired)


def build_fg_alignment_table() -> tuple[Path, Path]:
    """Aggregate every deposited JSON into ``fg_alignment_table.{json,md}``."""
    return _deposit.build_table(_spec())


def deposit_all(
    matched_pairs: Sequence[MatchedPair] | None = None,
    unpaired_cells: Sequence[UnpairedCell] | None = None,
    *,
    rebuild: bool = False,
    only_pair_keys: frozenset[str] | None = None,
) -> tuple[list[str], list[tuple[str, str]]]:
    """Deposit every FG-alignment record across the committed grid + extras."""
    return _deposit.deposit_all(
        _spec(),
        matched_pairs,
        unpaired_cells,
        rebuild=rebuild,
        only_pair_keys=only_pair_keys,
    )


__all__ = [
    "FG_ALIGNMENT_CELL_DIR",
    "FG_ALIGNMENT_DATA_DIR",
    "FG_ALIGNMENT_TABLE_JSON",
    "FG_ALIGNMENT_TABLE_MD",
    "SCHEMA_VERSION",
    "build_fg_alignment_table",
    "deposit_all",
    "deposit_pair",
    "deposit_unpaired",
    "fg_alignment_json_path",
    "is_fg_alignment_done",
    "read_fg_alignment_json",
    "write_fg_alignment_json",
]
