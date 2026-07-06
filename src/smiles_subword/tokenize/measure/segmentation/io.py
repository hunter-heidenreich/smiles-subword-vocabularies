"""Deposit segmentation-entropy records and rebuild the aggregator table.

Per matched ``(V, corpus, boundary)`` coordinate: resolve both cells, compute
the Unigram arm's exact segmentation entropy on the corpus's held-out split (the
BPE arm is zero by construction — no encode), and write one per-pair JSON under
``results/data/segmentation/`` carrying mean entropy per molecule + per glyph
with 95% bootstrap CIs. Single-arm coordinates emit single-arm JSONs of the same
schema.

The deposit dispatch and table join are the shared engine in
:mod:`smiles_subword.tokenize.measure._deposit`; this module supplies the
record builders, row projection, Markdown columns, and the arm-freshness
predicate. The Unigram arm is validated on both ``training_corpus_sha`` and
``eval_split_sha``; the zero-by-construction BPE arm has no eval dependency, so
only its ``training_corpus_sha`` is checked — the inverse of Scaffold's freshness
logic.

The aggregator surfaces any matched pair whose BPE arm is not the structural
zero (``bpe_zero_violations``) — a self-check that the by-construction premise
held.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from smiles_subword.paths import RESULTS_DATA_DIR
from smiles_subword.tokenize.measure import _deposit
from smiles_subword.tokenize.measure._cellmeta import (
    cell_training_sha_fresh,
    resolve_cell_meta,
)
from smiles_subword.tokenize.measure._cells import eval_split_sha, load_cell_adapter
from smiles_subword.tokenize.measure._pairing import (
    MatchedPair,
    UnpairedCell,
    pair_all_cells,
)
from smiles_subword.tokenize.measure._tables import fmt_md
from smiles_subword.tokenize.measure.segmentation.math import (
    Arm,
    ArmSegmentation,
    MatchedPairSegmentation,
    UnpairedSegmentation,
    compute_bpe_arm_segmentation,
    compute_matched_pair_segmentation,
    compute_unpaired_segmentation,
)
from smiles_subword.tokenize.measure.segmentation.runner import run_arm_segmentation

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

SCHEMA_VERSION = 1

SEGMENTATION_DATA_DIR = RESULTS_DATA_DIR
SEGMENTATION_CELL_DIR = SEGMENTATION_DATA_DIR / "segmentation"
SEGMENTATION_TABLE_JSON = SEGMENTATION_DATA_DIR / "segmentation_table.json"
SEGMENTATION_TABLE_MD = SEGMENTATION_DATA_DIR / "segmentation_table.md"


def _arm_block_is_fresh(arm_block: object) -> bool:
    if not isinstance(arm_block, dict):
        return False
    cell = cell_training_sha_fresh(
        arm_block.get("cell_id"), arm_block.get("training_corpus_sha")
    )
    if cell is None:
        return False
    if arm_block.get("verified_by_construction"):
        return True
    corpus, _name = cell
    try:
        return eval_split_sha(corpus) == arm_block.get("eval_split_sha")
    except (FileNotFoundError, ValueError):
        return False


def _arm_from_cell(cell_id: str, arm: Arm) -> ArmSegmentation | str:
    """Compute one cell's arm; return an :class:`ArmSegmentation` or an error reason.

    BPE arms are the zero by construction — derived from ``meta.yaml``
    alone, with no adapter load and no held-out encode. Unigram arms load the
    adapter and stream the held-out split through :func:`run_arm_segmentation`.
    """
    fields = resolve_cell_meta(cell_id)
    if isinstance(fields, str):
        return fields
    if arm == "bpe":
        return compute_bpe_arm_segmentation(
            cell_id=cell_id,
            boundary=fields.boundary,
            training_corpus_sha=fields.training_corpus_sha,
        )
    try:
        adapter = load_cell_adapter(fields.corpus, fields.name)
    except FileNotFoundError as exc:
        return str(exc)
    return run_arm_segmentation(
        adapter,
        cell_id=cell_id,
        corpus=fields.corpus,
        arm="unigram",
        boundary=fields.boundary,
        training_corpus_sha=fields.training_corpus_sha,
        tokenizer_json=fields.artifact_dir / "tokenizer.json",
    )


def _matched_pair_record(pair: MatchedPair) -> MatchedPairSegmentation | str:
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
    return compute_matched_pair_segmentation(
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


def _unpaired_record(unpaired: UnpairedCell) -> UnpairedSegmentation | str:
    arm = _arm_from_cell(unpaired.cell_id, unpaired.arm)
    if isinstance(arm, str):
        return arm
    missing_arm: Arm = "unigram" if unpaired.arm == "bpe" else "bpe"
    return compute_unpaired_segmentation(
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
        "bpe_verified_zero": bool(bpe["verified_by_construction"])
        and bpe["total_entropy_nats"] == 0.0,
        "unigram_entropy_per_molecule": unigram["entropy_per_molecule_mean"],
        "unigram_entropy_per_glyph": unigram["entropy_per_glyph"],
        "unigram_live_molecules": unigram["n_molecules"],
        "delta_entropy_per_molecule": payload["delta_entropy_per_molecule"],
        "delta_entropy_per_glyph": payload["delta_entropy_per_glyph"],
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
        "present_entropy_per_molecule": present["entropy_per_molecule_mean"],
        "present_entropy_per_glyph": present["entropy_per_glyph"],
        "verified_by_construction": present["verified_by_construction"],
        "missing_arm": payload["missing_arm"],
        "unpaired_reason": payload["unpaired_reason"],
    }


def _format_md_table(
    matched_rows: list[dict[str, Any]],
    unpaired_rows: list[dict[str, Any]],
    pending: list[str],
) -> str:
    lines: list[str] = [
        "## Matched pairs (Unigram entropy; BPE zero by construction)",
        "",
        "| pair | tier | V | corpus | bnd | H/mol^UL | H/glyph^UL | BPE=0 |",
        "|---|---|--:|---|---|--:|--:|:--:|",
    ]
    lines.extend(
        f"| {row['pair_key']} | {row['tier']} | {row['vocab_size']} "
        f"| {row['corpus']} | {row['boundary']} "
        f"| {fmt_md(row['unigram_entropy_per_molecule'])} "
        f"| {fmt_md(row['unigram_entropy_per_glyph'])} "
        f"| {'✓' if row['bpe_verified_zero'] else '✗'} |"
        for row in matched_rows
    )
    if unpaired_rows:
        lines.extend(["", "## Single-arm coordinates", ""])
        lines.append(
            "| pair | tier | V | corpus | bnd | arm | H/mol | H/glyph | reason |"
        )
        lines.append("|---|---|--:|---|---|---|--:|--:|---|")
        lines.extend(
            f"| {row['pair_key']} | {row['tier']} | {row['vocab_size']} "
            f"| {row['corpus']} | {row['boundary']} | {row['present_arm']} "
            f"| {fmt_md(row['present_entropy_per_molecule'])} "
            f"| {fmt_md(row['present_entropy_per_glyph'])} "
            f"| {row['unpaired_reason']} |"
            for row in unpaired_rows
        )
    if pending:
        lines.extend(
            [
                "",
                f"## Pending ({len(pending)} pair_keys without Segmentation records)",
                "",
            ]
        )
        lines.extend(f"- {pk}" for pk in pending)
    return "\n".join(lines) + "\n"


def _table_extra(
    matched_rows: list[dict[str, Any]], unpaired_rows: list[dict[str, Any]]
) -> dict[str, object]:
    del unpaired_rows
    return {
        "bpe_zero_violations": [
            r["pair_key"] for r in matched_rows if not r["bpe_verified_zero"]
        ]
    }


def _spec() -> _deposit.DepositSpec:
    return _deposit.DepositSpec(
        schema_version=SCHEMA_VERSION,
        cell_dir=SEGMENTATION_CELL_DIR,
        table_json=SEGMENTATION_TABLE_JSON,
        table_md=SEGMENTATION_TABLE_MD,
        pair_provider=pair_all_cells,
        build_matched=_matched_pair_record,
        build_unpaired=_unpaired_record,
        row_matched=_table_row_matched,
        row_unpaired=_table_row_unpaired,
        format_md=_format_md_table,
        is_done=is_segmentation_done,
        table_extra=_table_extra,
    )


def segmentation_json_path(pair_key: str) -> Path:
    """Return the per-pair Segmentation JSON path for ``pair_key``."""
    return _deposit.json_path(SEGMENTATION_CELL_DIR, pair_key)


def write_segmentation_json(
    record: MatchedPairSegmentation | UnpairedSegmentation,
) -> Path:
    """Deposit ``record`` as a per-pair JSON; return its path."""
    return _deposit.write_json(SEGMENTATION_CELL_DIR, SCHEMA_VERSION, record)


def read_segmentation_json(pair_key: str) -> dict[str, object] | None:
    """Return the deposited Segmentation payload for ``pair_key``, or None if absent."""
    return _deposit.read_json(SEGMENTATION_CELL_DIR, pair_key)


def is_segmentation_done(pair_key: str) -> bool:
    """True iff a deposited Segmentation record exists whose upstream SHAs still match.

    The encoded (Unigram) arm is validated on ``training_corpus_sha`` +
    ``eval_split_sha``; the zero-by-construction BPE arm has no eval dependency,
    so only its ``training_corpus_sha`` is re-checked. Any drift triggers a
    re-deposit on the next sweep.
    """
    return _deposit.nested_is_done(
        read_segmentation_json(pair_key), _arm_block_is_fresh
    )


def deposit_pair(pair: MatchedPair) -> tuple[Path | None, str | None]:
    """Compute and deposit ``pair``; return ``(path, None)`` or ``(None, reason)``."""
    return _deposit.deposit_pair(_spec(), pair)


def deposit_unpaired(unpaired: UnpairedCell) -> tuple[Path | None, str | None]:
    """Compute and deposit ``unpaired``; return ``(path, None)`` or pending reason."""
    return _deposit.deposit_unpaired(_spec(), unpaired)


def build_segmentation_table() -> tuple[Path, Path]:
    """Aggregate every deposited JSON into ``segmentation_table.{json,md}``."""
    return _deposit.build_table(_spec())


def deposit_all(
    matched_pairs: Sequence[MatchedPair] | None = None,
    unpaired_cells: Sequence[UnpairedCell] | None = None,
    *,
    rebuild: bool = False,
    only_pair_keys: frozenset[str] | None = None,
) -> tuple[list[str], list[tuple[str, str]]]:
    """Deposit every Segmentation record across the committed grid + extras."""
    return _deposit.deposit_all(
        _spec(),
        matched_pairs,
        unpaired_cells,
        rebuild=rebuild,
        only_pair_keys=only_pair_keys,
    )


__all__ = [
    "SCHEMA_VERSION",
    "SEGMENTATION_CELL_DIR",
    "SEGMENTATION_DATA_DIR",
    "SEGMENTATION_TABLE_JSON",
    "SEGMENTATION_TABLE_MD",
    "build_segmentation_table",
    "deposit_all",
    "deposit_pair",
    "deposit_unpaired",
    "is_segmentation_done",
    "read_segmentation_json",
    "segmentation_json_path",
    "write_segmentation_json",
]
