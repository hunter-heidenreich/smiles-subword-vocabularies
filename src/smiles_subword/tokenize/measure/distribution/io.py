"""Deposit Distribution records and aggregate the table.

Per matched ``(V, corpus, boundary)`` coordinate, loads both tokenizers, streams
the held-out split once, and writes one per-pair JSON under
``results/data/distribution/`` carrying ``D``, entropy ``η``, Rényi (α=2.5), the
live-token count, and 95% bootstrap CIs. Single-arm coordinates (ZINC-22 BPE
``V=2048`` conditional branch and the four single-arm-knob extras) emit
single-arm JSONs of the same schema. The deposit dispatch and table join are the
shared :mod:`..._deposit` engine; this module supplies the record builders, row
projection, and Markdown columns.

Idempotent: re-running skips pairs whose ``training_corpus_sha`` (cell
``meta.yaml``) and ``eval_split_sha`` (held-out split MANIFEST) still match;
``--rebuild`` forces recompute. The aggregator surfaces any matched pair whose
arms disagree on ``v_effective`` (the ``ΔD`` dead-glyph cancellation premise).
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
from smiles_subword.tokenize.measure.distribution.math import (
    Arm,
    ArmDistribution,
    MatchedPairDistribution,
    UnpairedDistribution,
    compute_matched_pair_distribution,
    compute_unpaired_distribution,
)
from smiles_subword.tokenize.measure.distribution.runner import (
    collect_all_special_ids,
    run_arm_distribution,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

SCHEMA_VERSION = 1

DISTRIBUTION_DATA_DIR = RESULTS_DATA_DIR
DISTRIBUTION_CELL_DIR = DISTRIBUTION_DATA_DIR / "distribution"
DISTRIBUTION_TABLE_JSON = DISTRIBUTION_DATA_DIR / "distribution_table.json"
DISTRIBUTION_TABLE_MD = DISTRIBUTION_DATA_DIR / "distribution_table.md"


def _arm_from_cell(
    cell_id: str,
    arm: Arm,
    *,
    target_vocab_size: int,
    use_realized_v: bool = False,
) -> ArmDistribution | str:
    """Encode one cell's arm, normalizing by the nominal target ``|V|``.

    ``target_vocab_size`` is the coordinate's nominal ``V`` (identical across
    both arms, so the ``ΔD`` dead-glyph cancellation premise holds).
    ``use_realized_v`` switches to the realized non-special vocabulary count for
    the single-arm merge-exhaustion H-anchor, whose realized vocabulary falls
    short of its nominal target.
    """
    fields = resolve_cell_meta(cell_id)
    if isinstance(fields, str):
        return fields
    try:
        adapter = load_cell_adapter(fields.corpus, fields.name)
    except FileNotFoundError as exc:
        return str(exc)
    special_ids = collect_all_special_ids(adapter, fields.artifact_dir)
    if use_realized_v:
        in_vocab_specials = sum(1 for s in special_ids if s < adapter.vocab_size)
        v_effective = max(1, adapter.vocab_size - in_vocab_specials)
    else:
        v_effective = target_vocab_size
    return run_arm_distribution(
        adapter,
        cell_id=cell_id,
        corpus=fields.corpus,
        arm=arm,
        boundary=fields.boundary,
        v_effective=v_effective,
        special_ids=special_ids,
        training_corpus_sha=fields.training_corpus_sha,
    )


def _matched_pair_record(pair: MatchedPair) -> MatchedPairDistribution | str:
    target_v = pair.key.vocab_size
    bpe_arm = _arm_from_cell(pair.bpe_cell_id, "bpe", target_vocab_size=target_v)
    if isinstance(bpe_arm, str):
        return bpe_arm
    unigram_arm = _arm_from_cell(
        pair.unigram_cell_id, "unigram", target_vocab_size=target_v
    )
    if isinstance(unigram_arm, str):
        return unigram_arm
    if bpe_arm.boundary != unigram_arm.boundary:
        return (
            f"arm boundary mismatch for {pair.key.slug}: "
            f"bpe={bpe_arm.boundary!r}, unigram={unigram_arm.boundary!r}"
        )
    return compute_matched_pair_distribution(
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


def _unpaired_record(unpaired: UnpairedCell) -> UnpairedDistribution | str:
    use_realized_v = unpaired.key.extras_kind == "merge_exhaustion"
    arm = _arm_from_cell(
        unpaired.cell_id,
        unpaired.arm,
        target_vocab_size=unpaired.key.vocab_size,
        use_realized_v=use_realized_v,
    )
    if isinstance(arm, str):
        return arm
    missing_arm: Arm = "unigram" if unpaired.arm == "bpe" else "bpe"
    return compute_unpaired_distribution(
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
        "bpe_d": bpe["d"],
        "bpe_d_ci_lo": bpe["d_ci"][0],
        "bpe_d_ci_hi": bpe["d_ci"][1],
        "unigram_d": unigram["d"],
        "unigram_d_ci_lo": unigram["d_ci"][0],
        "unigram_d_ci_hi": unigram["d_ci"][1],
        "delta_d": payload["delta_d"],
        "abs_delta_d": payload["abs_delta_d"],
        "delta_d_exceeds_threshold": payload["delta_d_exceeds_threshold"],
        "bpe_eta": bpe["eta"],
        "bpe_eta_ci_lo": bpe["eta_ci"][0],
        "bpe_eta_ci_hi": bpe["eta_ci"][1],
        "unigram_eta": unigram["eta"],
        "unigram_eta_ci_lo": unigram["eta_ci"][0],
        "unigram_eta_ci_hi": unigram["eta_ci"][1],
        "bpe_renyi": bpe["renyi"],
        "bpe_renyi_ci_lo": bpe["renyi_ci"][0],
        "bpe_renyi_ci_hi": bpe["renyi_ci"][1],
        "unigram_renyi": unigram["renyi"],
        "unigram_renyi_ci_lo": unigram["renyi_ci"][0],
        "unigram_renyi_ci_hi": unigram["renyi_ci"][1],
        "bpe_live": bpe["live_token_count"],
        "unigram_live": unigram["live_token_count"],
        "v_effective_consistent": payload["v_effective_consistent"],
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
        "present_d": present["d"],
        "present_eta": present["eta"],
        "present_renyi": present["renyi"],
        "present_live": present["live_token_count"],
        "vocab_size_effective": present["v_effective"],
        "missing_arm": payload["missing_arm"],
        "unpaired_reason": payload["unpaired_reason"],
    }


def _format_md_table(
    matched_rows: list[dict[str, Any]],
    unpaired_rows: list[dict[str, Any]],
    pending: list[str],
) -> str:
    header = "| pair | tier | V | corpus | bnd | D^BPE | D^UL | ΔD | >.002"
    header += " | η^BPE | η^UL | R^BPE | R^UL | live^BPE | live^UL | veff✓ |"
    lines: list[str] = [
        "## Matched pairs (ΔD reportable)",
        "",
        header,
        "|---|---|--:|---|---|--:|--:|--:|:--:|--:|--:|--:|--:|--:|--:|:--:|",
    ]
    lines.extend(
        f"| {row['pair_key']} | {row['tier']} | {row['vocab_size']} "
        f"| {row['corpus']} | {row['boundary']} "
        f"| {fmt_md(row['bpe_d'])} | {fmt_md(row['unigram_d'])} "
        f"| {fmt_md(row['delta_d'], spec='+.4f')} "
        f"| {'✓' if row['delta_d_exceeds_threshold'] else '·'} "
        f"| {fmt_md(row['bpe_eta'])} | {fmt_md(row['unigram_eta'])} "
        f"| {fmt_md(row['bpe_renyi'])} | {fmt_md(row['unigram_renyi'])} "
        f"| {row['bpe_live']} | {row['unigram_live']} "
        f"| {'✓' if row['v_effective_consistent'] else '✗'} |"
        for row in matched_rows
    )
    if unpaired_rows:
        lines.extend(["", "## Single-arm coordinates (ΔD undefined)", ""])
        lines.append(
            "| pair | tier | V | corpus | bnd | arm | D | η | R | live | reason |"
        )
        lines.append("|---|---|--:|---|---|---|--:|--:|--:|--:|---|")
        lines.extend(
            f"| {row['pair_key']} | {row['tier']} | {row['vocab_size']} "
            f"| {row['corpus']} | {row['boundary']} | {row['present_arm']} "
            f"| {fmt_md(row['present_d'])} | {fmt_md(row['present_eta'])} "
            f"| {fmt_md(row['present_renyi'])} | {row['present_live']} "
            f"| {row['unpaired_reason']} |"
            for row in unpaired_rows
        )
    if pending:
        lines.extend(
            [
                "",
                f"## Pending ({len(pending)} pair_keys without Distribution records)",
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
        "v_effective_violations": [
            r["pair_key"] for r in matched_rows if not r["v_effective_consistent"]
        ]
    }


def _spec() -> _deposit.DepositSpec:
    return _deposit.DepositSpec(
        schema_version=SCHEMA_VERSION,
        cell_dir=DISTRIBUTION_CELL_DIR,
        table_json=DISTRIBUTION_TABLE_JSON,
        table_md=DISTRIBUTION_TABLE_MD,
        pair_provider=pair_all_cells,
        build_matched=_matched_pair_record,
        build_unpaired=_unpaired_record,
        row_matched=_table_row_matched,
        row_unpaired=_table_row_unpaired,
        format_md=_format_md_table,
        is_done=is_distribution_done,
        table_extra=_table_extra,
    )


def distribution_json_path(pair_key: str) -> Path:
    """Return the per-pair Distribution JSON path for ``pair_key``."""
    return _deposit.json_path(DISTRIBUTION_CELL_DIR, pair_key)


def write_distribution_json(
    record: MatchedPairDistribution | UnpairedDistribution,
) -> Path:
    """Deposit ``record`` as a per-pair JSON; return its path."""
    return _deposit.write_json(DISTRIBUTION_CELL_DIR, SCHEMA_VERSION, record)


def read_distribution_json(pair_key: str) -> dict[str, object] | None:
    """Return the deposited Distribution payload for ``pair_key``, or None if absent."""
    return _deposit.read_json(DISTRIBUTION_CELL_DIR, pair_key)


def is_distribution_done(pair_key: str) -> bool:
    """True iff a deposited Distribution record exists whose upstream SHAs match."""
    return _deposit.nested_is_done(
        read_distribution_json(pair_key), _deposit.standard_arm_block_fresh
    )


def deposit_pair(pair: MatchedPair) -> tuple[Path | None, str | None]:
    """Compute and deposit ``pair``; return ``(path, None)`` or ``(None, reason)``."""
    return _deposit.deposit_pair(_spec(), pair)


def deposit_unpaired(unpaired: UnpairedCell) -> tuple[Path | None, str | None]:
    """Compute and deposit ``unpaired``; return ``(path, None)`` or pending reason."""
    return _deposit.deposit_unpaired(_spec(), unpaired)


def build_distribution_table() -> tuple[Path, Path]:
    """Aggregate every deposited JSON into ``distribution_table.{json,md}``."""
    return _deposit.build_table(_spec())


def deposit_all(
    matched_pairs: Sequence[MatchedPair] | None = None,
    unpaired_cells: Sequence[UnpairedCell] | None = None,
    *,
    rebuild: bool = False,
    only_pair_keys: frozenset[str] | None = None,
) -> tuple[list[str], list[tuple[str, str]]]:
    """Deposit every Distribution record across the committed grid + extras."""
    return _deposit.deposit_all(
        _spec(),
        matched_pairs,
        unpaired_cells,
        rebuild=rebuild,
        only_pair_keys=only_pair_keys,
    )


__all__ = [
    "DISTRIBUTION_CELL_DIR",
    "DISTRIBUTION_DATA_DIR",
    "DISTRIBUTION_TABLE_JSON",
    "DISTRIBUTION_TABLE_MD",
    "SCHEMA_VERSION",
    "build_distribution_table",
    "deposit_all",
    "deposit_pair",
    "deposit_unpaired",
    "distribution_json_path",
    "is_distribution_done",
    "read_distribution_json",
    "write_distribution_json",
]
