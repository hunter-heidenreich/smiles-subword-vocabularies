"""Deposition + aggregator for Jaccard vocabulary-Jaccard records.

Per-pair JSONs land under ``results/data/jaccard/``, one file per ``pair_key``.
The deposit dispatch and table join are the shared engine in
:mod:`smiles_subword.tokenize.measure._deposit`; this module supplies the
Jaccard-specific record builders (including the cached training-corpus chunk
inventory and the per-pair vocabulary characterization), row projection, and
Markdown columns.

:func:`is_jaccard_done` gates resume on the deposited ``schema_version`` plus
both ``training_corpus_sha`` (each cell's ``meta.yaml``) and ``eval_split_sha``
(the held-out test-split MANIFEST); the training-corpus check is load-bearing
because ``J_struct``'s bracket-internal split is a training-corpus property. The
chunk inventory is cached under ``jaccard/_inventory/`` keyed by
``training_corpus_sha``, built once and shared across every ``V`` and both arms
that train on the same input. The aggregator :func:`build_jaccard_table`
surfaces any cell whose inventory hit the non-bracket cap.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from smiles_subword.paths import RESULTS_DATA_DIR, tokenizer_artifact_dir
from smiles_subword.tokenize.audit.f95_io import f95_json_path
from smiles_subword.tokenize.measure import _deposit
from smiles_subword.tokenize.measure._cellmeta import resolve_cell_meta
from smiles_subword.tokenize.measure._cells import load_cell_adapter
from smiles_subword.tokenize.measure._glyphmap import glyph_tuple_map
from smiles_subword.tokenize.measure._pairing import (
    MatchedPair,
    UnpairedCell,
    pair_all_cells,
)
from smiles_subword.tokenize.measure._tables import fmt_md
from smiles_subword.tokenize.measure.jaccard.math import (
    Arm,
    ArmJaccardInputs,
    GlyphTuple,
    JwMoleculeData,
    MatchedPairJaccard,
    UnpairedJaccard,
    compute_matched_pair_jaccard,
    compute_unpaired_jaccard,
)
from smiles_subword.tokenize.measure.jaccard.runner import run_arm_jaccard
from smiles_subword.tokenize.measure.supplementary.vocab_characterization import (
    characterize_pair,
    write_vocab_characterization,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

SCHEMA_VERSION = 2

JACCARD_DATA_DIR = RESULTS_DATA_DIR
JACCARD_CELL_DIR = JACCARD_DATA_DIR / "jaccard"
INVENTORY_DIR = JACCARD_CELL_DIR / "_inventory"
JACCARD_TABLE_JSON = JACCARD_DATA_DIR / "jaccard_table.json"
JACCARD_TABLE_MD = JACCARD_DATA_DIR / "jaccard_table.md"


def inventory_cache_path(training_corpus_sha: str) -> Path:
    """Return the cached chunk-inventory path for a training input."""
    return INVENTORY_DIR / f"{training_corpus_sha}.json"


def _arm_inputs_from_cell(cell_id: str, arm: Arm) -> ArmJaccardInputs | str:
    meta = resolve_cell_meta(cell_id)
    if isinstance(meta, str):
        return meta
    try:
        adapter = load_cell_adapter(meta.corpus, meta.name)
    except FileNotFoundError as exc:
        return str(exc)
    return run_arm_jaccard(
        adapter,
        cell_id=cell_id,
        corpus=meta.corpus,
        name=meta.name,
        arm=arm,
        boundary=meta.boundary,
        training_corpus_sha=meta.training_corpus_sha,
        inventory_cache_path=inventory_cache_path(meta.training_corpus_sha),
    )


def _holdout_counts(
    multi: frozenset[GlyphTuple], jw: JwMoleculeData
) -> dict[GlyphTuple, int]:
    """Aggregate one arm's sparse held-out emission data into per-piece counts.

    Every multi-glyph subword gets an entry (``0`` if it never fired on the
    held-out split) so the rank-frequency curve carries the dead tail.
    """
    counts: dict[GlyphTuple, int] = dict.fromkeys(multi, 0)
    for local_id, count in zip(jw.sub_local.tolist(), jw.count.tolist(), strict=True):
        piece = jw.local_tuples[local_id]
        if piece in counts:
            counts[piece] += int(count)
    return counts


def _train_counts(arm: ArmJaccardInputs) -> dict[GlyphTuple, int] | None:
    """Per-piece training counts for an arm's multi-glyph vocabulary.

    Read from the deposited F95 result and remapped from token id to glyph
    tuple via the arm's saved tokenizer; ``None`` when F95 has not yet run for
    this cell (or predates the per-piece-count deposit), leaving the training
    rank-frequency curve empty rather than fabricated.
    """
    f95_path = f95_json_path(arm.cell_id)
    if not f95_path.is_file():
        return None
    raw = json.loads(f95_path.read_text()).get("training_counts_by_id")
    if not isinstance(raw, dict):
        return None
    corpus, _, name = arm.cell_id.partition("__")
    id_to_tuple = glyph_tuple_map(tokenizer_artifact_dir(corpus, name), arm.arm)
    counts: dict[GlyphTuple, int] = {}
    for tid_str, count in raw.items():
        piece = id_to_tuple.get(int(tid_str))
        if piece is not None and len(piece) >= 2:
            counts[piece] = int(count)
    return counts


def _deposit_pair_characterization(
    pair_key: str, bpe: ArmJaccardInputs, unigram: ArmJaccardInputs
) -> None:
    """Emit the per-pair vocabulary characterization beside the Jaccard record.

    Partition, length profiles, and the held-out rank-frequency curves come
    free from Jaccard's own inputs; the training rank-frequency is loaded from the
    deposited F95 counts when present (else left empty).
    """
    payload = characterize_pair(
        pair_key,
        bpe.multi_subwords,
        unigram.multi_subwords,
        bpe_train_counts=_train_counts(bpe),
        unigram_train_counts=_train_counts(unigram),
        bpe_holdout_counts=_holdout_counts(bpe.multi_subwords, bpe.jw),
        unigram_holdout_counts=_holdout_counts(unigram.multi_subwords, unigram.jw),
    )
    write_vocab_characterization(payload)


def _matched_pair_record(pair: MatchedPair) -> MatchedPairJaccard | str:
    bpe = _arm_inputs_from_cell(pair.bpe_cell_id, "bpe")
    if isinstance(bpe, str):
        return bpe
    unigram = _arm_inputs_from_cell(pair.unigram_cell_id, "unigram")
    if isinstance(unigram, str):
        return unigram
    if bpe.boundary != unigram.boundary:
        return (
            f"arm boundary mismatch for {pair.key.slug}: "
            f"bpe={bpe.boundary!r}, unigram={unigram.boundary!r}"
        )
    record = compute_matched_pair_jaccard(
        bpe,
        unigram,
        pair_key=pair.key.slug,
        tier=pair.tier,
        corpus=pair.key.corpus,
        vocab_size=pair.key.vocab_size,
        boundary=bpe.boundary,
        extras_kind=pair.key.extras_kind,
        extras_label=pair.key.extras_label,
    )
    _deposit_pair_characterization(pair.key.slug, bpe, unigram)
    return record


def _unpaired_record(unpaired: UnpairedCell) -> UnpairedJaccard | str:
    arm = _arm_inputs_from_cell(unpaired.cell_id, unpaired.arm)
    if isinstance(arm, str):
        return arm
    missing_arm: Arm = "unigram" if unpaired.arm == "bpe" else "bpe"
    return compute_unpaired_jaccard(
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
    ci = payload.get("weighted_jaccard_ci") or [None, None]
    ci_struct = payload.get("weighted_jaccard_struct_ci") or [None, None]
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
        "bpe_n_multi": bpe["n_multi_subwords"],
        "unigram_n_multi": unigram["n_multi_subwords"],
        "bpe_n_structural": bpe["n_structural"],
        "unigram_n_structural": unigram["n_structural"],
        "jaccard": payload["jaccard"],
        "jaccard_struct": payload["jaccard_struct"],
        "weighted_jaccard": payload["weighted_jaccard"],
        "weighted_jaccard_ci_lo": ci[0],
        "weighted_jaccard_ci_hi": ci[1],
        "weighted_jaccard_struct": payload.get("weighted_jaccard_struct"),
        "weighted_jaccard_struct_ci_lo": ci_struct[0],
        "weighted_jaccard_struct_ci_hi": ci_struct[1],
        "jaccard_minus_struct": payload["jaccard_minus_struct"],
        "cap_bound": bool(bpe.get("nonbracket_cap_bound"))
        or bool(unigram.get("nonbracket_cap_bound")),
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
        "present_n_multi": present["n_multi_subwords"],
        "present_n_structural": present["n_structural"],
        "missing_arm": payload["missing_arm"],
        "unpaired_reason": payload["unpaired_reason"],
    }


def _format_md_table(
    matched_rows: list[dict[str, Any]],
    unpaired_rows: list[dict[str, Any]],
    pending: list[str],
) -> str:
    header = (
        "| pair | tier | V | corpus | bnd | J | J_struct | J−J_struct"
        " | J_w | J_w CI | J_w_struct | J_w_struct CI | cap |"
    )
    lines: list[str] = [
        "## Matched pairs (Jaccards reportable)",
        "",
        header,
        "|---|---|--:|---|---|--:|--:|--:|--:|---|--:|---|:--:|",
    ]
    lines.extend(
        f"| {row['pair_key']} | {row['tier']} | {row['vocab_size']} "
        f"| {row['corpus']} | {row['boundary']} "
        f"| {fmt_md(row['jaccard'])} | {fmt_md(row['jaccard_struct'])} "
        f"| {fmt_md(row['jaccard_minus_struct'])} | {fmt_md(row['weighted_jaccard'])} "
        f"| [{fmt_md(row['weighted_jaccard_ci_lo'])}, "
        f"{fmt_md(row['weighted_jaccard_ci_hi'])}] "
        f"| {fmt_md(row['weighted_jaccard_struct'])} "
        f"| [{fmt_md(row['weighted_jaccard_struct_ci_lo'])}, "
        f"{fmt_md(row['weighted_jaccard_struct_ci_hi'])}] "
        f"| {'⚠' if row['cap_bound'] else ''} |"
        for row in matched_rows
    )
    if unpaired_rows:
        lines.extend(["", "## Single-arm coordinates (Jaccards undefined)", ""])
        lines.append("| pair | tier | V | corpus | bnd | arm | n_multi | reason |")
        lines.append("|---|---|--:|---|---|---|--:|---|")
        lines.extend(
            f"| {row['pair_key']} | {row['tier']} | {row['vocab_size']} "
            f"| {row['corpus']} | {row['boundary']} | {row['present_arm']} "
            f"| {row['present_n_multi']} | {row['unpaired_reason']} |"
            for row in unpaired_rows
        )
    if pending:
        lines.extend(
            ["", f"## Pending ({len(pending)} pair_keys without Jaccard records)", ""]
        )
        lines.extend(f"- {pk}" for pk in pending)
    return "\n".join(lines) + "\n"


def _table_extra(
    matched_rows: list[dict[str, Any]], unpaired_rows: list[dict[str, Any]]
) -> dict[str, object]:
    del unpaired_rows
    return {
        "nonbracket_cap_bound": [r["pair_key"] for r in matched_rows if r["cap_bound"]],
    }


def _spec() -> _deposit.DepositSpec:
    return _deposit.DepositSpec(
        schema_version=SCHEMA_VERSION,
        cell_dir=JACCARD_CELL_DIR,
        table_json=JACCARD_TABLE_JSON,
        table_md=JACCARD_TABLE_MD,
        pair_provider=lambda: pair_all_cells(include_large_v_anchor=True),
        build_matched=_matched_pair_record,
        build_unpaired=_unpaired_record,
        row_matched=_table_row_matched,
        row_unpaired=_table_row_unpaired,
        format_md=_format_md_table,
        is_done=is_jaccard_done,
        table_extra=_table_extra,
    )


def jaccard_json_path(pair_key: str) -> Path:
    """Return the per-pair Jaccard JSON path for ``pair_key``."""
    return _deposit.json_path(JACCARD_CELL_DIR, pair_key)


def write_jaccard_json(record: MatchedPairJaccard | UnpairedJaccard) -> Path:
    """Deposit ``record`` as a per-pair JSON; return its path."""
    return _deposit.write_json(JACCARD_CELL_DIR, SCHEMA_VERSION, record)


def read_jaccard_json(pair_key: str) -> dict[str, object] | None:
    """Return the deposited Jaccard payload for ``pair_key``, or None if absent."""
    return _deposit.read_json(JACCARD_CELL_DIR, pair_key)


def is_jaccard_done(pair_key: str) -> bool:
    """True iff a current-schema Jaccard record exists whose upstream SHAs match.

    Validates the deposited ``schema_version`` plus the corpus + eval-split SHAs
    against each present cell's ``meta.yaml`` and the corpus's current test-split
    MANIFEST; a schema bump or any SHA drift triggers a re-deposit.
    """
    payload = read_jaccard_json(pair_key)
    if payload is None or payload.get("schema_version") != SCHEMA_VERSION:
        return False
    return _deposit.nested_is_done(payload, _deposit.standard_arm_block_fresh)


def deposit_pair(pair: MatchedPair) -> tuple[Path | None, str | None]:
    """Compute and deposit ``pair``; return ``(path, None)`` or ``(None, reason)``."""
    return _deposit.deposit_pair(_spec(), pair)


def deposit_unpaired(unpaired: UnpairedCell) -> tuple[Path | None, str | None]:
    """Compute and deposit ``unpaired``; return ``(path, None)`` or pending reason."""
    return _deposit.deposit_unpaired(_spec(), unpaired)


def build_jaccard_table() -> tuple[Path, Path]:
    """Aggregate every deposited Jaccard JSON into ``jaccard_table.{json,md}``."""
    return _deposit.build_table(_spec())


def deposit_all(
    matched_pairs: Sequence[MatchedPair] | None = None,
    unpaired_cells: Sequence[UnpairedCell] | None = None,
    *,
    rebuild: bool = False,
    only_pair_keys: frozenset[str] | None = None,
) -> tuple[list[str], list[tuple[str, str]]]:
    """Deposit every Jaccard record across the committed grid + extras."""
    return _deposit.deposit_all(
        _spec(),
        matched_pairs,
        unpaired_cells,
        rebuild=rebuild,
        only_pair_keys=only_pair_keys,
    )


__all__ = [
    "INVENTORY_DIR",
    "JACCARD_CELL_DIR",
    "JACCARD_DATA_DIR",
    "JACCARD_TABLE_JSON",
    "JACCARD_TABLE_MD",
    "SCHEMA_VERSION",
    "build_jaccard_table",
    "deposit_all",
    "deposit_pair",
    "deposit_unpaired",
    "inventory_cache_path",
    "is_jaccard_done",
    "jaccard_json_path",
    "read_jaccard_json",
    "write_jaccard_json",
]
