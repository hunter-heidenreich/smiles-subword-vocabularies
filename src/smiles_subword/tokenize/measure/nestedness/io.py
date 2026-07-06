"""Deposition + aggregator for nestedness records.

Per-pair JSONs land under ``results/data/nestedness/``, one file per ``pair_key``.
The deposit dispatch and table join are the shared engine in
:mod:`smiles_subword.tokenize.measure._deposit`; this module supplies the
Nestedness-specific record builders, row projection, and Markdown columns.

Nestedness deposits a *flat* schema (``bpe_cell_id`` / ``present_cell_id`` …
rather than nested ``bpe`` / ``unigram`` blocks), so :func:`is_nestedness_done`
is bespoke: it validates the deposited ``eval_split_sha`` against the corpus's
current test-split MANIFEST and each present arm's ``training_corpus_sha``
against its ``meta.yaml``. Single-arm coordinates deposit a metric-free
:class:`UnpairedNestedness` (nestedness is intrinsically cross-arm).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from smiles_subword.paths import RESULTS_DATA_DIR
from smiles_subword.tokenize.measure import _deposit
from smiles_subword.tokenize.measure._cellmeta import (
    arm_info,
    cell_training_sha_fresh,
)
from smiles_subword.tokenize.measure._cells import eval_split_sha, load_cell_adapter
from smiles_subword.tokenize.measure._pairing import (
    MatchedPair,
    UnpairedCell,
    pair_all_cells,
)
from smiles_subword.tokenize.measure._tables import fmt_md
from smiles_subword.tokenize.measure.nestedness.math import (
    Arm,
    Boundary,
    MatchedPairNestedness,
    UnpairedNestedness,
    make_unpaired_nestedness,
)
from smiles_subword.tokenize.measure.nestedness.runner import run_pair_nestedness

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

SCHEMA_VERSION = 1

NESTEDNESS_DATA_DIR = RESULTS_DATA_DIR
NESTEDNESS_CELL_DIR = NESTEDNESS_DATA_DIR / "nestedness"
NESTEDNESS_TABLE_JSON = NESTEDNESS_DATA_DIR / "nestedness_table.json"
NESTEDNESS_TABLE_MD = NESTEDNESS_DATA_DIR / "nestedness_table.md"


def is_nestedness_done(pair_key: str) -> bool:
    """True iff a deposited record exists whose upstream SHAs still match.

    Validates the deposited ``eval_split_sha`` against the corpus's current
    test-split MANIFEST and each present arm's ``training_corpus_sha`` against
    its ``meta.yaml``. Any drift triggers a re-deposit on the next sweep.
    """
    payload = read_nestedness_json(pair_key)
    if payload is None:
        return False
    corpus = payload.get("corpus")
    if not isinstance(corpus, str):
        return False
    try:
        expected_eval = eval_split_sha(corpus)
    except (FileNotFoundError, ValueError):
        return False
    if payload.get("eval_split_sha") != expected_eval:
        return False
    if payload.get("pair_status") == "single_arm":
        cell_id = payload.get("present_cell_id")
        deposited = payload.get("present_training_corpus_sha")
        return cell_training_sha_fresh(cell_id, deposited) is not None
    for arm in ("bpe", "unigram"):
        cell_id = payload.get(f"{arm}_cell_id")
        deposited = payload.get(f"{arm}_training_corpus_sha")
        if cell_training_sha_fresh(cell_id, deposited) is None:
            return False
    return True


def _matched_pair_record(pair: MatchedPair) -> MatchedPairNestedness | str:
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
    return run_pair_nestedness(
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


def _unpaired_record(unpaired: UnpairedCell) -> UnpairedNestedness | str:
    info = arm_info(unpaired.cell_id)
    if isinstance(info, str):
        return info
    try:
        sha = eval_split_sha(unpaired.key.corpus)
    except (FileNotFoundError, ValueError) as exc:
        return str(exc)
    boundary: Boundary = "mb" if unpaired.key.boundary == "mb" else "nmb"
    missing_arm: Arm = "unigram" if unpaired.arm == "bpe" else "bpe"
    return make_unpaired_nestedness(
        pair_key=unpaired.key.slug,
        tier=unpaired.tier,
        corpus=unpaired.key.corpus,
        vocab_size=unpaired.key.vocab_size,
        boundary=boundary,
        extras_kind=unpaired.key.extras_kind,
        extras_label=unpaired.key.extras_label,
        present_arm=unpaired.arm,
        present_cell_id=unpaired.cell_id,
        present_training_corpus_sha=info.training_corpus_sha,
        eval_split_sha=sha,
        missing_arm=missing_arm,
        unpaired_reason=unpaired.reason,
    )


def _table_row_matched(payload: dict[str, Any]) -> dict[str, Any]:
    cut_rate = payload.get("cut_rate_by_class") or {}
    return {
        "pair_key": payload["pair_key"],
        "pair_status": payload["pair_status"],
        "tier": payload["tier"],
        "extras_kind": payload.get("extras_kind"),
        "extras_label": payload.get("extras_label"),
        "corpus": payload["corpus"],
        "vocab_size": payload["vocab_size"],
        "boundary": payload["boundary"],
        "boundary_jaccard": payload["boundary_jaccard"],
        "conflict_rate": payload["conflict_rate"],
        "nest_rate": payload["nest_rate"],
        "nested_molecule_fraction": payload["nested_molecule_fraction"],
        "conflict_share_of_disagreement": payload["conflict_share_of_disagreement"],
        "n_molecules": payload["n_molecules"],
        "n_length_mismatch": payload["n_length_mismatch"],
        "cut_rate_heteroatom": cut_rate.get("heteroatom"),
        "cut_rate_unsat_c": cut_rate.get("unsat-C"),
        "cut_rate_sat_c": cut_rate.get("sat-C"),
    }


def _table_row_unpaired(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "pair_key": payload["pair_key"],
        "pair_status": payload["pair_status"],
        "tier": payload["tier"],
        "extras_kind": payload.get("extras_kind"),
        "extras_label": payload.get("extras_label"),
        "corpus": payload["corpus"],
        "vocab_size": payload["vocab_size"],
        "boundary": payload["boundary"],
        "present_arm": payload["present_arm"],
        "present_cell_id": payload["present_cell_id"],
        "missing_arm": payload["missing_arm"],
        "unpaired_reason": payload["unpaired_reason"],
    }


def _format_md_table(
    matched_rows: list[dict[str, Any]],
    unpaired_rows: list[dict[str, Any]],
    pending: list[str],
) -> str:
    lines: list[str] = [
        "## Matched pairs (boundary nestedness)",
        "",
        "| pair | tier | V | corpus | bnd | bnd-J | conflict | nest | nested-mol "
        "| cut^het | cut^sat |",
        "|---|---|--:|---|---|--:|--:|--:|--:|--:|--:|",
    ]
    lines.extend(
        f"| {row['pair_key']} | {row['tier']} | {row['vocab_size']} "
        f"| {row['corpus']} | {row['boundary']} "
        f"| {fmt_md(row['boundary_jaccard'])} "
        f"| {fmt_md(row['conflict_rate'])} "
        f"| {fmt_md(row['nest_rate'])} "
        f"| {fmt_md(row['nested_molecule_fraction'])} "
        f"| {fmt_md(row['cut_rate_heteroatom'])} "
        f"| {fmt_md(row['cut_rate_sat_c'])} |"
        for row in matched_rows
    )
    if unpaired_rows:
        lines.extend(["", "## Single-arm coordinates (nestedness undefined)", ""])
        lines.append("| pair | tier | V | corpus | bnd | present | reason |")
        lines.append("|---|---|--:|---|---|---|---|")
        lines.extend(
            f"| {row['pair_key']} | {row['tier']} | {row['vocab_size']} "
            f"| {row['corpus']} | {row['boundary']} | {row['present_arm']} "
            f"| {row['unpaired_reason']} |"
            for row in unpaired_rows
        )
    if pending:
        lines.extend(["", f"## Pending ({len(pending)} pair_keys without records)", ""])
        lines.extend(f"- {pk}" for pk in pending)
    return "\n".join(lines) + "\n"


def _spec() -> _deposit.DepositSpec:
    return _deposit.DepositSpec(
        schema_version=SCHEMA_VERSION,
        cell_dir=NESTEDNESS_CELL_DIR,
        table_json=NESTEDNESS_TABLE_JSON,
        table_md=NESTEDNESS_TABLE_MD,
        pair_provider=pair_all_cells,
        build_matched=_matched_pair_record,
        build_unpaired=_unpaired_record,
        row_matched=_table_row_matched,
        row_unpaired=_table_row_unpaired,
        format_md=_format_md_table,
        is_done=is_nestedness_done,
    )


def nestedness_json_path(pair_key: str) -> Path:
    """Return the per-pair Nestedness JSON path for ``pair_key``."""
    return _deposit.json_path(NESTEDNESS_CELL_DIR, pair_key)


def write_nestedness_json(
    record: MatchedPairNestedness | UnpairedNestedness,
) -> Path:
    """Deposit ``record`` as a per-pair JSON; return its path."""
    return _deposit.write_json(NESTEDNESS_CELL_DIR, SCHEMA_VERSION, record)


def read_nestedness_json(pair_key: str) -> dict[str, object] | None:
    """Return the deposited Nestedness payload for ``pair_key``, or None if absent."""
    return _deposit.read_json(NESTEDNESS_CELL_DIR, pair_key)


def deposit_pair(pair: MatchedPair) -> tuple[Path | None, str | None]:
    """Compute and deposit ``pair``; return ``(path, None)`` or ``(None, reason)``."""
    return _deposit.deposit_pair(_spec(), pair)


def deposit_unpaired(unpaired: UnpairedCell) -> tuple[Path | None, str | None]:
    """Compute and deposit ``unpaired``; return ``(path, None)`` or pending reason."""
    return _deposit.deposit_unpaired(_spec(), unpaired)


def build_nestedness_table() -> tuple[Path, Path]:
    """Aggregate every deposited Nestedness JSON into ``nestedness_table.{json,md}``."""
    return _deposit.build_table(_spec())


def deposit_all(
    matched_pairs: Sequence[MatchedPair] | None = None,
    unpaired_cells: Sequence[UnpairedCell] | None = None,
    *,
    rebuild: bool = False,
    only_pair_keys: frozenset[str] | None = None,
) -> tuple[list[str], list[tuple[str, str]]]:
    """Deposit every Nestedness record across the committed grid + extras."""
    return _deposit.deposit_all(
        _spec(),
        matched_pairs,
        unpaired_cells,
        rebuild=rebuild,
        only_pair_keys=only_pair_keys,
    )


__all__ = [
    "NESTEDNESS_CELL_DIR",
    "NESTEDNESS_DATA_DIR",
    "NESTEDNESS_TABLE_JSON",
    "NESTEDNESS_TABLE_MD",
    "SCHEMA_VERSION",
    "build_nestedness_table",
    "deposit_all",
    "deposit_pair",
    "deposit_unpaired",
    "is_nestedness_done",
    "nestedness_json_path",
    "read_nestedness_json",
    "write_nestedness_json",
]
