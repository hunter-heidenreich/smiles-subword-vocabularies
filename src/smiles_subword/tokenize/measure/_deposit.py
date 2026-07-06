"""Shared deposit + aggregate engine for the per-pair measurements.

Every measurement deposits one JSON per ``pair_key`` under
``results/data/<topic>/`` and aggregates the deposited set into a
``<topic>_table.{json,md}`` pair. The deposit dispatch (skip-if-fresh resume,
matched/unpaired/pending bookkeeping) and the table join (read every pair_key,
sort, write JSON + Markdown atomically) are identical across topics — only the
record builders, the flat row projection, the Markdown columns, and a handful
of extra table keys vary. Those variations are supplied per topic as a
:class:`DepositSpec`; everything else lives here once.

A topic's ``io.py`` builds its spec *fresh on each call* (``_spec()``) from its
module-level constants and helpers, so the test suite's monkeypatching of those
module globals (data dirs, ``pair_all_cells``, the record builders) stays
transparent — the engine never snapshots them at import time.

Freshness: most topics gate resume on each present arm block's
``training_corpus_sha`` + ``eval_split_sha`` (:func:`standard_arm_block_fresh`,
fed through :func:`nested_is_done`). Within-arm topics (closure, fg_alignment,
noncanon) validate a single arm block via :func:`within_arm_is_done`. Topics
whose staleness depends on something else (Deadzone joins F95 records; Nestedness
uses a flat schema) supply their own ``is_done`` predicate.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from smiles_subword._io import atomic_write_json, atomic_write_text, read_json_or_none
from smiles_subword.tokenize.measure._cellmeta import cell_training_sha_fresh
from smiles_subword.tokenize.measure._cells import eval_split_sha
from smiles_subword.tokenize.measure._pairing import MatchedPair, UnpairedCell


class Record(Protocol):
    """A per-pair record: carries its ``pair_key`` and serializes to a dict."""

    @property
    def pair_key(self) -> str: ...

    def as_dict(self) -> dict[str, object]: ...


def json_path(cell_dir: Path, pair_key: str) -> Path:
    """Return the per-pair JSON path for ``pair_key`` under ``cell_dir``."""
    return cell_dir / f"{pair_key}.json"


def write_json(cell_dir: Path, schema_version: int, record: Record) -> Path:
    """Deposit ``record`` (plus ``schema_version``) as a per-pair JSON."""
    payload: dict[str, object] = {"schema_version": schema_version, **record.as_dict()}
    path = json_path(cell_dir, record.pair_key)
    atomic_write_json(path, payload)
    return path


def read_json(cell_dir: Path, pair_key: str) -> dict[str, object] | None:
    """Return the deposited payload for ``pair_key``, or None if absent/corrupt."""
    return read_json_or_none(json_path(cell_dir, pair_key))


def standard_arm_block_fresh(block: object) -> bool:
    """True iff an arm block's ``training_corpus_sha`` + ``eval_split_sha`` still match.

    Re-reads the cell's ``meta.yaml`` (via :func:`cell_training_sha_fresh`) and the
    corpus's current held-out test-split MANIFEST (via :func:`eval_split_sha`).
    """
    if not isinstance(block, dict):
        return False
    cell = cell_training_sha_fresh(
        block.get("cell_id"), block.get("training_corpus_sha")
    )
    if cell is None:
        return False
    corpus, _name = cell
    try:
        return eval_split_sha(corpus) == block.get("eval_split_sha")
    except (FileNotFoundError, ValueError):
        return False


def nested_is_done(
    payload: dict[str, object] | None,
    arm_block_fresh: Callable[[object], bool],
) -> bool:
    """True iff every present ``bpe`` / ``unigram`` block in ``payload`` is fresh.

    The common ``is_<topic>_done`` shape: a missing payload is stale, a missing
    arm block is skipped (single-arm coordinate), and any non-fresh present block
    makes the record stale. ``arm_block_fresh`` decides freshness per block.
    """
    if payload is None:
        return False
    for arm_key in ("bpe", "unigram"):
        block = payload.get(arm_key)
        if block is None:
            continue
        if not arm_block_fresh(block):
            return False
    return True


def within_arm_is_done(
    payload: dict[str, object] | None,
    arm_block_fresh: Callable[[object], bool],
) -> bool:
    """True iff a within-arm record's present arm block(s) are all fresh.

    The ``is_<topic>_done`` shape for the *within-arm* topics (closure,
    fg_alignment, noncanon), where each present arm deposits a real reading
    rather than a metric-free stub. Unlike :func:`nested_is_done`, a single-arm
    coordinate gates on its ``present_arm`` block alone and a matched record
    requires *both* arm blocks fresh — a missing block is stale, not skipped.
    ``arm_block_fresh`` decides freshness per block (training-sha only for
    closure, training + eval-split sha for the held-out within-arm topics).
    """
    if payload is None:
        return False
    if payload.get("pair_status") == "single_arm":
        present = payload.get("present_arm")
        if not isinstance(present, str):
            return False
        return arm_block_fresh(payload.get(present))
    return all(arm_block_fresh(payload.get(arm)) for arm in ("bpe", "unigram"))


def _no_extra(matched_rows: list[Any], unpaired_rows: list[Any]) -> dict[str, object]:
    del matched_rows, unpaired_rows
    return {}


@dataclass(frozen=True)
class DepositSpec:
    """The per-topic surface the deposit/aggregate engine needs.

    Built fresh on each call from a topic's module globals so monkeypatching
    stays transparent. ``build_matched`` / ``build_unpaired`` return a
    :class:`Record` on success or a pending-reason ``str``; ``row_matched`` /
    ``row_unpaired`` project a deposited payload to a flat table row;
    ``format_md`` renders the Markdown table; ``table_extra`` contributes any
    topic-specific top-level keys to the table JSON.
    """

    schema_version: int
    cell_dir: Path
    table_json: Path
    table_md: Path
    pair_provider: Callable[[], tuple[Sequence[MatchedPair], Sequence[UnpairedCell]]]
    build_matched: Callable[[MatchedPair], Record | str]
    build_unpaired: Callable[[UnpairedCell], Record | str]
    row_matched: Callable[[dict[str, Any]], dict[str, Any]]
    row_unpaired: Callable[[dict[str, Any]], dict[str, Any]]
    format_md: Callable[[list[dict[str, Any]], list[dict[str, Any]], list[str]], str]
    is_done: Callable[[str], bool]
    table_extra: Callable[
        [list[dict[str, Any]], list[dict[str, Any]]], dict[str, object]
    ] = field(default=_no_extra)


def deposit_pair(
    spec: DepositSpec, pair: MatchedPair
) -> tuple[Path | None, str | None]:
    """Compute and deposit ``pair``; return ``(path, None)`` or ``(None, reason)``."""
    result = spec.build_matched(pair)
    if isinstance(result, str):
        return None, result
    return write_json(spec.cell_dir, spec.schema_version, result), None


def deposit_unpaired(
    spec: DepositSpec, unpaired: UnpairedCell
) -> tuple[Path | None, str | None]:
    """Compute and deposit ``unpaired``; return ``(path, None)`` or pending reason."""
    result = spec.build_unpaired(unpaired)
    if isinstance(result, str):
        return None, result
    return write_json(spec.cell_dir, spec.schema_version, result), None


def deposit_all(
    spec: DepositSpec,
    matched_pairs: Sequence[MatchedPair] | None = None,
    unpaired_cells: Sequence[UnpairedCell] | None = None,
    *,
    rebuild: bool = False,
    only_pair_keys: frozenset[str] | None = None,
) -> tuple[list[str], list[tuple[str, str]]]:
    """Deposit every record across the committed grid + extras.

    Returns ``(deposited_pair_keys, pending_pairs)`` where ``pending_pairs`` is a
    list of ``(pair_key, reason)``. When ``matched_pairs`` / ``unpaired_cells``
    are None they default to ``spec.pair_provider()``. ``rebuild`` recomputes
    even fresh records; ``only_pair_keys`` restricts the sweep.
    """
    if matched_pairs is None or unpaired_cells is None:
        manifest_matched, manifest_unpaired = spec.pair_provider()
        matched_pairs = manifest_matched if matched_pairs is None else matched_pairs
        unpaired_cells = manifest_unpaired if unpaired_cells is None else unpaired_cells

    deposited: list[str] = []
    pending: list[tuple[str, str]] = []

    def _filter(pair_key: str) -> bool:
        return only_pair_keys is None or pair_key in only_pair_keys

    for pair in matched_pairs:
        if not _filter(pair.key.slug):
            continue
        if not rebuild and spec.is_done(pair.key.slug):
            deposited.append(pair.key.slug)
            continue
        _path, reason = deposit_pair(spec, pair)
        if reason is None:
            deposited.append(pair.key.slug)
        else:
            pending.append((pair.key.slug, reason))
    for unpaired in unpaired_cells:
        if not _filter(unpaired.key.slug):
            continue
        if not rebuild and spec.is_done(unpaired.key.slug):
            deposited.append(unpaired.key.slug)
            continue
        _path, reason = deposit_unpaired(spec, unpaired)
        if reason is None:
            deposited.append(unpaired.key.slug)
        else:
            pending.append((unpaired.key.slug, reason))

    return deposited, pending


def build_table(spec: DepositSpec) -> tuple[Path, Path]:
    """Aggregate every deposited per-pair JSON into ``<topic>_table.{json,md}``.

    A pure on-disk join over ``spec.pair_provider()``: reads each pair_key's
    payload, projects matched / unpaired rows (missing payloads become pending),
    sorts, and writes the table JSON + Markdown atomically.
    """
    matched_pairs, unpaired_cells = spec.pair_provider()
    matched_rows: list[dict[str, Any]] = []
    unpaired_rows: list[dict[str, Any]] = []
    pending: list[str] = []

    for pair in matched_pairs:
        payload = read_json(spec.cell_dir, pair.key.slug)
        if payload is None:
            pending.append(pair.key.slug)
            continue
        matched_rows.append(spec.row_matched(payload))
    for unpaired in unpaired_cells:
        payload = read_json(spec.cell_dir, unpaired.key.slug)
        if payload is None:
            pending.append(unpaired.key.slug)
            continue
        unpaired_rows.append(spec.row_unpaired(payload))

    matched_rows.sort(key=lambda r: str(r["pair_key"]))
    unpaired_rows.sort(key=lambda r: str(r["pair_key"]))
    pending.sort()

    table_json: dict[str, object] = {
        "schema_version": spec.schema_version,
        "n_pairs": len(matched_pairs) + len(unpaired_cells),
        "n_present": len(matched_rows) + len(unpaired_rows),
        "n_matched_present": len(matched_rows),
        "n_unpaired_present": len(unpaired_rows),
        **spec.table_extra(matched_rows, unpaired_rows),
        "pending": pending,
        "matched": matched_rows,
        "unpaired": unpaired_rows,
    }
    atomic_write_json(spec.table_json, table_json)
    atomic_write_text(
        spec.table_md, spec.format_md(matched_rows, unpaired_rows, pending)
    )

    return spec.table_json, spec.table_md


__all__ = [
    "DepositSpec",
    "Record",
    "build_table",
    "deposit_all",
    "deposit_pair",
    "deposit_unpaired",
    "json_path",
    "nested_is_done",
    "read_json",
    "standard_arm_block_fresh",
    "within_arm_is_done",
    "write_json",
]
