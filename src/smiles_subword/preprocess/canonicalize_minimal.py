"""Minimal canonicalization shared by ``canon_dedup`` and the OOD corpora.

Applies **only** RDKit's ``MolFromSmiles → MolToSmiles isomeric`` round-trip and
drops unparseable rows — no neutralization, salt-stripping, heavy-atom filtering,
or dedup, each of which would bias what the study measures (neutralization
removes the ionic forms that exercise Smirk's no-UNK fallback, a heavy-atom cap
drops macrocycles, dedup masks in-corpus redundancy). Preserves the ``raw_v1``
schema (``source_id``, ``smiles``, ``source``, ``ingest_ts``) so the audit
runner streams canonical_v1 shards exactly as raw_v1. Dropped-row count is
recorded in the sidecar ``MANIFEST.yaml``.

The round-trip is the runtime bottleneck (~97% of canon_dedup wall time).
``n_workers > 1`` fans per-batch canonicalization across a process pool; futures
drain in submission order, so output row order — and shard bytes — match the
serial path.
"""

from __future__ import annotations

from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger

from smiles_subword.ingest._common import RAW_V1_SCHEMA
from smiles_subword.preprocess._io import (
    ShardWriter,
    list_input_shards,
    read_source_manifest,
    shard_dicts,
    stage_run,
    write_manifest,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

RDLogger.DisableLog("rdApp.*")  # pyright: ignore[reportAttributeAccessIssue]

_DEFAULT_TARGET_BYTES = 64 * 1024 * 1024
_DEFAULT_ROWS_PER_BATCH = 8192

_RawRows = list[tuple[str, str, str, object]]


@dataclass(frozen=True)
class CanonicalizeMinimalResult:
    """Outcome of one minimal-canonicalize run."""

    output_dir: Path
    n_input_rows: int
    n_output_rows: int
    n_dropped_unparseable: int
    n_shards: int


@dataclass
class _Counts:
    n_input: int = 0
    n_output: int = 0
    n_dropped: int = 0


def canonicalize_minimal(
    input_dir: Path,
    output_dir: Path,
    *,
    target_bytes: int = _DEFAULT_TARGET_BYTES,
    n_workers: int = 1,
    rows_per_batch: int = _DEFAULT_ROWS_PER_BATCH,
) -> CanonicalizeMinimalResult:
    """Run minimal canonicalization over every shard under ``input_dir``.

    Args:
        input_dir: directory containing ``raw_v1`` Parquet shards plus a
            ``MANIFEST.yaml``. Both convention markers must be present.
        output_dir: where the ``canonical_v1`` shards land. Atomic-renamed
            into place after a successful run.
        target_bytes: shard rollover threshold; defaults to 64 MiB so
            chunks fit comfortably in row-group memory.
        n_workers: process-pool width for the RDKit round-trip. ``1`` runs
            in-process; ``>1`` fans batches across a pool. Output is
            byte-identical either way — futures drain in submission order.
        rows_per_batch: rows per canonicalization task / Parquet read batch.

    Returns:
        :class:`CanonicalizeMinimalResult` with input / output row counts
        and the dropped-row tally. The same payload is also written to
        ``output_dir/MANIFEST.yaml``.
    """
    _, source_manifest = read_source_manifest(input_dir)
    input_shards = list_input_shards(input_dir)
    if not input_shards:
        raise FileNotFoundError(
            f"no parquet shards under {input_dir!r}; not a raw_v1 directory"
        )

    counts = _Counts()

    with stage_run(output_dir) as (staging_dir, _started_ts):
        writer = ShardWriter(
            staging_dir,
            schema=RAW_V1_SCHEMA,
            shard_prefix="canonical_v1",
            target_bytes=target_bytes,
        )
        batches = _iter_input_batches(input_shards, rows_per_batch)
        if n_workers > 1:
            _run_pool(batches, writer, counts, n_workers)
        else:
            _run_serial(batches, writer, counts)
        writer.close_current()
        write_manifest(
            staging_dir,
            _build_manifest(
                source_manifest=source_manifest,
                counts=counts,
                shards=writer.shards,
            ),
        )

    return CanonicalizeMinimalResult(
        output_dir=output_dir,
        n_input_rows=counts.n_input,
        n_output_rows=counts.n_output,
        n_dropped_unparseable=counts.n_dropped,
        n_shards=len(writer.shards),
    )


def _iter_input_batches(
    input_shards: list[Path], rows_per_batch: int
) -> Iterator[_RawRows]:
    for shard_path in input_shards:
        pf = pq.ParquetFile(shard_path)
        for batch in pf.iter_batches(batch_size=rows_per_batch):
            yield list(
                zip(
                    batch.column("source_id").to_pylist(),
                    batch.column("smiles").to_pylist(),
                    batch.column("source").to_pylist(),
                    batch.column("ingest_ts").to_pylist(),
                    strict=True,
                )
            )


def _canonicalize_batch(rows: _RawRows) -> tuple[_RawRows, int, int]:
    """Worker: canonicalize one batch; return (kept rows, n_input, n_dropped).

    Runs in a pool worker under ``n_workers > 1``; the module-level
    ``RDLogger.DisableLog`` re-runs on import in each spawned process.
    """
    kept: _RawRows = []
    for sid, smi, src, ts in rows:
        canonical = _canonicalize_one(smi)  # pyright: ignore[reportArgumentType]
        if canonical is not None:
            kept.append((sid, canonical, src, ts))
    return kept, len(rows), len(rows) - len(kept)


def _run_serial(
    batches: Iterator[_RawRows], writer: ShardWriter, counts: _Counts
) -> None:
    for rows in batches:
        _accumulate(_canonicalize_batch(rows), writer, counts)


def _run_pool(
    batches: Iterator[_RawRows],
    writer: ShardWriter,
    counts: _Counts,
    n_workers: int,
) -> None:
    max_in_flight = max(2, n_workers * 2)
    in_flight: list[Future[tuple[_RawRows, int, int]]] = []

    def _drain_one() -> None:
        _accumulate(in_flight.pop(0).result(), writer, counts)

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for rows in batches:
            while len(in_flight) >= max_in_flight:
                _drain_one()
            in_flight.append(pool.submit(_canonicalize_batch, rows))
        while in_flight:
            _drain_one()


def _accumulate(
    result: tuple[_RawRows, int, int], writer: ShardWriter, counts: _Counts
) -> None:
    kept, n_input, n_dropped = result
    counts.n_input += n_input
    counts.n_dropped += n_dropped
    if not kept:
        return
    batch = pa.RecordBatch.from_arrays(
        [
            pa.array([r[0] for r in kept], type=pa.string()),
            pa.array([r[1] for r in kept], type=pa.string()),
            pa.array([r[2] for r in kept], type=pa.string()),
            pa.array([r[3] for r in kept], type=pa.timestamp("us")),
        ],
        schema=RAW_V1_SCHEMA,
    )
    writer.write_batch(batch)
    counts.n_output += batch.num_rows


def _canonicalize_one(smi: str | None) -> str | None:
    if smi is None or not smi:
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def _build_manifest(
    *,
    source_manifest: dict,
    counts: _Counts,
    shards: list,
) -> dict[str, object]:
    return {
        "schema": "canonical_v1_minimal",
        "policy": "rdkit_canonical_isomeric_only",
        "source_manifest_stage": source_manifest.get("schema", "raw_v1"),
        "source_corpus": source_manifest.get("corpus") or source_manifest.get("source"),
        "n_input_rows": counts.n_input,
        "n_output_rows": counts.n_output,
        "n_dropped_unparseable": counts.n_dropped,
        "n_shards": len(shards),
        "shards": shard_dicts(shards),
    }


__all__ = ["CanonicalizeMinimalResult", "canonicalize_minimal"]
