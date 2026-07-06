"""Deterministic hash-partition subsample.

Keeps a molecule iff a stable hash of its *canonical SMILES* falls in the
acceptance band ``[0, target_n / n_input_rows)``. Uniform in expectation, not
exact-N: the kept count fluctuates around ``target_n`` by binomial noise. Used
for the study corpora that exceed their target size (PubChem, ZINC-22, the
REAL-Space anchor).

The hash keys off the canonical SMILES, never ``source_id`` — the train/test
split (``holdout_split.py``) keys off ``source_id``, so the two partitions draw
on independent hash domains and stay uncorrelated; ``hash_domain`` namespaces the
SHA1 input as a second independence guarantee.

Input is a ``canon_dedup_v1`` Parquet directory; output keeps the 4-field
``raw_v1`` schema, so the split stage sees one schema whether or not a corpus was
subsampled.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq

from smiles_subword._hashing import sha256_bytes
from smiles_subword.ingest._common import RAW_V1_SCHEMA
from smiles_subword.preprocess._io import (
    ShardWriter,
    coerce_to_schema,
    count_parquet_rows,
    list_input_shards,
    read_and_verify_source_manifest,
    shard_dicts,
    stage_run,
    write_manifest,
)
from smiles_subword.preprocess.types import HashSubsampleResult, ShardInfo

if TYPE_CHECKING:
    from datetime import datetime
    from pathlib import Path

    from smiles_subword.config import HashSubsampleConfig

__all__ = ["hash_subsample", "smiles_acceptance_coord"]


def smiles_acceptance_coord(smiles: str, *, domain: str) -> float:
    """Map a canonical SMILES to a uniform ``[0, 1)`` coordinate.

    SHA1 of ``f"{domain}|{smiles}"``, leading 64 bits as a big-endian uint64
    divided by ``2**64``. ``domain`` namespaces the hash so this partition is
    independent of any other SHA1 partition over the same molecules.
    """
    digest = hashlib.sha1(f"{domain}|{smiles}".encode(), usedforsecurity=False)
    return int.from_bytes(digest.digest()[:8], "big") / 2**64


def hash_subsample(
    cfg: HashSubsampleConfig, *, verify_input_sha: bool = True
) -> HashSubsampleResult:
    """Subsample `cfg.input_dir` into `cfg.output_dir` at ~`cfg.target_n` rows.

    A molecule is kept iff `smiles_acceptance_coord` of its canonical SMILES
    is below the acceptance band `target_n / n_input_rows` (clamped to 1.0).

    Args:
        cfg: validated hash-subsample config.
        verify_input_sha: when True, hash every input shard and require it to
            match the SHA recorded in the source `MANIFEST.yaml`.

    Raises:
        ValueError: if `verify_input_sha` is True and any input shard's SHA256
            disagrees with the source manifest.
        FileNotFoundError: if `cfg.input_dir/MANIFEST.yaml` is missing.
    """
    source_manifest_bytes, _ = read_and_verify_source_manifest(
        cfg.input_dir, verify=verify_input_sha
    )

    input_shards = list_input_shards(cfg.input_dir)
    n_input_rows = count_parquet_rows(input_shards)
    accept_frac = min(1.0, cfg.target_n / n_input_rows) if n_input_rows else 1.0

    with stage_run(cfg.output_dir) as (staging_dir, started_ts):
        shards = _write_kept(cfg, staging_dir, input_shards, accept_frac)
        n_kept = sum(s.n_rows for s in shards)
        _write_stage_manifest(
            staging_dir,
            cfg=cfg,
            started_ts=started_ts,
            n_input_rows=n_input_rows,
            n_kept=n_kept,
            accept_frac=accept_frac,
            source_manifest_bytes=source_manifest_bytes,
            shards=shards,
        )

    return HashSubsampleResult(
        n_input_rows=n_input_rows,
        n_kept=n_kept,
        n_dropped=n_input_rows - n_kept,
        target_n=cfg.target_n,
        acceptance_fraction=accept_frac,
        hash_domain=cfg.hash_domain,
        shards=tuple(shards),
        output_dir=cfg.output_dir,
        started_ts=started_ts,
    )


def _write_kept(
    cfg: HashSubsampleConfig,
    staging_dir: Path,
    input_shards: list[Path],
    accept_frac: float,
) -> list[ShardInfo]:
    """Stream input shards, emit rows whose SMILES coordinate is in the band."""
    writer = ShardWriter(
        staging_dir,
        schema=RAW_V1_SCHEMA,
        shard_prefix="canon_dedup_v1",
        target_bytes=cfg.shard_target_bytes,
        compression=cfg.parquet_compression,
        compression_level=cfg.parquet_compression_level,
    )
    for shard_path in input_shards:
        pf = pq.ParquetFile(shard_path)
        for batch in pf.iter_batches(batch_size=cfg.rows_per_batch):
            if batch.num_rows == 0:
                continue
            mask = pa.array(
                [
                    smiles_acceptance_coord(smi, domain=cfg.hash_domain) < accept_frac
                    for smi in batch.column("smiles").to_pylist()
                ],
                type=pa.bool_(),
            )
            writer.write_batch(coerce_to_schema(batch.filter(mask), RAW_V1_SCHEMA))
    writer.close_current()
    return writer.shards


def _write_stage_manifest(
    staging_dir: Path,
    *,
    cfg: HashSubsampleConfig,
    started_ts: datetime,
    n_input_rows: int,
    n_kept: int,
    accept_frac: float,
    source_manifest_bytes: bytes,
    shards: list[ShardInfo],
) -> None:
    payload = {
        "schema": "canon_dedup_v1_sub",
        "name": cfg.name,
        "source_manifest_sha256": sha256_bytes(source_manifest_bytes),
        "subsample": {
            "method": "hash_partition_acceptance_band",
            "hash": "sha1",
            "hash_domain": cfg.hash_domain,
            "target_n": cfg.target_n,
            "n_input_rows": n_input_rows,
            "acceptance_fraction": accept_frac,
        },
        "started_ts": started_ts.isoformat() + "Z",
        "n_input_rows": n_input_rows,
        "n_output_rows": n_kept,
        "n_dropped": n_input_rows - n_kept,
        "n_shards": len(shards),
        "parquet_compression": cfg.parquet_compression,
        "parquet_compression_level": cfg.parquet_compression_level,
        "shards": shard_dicts(shards),
    }
    write_manifest(staging_dir, payload)
