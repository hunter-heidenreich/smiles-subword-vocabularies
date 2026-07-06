"""Deterministic train/test split.

A molecule lands in the held-out `test` split iff a SHA1-of-`source_id`
coordinate falls below the effective threshold
`min(test_fraction, test_cap / n_input_rows)` — the absolute `test_cap` (1e6)
binds when `test_fraction` (5%) alone would exceed it (as on PubChem and the
REAL-Space anchor). The continuous coordinate makes the realised test fraction
exact rather than rounded to a leading-hex-digit partition.

Keying off `source_id` (the subsample `hash_subsample.py` keys off the canonical
SMILES) keeps the two partitions on independent hash domains, uncorrelated.

Consumes the conformance stage's Parquet directory (`conformant_v1_sub` for the
subsampled PubChem/ZINC-22 corpora, `conformant_v1_full` for COCONUT/REAL-Space)
and writes `train/` and `test/` subdirectories under `canon_dedup_v1`, each
`raw_v1`-schema with its own `MANIFEST.yaml`, so the tokenize layer's
whole-directory corpus consumers work unchanged.
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
from smiles_subword.preprocess.types import HoldoutSplitResult, ShardInfo

if TYPE_CHECKING:
    from datetime import datetime
    from pathlib import Path

    from smiles_subword.config import HoldoutSplitConfig

__all__ = ["source_id_split_coord", "split_train_test"]


def source_id_split_coord(source_id: object, *, seed: int, domain: str) -> float:
    """Map a `source_id` to a uniform ``[0, 1)`` coordinate.

    SHA1 of ``f"{domain}|{seed}|{source_id}"``, leading 64 bits as a
    big-endian uint64 divided by ``2**64`` — a continuous generalization of
    a leading-hex-digit partition, so a fractional
    cutoff (e.g. exactly 5%) is reachable. `source_id` is coerced to `str`.
    """
    key = f"{domain}|{seed}|{source_id}"
    digest = hashlib.sha1(key.encode(), usedforsecurity=False)
    return int.from_bytes(digest.digest()[:8], "big") / 2**64


def split_train_test(
    cfg: HoldoutSplitConfig, *, verify_input_sha: bool = True
) -> HoldoutSplitResult:
    """Split `cfg.input_dir` into `cfg.output_dir/{train,test}/`.

    Args:
        cfg: validated holdout-split config.
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
    cap_thr = cfg.test_cap / n_input_rows if n_input_rows else cfg.test_fraction
    cap_bound = cap_thr < cfg.test_fraction
    effective_thr = min(cfg.test_fraction, cap_thr)
    source_sha = sha256_bytes(source_manifest_bytes)

    with stage_run(cfg.output_dir) as (staging_dir, started_ts):
        train_dir = staging_dir / "train"
        test_dir = staging_dir / "test"
        train_dir.mkdir()
        test_dir.mkdir()
        train_shards, test_shards = _write_split(
            cfg, train_dir, test_dir, input_shards, effective_thr
        )
        for subdir, split, shards in (
            (train_dir, "train", train_shards),
            (test_dir, "test", test_shards),
        ):
            _write_split_manifest(
                subdir,
                cfg=cfg,
                split=split,
                started_ts=started_ts,
                effective_thr=effective_thr,
                cap_bound=cap_bound,
                n_input_rows=n_input_rows,
                source_manifest_sha256=source_sha,
                shards=shards,
            )

    return HoldoutSplitResult(
        n_input_rows=n_input_rows,
        n_train=sum(s.n_rows for s in train_shards),
        n_test=sum(s.n_rows for s in test_shards),
        test_fraction=cfg.test_fraction,
        test_cap=cfg.test_cap,
        effective_threshold=effective_thr,
        cap_bound=cap_bound,
        seed=cfg.seed,
        train_shards=tuple(train_shards),
        test_shards=tuple(test_shards),
        output_dir=cfg.output_dir,
        started_ts=started_ts,
    )


def _new_writer(cfg: HoldoutSplitConfig, subdir: Path) -> ShardWriter:
    return ShardWriter(
        subdir,
        schema=RAW_V1_SCHEMA,
        shard_prefix="canon_dedup_v1",
        target_bytes=cfg.shard_target_bytes,
        compression=cfg.parquet_compression,
        compression_level=cfg.parquet_compression_level,
    )


def _write_split(
    cfg: HoldoutSplitConfig,
    train_dir: Path,
    test_dir: Path,
    input_shards: list[Path],
    effective_thr: float,
) -> tuple[list[ShardInfo], list[ShardInfo]]:
    """Stream input shards once, demux each row by its split coordinate."""
    train_writer = _new_writer(cfg, train_dir)
    test_writer = _new_writer(cfg, test_dir)
    for shard_path in input_shards:
        pf = pq.ParquetFile(shard_path)
        for batch in pf.iter_batches(batch_size=cfg.rows_per_batch):
            if batch.num_rows == 0:
                continue
            is_test = [
                source_id_split_coord(sid, seed=cfg.seed, domain=cfg.hash_domain)
                < effective_thr
                for sid in batch.column("source_id").to_pylist()
            ]
            test_mask = pa.array(is_test, type=pa.bool_())
            train_mask = pa.array([not b for b in is_test], type=pa.bool_())
            test_writer.write_batch(
                coerce_to_schema(batch.filter(test_mask), RAW_V1_SCHEMA)
            )
            train_writer.write_batch(
                coerce_to_schema(batch.filter(train_mask), RAW_V1_SCHEMA)
            )
    train_writer.close_current()
    test_writer.close_current()
    return train_writer.shards, test_writer.shards


def _write_split_manifest(
    subdir: Path,
    *,
    cfg: HoldoutSplitConfig,
    split: str,
    started_ts: datetime,
    effective_thr: float,
    cap_bound: bool,
    n_input_rows: int,
    source_manifest_sha256: str,
    shards: list[ShardInfo],
) -> None:
    payload = {
        "schema": "canon_dedup_v1",
        "name": cfg.name,
        "split": split,
        "source_manifest_sha256": source_manifest_sha256,
        "split_params": {
            "method": "sha1_source_id_partition",
            "hash": "sha1",
            "hash_domain": cfg.hash_domain,
            "seed": cfg.seed,
            "test_fraction": cfg.test_fraction,
            "test_cap": cfg.test_cap,
            "effective_threshold": effective_thr,
            "cap_bound": cap_bound,
            "n_input_rows": n_input_rows,
        },
        "started_ts": started_ts.isoformat() + "Z",
        "n_output_rows": sum(s.n_rows for s in shards),
        "n_shards": len(shards),
        "parquet_compression": cfg.parquet_compression,
        "parquet_compression_level": cfg.parquet_compression_level,
        "shards": shard_dicts(shards),
    }
    write_manifest(subdir, payload)
