"""Flatten an enumerated nested raw_v1 tranche set into one raw_v1 directory.

The ZINC-22 raw_v1 ingest is a nested per-tranche layout: one `<tranche_id>/`
subdirectory per ingested tranche under `input_dir`, each a self-contained
raw_v1 shard set with its own `MANIFEST.yaml`, plus a top-level aggregate
`MANIFEST.yaml` (`layout: multi_tranche`) enumerating every tranche. `canon_dedup`
consumes a *flat* raw_v1 directory (globs `*.parquet` non-recursively), so this
bridge hard-links every tranche's shards into one flat `output_dir` and writes a
flat raw_v1 `MANIFEST.yaml`.

The flat manifest carries a `tranche_union` provenance block — the enumerated
tranche-list path + SHA256, the discovered/used counts, and the
discovered-but-not-ingested exclusions with reasons — locking which ZINC-22
tranches the corpus draw is confined to (surfaced into `data/MANIFEST.yaml` by
`register_processed_corpus`).

Hard-linking keeps the bridge near-free in time and disk; a cross-device
`output_dir` falls back to a byte copy.
"""

from __future__ import annotations

import csv
import shutil
from typing import TYPE_CHECKING

from smiles_subword._hashing import sha256_bytes
from smiles_subword.paths import REPO_ROOT
from smiles_subword.preprocess._io import (
    read_source_manifest,
    shard_dicts,
    stage_run,
    verify_shard_sha256,
    write_manifest,
)
from smiles_subword.preprocess.types import ShardInfo, TrancheUnionResult

if TYPE_CHECKING:
    from pathlib import Path

    from smiles_subword.config import TrancheUnionConfig

__all__ = ["consolidate_tranches"]


def consolidate_tranches(
    cfg: TrancheUnionConfig, *, verify_input_sha: bool = True
) -> TrancheUnionResult:
    """Flatten the enumerated nested tranche set under `cfg.input_dir`.

    Hard-links every ingested tranche's raw_v1 shards into `cfg.output_dir`
    as a flat, sequentially-named shard set and writes a flat raw_v1
    `MANIFEST.yaml` carrying the `tranche_union` provenance block.

    Args:
        cfg: validated tranche-union config.
        verify_input_sha: when True, hash every tranche shard and require it
            to match the SHA recorded in that tranche's `MANIFEST.yaml`.

    Raises:
        ValueError: if `cfg.input_dir` is not a nested per-tranche raw_v1
            directory; if
            the enumerated tranche list's SHA256 disagrees with the one the
            aggregate manifest records; if the discovered-but-not-ingested
            tranche set differs from `cfg.expected_exclusions`; if a tranche
            shard's SHA256 fails verification; or if the consolidated row
            count disagrees with the aggregate manifest.
        FileNotFoundError: if a `MANIFEST.yaml` or tranche list is missing.
    """
    _, aggregate = read_source_manifest(cfg.input_dir)
    if aggregate.get("layout") != "multi_tranche":
        raise ValueError(
            f"{cfg.input_dir} is not a nested per-tranche raw_v1 directory "
            f"(manifest layout={aggregate.get('layout')!r}); tranche_union "
            "consumes only the per-tranche nested layout"
        )

    used = {
        t["tranche_id"]: t
        for t in aggregate.get("tranches", [])
        if not t.get("skipped", False)
    }
    tranches_path = str(aggregate["tranches_path"])
    tranches_sha256 = str(aggregate["tranches_sha256"])
    discovered = _read_tranche_list(tranches_path, tranches_sha256)

    excluded = {
        tid: cfg.expected_exclusions.get(tid, "") for tid in discovered - set(used)
    }
    if set(excluded) != set(cfg.expected_exclusions):
        raise ValueError(
            "discovered-but-not-ingested tranche set "
            f"{sorted(excluded)} does not match expected_exclusions "
            f"{sorted(cfg.expected_exclusions)} — reconcile before rerunning"
        )

    with stage_run(cfg.output_dir) as (staging_dir, started_ts):
        shards = _link_tranche_shards(
            cfg, staging_dir, used, verify_input_sha=verify_input_sha
        )
        n_rows = sum(s.n_rows for s in shards)
        expected_rows = aggregate.get("n_rows")
        if expected_rows is not None and n_rows != expected_rows:
            raise ValueError(
                f"consolidated {n_rows} rows but aggregate manifest "
                f"records {expected_rows} — a tranche is incomplete"
            )
        _write_flat_manifest(
            staging_dir,
            shards=shards,
            n_rows=n_rows,
            n_discovered=len(discovered),
            tranches_path=tranches_path,
            tranches_sha256=tranches_sha256,
            excluded=excluded,
        )

    return TrancheUnionResult(
        n_discovered=len(discovered),
        n_used=len(used),
        n_rows=n_rows,
        excluded=excluded,
        tranches_path=tranches_path,
        tranches_sha256=tranches_sha256,
        shards=tuple(shards),
        output_dir=cfg.output_dir,
        started_ts=started_ts,
    )


def _read_tranche_list(tranches_path: str, expected_sha256: str) -> set[str]:
    """Return the `tranche_id` set from the enumerated tranche-list TSV.

    The list is the discovered tranche universe — a superset of what the
    raw_v1 ingest actually landed. Its SHA256 must match the digest the
    aggregate manifest pinned at ingest time.
    """
    path = REPO_ROOT / tranches_path
    raw = path.read_bytes()
    actual = sha256_bytes(raw)
    if actual != expected_sha256:
        raise ValueError(
            f"tranche list {tranches_path} sha256 {actual} != "
            f"manifest-recorded {expected_sha256}"
        )
    reader = csv.DictReader(raw.decode().splitlines(), delimiter="\t")
    return {row["tranche_id"] for row in reader}


def _link_tranche_shards(
    cfg: TrancheUnionConfig,
    staging_dir: Path,
    used: dict[str, dict],
    *,
    verify_input_sha: bool,
) -> list[ShardInfo]:
    """Hard-link every ingested tranche's shards into `staging_dir`, flat."""
    shards: list[ShardInfo] = []
    for tranche_id in sorted(used):
        tranche_dir = cfg.input_dir / tranche_id
        _, manifest = read_source_manifest(tranche_dir)
        tranche_shards = manifest.get("shards", [])
        if verify_input_sha:
            verify_shard_sha256(
                (tranche_dir / s["file"], s["sha256"]) for s in tranche_shards
            )
        for shard in tranche_shards:
            flat_name = f"raw_v1-{len(shards):05d}.parquet"
            _link_or_copy(tranche_dir / shard["file"], staging_dir / flat_name)
            shards.append(
                ShardInfo(
                    file=flat_name,
                    sha256=str(shard["sha256"]),
                    n_rows=int(shard["n_rows"]),
                    n_bytes=int(shard["n_bytes"]),
                )
            )
    return shards


def _link_or_copy(src: Path, dst: Path) -> None:
    """Hard-link `src` to `dst`; fall back to a byte copy across devices."""
    try:
        dst.hardlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def _write_flat_manifest(
    staging_dir: Path,
    *,
    shards: list[ShardInfo],
    n_rows: int,
    n_discovered: int,
    tranches_path: str,
    tranches_sha256: str,
    excluded: dict[str, str],
) -> None:
    payload = {
        "schema": "raw_v1",
        "source": "zinc22",
        "n_rows": n_rows,
        "n_shards": len(shards),
        "tranche_union": {
            "tranches_path": tranches_path,
            "tranches_sha256": tranches_sha256,
            "n_discovered": n_discovered,
            "n_used": n_discovered - len(excluded),
            "n_rows": n_rows,
            "excluded": [
                {"tranche_id": tid, "reason": excluded[tid]} for tid in sorted(excluded)
            ],
        },
        "shards": shard_dicts(shards),
    }
    write_manifest(staging_dir, payload)
