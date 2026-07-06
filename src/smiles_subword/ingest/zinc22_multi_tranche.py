"""Stage 0 ZINC-22 multi-tranche ingest: many tranches -> raw_v1 Parquet shards.

Orchestrates `smiles_subword.ingest.zinc22.ingest()` over the curated tranche
TSV at `cfg.tranches_path`, fanning out to `cfg.concurrency` workers. Per-tranche
atomicity, skip-if-already-ingested, and SHA recording are delegated to the
single-tranche primitive; this module adds the aggregate `MANIFEST.yaml` for the
run.
"""

from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import TYPE_CHECKING

import yaml

from smiles_subword._hashing import sha256_file
from smiles_subword._io import atomic_write_text
from smiles_subword.config import Zinc22CorpusConfig, Zinc22MultiTrancheConfig
from smiles_subword.ingest._common import (
    ingest_timestamp,
    relative_to_repo,
)
from smiles_subword.ingest.types import (
    FailedTranche,
    TranchedIngestResult,
    TrancheInfo,
)
from smiles_subword.ingest.zinc22 import ingest as ingest_one
from smiles_subword.manifest import ManifestEntry, record_manifest_entry

if TYPE_CHECKING:
    from datetime import datetime
    from pathlib import Path

__all__ = ["ingest_multi_tranche"]


@dataclass(frozen=True)
class TrancheSpec:
    """One row of the tranche TSV."""

    tranche_id: str
    generation: str
    heavy_atom_bin: int
    logp_bin: int
    url: str
    expected_bytes: int


def read_tranche_list(path: Path) -> list[TrancheSpec]:
    """Parse the curated tranche TSV into typed specs.

    Raises:
        ValueError: if the header row does not match the expected schema
            or any data row is malformed.
    """
    expected = (
        "tranche_id",
        "generation",
        "heavy_atom_bin",
        "logp_bin",
        "url",
        "expected_bytes",
    )
    with path.open() as fh:
        reader = csv.reader(fh, delimiter="\t")
        header = tuple(next(reader, []))
        if header != expected:
            raise ValueError(
                f"tranche TSV header {header} does not match expected {expected}"
            )
        specs: list[TrancheSpec] = []
        for row_index, row in enumerate(reader, start=2):
            if not row:
                continue
            if len(row) != len(expected):
                raise ValueError(f"malformed row {row_index} in {path}: {row}")
            try:
                specs.append(
                    TrancheSpec(
                        tranche_id=row[0],
                        generation=row[1],
                        heavy_atom_bin=int(row[2]),
                        logp_bin=int(row[3]),
                        url=row[4],
                        expected_bytes=int(row[5]),
                    )
                )
            except ValueError as exc:
                raise ValueError(f"malformed row {row_index} in {path}: {row}") from exc
    return specs


def ingest_multi_tranche(
    cfg: Zinc22MultiTrancheConfig,
    *,
    fetch: bool = True,
    verify_input_sha: bool = True,
    manifest_path: Path | None = None,
    limit: int | None = None,
) -> TranchedIngestResult:
    """Run the multi-tranche ZINC-22 ingest.

    Args:
        cfg: validated multi-tranche config.
        fetch: when False, skip per-tranche downloads (tests pass False
            against pre-staged synthetic raw files).
        verify_input_sha: passed through to each per-tranche `ingest()`.
        manifest_path: passed through to each per-tranche `ingest()`
            (tests redirect the global manifest).
        limit: optional cap on the number of tranches processed
            (sizing run).
    """
    specs = read_tranche_list(cfg.tranches_path)
    if limit is not None:
        specs = specs[:limit]

    cfg.raw_root.mkdir(parents=True, exist_ok=True)
    cfg.output_root.mkdir(parents=True, exist_ok=True)

    ingest_ts = ingest_timestamp()
    tranches: list[TrancheInfo] = []
    failures: list[FailedTranche] = []

    pending: list[tuple[TrancheSpec, Zinc22CorpusConfig]] = []
    for spec in specs:
        per_cfg = _per_tranche_config(spec, cfg)
        if _is_complete(per_cfg.output_dir):
            tranches.append(_load_existing(spec, per_cfg))
            continue
        pending.append((spec, per_cfg))

    pending_entries: dict[str, ManifestEntry] = {}
    if pending:
        with ThreadPoolExecutor(max_workers=cfg.concurrency) as pool:
            futures = {
                pool.submit(
                    ingest_one,
                    per_cfg,
                    verify_input_sha=verify_input_sha,
                    fetch=fetch,
                    manifest_path=manifest_path,
                    record_manifest=False,
                ): (spec, per_cfg)
                for spec, per_cfg in pending
            }
            for fut in as_completed(futures):
                spec, per_cfg = futures[fut]
                try:
                    result = fut.result()
                except Exception as exc:  # noqa: BLE001
                    failures.append(
                        FailedTranche(spec.tranche_id, f"{type(exc).__name__}: {exc}")
                    )
                    continue
                if result.pending_manifest_entry is not None:
                    pending_entries[spec.tranche_id] = result.pending_manifest_entry
                tranches.append(
                    TrancheInfo(
                        tranche_id=spec.tranche_id,
                        manifest_id=per_cfg.manifest_id,
                        output_dir=result.output_dir,
                        n_rows=result.n_rows,
                        n_shards=result.n_shards,
                        n_bytes=sum(s.n_bytes for s in result.shards),
                        skipped=False,
                    )
                )

    for tranche_id in sorted(pending_entries):
        record_manifest_entry(pending_entries[tranche_id], manifest_path=manifest_path)

    tranches.sort(key=lambda t: t.tranche_id)
    n_rows = sum(t.n_rows for t in tranches)

    _write_aggregate_manifest(
        cfg=cfg,
        ingest_ts=ingest_ts,
        tranches=tranches,
        failures=failures,
    )

    return TranchedIngestResult(
        n_rows=n_rows,
        output_root=cfg.output_root,
        ingest_ts=ingest_ts,
        tranches=tuple(tranches),
        failures=tuple(failures),
    )


def _per_tranche_config(
    spec: TrancheSpec, cfg: Zinc22MultiTrancheConfig
) -> Zinc22CorpusConfig:
    return Zinc22CorpusConfig(
        name=f"zinc22-multi-tranche-{spec.tranche_id}",
        manifest_id=f"zinc22-{spec.tranche_id}",
        tranche_id=spec.tranche_id,
        tranche_url=spec.url,
        transport=cfg.transport,
        expected_bytes=spec.expected_bytes,
        raw_path=cfg.raw_root / f"{spec.tranche_id}.smi.gz",
        output_dir=cfg.output_root / spec.tranche_id,
        smiles_column=cfg.smiles_column,
        id_column=cfg.id_column,
        has_header=cfg.has_header,
        delim=cfg.delim,
        file_compression=cfg.file_compression,
        shard_target_bytes=cfg.shard_target_bytes,
        rows_per_batch=cfg.rows_per_batch,
        parquet_compression=cfg.parquet_compression,
        parquet_compression_level=cfg.parquet_compression_level,
    )


def _is_complete(output_dir: Path) -> bool:
    """True if `output_dir` holds a complete raw_v1 stage manifest.

    Validates a parseable `MANIFEST.yaml` with `schema: raw_v1` and the fields
    `_load_existing` reads back (`n_rows`, `n_shards`, a `shards` list of
    `n_bytes`-bearing entries), not bare existence — so a partial/corrupt
    manifest counts as incomplete (re-ingested) instead of tripping a `KeyError`
    on resume.
    """
    manifest = output_dir / "MANIFEST.yaml"
    if not manifest.exists():
        return False
    try:
        payload = yaml.safe_load(manifest.read_text())
    except (OSError, yaml.YAMLError):
        return False
    if not isinstance(payload, dict) or payload.get("schema") != "raw_v1":
        return False
    if "n_rows" not in payload or "n_shards" not in payload:
        return False
    shards = payload.get("shards")
    return isinstance(shards, list) and all(
        isinstance(s, dict) and "n_bytes" in s for s in shards
    )


def _load_existing(spec: TrancheSpec, per_cfg: Zinc22CorpusConfig) -> TrancheInfo:
    with (per_cfg.output_dir / "MANIFEST.yaml").open() as fh:
        payload = yaml.safe_load(fh)
    shards = payload["shards"]
    return TrancheInfo(
        tranche_id=spec.tranche_id,
        manifest_id=per_cfg.manifest_id,
        output_dir=per_cfg.output_dir,
        n_rows=int(payload["n_rows"]),
        n_shards=int(payload["n_shards"]),
        n_bytes=sum(int(s["n_bytes"]) for s in shards),
        skipped=True,
    )


def _write_aggregate_manifest(
    *,
    cfg: Zinc22MultiTrancheConfig,
    ingest_ts: datetime,
    tranches: list[TrancheInfo],
    failures: list[FailedTranche],
) -> None:
    payload: dict[str, object] = {
        "schema": "raw_v1",
        "layout": "multi_tranche",
        "source": cfg.source,
        "ingest_ts": ingest_ts.isoformat() + "Z",
        "tranches_path": str(relative_to_repo(cfg.tranches_path)),
        "tranches_sha256": sha256_file(cfg.tranches_path),
        "n_tranches": len(tranches),
        "n_rows": sum(t.n_rows for t in tranches),
        "n_shards": sum(t.n_shards for t in tranches),
        "n_bytes": sum(t.n_bytes for t in tranches),
        "tranches": [
            {
                "tranche_id": t.tranche_id,
                "manifest_id": t.manifest_id,
                "output_dir": str(relative_to_repo(t.output_dir)),
                "n_rows": t.n_rows,
                "n_shards": t.n_shards,
                "n_bytes": t.n_bytes,
                "skipped": t.skipped,
            }
            for t in tranches
        ],
        "failures": [{"tranche_id": f.tranche_id, "error": f.reason} for f in failures],
    }
    atomic_write_text(
        cfg.output_root / "MANIFEST.yaml",
        yaml.safe_dump(payload, sort_keys=False),
    )
