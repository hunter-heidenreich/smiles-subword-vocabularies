"""Stage 0 ZINC-22 ingest: one tranche -> raw_v1 Parquet shards.

Downloads one `<tranche_id>.smi.gz` from `files.docking.org` (or an rsync / S3
mirror; ZINC-22 paper, JCIM 2023, doi 10.1021/acs.jcim.2c01253), then streams it
through the shared CSV reader into raw_v1 shards (`source = "zinc22"`).

SHA semantics differ from PubChem: the publisher pins no per-file SHA256, so the
first download records its observed SHA into `data/MANIFEST.yaml` and later runs
verify against it.
"""

from __future__ import annotations

import subprocess
from dataclasses import replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from smiles_subword._hashing import sha256_file
from smiles_subword.ingest._common import (
    ingest_timestamp,
    relative_to_repo,
    run_single_file_ingest,
    stream_csv_path,
)
from smiles_subword.manifest import (
    ManifestEntry,
    load_manifest_entry,
    record_manifest_entry,
)

if TYPE_CHECKING:
    from pathlib import Path

    from smiles_subword.config import Zinc22CorpusConfig
    from smiles_subword.ingest.types import IngestResult

__all__ = ["fetch_tranche", "ingest"]


def ingest(
    cfg: Zinc22CorpusConfig,
    *,
    verify_input_sha: bool = True,
    fetch: bool = True,
    manifest_path: Path | None = None,
    record_manifest: bool = True,
) -> IngestResult:
    """Download one ZINC-22 tranche and stream it into raw_v1 shards.

    Args:
        cfg: validated single-tranche config (one per tranche).
        verify_input_sha: when True and a `data/MANIFEST.yaml` entry exists for
            `cfg.manifest_id`, require `cfg.raw_path`'s SHA to match; on first
            observation the observed SHA is recorded instead. Tests pass False.
        fetch: when True, download `cfg.tranche_url` if the local file is
            missing. Tests pass False to use a pre-staged fixture.
        record_manifest: when True (default), commit a new `ManifestEntry`
            in-process if none exists; when False, return it on
            `IngestResult.pending_manifest_entry` so a concurrent orchestrator
            can serialize the writes in its main thread.

    Raises:
        ValueError: if `verify_input_sha` and the observed SHA disagrees with
            the recorded one.
    """
    if fetch and not cfg.raw_path.exists():
        cfg.raw_path.parent.mkdir(parents=True, exist_ok=True)
        fetch_tranche(cfg)

    observed_sha = sha256_file(cfg.raw_path)
    pending_entry = _reconcile_manifest(
        cfg,
        observed_sha,
        verify=verify_input_sha,
        manifest_path=manifest_path,
        record_manifest=record_manifest,
    )

    result = run_single_file_ingest(
        cfg,
        stream_batches=lambda c, ts: stream_csv_path(c, c.raw_path, ts),
        ingest_ts=ingest_timestamp(),
        input_sha256=observed_sha,
        manifest_id=cfg.manifest_id,
    )
    return replace(result, pending_manifest_entry=pending_entry)


def fetch_tranche(cfg: Zinc22CorpusConfig) -> Path:
    """Download `cfg.tranche_url` to `cfg.raw_path` via curl or rsync.

    Both transports resume (`curl -C -`, `rsync --partial --inplace`); curl over
    HTTPS is the default for `https://` URLs.

    Raises:
        ValueError: if `cfg.expected_bytes` is set and the downloaded size
            differs — a truncated download would otherwise be accepted and its
            SHA recorded as canonical on first observation.
    """
    transport = cfg.resolved_transport()
    target = str(cfg.raw_path)
    # `--` ends option parsing so a tranche URL beginning with `-` is treated as
    # a positional arg, not an injected flag (config is trusted, but cheap).
    if transport == "rsync":
        cmd = [
            "rsync",
            "-L",
            "-a",
            "--partial",
            "--inplace",
            "--",
            cfg.tranche_url,
            target,
        ]
    else:
        cmd = ["curl", "-fsSL", "-C", "-", "-o", target, "--", cfg.tranche_url]
    subprocess.run(cmd, check=True)
    if cfg.expected_bytes is not None:
        actual = cfg.raw_path.stat().st_size
        if actual != cfg.expected_bytes:
            raise ValueError(
                f"download size mismatch for {cfg.raw_path}: "
                f"expected {cfg.expected_bytes} bytes, got {actual}"
            )
    return cfg.raw_path


def _reconcile_manifest(
    cfg: Zinc22CorpusConfig,
    observed_sha: str,
    *,
    verify: bool,
    manifest_path: Path | None,
    record_manifest: bool,
) -> ManifestEntry | None:
    """Verify observed SHA against any existing manifest entry.

    Returns the ManifestEntry that should be recorded if there isn't one
    yet AND the caller asked us not to commit it ourselves. In every
    other case (`verify=False`, existing entry matches, or
    `record_manifest=True` so we wrote it inline) returns None.
    """
    if not verify:
        return None
    try:
        existing = load_manifest_entry(cfg.manifest_id, manifest_path=manifest_path)
    except KeyError:
        existing = None
    if existing is not None:
        if existing.sha256 != observed_sha:
            raise ValueError(
                f"sha256 mismatch for {cfg.raw_path}: "
                f"recorded {existing.sha256}, observed {observed_sha}"
            )
        return None
    entry = ManifestEntry(
        id=cfg.manifest_id,
        path=relative_to_repo(cfg.raw_path),
        source_url=cfg.tranche_url,
        sha256=observed_sha,
        size_bytes=cfg.raw_path.stat().st_size,
        ingest_date=(today := datetime.now(tz=UTC).date()),
        notes=(
            f"SHA recorded on first observation {today.isoformat()}; "
            "not publisher-supplied."
        ),
    )
    if record_manifest:
        record_manifest_entry(entry, manifest_path=manifest_path)
        return None
    return entry
