"""Stage 0 single-file CSV/TSV ingest: one delimited corpus -> raw_v1 shards.

Shared driver behind PubChem, COCONUT, CycPeptMPDB, and tmQM. Every per-corpus
difference lives on the `CorpusConfig` from `configs/corpus/*.yaml`, so a new
delimited corpus is a YAML file, not a new module. Emits byte-budgeted Parquet
shards on the `raw_v1` schema:

    source_id: string, smiles: string, source: string, ingest_ts: timestamp[us]

Determinism: input is content-addressed via `data/MANIFEST.yaml`; reruns against
the same input + config produce identical shard contents (including `ingest_ts`,
captured once per call) and identical shard counts.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from smiles_subword._hashing import sha256_file
from smiles_subword.config import CorpusConfig
from smiles_subword.ingest._common import (
    ingest_timestamp,
    run_single_file_ingest,
    stream_csv_path,
)
from smiles_subword.ingest._common import (
    verify_input_sha as _verify_input_sha,
)

if TYPE_CHECKING:
    from smiles_subword.ingest.types import IngestResult

__all__ = ["ingest", "run_csv_ingest_cli"]


def ingest(cfg: CorpusConfig, *, verify_input_sha: bool = True) -> IngestResult:
    """Stream `cfg.raw_path` into `raw_v1` Parquet shards under `cfg.output_dir`.

    Writes shards into a sibling `.tmp` dir, then atomic-renames into place,
    always emitting a per-stage `MANIFEST.yaml`.

    Args:
        cfg: validated corpus config (column/format fields select the read mode).
        verify_input_sha: when True, require `cfg.raw_path`'s SHA to match the
            `data/MANIFEST.yaml` entry for `cfg.manifest_id`; the hash is reused
            as the manifest's `input_sha256` (file read once). Tests pass False.

    Raises:
        ValueError: if `verify_input_sha` and the file's SHA256 doesn't match.
        KeyError: if `verify_input_sha` but no manifest entry exists yet — this
            path assumes a publisher-pinned SHA (unlike ZINC-22's
            record-on-first-observation).
    """
    input_sha256 = (
        _verify_input_sha(cfg) if verify_input_sha else sha256_file(cfg.raw_path)
    )
    return run_single_file_ingest(
        cfg,
        stream_batches=lambda c, ts: stream_csv_path(c, c.raw_path, ts),
        ingest_ts=ingest_timestamp(),
        input_sha256=input_sha256,
        manifest_id=cfg.manifest_id,
    )


def run_csv_ingest_cli(doc: str | None) -> str:
    """Parse ``--config`` / ``--no-verify-sha``, run :func:`ingest`, return a summary.

    Shared body of the per-corpus CSV ingest drivers
    (``scripts/ingest/ingest_{pubchem,coconut,tmqm,cycpeptmpdb}.py``); each driver
    is a thin shim that prints the returned line. ``doc`` becomes the ``--help``
    description.
    """
    parser = argparse.ArgumentParser(description=doc)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--no-verify-sha",
        action="store_true",
        help="Skip SHA256 verification against data/MANIFEST.yaml",
    )
    args = parser.parse_args()
    cfg = CorpusConfig.from_yaml(args.config)
    result = ingest(cfg, verify_input_sha=not args.no_verify_sha)
    return (
        f"wrote {result.n_shards} shards, {result.n_rows:,} rows to {result.output_dir}"
    )
