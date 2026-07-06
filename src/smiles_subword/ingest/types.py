"""Internal value types for the ingest stage."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from smiles_subword.manifest import ManifestEntry, ShardInfo


@dataclass(frozen=True)
class IngestResult:
    """Outcome of one `ingest()` call.

    `pending_manifest_entry` is set only when the caller passed
    `record_manifest=False` and the run observed a SHA not yet in
    `data/MANIFEST.yaml`. Concurrent orchestrators (the multi-tranche ingest) use
    it to commit entries serially in the main thread, sidestepping the
    read-modify-write race in `record_manifest_entry`.
    """

    n_rows: int
    output_dir: Path
    ingest_ts: datetime
    shards: tuple[ShardInfo, ...]
    pending_manifest_entry: ManifestEntry | None = None

    @property
    def n_shards(self) -> int:
        """Number of Parquet shards written."""
        return len(self.shards)


@dataclass(frozen=True)
class TrancheInfo:
    """Per-tranche summary inside a `TranchedIngestResult`."""

    tranche_id: str
    manifest_id: str
    output_dir: Path
    n_rows: int
    n_shards: int
    n_bytes: int
    skipped: bool


@dataclass(frozen=True)
class FailedTranche:
    """One tranche that errored during a `TranchedIngestResult` run.

    `reason` is the formatted exception (`<ExcType>: <message>`).
    """

    tranche_id: str
    reason: str


@dataclass(frozen=True)
class TranchedIngestResult:
    """Outcome of one `ingest_multi_tranche()` call.

    The concurrent, many-tranche path, as opposed to the single-pass
    `IngestResult`.
    """

    n_rows: int
    output_root: Path
    ingest_ts: datetime
    tranches: tuple[TrancheInfo, ...]
    failures: tuple[FailedTranche, ...]

    @property
    def n_tranches(self) -> int:
        """Number of tranches successfully ingested."""
        return len(self.tranches)

    @property
    def n_shards(self) -> int:
        """Total Parquet shards across all ingested tranches."""
        return sum(t.n_shards for t in self.tranches)
