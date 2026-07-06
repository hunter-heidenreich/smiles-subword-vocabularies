"""Internal value types for the preprocess stage."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from smiles_subword.manifest import ShardInfo


@dataclass(frozen=True)
class TrancheUnionResult:
    """Outcome of one `consolidate_tranches()` call.

    `n_discovered` is the tranche count in the enumerated tranche list;
    `n_used` is the count actually present in the nested per-tranche raw_v1
    ingest.
    `excluded` maps each discovered-but-absent tranche id to its recorded
    reason. `tranches_path` / `tranches_sha256` pin the enumerated list.
    """

    n_discovered: int
    n_used: int
    n_rows: int
    excluded: dict[str, str]
    tranches_path: str
    tranches_sha256: str
    shards: tuple[ShardInfo, ...]
    output_dir: Path
    started_ts: datetime


@dataclass(frozen=True)
class CanonDedupResult:
    """Outcome of one `canon_dedup()` call.

    `n_rdkit_rejected` counts rows RDKit could not parse or canonicalize;
    `rdkit_rejection_rate` is that count over `n_input_rows` (0.0 for an
    empty corpus). `rdkit_version` is the resolved RDKit version that
    produced the canonical SMILES — canonical output is version-dependent.
    """

    n_input_rows: int
    n_rdkit_rejected: int
    n_canonical_rows: int
    n_duplicates: int
    n_output_rows: int
    rdkit_rejection_rate: float
    rdkit_version: str
    shards: tuple[ShardInfo, ...]
    output_dir: Path
    started_ts: datetime


@dataclass(frozen=True)
class HashSubsampleResult:
    """Outcome of one `hash_subsample()` call.

    `acceptance_fraction` is the realised band width `target_n / n_input_rows`
    (clamped to 1.0). The subsample is uniform-in-expectation, so `n_kept`
    fluctuates around `target_n` by binomial noise rather than equalling it.
    """

    n_input_rows: int
    n_kept: int
    n_dropped: int
    target_n: int
    acceptance_fraction: float
    hash_domain: str
    shards: tuple[ShardInfo, ...]
    output_dir: Path
    started_ts: datetime


@dataclass(frozen=True)
class HoldoutSplitResult:
    """Outcome of one `split_train_test()` call.

    `effective_threshold` is the SHA1-coordinate cutoff for the test split:
    `min(test_fraction, test_cap / n_input_rows)`. `cap_bound` is True when the
    absolute `test_cap` would have been exceeded by `test_fraction` alone.
    """

    n_input_rows: int
    n_train: int
    n_test: int
    test_fraction: float
    test_cap: int
    effective_threshold: float
    cap_bound: bool
    seed: int
    train_shards: tuple[ShardInfo, ...]
    test_shards: tuple[ShardInfo, ...]
    output_dir: Path
    started_ts: datetime


@dataclass(frozen=True)
class ConformanceResult:
    """Counts from one conformance-filter pass."""

    n_input_rows: int
    n_kept: int
    n_dropped: int
    output_dir: Path
    deposit_path: Path

    @property
    def drop_rate(self) -> float:
        """Fraction of input molecules dropped as non-conformant."""
        return self.n_dropped / self.n_input_rows if self.n_input_rows else 0.0
