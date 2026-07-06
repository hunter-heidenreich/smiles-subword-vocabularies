"""Pydantic schemas for the pipeline's on-disk YAML configs.

Shared infrastructure first, then one config per pipeline stage in run order:
Stage 0 ingest, the tranche-union bridge, the ``canon_dedup_v1`` preprocessing
stages, and the Stage 5 tokenizer. Every config is ``extra="forbid",
frozen=True`` and loads via ``from_yaml``.
"""

import os
from pathlib import Path
from typing import Annotated, Literal, Self

import yaml
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from smiles_subword.paths import REPO_ROOT

# --- Shared infrastructure ---------------------------------------------------


def _default_n_workers() -> int:
    return max(1, (os.cpu_count() or 2) - 1)


def _resolve_path(value: Path) -> Path:
    return value if value.is_absolute() else (REPO_ROOT / value)


def _resolve_path_or_none(value: Path | None) -> Path | None:
    return _resolve_path(value) if value is not None else None


RepoPath = Annotated[Path, AfterValidator(_resolve_path)]
"""Repo-relative path resolved against ``REPO_ROOT`` at validation; absolute
paths pass through unchanged."""

RepoPathOrNone = Annotated[Path | None, AfterValidator(_resolve_path_or_none)]
"""Optional :data:`RepoPath` — ``None`` is preserved, paths resolve."""


class YamlLoadable(BaseModel):
    """Mixin: ``cls.from_yaml(path)`` reads + validates a YAML file."""

    @classmethod
    def from_yaml(cls, path: Path) -> Self:
        return cls.model_validate(yaml.safe_load(Path(path).read_text()))


class ParquetShardOutputConfig(BaseModel):
    """Parquet shard-output knobs shared by the ingest + preprocess stages.

    ``rows_per_batch`` is *not* here — its lower bound differs by stage (ingest
    ``ge=64``, preprocess ``ge=1``), so each config declares it.
    """

    shard_target_bytes: int = Field(default=256 * 2**20, ge=1024)
    parquet_compression: Literal["zstd", "snappy", "gzip"] = "zstd"
    parquet_compression_level: int = Field(default=3, ge=1, le=22)


# --- Stage 0: ingest (raw_v1) ------------------------------------------------


class CsvFormatConfig(YamlLoadable):
    """Shared delimited-file read settings for Stage 0 ingest, driving the
    config-driven CSV reader (`smiles_subword.ingest._common.stream_csv_path`).

    Two read modes: ``positional`` (headerless 2-column files — PubChem,
    ZINC-22, REAL-Space; typed ``columns`` schema, auto-detect off) and
    ``named`` (header CSVs, two columns selected by optionally-normalized name —
    COCONUT, CycPeptMPDB, tmQM; auto-detect on). Defaults match PubChem
    CID-SMILES. `disable_quoting` captures ``"`` / ``\\`` fields verbatim;
    `coalesce_null_smiles` maps NULL SMILES to ``''`` (schema non-nullable);
    `drop_null_smiles` filters those rows out.

    In ``named`` mode the reader hardcodes a header, reads uncompressed, and
    forces the id to ``VARCHAR``, so ``has_header`` / ``file_compression`` /
    ``id_column_type`` / ``positional_id_first`` / ``disable_quoting`` are inert
    and ``normalize_names`` applies only there. `_reject_inert_named_mode_fields`
    rejects the two inert cases that would fail silently.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    smiles_column: str = "smiles"
    id_column: str = "id"
    id_column_type: Literal["VARCHAR", "BIGINT"] = "VARCHAR"
    delim: str = "\t"
    has_header: bool = False
    file_compression: Literal["gzip", "zstd", "none"] = "none"
    csv_read_mode: Literal["positional", "named"] = "positional"
    positional_id_first: bool = True
    normalize_names: bool = False
    drop_null_smiles: bool = False
    coalesce_null_smiles: bool = False
    disable_quoting: bool = False

    @model_validator(mode="after")
    def _reject_inert_named_mode_fields(self) -> Self:
        if self.csv_read_mode != "named":
            return self
        if self.file_compression != "none":
            raise ValueError(
                "file_compression is unsupported in csv_read_mode='named' "
                "(the named reader cannot decompress); set file_compression='none'"
            )
        if not self.has_header:
            raise ValueError(
                "csv_read_mode='named' requires has_header=True "
                "(columns are selected by name, which needs a header row)"
            )
        return self


class CorpusConfig(CsvFormatConfig, ParquetShardOutputConfig):
    """Stage 0 ingest config for a single delimited corpus.

    Pairs with a `data/MANIFEST.yaml` entry via `manifest_id` (not a duplicated
    SHA256). Format-field defaults reproduce the PubChem CID-SMILES layout (gzip
    TSV, no header, positional read, integer id); other corpora override.
    """

    name: str
    source: str
    manifest_id: str
    raw_path: RepoPath
    output_dir: RepoPath
    id_column: str = "cid"
    id_column_type: Literal["VARCHAR", "BIGINT"] = "BIGINT"
    file_compression: Literal["gzip", "zstd", "none"] = "gzip"
    rows_per_batch: int = Field(default=131072, ge=64)


class Zinc22CorpusConfig(CsvFormatConfig, ParquetShardOutputConfig):
    """Stage 0 ingest config for one ZINC-22 tranche.

    Sibling of `CorpusConfig`, not an extension: SHA-anchor semantics differ —
    PubChem is publisher-supplied (`load_manifest_entry` must succeed first),
    ZINC-22 is record-on-first-observation (the first download writes the
    `data/MANIFEST.yaml` entry). `tranche_url` may be `https://...` or
    `rsync://...`; transport is picked by URL scheme unless `transport` is set.
    Format fields describe the `<tranche_id>.smi.gz` payload.
    """

    name: str
    source: Literal["zinc22"] = "zinc22"
    manifest_id: str
    tranche_id: str = Field(min_length=1)
    tranche_url: str = Field(min_length=1)
    transport: Literal["auto", "curl", "rsync"] = "auto"
    expected_bytes: int | None = Field(default=None, ge=0)
    raw_path: RepoPath
    output_dir: RepoPath
    id_column: str = "zinc_id"
    positional_id_first: bool = False
    file_compression: Literal["gzip", "zstd", "none"] = "gzip"
    rows_per_batch: int = Field(default=131072, ge=64)

    def resolved_transport(self) -> Literal["curl", "rsync"]:
        """Pick a transport from `transport` or fall back to URL scheme."""
        if self.transport != "auto":
            return self.transport
        if self.tranche_url.startswith("rsync://"):
            return "rsync"
        return "curl"


class Zinc22MultiTrancheConfig(CsvFormatConfig, ParquetShardOutputConfig):
    """Stage 0 ingest config for the concurrent, multi-tranche ZINC-22 run.

    Orchestrates `Zinc22CorpusConfig`-shaped per-tranche ingests over the list
    at `tranches_path`, fanning out to `concurrency` parallel workers.
    """

    name: str
    source: Literal["zinc22"] = "zinc22"
    tranches_path: RepoPath
    raw_root: RepoPath
    output_root: RepoPath
    concurrency: int = Field(default=4, ge=1, le=32)
    transport: Literal["auto", "curl", "rsync"] = "auto"
    id_column: str = "zinc_id"
    positional_id_first: bool = False
    file_compression: Literal["gzip", "zstd", "none"] = "gzip"
    rows_per_batch: int = Field(default=131072, ge=64)


class RealSpaceCorpusConfig(CsvFormatConfig, ParquetShardOutputConfig):
    """Stage 0 ingest config for the REAL-Space corpus.

    REAL-Space arrives as a *set* of pre-staged Enamine REAL `.cxsmiles` files,
    so this carries `raw_dir` + `glob` (not a single `raw_path`) and has no
    `manifest_id`: the file set is pinned by per-file SHA256 in the emitted
    `raw_v1` manifest, and the global `data/MANIFEST.yaml` lock lands at
    preprocessing time. Format fields default to the Enamine REAL convention
    (tab-delimited, `smiles`/`id` header). SMILES is captured verbatim,
    including any CXSMILES ` |...|` block — `canon_dedup_v1` parses/rejects it.
    """

    name: str
    source: Literal["real_space"] = "real_space"
    raw_dir: RepoPath
    glob: str = Field(default="*.cxsmiles", min_length=1)
    has_header: bool = True
    positional_id_first: bool = False
    disable_quoting: bool = True
    coalesce_null_smiles: bool = True
    output_dir: RepoPath
    rows_per_batch: int = Field(default=131072, ge=64)


# --- Tranche-union bridge (flatten nested raw_v1) ----------------------------


class TrancheUnionConfig(YamlLoadable):
    """Flatten an enumerated nested raw_v1 tranche set into one raw_v1 dir.

    The ZINC-22 raw_v1 ingest is nested per-tranche (`input_dir` holds one
    `<tranche_id>/` subdir per tranche plus a top-level `MANIFEST.yaml`,
    `layout: multi_tranche`); `canon_dedup` needs a flat raw_v1 directory, so
    this stage hard-links every tranche's shards into one flat `output_dir` with
    a flat manifest carrying the tranche-set provenance.

    `expected_exclusions` maps each discovered-but-not-ingested tranche to a
    reason. The stage asserts discovered-minus-ingested equals exactly these
    keys, so a silently missing tranche is a hard failure, not a shrunk corpus.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    input_dir: RepoPath
    output_dir: RepoPath
    expected_exclusions: dict[str, str] = Field(default_factory=dict)


# --- Preprocessing (canon_dedup_v1) ------------------------------------------


class CanonDedupConfig(YamlLoadable, ParquetShardOutputConfig):
    """`canon_dedup_v1` pipeline config for a single corpus.

    Consumes a `raw_v1` Parquet dir under `input_dir`, writes a `canon_dedup_v1`
    dir under `output_dir`: RDKit isomeric canonicalization then exact-string
    dedup, nothing else — no charge neutralization, salt-stripping, or heavy-atom
    cap, and no knobs to request them (`extra="forbid"`). Output keeps the
    4-field `raw_v1` schema.

    `mode`: `single_pass` (default) is fine through ~14M rows; `bucket` bounds
    the intermediate working set, required for the 50M+ corpora. `n_workers`
    fans the bottleneck canonicalization across a process pool; output is
    `n_workers`-independent (the dedup re-sorts by `smiles`).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    input_dir: RepoPath
    output_dir: RepoPath
    mode: Literal["single_pass", "bucket"] = "single_pass"
    n_workers: int = Field(default_factory=_default_n_workers, ge=1)
    rows_per_batch: int = Field(default=131072, ge=1)
    duckdb_threads: int | None = Field(default=None, ge=1)
    duckdb_memory_limit: str | None = None


class HashSubsampleConfig(YamlLoadable, ParquetShardOutputConfig):
    """`canon_dedup_v1` hash-partition subsample config.

    Consumes a `canon_dedup_v1` dir under `input_dir`, writes a same-schema
    (`raw_v1`) dir under `output_dir`, keeping a molecule iff a stable hash of
    its *canonical SMILES* falls in the band `[0, target_n / n_input_rows)`. Kept
    count fluctuates around `target_n` by binomial noise.

    Keying off canonical SMILES (never `source_id`) keeps this uncorrelated with
    the `source_id`-keyed split (`HoldoutSplitConfig`); `hash_domain` namespaces
    the SHA1 input as a second independence guarantee.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    input_dir: RepoPath
    output_dir: RepoPath
    target_n: int = Field(ge=1)
    hash_domain: str = Field(default="canon_dedup_v1.subsample", min_length=1)
    rows_per_batch: int = Field(default=131072, ge=1)


class HoldoutSplitConfig(YamlLoadable, ParquetShardOutputConfig):
    """`canon_dedup_v1` train/test split config.

    Consumes a `canon_dedup_v1` dir under `input_dir`, writes `train/` and
    `test/` subdirs under `output_dir`, each `raw_v1`-schema with its own
    `MANIFEST.yaml`. A molecule lands in `test` iff a SHA1-of-`source_id`
    coordinate falls below `min(test_fraction, test_cap / n_input_rows)` — so
    the absolute `test_cap` binds when `test_fraction` alone would exceed it.
    `seed` salts the SHA1; the convention is shared across all four study corpora
    so the held-out splits are mutually consistent.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    input_dir: RepoPath
    output_dir: RepoPath
    test_fraction: float = Field(default=0.05, gt=0.0, lt=1.0)
    test_cap: int = Field(default=1_000_000, ge=1)
    seed: int = 20260426
    hash_domain: str = Field(default="canon_dedup_v1.split", min_length=1)
    rows_per_batch: int = Field(default=131072, ge=1)


# --- Stage 5: tokenizer ------------------------------------------------------


TokenizerKind = Literal[
    "smirk_base",
    "smirk_gpe",
    "smirk_unigram",
]

TokenizerAlgo = Literal["bpe", "unigram"]
"""The two algorithm axes of the study. ``bpe`` is realized as the ``smirk_gpe``
kind (Smirk-GPE = BPE over the OpenSMILES glyph base, *not* a third algorithm);
``unigram`` as ``smirk_unigram`` (Unigram-LM)."""


def algo_to_engine_tag(algo: TokenizerAlgo) -> str:
    """Map an algorithm axis to its artifact-name engine tag (``gpe``/``unigram``)."""
    return "gpe" if algo == "bpe" else "unigram"


def algo_to_kind(algo: TokenizerAlgo) -> TokenizerKind:
    """Map an algorithm axis to its trainer :data:`TokenizerKind`
    (``bpe`` → ``smirk_gpe``, ``unigram`` → ``smirk_unigram``). Single source of
    the rule, shared by grid/extras cell models and the transfer measurement."""
    return "smirk_gpe" if algo == "bpe" else "smirk_unigram"


def cell_artifact_name(
    algo: TokenizerAlgo,
    vocab_size: int,
    boundary: str,
    *,
    suffix: str | None = None,
) -> str:
    """Artifact directory name for a tokenizer cell:
    ``smirk_{gpe|unigram}_v{V}_{boundary}`` plus an optional trailing
    ``_{suffix}`` (the robustness-extras per-kind discriminator). Single source
    of the on-disk naming shared by ``GridCell.name``, ``ExtrasCell.name``, and
    the transfer measurement's ``cell_name``.
    """
    stem = f"smirk_{algo_to_engine_tag(algo)}_v{vocab_size}_{boundary}"
    return f"{stem}_{suffix}" if suffix else stem


class TokenizerConfig(YamlLoadable):
    """Stage 5 tokenizer config, driving `build_tokenizer(cfg)`.

    Training kinds (`smirk_gpe` / `smirk_unigram`) require `training_input` (a
    `canon_dedup_v1` dir: train split or extras subsample) and `vocab_size`;
    pretrained `smirk_base` uses neither.

    `ref_artifact_dir` is the GPE merge-trajectory chaining hook: when set, the
    smirk_gpe builder loads it as the `ref` tokenizer for `train_gpe`, so the
    V₁ → V₂ → V₃ checkpoints share a monotonic merge history.

    ``merge_brackets`` / ``split_structure`` thread smirk's boundary knobs to
    ``train_gpe`` / ``train_unigram`` (``smirk_gpe`` / ``smirk_unigram`` only);
    ``seed_size`` / ``max_piece_length`` / ``n_sub_iterations`` /
    ``shrinking_factor`` are Unigram-LM knobs (``smirk_unigram`` only).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    kind: TokenizerKind
    vocab_size: int | None = Field(default=None, ge=1)
    min_frequency: int = Field(default=2, ge=0)
    corpus: str | None = None
    training_input: RepoPathOrNone = None
    output_dir: RepoPath
    ref_artifact_dir: RepoPathOrNone = None
    merge_brackets: bool = False
    split_structure: bool = True
    seed_size: int | None = Field(default=None, ge=1)
    max_piece_length: int | None = Field(default=None, ge=1)
    n_sub_iterations: int | None = Field(default=None, ge=1)
    shrinking_factor: float | None = Field(default=None, gt=0.0, lt=1.0)
    scaffold_log: bool = False
    """When True, ``smirk_gpe`` training streams a per-merge-step JSONL log to
    ``<output_dir>/scaffold.jsonl``. Logging-only — does not alter merge
    selection (artifact stays byte-identical). ``smirk_gpe`` only."""

    @model_validator(mode="after")
    def _validate_kind_requirements(self) -> "TokenizerConfig":
        if self.kind in {"smirk_gpe", "smirk_unigram"}:
            if self.vocab_size is None:
                raise ValueError(f"kind={self.kind!r} requires vocab_size")
            if self.training_input is None:
                raise ValueError(f"kind={self.kind!r} requires training_input")
        if self.ref_artifact_dir is not None and self.kind != "smirk_gpe":
            raise ValueError("ref_artifact_dir is only meaningful for kind='smirk_gpe'")
        boundary_knobs_set = self.merge_brackets or not self.split_structure
        if self.kind not in {"smirk_gpe", "smirk_unigram"} and boundary_knobs_set:
            raise ValueError(
                "merge_brackets / split_structure are only meaningful for "
                "kind='smirk_gpe' or kind='smirk_unigram'"
            )
        unigram_knobs_set = (
            self.seed_size is not None
            or self.max_piece_length is not None
            or self.n_sub_iterations is not None
            or self.shrinking_factor is not None
        )
        if self.kind != "smirk_unigram" and unigram_knobs_set:
            raise ValueError(
                "seed_size / max_piece_length / n_sub_iterations / "
                "shrinking_factor are only meaningful for kind='smirk_unigram'"
            )
        if self.scaffold_log and self.kind != "smirk_gpe":
            raise ValueError(
                "scaffold_log is only meaningful for kind='smirk_gpe' "
                "(GpeTrainer per-merge-step instrumentation)"
            )
        return self
