"""Tests for `smiles_subword.ingest.real_space` (Stage 0 REAL-Space ingest)."""

from __future__ import annotations

import gzip
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from pydantic import ValidationError

from smiles_subword import _time
from smiles_subword.config import RealSpaceCorpusConfig
from smiles_subword.ingest._common import RAW_V1_SCHEMA, sha256_file
from smiles_subword.ingest.real_space import ingest

if TYPE_CHECKING:
    from collections.abc import Iterable

_PART_0: tuple[tuple[str, str], ...] = (
    ("CCO", "RS0-1"),
    ("c1ccccc1", "RS0-2"),
    ("CC(=O)O", "RS0-3"),
)
_CXSMILES_VERBATIM = 'C[*] |$;R1$| spaced "quote" tail'
_PART_1: tuple[tuple[str, str], ...] = (
    (_CXSMILES_VERBATIM, "RS1-1"),
    ("CCN", "RS1-2"),
)
_ALL_IDS = ["RS0-1", "RS0-2", "RS0-3", "RS1-1", "RS1-2"]


def _write_cxsmiles(
    path: Path,
    rows: Iterable[tuple[str, str]],
    *,
    header: bool = True,
    compressed: bool = False,
) -> None:
    """Write tab-delimited `<smiles>\\t<id>` rows, optionally gzip-compressed."""
    lines = ["smiles\tid"] if header else []
    lines.extend(f"{smiles}\t{mol_id}" for smiles, mol_id in rows)
    body = "".join(f"{line}\n" for line in lines)
    if compressed:
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            fh.write(body)
    else:
        path.write_text(body, encoding="utf-8")


def _all_shards(output_dir: Path) -> list[Path]:
    return sorted(output_dir.glob("raw_v1-*.parquet"))


def _read_table(output_dir: Path) -> pa.Table:
    return pa.concat_tables([pq.read_table(s) for s in _all_shards(output_dir)])


def _manifest(output_dir: Path) -> dict:
    return yaml.safe_load((output_dir / "MANIFEST.yaml").read_text())


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    """Two `.cxsmiles` files, zero-padded names, distinct per-file id prefixes."""
    src = tmp_path / "real_space"
    src.mkdir()
    _write_cxsmiles(src / "part-00000.cxsmiles", _PART_0)
    _write_cxsmiles(src / "part-00001.cxsmiles", _PART_1)
    return src


@pytest.fixture
def cfg(tmp_path: Path, raw_dir: Path) -> RealSpaceCorpusConfig:
    return RealSpaceCorpusConfig(
        name="real_space",
        raw_dir=raw_dir,
        output_dir=tmp_path / "out",
        shard_target_bytes=2**20,
        rows_per_batch=128,
    )


class TestEmittedRows:
    """The ingest projects every `.cxsmiles` row onto the raw_v1 schema."""

    def test_emitted_schema_matches_raw_v1(self, cfg: RealSpaceCorpusConfig) -> None:
        ingest(cfg)

        for shard in _all_shards(cfg.output_dir):
            assert pq.ParquetFile(shard).schema_arrow.equals(RAW_V1_SCHEMA)

    def test_row_count_is_sum_over_all_files(self, cfg: RealSpaceCorpusConfig) -> None:
        result = ingest(cfg)

        assert result.n_rows == len(_PART_0) + len(_PART_1)

    def test_source_column_is_real_space(self, cfg: RealSpaceCorpusConfig) -> None:
        ingest(cfg)

        sources = set(_read_table(cfg.output_dir).column("source").to_pylist())
        assert sources == {"real_space"}

    def test_source_id_is_the_id_column(self, cfg: RealSpaceCorpusConfig) -> None:
        ingest(cfg)

        ids = _read_table(cfg.output_dir).column("source_id").to_pylist()
        assert sorted(ids) == sorted(_ALL_IDS)

    def test_rows_appear_in_sorted_file_order(self, cfg: RealSpaceCorpusConfig) -> None:
        ingest(cfg)

        ids = _read_table(cfg.output_dir).column("source_id").to_pylist()
        assert ids == _ALL_IDS

    def test_cxsmiles_extension_block_passes_through_verbatim(
        self, cfg: RealSpaceCorpusConfig
    ) -> None:
        ingest(cfg)
        table = _read_table(cfg.output_dir)

        row = table.filter(pa.compute.field("source_id") == "RS1-1")
        assert row.column("smiles").to_pylist() == [_CXSMILES_VERBATIM]


class TestHeader:
    """`has_header` controls whether the first line is consumed or kept."""

    def test_header_row_is_consumed_when_has_header_true(
        self, cfg: RealSpaceCorpusConfig
    ) -> None:
        ingest(cfg)

        ids = _read_table(cfg.output_dir).column("source_id").to_pylist()
        assert "id" not in ids

    def test_first_line_is_data_when_has_header_false(self, tmp_path: Path) -> None:
        src = tmp_path / "real_space"
        src.mkdir()
        _write_cxsmiles(src / "part-00000.cxsmiles", _PART_0, header=False)
        cfg = RealSpaceCorpusConfig(
            name="real_space",
            raw_dir=src,
            has_header=False,
            output_dir=tmp_path / "out",
        )

        result = ingest(cfg)

        assert result.n_rows == len(_PART_0)


class TestEdgeCases:
    """Zero, boundary, and exception scenarios."""

    def test_raises_when_glob_matches_no_files(self, tmp_path: Path) -> None:
        empty = tmp_path / "real_space"
        empty.mkdir()
        (empty / "readme.txt").write_text("not a cxsmiles file")
        cfg = RealSpaceCorpusConfig(
            name="real_space", raw_dir=empty, output_dir=tmp_path / "out"
        )

        with pytest.raises(FileNotFoundError, match=r"matched no files"):
            ingest(cfg)

    def test_empty_file_yields_zero_rows_for_that_file(self, tmp_path: Path) -> None:
        src = tmp_path / "real_space"
        src.mkdir()
        _write_cxsmiles(src / "part-00000.cxsmiles", _PART_0)
        _write_cxsmiles(src / "part-00001.cxsmiles", [])
        cfg = RealSpaceCorpusConfig(
            name="real_space", raw_dir=src, output_dir=tmp_path / "out"
        )

        result = ingest(cfg)

        by_path = {e["path"]: e for e in _manifest(cfg.output_dir)["source_files"]}
        empty_entry = by_path[str(src / "part-00001.cxsmiles")]
        assert result.n_rows == len(_PART_0)
        assert empty_entry["n_rows"] == 0

    def test_all_empty_input_writes_manifest_with_zero_shards(
        self, tmp_path: Path
    ) -> None:
        src = tmp_path / "real_space"
        src.mkdir()
        _write_cxsmiles(src / "part-00000.cxsmiles", [])
        _write_cxsmiles(src / "part-00001.cxsmiles", [])
        cfg = RealSpaceCorpusConfig(
            name="real_space", raw_dir=src, output_dir=tmp_path / "out"
        )

        result = ingest(cfg)

        assert result.n_rows == 0
        assert result.n_shards == 0
        assert (cfg.output_dir / "MANIFEST.yaml").is_file()

    def test_null_smiles_is_coalesced_to_empty_string(self, tmp_path: Path) -> None:
        src = tmp_path / "real_space"
        src.mkdir()
        _write_cxsmiles(src / "part-00000.cxsmiles", [("", "RS-NULL")])
        cfg = RealSpaceCorpusConfig(
            name="real_space", raw_dir=src, output_dir=tmp_path / "out"
        )

        result = ingest(cfg)
        table = _read_table(cfg.output_dir)

        assert result.n_rows == 1
        assert table.column("smiles").to_pylist() == [""]

    def test_multi_shard_rollover_when_target_is_small(self, tmp_path: Path) -> None:
        src = tmp_path / "real_space"
        src.mkdir()
        rows = [(f"C{'C' * (i % 8)}O", f"RS-{i:05d}") for i in range(400)]
        _write_cxsmiles(src / "part-00000.cxsmiles", rows)
        cfg = RealSpaceCorpusConfig(
            name="real_space",
            raw_dir=src,
            output_dir=tmp_path / "out",
            shard_target_bytes=1024,
            rows_per_batch=64,
        )

        result = ingest(cfg)

        assert result.n_shards > 1
        assert _read_table(cfg.output_dir).num_rows == 400

    def test_reingest_replaces_existing_output_dir(
        self, cfg: RealSpaceCorpusConfig
    ) -> None:
        ingest(cfg)

        result = ingest(cfg)

        assert result.n_rows == len(_PART_0) + len(_PART_1)
        assert _read_table(cfg.output_dir).num_rows == result.n_rows

    def test_reads_gzip_compressed_cxsmiles(self, tmp_path: Path) -> None:
        src = tmp_path / "real_space"
        src.mkdir()
        _write_cxsmiles(src / "part-00000.cxsmiles.gz", _PART_0, compressed=True)
        cfg = RealSpaceCorpusConfig(
            name="real_space",
            raw_dir=src,
            glob="*.cxsmiles.gz",
            file_compression="gzip",
            output_dir=tmp_path / "out",
        )

        result = ingest(cfg)

        assert result.n_rows == len(_PART_0)


@pytest.fixture
def frozen_ingest_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin ``ingest()``'s wall-clock stamp so reruns share one ``ingest_ts``.

    ``ingest_ts`` is a raw_v1 *data* column, so two ``ingest()`` calls that
    straddle a one-second boundary write shards differing only in that
    timestamp — the race that made the byte-identity assertion below flaky
    (~8%). Freezing the clock isolates the one legitimate source of variation
    and lets the test verify what it claims: the rest of the shard is
    byte-deterministic.

    The stamp comes from the shared ``_time.utc_now_naive_seconds`` (behind
    ``ingest_timestamp``), so the freeze patches ``_time.datetime`` — the single
    clock source for every stage.
    """
    fixed = datetime(2026, 5, 17, 12, 0, 0, tzinfo=UTC)

    class _FrozenClock:
        @staticmethod
        def now(**_kwargs: object) -> datetime:
            return fixed

    monkeypatch.setattr(_time, "datetime", _FrozenClock)


class TestDeterminism:
    """Reruns against the same input produce byte-identical artifacts."""

    def test_shard_sha256_is_byte_identical_across_reruns(
        self, tmp_path: Path, raw_dir: Path, frozen_ingest_clock: None
    ) -> None:
        cfg_a = RealSpaceCorpusConfig(
            name="real_space", raw_dir=raw_dir, output_dir=tmp_path / "a"
        )
        cfg_b = RealSpaceCorpusConfig(
            name="real_space", raw_dir=raw_dir, output_dir=tmp_path / "b"
        )

        ingest(cfg_a)
        ingest(cfg_b)

        sha_a = [s["sha256"] for s in _manifest(cfg_a.output_dir)["shards"]]
        sha_b = [s["sha256"] for s in _manifest(cfg_b.output_dir)["shards"]]
        assert sha_a == sha_b

    def test_input_sha256_is_a_stable_aggregate(
        self, tmp_path: Path, raw_dir: Path
    ) -> None:
        cfg_a = RealSpaceCorpusConfig(
            name="real_space", raw_dir=raw_dir, output_dir=tmp_path / "a"
        )
        cfg_b = RealSpaceCorpusConfig(
            name="real_space", raw_dir=raw_dir, output_dir=tmp_path / "b"
        )

        ingest(cfg_a)
        ingest(cfg_b)

        assert (
            _manifest(cfg_a.output_dir)["input_sha256"]
            == _manifest(cfg_b.output_dir)["input_sha256"]
        )

    def test_input_sha256_changes_when_a_file_changes(
        self, tmp_path: Path, raw_dir: Path
    ) -> None:
        # The aggregate is the provenance anchor for the whole set, so it must
        # respond to content changes — stability alone (above) would pass for a
        # constant. Mutating one file's bytes must change input_sha256.
        cfg_before = RealSpaceCorpusConfig(
            name="real_space", raw_dir=raw_dir, output_dir=tmp_path / "before"
        )
        ingest(cfg_before)
        before = _manifest(cfg_before.output_dir)["input_sha256"]

        _write_cxsmiles(raw_dir / "part-00000.cxsmiles", (("CCCC", "RS0-9"),))
        cfg_after = RealSpaceCorpusConfig(
            name="real_space", raw_dir=raw_dir, output_dir=tmp_path / "after"
        )
        ingest(cfg_after)
        after = _manifest(cfg_after.output_dir)["input_sha256"]

        assert before != after


class TestManifest:
    """The stage manifest carries per-file SHA256 provenance."""

    def test_manifest_schema_is_raw_v1(self, cfg: RealSpaceCorpusConfig) -> None:
        ingest(cfg)
        manifest = _manifest(cfg.output_dir)

        assert manifest["schema"] == "raw_v1"
        assert manifest["source"] == "real_space"
        assert manifest["manifest_id"] == "real_space"

    def test_manifest_records_per_file_sha256_and_row_count(
        self, cfg: RealSpaceCorpusConfig, raw_dir: Path
    ) -> None:
        ingest(cfg)
        source_files = _manifest(cfg.output_dir)["source_files"]

        by_path = {e["path"]: e for e in source_files}
        for name, n_rows in (("part-00000", 3), ("part-00001", 2)):
            entry = by_path[str(raw_dir / f"{name}.cxsmiles")]
            assert entry["sha256"] == sha256_file(raw_dir / f"{name}.cxsmiles")
            assert entry["n_rows"] == n_rows


class TestConfig:
    """`RealSpaceCorpusConfig` validation boundaries."""

    def test_rejects_unknown_field(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError):
            RealSpaceCorpusConfig(
                name="real_space",
                raw_dir=tmp_path,
                output_dir=tmp_path / "out",
                bogus_field=1,
            )
