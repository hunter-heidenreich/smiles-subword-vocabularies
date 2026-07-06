"""Behavioral tests for the `tranche_union` nested raw_v1 flattening bridge.

`consolidate_tranches` hard-links every enumerated tranche's raw_v1 shards
into one flat raw_v1 directory `canon_dedup` can consume, and records the
tranche-restricted draw's enumerated set. These
tests pin that contract: the flatten is faithful, the flat directory is
canon_dedup-consumable, the discovered-but-not-ingested gap is validated
against `expected_exclusions`, and tampered inputs fail loudly.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from smiles_subword.config import CanonDedupConfig, TrancheUnionConfig
from smiles_subword.ingest._common import RAW_V1_SCHEMA
from smiles_subword.preprocess.canon_dedup import canon_dedup
from smiles_subword.preprocess.tranche_union import consolidate_tranches

_SMILES = ("CCO", "c1ccccc1", "CC(=O)O", "CCN", "CCC", "OCC", "CCCC", "CN")


def _rows(n: int, start: int = 0) -> list[tuple[str, str]]:
    """`n` `(source_id, smiles)` rows; SMILES cycle a small valid set."""
    return [(f"id{i:05d}", _SMILES[i % len(_SMILES)]) for i in range(start, start + n)]


def _write_raw_v1(path: Path, rows: list[tuple[str, str]]) -> None:
    ts = datetime(2026, 4, 28, 12, 0, 0, tzinfo=UTC)
    table = pa.table(
        {
            "source_id": [r[0] for r in rows],
            "smiles": [r[1] for r in rows],
            "source": ["zinc22"] * len(rows),
            "ingest_ts": [ts] * len(rows),
        },
        schema=RAW_V1_SCHEMA,
    )
    pq.write_table(table, path, compression="zstd")


def _write_tranche_manifest(tranche_dir: Path, shards: list[Path]) -> None:
    payload = {
        "schema": "raw_v1",
        "source": "zinc22",
        "shards": [
            {
                "file": s.name,
                "sha256": hashlib.sha256(s.read_bytes()).hexdigest(),
                "n_rows": pq.ParquetFile(s).metadata.num_rows,
                "n_bytes": s.stat().st_size,
            }
            for s in shards
        ],
    }
    (tranche_dir / "MANIFEST.yaml").write_text(yaml.safe_dump(payload, sort_keys=False))


@pytest.fixture
def make_multi_tranche_raw_v1(tmp_path: Path) -> Callable[..., Path]:
    """Build a synthetic nested raw_v1 directory + tranche-list TSV.

    `tranches` maps `tranche_id` to its `(source_id, smiles)` rows; each
    becomes a `<tranche_id>/` subdir with one raw_v1 shard and per-tranche
    `MANIFEST.yaml`. `extra_discovered` names tranche ids written into the
    tranche-list TSV but *not* ingested — the discovered-not-used gap.
    """

    def _make(
        tranches: dict[str, list[tuple[str, str]]],
        *,
        extra_discovered: tuple[str, ...] = (),
        layout: str = "multi_tranche",
        n_rows_override: int | None = None,
    ) -> Path:
        raw_dir = tmp_path / "raw_v1"
        raw_dir.mkdir()
        agg_tranches: list[dict[str, object]] = []
        total_rows = 0
        for tid in sorted(tranches):
            rows = tranches[tid]
            tranche_dir = raw_dir / tid
            tranche_dir.mkdir()
            shard = tranche_dir / "raw_v1-00000.parquet"
            _write_raw_v1(shard, rows)
            _write_tranche_manifest(tranche_dir, [shard])
            total_rows += len(rows)
            agg_tranches.append(
                {
                    "tranche_id": tid,
                    "manifest_id": f"zinc22-{tid}",
                    "output_dir": str(tranche_dir.relative_to(tmp_path)),
                    "n_rows": len(rows),
                    "n_shards": 1,
                    "n_bytes": shard.stat().st_size,
                    "skipped": False,
                }
            )

        tsv_path = tmp_path / "zinc22_tranches.tsv"
        discovered = sorted([*tranches, *extra_discovered])
        tsv_path.write_text(
            "tranche_id\tgeneration\n"
            + "\n".join(f"{tid}\tx" for tid in discovered)
            + "\n"
        )
        aggregate = {
            "schema": "raw_v1",
            "layout": layout,
            "source": "zinc22",
            "tranches_path": str(tsv_path),
            "tranches_sha256": hashlib.sha256(tsv_path.read_bytes()).hexdigest(),
            "n_tranches": len(agg_tranches),
            "n_rows": total_rows if n_rows_override is None else n_rows_override,
            "n_shards": len(agg_tranches),
            "tranches": agg_tranches,
        }
        (raw_dir / "MANIFEST.yaml").write_text(
            yaml.safe_dump(aggregate, sort_keys=False)
        )
        return raw_dir

    return _make


class TestConsolidation:
    """The flatten faithfully unions every ingested tranche's shards."""

    def test_flattens_all_tranche_shards(
        self, make_multi_tranche_raw_v1: Callable[..., Path], tmp_path: Path
    ) -> None:
        raw = make_multi_tranche_raw_v1(
            {"zinc-22x-H10P000": _rows(5), "zinc-22x-H10P010": _rows(7, start=5)}
        )
        cfg = TrancheUnionConfig(
            name="zinc22", input_dir=raw, output_dir=tmp_path / "out"
        )

        result = consolidate_tranches(cfg)

        assert result.n_used == 2
        assert result.n_rows == 12
        flat = sorted((tmp_path / "out").glob("raw_v1-*.parquet"))
        assert sum(pq.ParquetFile(s).metadata.num_rows for s in flat) == 12

    def test_single_tranche(
        self, make_multi_tranche_raw_v1: Callable[..., Path], tmp_path: Path
    ) -> None:
        raw = make_multi_tranche_raw_v1({"zinc-22f-H10P000": _rows(3)})
        cfg = TrancheUnionConfig(
            name="zinc22", input_dir=raw, output_dir=tmp_path / "out"
        )

        result = consolidate_tranches(cfg)

        assert result.n_used == 1
        assert len(result.shards) == 1

    def test_consolidated_shard_content_matches_source(
        self, make_multi_tranche_raw_v1: Callable[..., Path], tmp_path: Path
    ) -> None:
        raw = make_multi_tranche_raw_v1({"zinc-22f-H10P000": _rows(6)})
        cfg = TrancheUnionConfig(
            name="zinc22", input_dir=raw, output_dir=tmp_path / "out"
        )

        consolidate_tranches(cfg)

        src = (raw / "zinc-22f-H10P000" / "raw_v1-00000.parquet").read_bytes()
        flat = (tmp_path / "out" / "raw_v1-00000.parquet").read_bytes()
        assert flat == src

    def test_falls_back_to_copy_when_hardlink_unavailable(
        self,
        make_multi_tranche_raw_v1: Callable[..., Path],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        raw = make_multi_tranche_raw_v1({"zinc-22f-H10P000": _rows(4)})

        def _no_hardlink(self: Path, target: Path) -> None:
            raise OSError("Invalid cross-device link")

        monkeypatch.setattr(Path, "hardlink_to", _no_hardlink)
        cfg = TrancheUnionConfig(
            name="zinc22", input_dir=raw, output_dir=tmp_path / "out"
        )

        result = consolidate_tranches(cfg)

        assert result.n_rows == 4
        assert (tmp_path / "out" / "raw_v1-00000.parquet").exists()


class TestCanonDedupConsumable:
    """The flat output directory is consumable by `canon_dedup` unchanged."""

    def test_canon_dedup_runs_on_consolidated_output(
        self, make_multi_tranche_raw_v1: Callable[..., Path], tmp_path: Path
    ) -> None:
        raw = make_multi_tranche_raw_v1(
            {"zinc-22x-H10P000": _rows(5), "zinc-22x-H10P010": _rows(7, start=5)}
        )
        union_cfg = TrancheUnionConfig(
            name="zinc22", input_dir=raw, output_dir=tmp_path / "union"
        )
        consolidate_tranches(union_cfg)

        result = canon_dedup(
            CanonDedupConfig(
                name="zinc22",
                input_dir=tmp_path / "union",
                output_dir=tmp_path / "full",
                n_workers=1,
            )
        )

        assert result.n_input_rows == 12
        assert result.n_output_rows > 0


class TestProvenance:
    """The flat manifest records the enumerated tranche-restriction set."""

    def test_manifest_records_tranche_union_block(
        self, make_multi_tranche_raw_v1: Callable[..., Path], tmp_path: Path
    ) -> None:
        raw = make_multi_tranche_raw_v1(
            {"zinc-22x-H10P000": _rows(5), "zinc-22x-H10P010": _rows(7, start=5)}
        )
        cfg = TrancheUnionConfig(
            name="zinc22", input_dir=raw, output_dir=tmp_path / "out"
        )

        consolidate_tranches(cfg)

        block = yaml.safe_load((tmp_path / "out" / "MANIFEST.yaml").read_text())[
            "tranche_union"
        ]
        assert block["n_discovered"] == 2
        assert block["n_used"] == 2
        assert block["n_rows"] == 12
        assert block["excluded"] == []

    def test_excluded_tranche_recorded_with_reason(
        self, make_multi_tranche_raw_v1: Callable[..., Path], tmp_path: Path
    ) -> None:
        raw = make_multi_tranche_raw_v1(
            {"zinc-22x-H10P000": _rows(5)}, extra_discovered=("zinc-22x-H29P280",)
        )
        cfg = TrancheUnionConfig(
            name="zinc22",
            input_dir=raw,
            output_dir=tmp_path / "out",
            expected_exclusions={"zinc-22x-H29P280": "upstream file truncated"},
        )

        result = consolidate_tranches(cfg)

        assert result.n_discovered == 2
        assert result.n_used == 1
        assert result.excluded == {"zinc-22x-H29P280": "upstream file truncated"}
        block = yaml.safe_load((tmp_path / "out" / "MANIFEST.yaml").read_text())[
            "tranche_union"
        ]
        assert block["excluded"] == [
            {"tranche_id": "zinc-22x-H29P280", "reason": "upstream file truncated"}
        ]


class TestExclusionValidation:
    """The discovered-minus-ingested gap must equal `expected_exclusions`."""

    def test_raises_when_unexpected_tranche_missing(
        self, make_multi_tranche_raw_v1: Callable[..., Path], tmp_path: Path
    ) -> None:
        raw = make_multi_tranche_raw_v1(
            {"zinc-22x-H10P000": _rows(5)}, extra_discovered=("zinc-22x-H29P280",)
        )
        cfg = TrancheUnionConfig(
            name="zinc22", input_dir=raw, output_dir=tmp_path / "out"
        )

        with pytest.raises(ValueError, match="does not match expected_exclusions"):
            consolidate_tranches(cfg)

    def test_raises_when_configured_exclusion_actually_present(
        self, make_multi_tranche_raw_v1: Callable[..., Path], tmp_path: Path
    ) -> None:
        raw = make_multi_tranche_raw_v1({"zinc-22x-H10P000": _rows(5)})
        cfg = TrancheUnionConfig(
            name="zinc22",
            input_dir=raw,
            output_dir=tmp_path / "out",
            expected_exclusions={"zinc-22x-H10P000": "should not be here"},
        )

        with pytest.raises(ValueError, match="does not match expected_exclusions"):
            consolidate_tranches(cfg)


class TestInputVerification:
    """Tampered or malformed inputs fail loudly, never silently."""

    def test_raises_on_tranche_shard_sha_mismatch(
        self, make_multi_tranche_raw_v1: Callable[..., Path], tmp_path: Path
    ) -> None:
        raw = make_multi_tranche_raw_v1({"zinc-22f-H10P000": _rows(5)})
        shard = raw / "zinc-22f-H10P000" / "raw_v1-00000.parquet"
        shard.write_bytes(shard.read_bytes() + b"tamper")
        cfg = TrancheUnionConfig(
            name="zinc22", input_dir=raw, output_dir=tmp_path / "out"
        )

        with pytest.raises(ValueError, match="sha256 mismatch"):
            consolidate_tranches(cfg)

    def test_raises_on_tranche_list_sha_mismatch(
        self, make_multi_tranche_raw_v1: Callable[..., Path], tmp_path: Path
    ) -> None:
        raw = make_multi_tranche_raw_v1({"zinc-22f-H10P000": _rows(5)})
        (tmp_path / "zinc22_tranches.tsv").write_text("tranche_id\tgeneration\n")
        cfg = TrancheUnionConfig(
            name="zinc22", input_dir=raw, output_dir=tmp_path / "out"
        )

        with pytest.raises(ValueError, match=r"tranche list .* sha256"):
            consolidate_tranches(cfg)

    def test_raises_when_not_multi_tranche(
        self, make_multi_tranche_raw_v1: Callable[..., Path], tmp_path: Path
    ) -> None:
        raw = make_multi_tranche_raw_v1({"zinc-22f-H10P000": _rows(5)}, layout="single")
        cfg = TrancheUnionConfig(
            name="zinc22", input_dir=raw, output_dir=tmp_path / "out"
        )

        with pytest.raises(
            ValueError, match="not a nested per-tranche raw_v1 directory"
        ):
            consolidate_tranches(cfg)

    def test_raises_on_row_count_mismatch(
        self, make_multi_tranche_raw_v1: Callable[..., Path], tmp_path: Path
    ) -> None:
        raw = make_multi_tranche_raw_v1(
            {"zinc-22f-H10P000": _rows(5)}, n_rows_override=9999
        )
        cfg = TrancheUnionConfig(
            name="zinc22", input_dir=raw, output_dir=tmp_path / "out"
        )

        with pytest.raises(ValueError, match="a tranche is incomplete"):
            consolidate_tranches(cfg)
