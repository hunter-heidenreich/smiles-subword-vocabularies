"""Tests for the MANIFEST.yaml provenance shapes and readers/writers.

Covers both granularities in manifest.py: the top-level ledger
(ManifestEntry + load_manifest_entry/record_manifest_entry) and the per-shard
stage-manifest rendering (shard_dicts).
"""

from __future__ import annotations

import multiprocessing as mp
from datetime import date
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from smiles_subword.manifest import (
    ManifestEntry,
    ShardInfo,
    load_manifest_entry,
    record_manifest_entry,
    shard_dicts,
)
from smiles_subword.paths import DATA_DIR


def test_loads_pubchem_entry_from_real_manifest() -> None:
    entry = load_manifest_entry("pubchem-cid-smiles")

    assert entry.id == "pubchem-cid-smiles"
    assert entry.path == Path("data/raw/pubchem/CID-SMILES.gz")
    assert entry.source_url.startswith("https://")
    assert len(entry.sha256) == 64
    assert entry.size_bytes > 0


def test_raises_keyerror_when_id_missing() -> None:
    with pytest.raises(KeyError, match="not_a_real_id"):
        load_manifest_entry("not_a_real_id")


def test_raises_filenotfound_when_manifest_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_manifest_entry("anything", manifest_path=tmp_path / "absent.yaml")


def test_raises_when_sha_malformed(tmp_path: Path) -> None:
    payload = {
        "version": 1,
        "artifacts": [
            {
                "id": "broken",
                "path": "data/x",
                "source_url": "https://example.org",
                "sha256": "not-hex",
                "size_bytes": 1,
                "ingest_date": "2026-04-26",
            }
        ],
    }
    path = tmp_path / "MANIFEST.yaml"
    path.write_text(yaml.safe_dump(payload))

    with pytest.raises(ValidationError):
        load_manifest_entry("broken", manifest_path=path)


def test_default_manifest_path_resolves_under_data_dir() -> None:
    assert (DATA_DIR / "MANIFEST.yaml").exists()


def test_manifest_entry_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        ManifestEntry.model_validate(
            {
                "id": "x",
                "path": "data/x",
                "source_url": "https://x",
                "sha256": "0" * 64,
                "size_bytes": 0,
                "ingest_date": "2026-04-26",
                "rogue": True,
            }
        )


def _make_entry(idx: int) -> ManifestEntry:
    return ManifestEntry(
        id=f"artifact-{idx:03d}",
        path=Path(f"data/x-{idx:03d}"),
        source_url=f"https://example.org/{idx}",
        sha256=f"{idx:064x}",
        size_bytes=idx + 1,
        ingest_date=date(2026, 4, 26),
    )


@pytest.fixture
def manifest_path(tmp_path: Path) -> Path:
    path = tmp_path / "MANIFEST.yaml"
    path.write_text(yaml.safe_dump({"version": 1, "artifacts": []}))
    return path


class TestRecordManifestEntry:
    def test_appends_new_entry(self, manifest_path: Path) -> None:
        record_manifest_entry(_make_entry(1), manifest_path=manifest_path)

        loaded = load_manifest_entry("artifact-001", manifest_path=manifest_path)
        assert loaded.sha256 == f"{1:064x}"

    def test_idempotent_when_same_sha_recorded_twice(self, manifest_path: Path) -> None:
        record_manifest_entry(_make_entry(1), manifest_path=manifest_path)
        record_manifest_entry(_make_entry(1), manifest_path=manifest_path)

        payload = yaml.safe_load(manifest_path.read_text())
        ids = [a["id"] for a in payload["artifacts"]]
        assert ids == ["artifact-001"]

    def test_raises_when_sha_conflicts(self, manifest_path: Path) -> None:
        record_manifest_entry(_make_entry(1), manifest_path=manifest_path)
        conflicting = _make_entry(1).model_copy(update={"sha256": "f" * 64})

        with pytest.raises(ValueError, match="conflicts with new"):
            record_manifest_entry(conflicting, manifest_path=manifest_path)

    def test_provenance_block_round_trips(self, manifest_path: Path) -> None:
        entry = _make_entry(1).model_copy(
            update={"notes": "zinc22 draw", "provenance": {"tranches": ["AA", "BB"]}}
        )
        record_manifest_entry(entry, manifest_path=manifest_path)

        loaded = load_manifest_entry("artifact-001", manifest_path=manifest_path)
        assert loaded.notes == "zinc22 draw"
        assert loaded.provenance == {"tranches": ["AA", "BB"]}

    def test_none_fields_omitted_from_serialized_entry(
        self, manifest_path: Path
    ) -> None:
        # _make_entry leaves notes/provenance None; exclude_none keeps them out
        # of the written YAML rather than serializing explicit nulls.
        record_manifest_entry(_make_entry(1), manifest_path=manifest_path)

        raw = yaml.safe_load(manifest_path.read_text())["artifacts"][0]
        assert "notes" not in raw
        assert "provenance" not in raw

    def test_atomic_when_dump_fails(
        self,
        manifest_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        record_manifest_entry(_make_entry(1), manifest_path=manifest_path)
        before = manifest_path.read_bytes()

        def _boom(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("simulated mid-write failure")

        monkeypatch.setattr("smiles_subword.manifest.yaml.safe_dump", _boom)

        with pytest.raises(RuntimeError, match="simulated mid-write failure"):
            record_manifest_entry(_make_entry(2), manifest_path=manifest_path)

        assert manifest_path.read_bytes() == before
        assert not manifest_path.with_suffix(".yaml.tmp").exists() or (
            manifest_path.with_suffix(".yaml.tmp").stat().st_size == 0
        )

    def test_concurrent_writers_all_land(self, manifest_path: Path) -> None:
        n_workers = 6
        ctx = mp.get_context("spawn")
        procs = [
            ctx.Process(target=_record_in_subprocess, args=(manifest_path, idx))
            for idx in range(n_workers)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=30)
            assert p.exitcode == 0, f"worker {p.pid} exited with {p.exitcode}"

        payload = yaml.safe_load(manifest_path.read_text())
        ids = sorted(a["id"] for a in payload["artifacts"])
        assert ids == [f"artifact-{i:03d}" for i in range(n_workers)]


def _record_in_subprocess(manifest_path: Path, idx: int) -> None:
    """Top-level worker so multiprocessing 'spawn' can import it cleanly."""
    record_manifest_entry(_make_entry(idx), manifest_path=manifest_path)


def test_shard_dicts_renders_per_shard_field_shape() -> None:
    shards = [
        ShardInfo(file="raw_v1-00000.parquet", sha256="a" * 64, n_rows=3, n_bytes=128),
        ShardInfo(file="raw_v1-00001.parquet", sha256="b" * 64, n_rows=5, n_bytes=256),
    ]

    assert shard_dicts(shards) == [
        {
            "file": "raw_v1-00000.parquet",
            "sha256": "a" * 64,
            "n_rows": 3,
            "n_bytes": 128,
        },
        {
            "file": "raw_v1-00001.parquet",
            "sha256": "b" * 64,
            "n_rows": 5,
            "n_bytes": 256,
        },
    ]
