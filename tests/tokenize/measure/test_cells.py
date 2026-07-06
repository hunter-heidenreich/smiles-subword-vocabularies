"""Tests for ``measure._cells`` (loading trained cells + their held-out split)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import yaml

from smiles_subword.tokenize.measure import _cells

if TYPE_CHECKING:
    from pathlib import Path


class TestEvalSplitSha:
    def test_reads_per_shard_shas_from_manifest(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr("smiles_subword.paths.DATA_DIR", tmp_path)
        test_dir = tmp_path / "processed" / "pubchem" / "canon_dedup_v1" / "test"
        test_dir.mkdir(parents=True, exist_ok=True)
        (test_dir / "MANIFEST.yaml").write_text(
            yaml.safe_dump(
                {
                    "schema": "canon_dedup_v1",
                    "shards": [
                        {
                            "file": "x.parquet",
                            "sha256": "a" * 64,
                            "n_rows": 1,
                            "n_bytes": 1,
                        },
                        {
                            "file": "y.parquet",
                            "sha256": "b" * 64,
                            "n_rows": 1,
                            "n_bytes": 1,
                        },
                    ],
                }
            )
        )

        sha = _cells.eval_split_sha("pubchem")

        assert len(sha) == 32

    def test_raises_on_empty_manifest(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr("smiles_subword.paths.DATA_DIR", tmp_path)
        test_dir = tmp_path / "processed" / "pubchem" / "canon_dedup_v1" / "test"
        test_dir.mkdir(parents=True, exist_ok=True)
        (test_dir / "MANIFEST.yaml").write_text(
            yaml.safe_dump({"schema": "canon_dedup_v1", "shards": []})
        )

        with pytest.raises(ValueError, match="no shards"):
            _cells.eval_split_sha("pubchem")


class TestIterTestSplit:
    def test_streams_full_split_from_test_dir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr("smiles_subword.paths.DATA_DIR", tmp_path)
        seen: list[Path] = []

        def _fake_reader(directory: Path) -> list[str]:
            seen.append(directory)
            return ["A", "B", "C"]

        monkeypatch.setattr(_cells, "iter_smiles_from_parquet", _fake_reader)

        out = list(_cells.iter_test_split("pubchem"))

        assert out == ["A", "B", "C"]
        assert seen == [tmp_path / "processed" / "pubchem" / "canon_dedup_v1" / "test"]

    def test_limit_molecules_truncates_stream(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            _cells, "iter_smiles_from_parquet", lambda _d: iter(["A", "B", "C", "D"])
        )

        out = list(_cells.iter_test_split("pubchem", limit_molecules=2))

        assert out == ["A", "B"]


class TestLoadCellAdapter:
    def test_raises_when_meta_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            _cells, "tokenizer_artifact_dir", lambda c, n: tmp_path / c / n
        )

        with pytest.raises(FileNotFoundError, match=r"no meta\.yaml"):
            _cells.load_cell_adapter("pubchem", "smirk_gpe_v256_nmb")

    def test_raises_on_unknown_base_kind(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            _cells, "tokenizer_artifact_dir", lambda c, n: tmp_path / c / n
        )
        artifact_dir = tmp_path / "pubchem" / "weird_v256_nmb"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "meta.yaml").write_text(
            yaml.safe_dump({"name": "weird_v256_nmb", "base_kind": "unsupported_kind"})
        )

        with pytest.raises(ValueError, match="not implemented"):
            _cells.load_cell_adapter("pubchem", "weird_v256_nmb")

    def test_dispatches_smirk_gpe_to_smirk_adapter(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            _cells, "tokenizer_artifact_dir", lambda c, n: tmp_path / c / n
        )
        cell = tmp_path / "pubchem" / "smirk_gpe_v256_nmb"
        cell.mkdir(parents=True, exist_ok=True)
        (cell / "meta.yaml").write_text(yaml.safe_dump({"base_kind": "smirk_gpe"}))
        sentinel = object()
        monkeypatch.setattr(
            _cells.SmirkAdapter, "load", classmethod(lambda _cls, _p: sentinel)
        )

        assert _cells.load_cell_adapter("pubchem", "smirk_gpe_v256_nmb") is sentinel

    def test_dispatches_smirk_unigram_to_unigram_adapter(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            _cells, "tokenizer_artifact_dir", lambda c, n: tmp_path / c / n
        )
        cell = tmp_path / "pubchem" / "smirk_unigram_v256_nmb"
        cell.mkdir(parents=True, exist_ok=True)
        (cell / "meta.yaml").write_text(yaml.safe_dump({"base_kind": "smirk_unigram"}))
        sentinel = object()
        monkeypatch.setattr(
            _cells.UnigramSmirkAdapter, "load", classmethod(lambda _cls, _p: sentinel)
        )

        assert _cells.load_cell_adapter("pubchem", "smirk_unigram_v256_nmb") is sentinel
