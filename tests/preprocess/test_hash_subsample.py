"""Behavioral tests for the `hash_subsample` acceptance-band subsample.

`hash_subsample` is specified as a deterministic
hash-partition: keep a molecule iff a stable hash of its canonical SMILES
falls in the band `[0, target_n / n_input_rows)`. These tests pin that
contract — uniform in expectation across SMILES features, keyed off SMILES
not `source_id`, byte-deterministic — without asserting an exact kept count.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow.parquet as pq
import pytest
import yaml

from smiles_subword.ingest._common import RAW_V1_SCHEMA
from smiles_subword.preprocess.hash_subsample import (
    hash_subsample,
    smiles_acceptance_coord,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

_ATOMS = ("C", "N", "O", "S")


def _rows(n: int) -> list[tuple[str, str]]:
    """`n` distinct `(source_id, smiles)` rows; SMILES first char cycles atoms."""
    return [(f"id{i:06d}", f"{_ATOMS[i % 4]}x{i}") for i in range(n)]


def _output_smiles(output_dir: Path) -> list[str]:
    shards = sorted(output_dir.glob("canon_dedup_v1-*.parquet"))
    out: list[str] = []
    for shard in shards:
        out.extend(pq.read_table(shard).column("smiles").to_pylist())
    return out


class TestAcceptanceBand:
    """The kept fraction tracks `target_n / n_input_rows`, uniformly."""

    def test_keeps_roughly_target_fraction(
        self,
        make_canon_dedup_v1_dir: Callable[..., Path],
        make_hash_subsample_config: Callable[..., object],
    ) -> None:
        input_dir = make_canon_dedup_v1_dir(_rows(4000))
        cfg = make_hash_subsample_config(input_dir, target_n=1000, rows_per_batch=256)

        result = hash_subsample(cfg)  # type: ignore[arg-type]

        assert result.n_input_rows == 4000
        assert 0.22 <= result.n_kept / 4000 <= 0.28

    def test_band_uniform_across_smiles_first_char(
        self,
        make_canon_dedup_v1_dir: Callable[..., Path],
        make_hash_subsample_config: Callable[..., object],
    ) -> None:
        input_dir = make_canon_dedup_v1_dir(_rows(4000))
        cfg = make_hash_subsample_config(input_dir, target_n=1000, rows_per_batch=256)

        result = hash_subsample(cfg)  # type: ignore[arg-type]

        kept = _output_smiles(result.output_dir)
        for atom in _ATOMS:
            bucket_kept = sum(1 for s in kept if s.startswith(atom))
            assert abs(bucket_kept / 1000 - 0.25) < 0.08

    def test_keeps_all_when_target_exceeds_input(
        self,
        make_canon_dedup_v1_dir: Callable[..., Path],
        make_hash_subsample_config: Callable[..., object],
    ) -> None:
        input_dir = make_canon_dedup_v1_dir(_rows(50))
        cfg = make_hash_subsample_config(input_dir, target_n=500)

        result = hash_subsample(cfg)  # type: ignore[arg-type]

        assert result.n_kept == 50
        assert result.acceptance_fraction == 1.0

    def test_keeps_all_when_target_equals_input(
        self,
        make_canon_dedup_v1_dir: Callable[..., Path],
        make_hash_subsample_config: Callable[..., object],
    ) -> None:
        input_dir = make_canon_dedup_v1_dir(_rows(50))
        cfg = make_hash_subsample_config(input_dir, target_n=50)

        result = hash_subsample(cfg)  # type: ignore[arg-type]

        assert result.n_kept == 50

    def test_empty_corpus_keeps_nothing(
        self,
        make_canon_dedup_v1_dir: Callable[..., Path],
        make_hash_subsample_config: Callable[..., object],
    ) -> None:
        input_dir = make_canon_dedup_v1_dir([])
        cfg = make_hash_subsample_config(input_dir, target_n=100)

        result = hash_subsample(cfg)  # type: ignore[arg-type]

        assert result.n_input_rows == 0
        assert result.n_kept == 0


class TestSmilesKeyed:
    """The subsample decision depends only on the canonical SMILES."""

    def test_kept_set_is_invariant_to_source_id(
        self,
        make_canon_dedup_v1_dir: Callable[..., Path],
        make_hash_subsample_config: Callable[..., object],
        tmp_path: Path,
    ) -> None:
        smiles = [f"{_ATOMS[i % 4]}x{i}" for i in range(2000)]
        rows_a = [(f"a{i}", s) for i, s in enumerate(smiles)]
        rows_b = [(f"b{i:09d}", s) for i, s in enumerate(smiles)]
        dir_a = make_canon_dedup_v1_dir(rows_a, name="corpus_a")
        dir_b = make_canon_dedup_v1_dir(rows_b, name="corpus_b")
        cfg_a = make_hash_subsample_config(
            dir_a, target_n=500, output_dir=tmp_path / "out_a"
        )
        cfg_b = make_hash_subsample_config(
            dir_b, target_n=500, output_dir=tmp_path / "out_b"
        )

        res_a = hash_subsample(cfg_a)  # type: ignore[arg-type]
        res_b = hash_subsample(cfg_b)  # type: ignore[arg-type]

        assert set(_output_smiles(res_a.output_dir)) == set(
            _output_smiles(res_b.output_dir)
        )


class TestDeterminism:
    """Re-runs are byte-identical."""

    def test_byte_identical_when_rerun_on_same_input(
        self,
        make_canon_dedup_v1_dir: Callable[..., Path],
        make_hash_subsample_config: Callable[..., object],
        tmp_path: Path,
    ) -> None:
        input_dir = make_canon_dedup_v1_dir(_rows(1500))
        cfg_a = make_hash_subsample_config(
            input_dir, target_n=400, output_dir=tmp_path / "out_a"
        )
        cfg_b = make_hash_subsample_config(
            input_dir, target_n=400, output_dir=tmp_path / "out_b"
        )

        res_a = hash_subsample(cfg_a)  # type: ignore[arg-type]
        res_b = hash_subsample(cfg_b)  # type: ignore[arg-type]

        assert {s.sha256 for s in res_a.shards} == {s.sha256 for s in res_b.shards}


class TestIO:
    """Output schema, manifest contents, input verification."""

    def test_output_schema_is_raw_v1(
        self,
        make_canon_dedup_v1_dir: Callable[..., Path],
        make_hash_subsample_config: Callable[..., object],
    ) -> None:
        input_dir = make_canon_dedup_v1_dir(_rows(800))
        cfg = make_hash_subsample_config(input_dir, target_n=200)

        result = hash_subsample(cfg)  # type: ignore[arg-type]

        for shard in sorted(result.output_dir.glob("canon_dedup_v1-*.parquet")):
            assert pq.ParquetFile(shard).schema_arrow.equals(RAW_V1_SCHEMA)

    def test_manifest_records_subsample_params(
        self,
        make_canon_dedup_v1_dir: Callable[..., Path],
        make_hash_subsample_config: Callable[..., object],
    ) -> None:
        input_dir = make_canon_dedup_v1_dir(_rows(800))
        cfg = make_hash_subsample_config(input_dir, target_n=200)

        result = hash_subsample(cfg)  # type: ignore[arg-type]

        manifest = yaml.safe_load((result.output_dir / "MANIFEST.yaml").read_text())
        assert manifest["schema"] == "canon_dedup_v1_sub"
        assert manifest["subsample"]["method"] == "hash_partition_acceptance_band"
        assert manifest["subsample"]["target_n"] == 200
        assert manifest["n_output_rows"] == result.n_kept

    def test_rejects_input_sha_mismatch(
        self,
        make_canon_dedup_v1_dir: Callable[..., Path],
        make_hash_subsample_config: Callable[..., object],
    ) -> None:
        input_dir = make_canon_dedup_v1_dir(_rows(100))
        shard = next(input_dir.glob("canon_dedup_v1-*.parquet"))
        shard.write_bytes(shard.read_bytes() + b"tamper")
        cfg = make_hash_subsample_config(input_dir, target_n=20)

        with pytest.raises(ValueError, match="sha256 mismatch"):
            hash_subsample(cfg)  # type: ignore[arg-type]


class TestCoordinate:
    """`smiles_acceptance_coord` is a deterministic, domain-namespaced map."""

    def test_coord_in_unit_interval(self) -> None:
        for i in range(200):
            coord = smiles_acceptance_coord(f"CC{i}O", domain="d")
            assert 0.0 <= coord < 1.0

    def test_coord_is_deterministic(self) -> None:
        first = smiles_acceptance_coord("c1ccccc1", domain="d")
        second = smiles_acceptance_coord("c1ccccc1", domain="d")
        assert first == second

    def test_distinct_domains_give_distinct_coords(self) -> None:
        sub = smiles_acceptance_coord("CCO", domain="canon_dedup_v1.subsample")
        split = smiles_acceptance_coord("CCO", domain="canon_dedup_v1.split")
        assert sub != split

    def test_golden_values_pin_the_hash_pipeline(self) -> None:
        # Frozen against the current pinned implementation. The determinism /
        # distinctness tests above only compare the hash to itself, so they
        # cannot catch a change to the algorithm, the 64-bit slice, or the
        # f"{domain}|{smiles}" key format that uniformly shifts every
        # coordinate -- which would silently change which molecules land in
        # every subsampled corpus while all other tests still pass. These
        # literals are the reproducibility anchor: a deliberate hash change
        # must update them AND re-deposit the affected corpora.
        assert (
            smiles_acceptance_coord("CCO", domain="canon_dedup_v1.subsample")
            == 0.015198286035248986
        )
        assert (
            smiles_acceptance_coord("c1ccccc1", domain="canon_dedup_v1.subsample")
            == 0.06477514790296351
        )
