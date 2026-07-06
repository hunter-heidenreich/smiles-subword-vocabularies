"""Behavioral tests for the `split_train_test` holdout split.

`split_train_test` is specified as a
deterministic SHA1-of-`source_id` partition: a 5% held-out test split capped at
an absolute 1e6 molecules. These tests pin that contract — train/test
partition the input, the cap binds when the fraction would exceed it, the
realised fraction is continuous (not rounded to 1/16), the partition is
independent of the SMILES-keyed subsample, and the run is byte-deterministic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow.parquet as pq
import pytest
import yaml

from smiles_subword.ingest._common import RAW_V1_SCHEMA
from smiles_subword.preprocess.hash_subsample import smiles_acceptance_coord
from smiles_subword.preprocess.holdout_split import (
    source_id_split_coord,
    split_train_test,
)
from smiles_subword.tokenize import materialize_smiles_txt, training_corpus_sha

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

_ATOMS = ("C", "N", "O", "S")


def _rows(n: int) -> list[tuple[str, str]]:
    """`n` distinct `(source_id, smiles)` rows."""
    return [(f"id{i:06d}", f"{_ATOMS[i % 4]}x{i}") for i in range(n)]


def _source_ids(split_dir: Path) -> list[str]:
    out: list[str] = []
    for shard in sorted(split_dir.glob("canon_dedup_v1-*.parquet")):
        out.extend(pq.read_table(shard).column("source_id").to_pylist())
    return out


class TestPartition:
    """Train and test together partition the input, disjointly."""

    def test_train_and_test_partition_input(
        self,
        make_canon_dedup_v1_dir: Callable[..., Path],
        make_holdout_split_config: Callable[..., object],
    ) -> None:
        input_dir = make_canon_dedup_v1_dir(_rows(3000))
        cfg = make_holdout_split_config(input_dir, rows_per_batch=256)

        result = split_train_test(cfg)  # type: ignore[arg-type]

        assert result.n_train + result.n_test == result.n_input_rows == 3000

    def test_train_and_test_are_disjoint(
        self,
        make_canon_dedup_v1_dir: Callable[..., Path],
        make_holdout_split_config: Callable[..., object],
    ) -> None:
        input_dir = make_canon_dedup_v1_dir(_rows(3000))
        cfg = make_holdout_split_config(input_dir, rows_per_batch=256)

        result = split_train_test(cfg)  # type: ignore[arg-type]

        train_ids = set(_source_ids(result.output_dir / "train"))
        test_ids = set(_source_ids(result.output_dir / "test"))
        assert train_ids.isdisjoint(test_ids)

    def test_empty_corpus_yields_empty_splits(
        self,
        make_canon_dedup_v1_dir: Callable[..., Path],
        make_holdout_split_config: Callable[..., object],
    ) -> None:
        input_dir = make_canon_dedup_v1_dir([])
        cfg = make_holdout_split_config(input_dir)

        result = split_train_test(cfg)  # type: ignore[arg-type]

        assert result.n_train == 0
        assert result.n_test == 0


class TestCap:
    """The absolute `test_cap` binds when `test_fraction` would exceed it."""

    def test_cap_binds_when_fraction_exceeds_cap(
        self,
        make_canon_dedup_v1_dir: Callable[..., Path],
        make_holdout_split_config: Callable[..., object],
    ) -> None:
        input_dir = make_canon_dedup_v1_dir(_rows(2000))
        cfg = make_holdout_split_config(
            input_dir, test_fraction=0.5, test_cap=200, rows_per_batch=256
        )

        result = split_train_test(cfg)  # type: ignore[arg-type]

        assert result.cap_bound is True
        assert result.effective_threshold == pytest.approx(0.1)
        assert 0.07 <= result.n_test / 2000 <= 0.13

    def test_cap_does_not_bind_when_fraction_below_cap(
        self,
        make_canon_dedup_v1_dir: Callable[..., Path],
        make_holdout_split_config: Callable[..., object],
    ) -> None:
        input_dir = make_canon_dedup_v1_dir(_rows(2000))
        cfg = make_holdout_split_config(
            input_dir, test_fraction=0.05, test_cap=1_000_000, rows_per_batch=256
        )

        result = split_train_test(cfg)  # type: ignore[arg-type]

        assert result.cap_bound is False
        assert result.effective_threshold == pytest.approx(0.05)

    def test_realised_test_fraction_is_continuous(
        self,
        make_canon_dedup_v1_dir: Callable[..., Path],
        make_holdout_split_config: Callable[..., object],
    ) -> None:
        input_dir = make_canon_dedup_v1_dir(_rows(4000))
        cfg = make_holdout_split_config(
            input_dir, test_fraction=0.05, rows_per_batch=256
        )

        result = split_train_test(cfg)  # type: ignore[arg-type]

        assert abs(result.n_test / 4000 - 0.05) < 0.02


class TestSubdirs:
    """`train/` and `test/` are first-class raw_v1 corpus directories."""

    def test_output_schema_is_raw_v1(
        self,
        make_canon_dedup_v1_dir: Callable[..., Path],
        make_holdout_split_config: Callable[..., object],
    ) -> None:
        input_dir = make_canon_dedup_v1_dir(_rows(1000))
        cfg = make_holdout_split_config(input_dir)

        result = split_train_test(cfg)  # type: ignore[arg-type]

        for sub in ("train", "test"):
            for shard in (result.output_dir / sub).glob("canon_dedup_v1-*.parquet"):
                assert pq.ParquetFile(shard).schema_arrow.equals(RAW_V1_SCHEMA)

    def test_subdir_manifest_records_split_params(
        self,
        make_canon_dedup_v1_dir: Callable[..., Path],
        make_holdout_split_config: Callable[..., object],
    ) -> None:
        input_dir = make_canon_dedup_v1_dir(_rows(1000))
        cfg = make_holdout_split_config(input_dir, seed=20260426)

        result = split_train_test(cfg)  # type: ignore[arg-type]

        manifest = yaml.safe_load(
            (result.output_dir / "test" / "MANIFEST.yaml").read_text()
        )
        assert manifest["split"] == "test"
        assert manifest["split_params"]["method"] == "sha1_source_id_partition"
        assert manifest["split_params"]["seed"] == 20260426

    def test_train_dir_consumable_by_tokenize_corpus_helpers(
        self,
        make_canon_dedup_v1_dir: Callable[..., Path],
        make_holdout_split_config: Callable[..., object],
        tmp_path: Path,
    ) -> None:
        input_dir = make_canon_dedup_v1_dir(_rows(1000))
        cfg = make_holdout_split_config(input_dir)

        result = split_train_test(cfg)  # type: ignore[arg-type]

        train_dir = result.output_dir / "train"
        assert len(training_corpus_sha(train_dir)) == 32
        smi_path = materialize_smiles_txt(train_dir, tmp_path / "train.smi")
        assert smi_path.read_text().count("\n") == result.n_train


class TestDeterminism:
    """Re-runs are byte-identical across both subdirectories."""

    def test_byte_identical_when_rerun_on_same_input(
        self,
        make_canon_dedup_v1_dir: Callable[..., Path],
        make_holdout_split_config: Callable[..., object],
        tmp_path: Path,
    ) -> None:
        input_dir = make_canon_dedup_v1_dir(_rows(1500))
        cfg_a = make_holdout_split_config(input_dir, output_dir=tmp_path / "out_a")
        cfg_b = make_holdout_split_config(input_dir, output_dir=tmp_path / "out_b")

        res_a = split_train_test(cfg_a)  # type: ignore[arg-type]
        res_b = split_train_test(cfg_b)  # type: ignore[arg-type]

        assert {s.sha256 for s in res_a.train_shards} == {
            s.sha256 for s in res_b.train_shards
        }
        assert {s.sha256 for s in res_a.test_shards} == {
            s.sha256 for s in res_b.test_shards
        }


class TestIndependence:
    """The source_id split and the SMILES subsample are uncorrelated."""

    def test_split_partition_independent_of_subsample_partition(self) -> None:
        n = 4000
        joint = {(True, True): 0, (True, False): 0, (False, True): 0, (False, False): 0}
        for i in range(n):
            in_test = (
                source_id_split_coord(
                    f"id{i:06d}", seed=20260426, domain="canon_dedup_v1.split"
                )
                < 0.5
            )
            in_sub = (
                smiles_acceptance_coord(
                    f"{_ATOMS[i % 4]}x{i}", domain="canon_dedup_v1.subsample"
                )
                < 0.5
            )
            joint[(in_test, in_sub)] += 1

        for count in joint.values():
            assert abs(count / n - 0.25) < 0.04


class TestCoordinate:
    """`source_id_split_coord` is a deterministic, seed-salted map."""

    def test_coord_in_unit_interval(self) -> None:
        for i in range(200):
            assert 0.0 <= source_id_split_coord(i, seed=1, domain="d") < 1.0

    def test_coord_changes_with_seed(self) -> None:
        first = source_id_split_coord("cid42", seed=1, domain="d")
        second = source_id_split_coord("cid42", seed=2, domain="d")
        assert first != second

    def test_golden_values_pin_the_hash_pipeline(self) -> None:
        # Frozen against the current pinned implementation; see the equivalent
        # anchor in test_hash_subsample.py. A change to the algorithm, the
        # 64-bit slice, or the f"{domain}|{seed}|{source_id}" key format would
        # reshuffle every train/test split while the determinism / seed tests
        # still pass. These literals must change deliberately, with the
        # held-out splits re-deposited.
        assert (
            source_id_split_coord(
                "CID12345", seed=20260426, domain="canon_dedup_v1.split"
            )
            == 0.9345032237394443
        )
        assert (
            source_id_split_coord(42, seed=20260426, domain="canon_dedup_v1.split")
            == 0.6478037945345199
        )

    def test_int_and_str_source_id_coincide(self) -> None:
        # `source_id` is coerced to str, so a numeric id hashes identically
        # whether it arrives as int or string -- the Parquet column dtype must
        # never change a molecule's split assignment.
        assert source_id_split_coord(42, seed=1, domain="d") == source_id_split_coord(
            "42", seed=1, domain="d"
        )
