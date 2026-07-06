"""Behavioral tests for the `canon_dedup_v1` pipeline.

`canon_dedup_v1` is specified as RDKit isomeric
canonicalization + exact-string dedup and nothing else. These tests pin
that contract: charged glyphs, salts, and oversized macrocycles must
survive (no neutralization / salt-strip / heavy-atom cap), duplicates
collapse on the *canonical* form, and the run is byte-deterministic.
"""

from __future__ import annotations

import hashlib
import inspect
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from pydantic import ValidationError
from rdkit import rdBase

import smiles_subword.preprocess.canon_dedup as canon_dedup_mod
import smiles_subword.preprocess.canonicalize_minimal as canonicalize_minimal_mod
from smiles_subword.config import CanonDedupConfig
from smiles_subword.ingest._common import RAW_V1_SCHEMA
from smiles_subword.preprocess.canon_dedup import canon_dedup

if TYPE_CHECKING:
    from datetime import datetime
    from pathlib import Path

    from smiles_subword.preprocess.types import CanonDedupResult

_RAW_TS: datetime = pa.scalar(0, type=pa.timestamp("us")).as_py()

_MACROCYCLE: str = "C1" + "C" * 80 + "1"
"""An 81-heavy-atom carbocycle — survives because `canon_dedup_v1` applies
no heavy-atom cap (a typical 70-atom filter would have dropped it)."""

FIXTURE_ROWS: list[tuple[str, str]] = [
    ("r1", "CCO"),
    ("r2", "OCC"),  # non-canonical form of CCO -> collapses onto r1
    ("r3", "[O-]C(=O)C"),  # anion
    ("r4", "[NH3+]CC([O-])=O"),  # zwitterion
    ("r5", "CCO.[Na+].[Cl-]"),  # salt
    ("r6", _MACROCYCLE),
    ("r7", "N[C@@H](C)C(=O)O"),  # stereocenter
    ("r8", "not a smiles"),  # RDKit-unparseable -> rejected
    ("r9", "[NH4+]"),  # cation
]


def _write_raw_v1_dir(
    tmp_path: Path, rows: list[tuple[str, str]], *, source: str = "fx"
) -> Path:
    raw_dir = tmp_path / "raw_v1"
    raw_dir.mkdir()
    shard = raw_dir / "raw_v1-000.parquet"
    table = pa.table(
        {
            "source_id": [r[0] for r in rows],
            "smiles": [r[1] for r in rows],
            "source": [source] * len(rows),
            "ingest_ts": [_RAW_TS] * len(rows),
        },
        schema=RAW_V1_SCHEMA,
    )
    pq.write_table(table, shard)
    manifest = {
        "schema": "raw_v1",
        "source": source,
        "shards": [
            {
                "file": "raw_v1-000.parquet",
                "sha256": hashlib.sha256(shard.read_bytes()).hexdigest(),
                "n_rows": len(rows),
            }
        ],
    }
    (raw_dir / "MANIFEST.yaml").write_text(yaml.safe_dump(manifest))
    return raw_dir


def _unique_rows(n: int) -> list[tuple[str, str]]:
    """`n` distinct, parseable, already-canonical linear alcohols."""
    return [(f"r{i:04d}", f"C{'C' * i}O") for i in range(n)]


def _canon_config(
    input_dir: Path, output_dir: Path, **overrides: object
) -> CanonDedupConfig:
    params: dict[str, object] = {
        "name": "fx",
        "input_dir": input_dir,
        "output_dir": output_dir,
        "n_workers": 1,
    }
    params.update(overrides)
    return CanonDedupConfig.model_validate(params)


def _output_table(output_dir: Path) -> pa.Table:
    shards = sorted(output_dir.glob("canon_dedup_v1-*.parquet"))
    if not shards:
        return RAW_V1_SCHEMA.empty_table()
    return pa.concat_tables(pq.read_table(s) for s in shards)


def _run(
    tmp_path: Path, rows: list[tuple[str, str]], **overrides: object
) -> CanonDedupResult:
    raw_dir = _write_raw_v1_dir(tmp_path, rows)
    cfg = _canon_config(raw_dir, tmp_path / "out", **overrides)
    return canon_dedup(cfg, verify_input_sha=False)


class TestCanonDedup:
    """Survival of chemistry, dedup on canonical form, rejection counting."""

    def test_charged_glyphs_survive_when_canon_dedup_runs(self, tmp_path: Path) -> None:
        result = _run(tmp_path, FIXTURE_ROWS)

        smiles = set(_output_table(result.output_dir).column("smiles").to_pylist())
        assert {"CC(=O)[O-]", "[NH3+]CC(=O)[O-]", "[NH4+]"} <= smiles

    def test_salt_components_survive_when_canon_dedup_runs(
        self, tmp_path: Path
    ) -> None:
        result = _run(tmp_path, FIXTURE_ROWS)

        smiles = _output_table(result.output_dir).column("smiles").to_pylist()
        salt = next(s for s in smiles if "." in s)
        assert sorted(salt.split(".")) == ["CCO", "[Cl-]", "[Na+]"]

    def test_oversized_macrocycle_survives_when_canon_dedup_runs(
        self, tmp_path: Path
    ) -> None:
        result = _run(tmp_path, FIXTURE_ROWS)

        smiles = _output_table(result.output_dir).column("smiles").to_pylist()
        assert _MACROCYCLE in smiles

    def test_stereochemistry_survives_when_canon_dedup_runs(
        self, tmp_path: Path
    ) -> None:
        result = _run(tmp_path, FIXTURE_ROWS)

        smiles = _output_table(result.output_dir).column("smiles").to_pylist()
        assert "C[C@H](N)C(=O)O" in smiles

    def test_noncanonical_duplicate_collapses_after_canonicalization(
        self, tmp_path: Path
    ) -> None:
        result = _run(tmp_path, [("r1", "CCO"), ("r2", "OCC")])

        assert result.n_duplicates == 1

    def test_smallest_source_id_kept_when_canonical_forms_collide(
        self, tmp_path: Path
    ) -> None:
        result = _run(tmp_path, [("r2", "OCC"), ("r1", "CCO")])

        table = _output_table(result.output_dir)
        assert table.column("source_id").to_pylist() == ["r1"]

    def test_unparseable_row_dropped_and_counted(self, tmp_path: Path) -> None:
        result = _run(tmp_path, [("r1", "CCO"), ("r2", "not a smiles")])

        assert result.n_rdkit_rejected == 1

    def test_fixture_corpus_produces_expected_counts(self, tmp_path: Path) -> None:
        result = _run(tmp_path, FIXTURE_ROWS)

        counts = (
            result.n_input_rows,
            result.n_rdkit_rejected,
            result.n_duplicates,
            result.n_output_rows,
        )
        assert counts == (9, 1, 1, 7)

    def test_output_rows_equal_input_minus_rejected_minus_duplicates(
        self, tmp_path: Path
    ) -> None:
        result = _run(tmp_path, FIXTURE_ROWS)

        assert result.n_output_rows == (
            result.n_input_rows - result.n_rdkit_rejected - result.n_duplicates
        )

    @pytest.mark.parametrize("mode", ["single_pass", "bucket"])
    def test_rejection_rate_is_zero_when_corpus_empty(
        self, tmp_path: Path, mode: str
    ) -> None:
        result = _run(tmp_path, [], mode=mode)

        assert result.rdkit_rejection_rate == 0.0


class TestManifest:
    """Per-stage `canon_dedup_v1/MANIFEST.yaml` contents."""

    def _manifest(self, result: CanonDedupResult) -> dict:
        return yaml.safe_load((result.output_dir / "MANIFEST.yaml").read_text())

    def test_manifest_records_rdkit_rejection_count_and_rate(
        self, tmp_path: Path
    ) -> None:
        manifest = self._manifest(_run(tmp_path, FIXTURE_ROWS))

        assert manifest["n_rdkit_rejected"] == 1
        assert manifest["rdkit_rejection_rate"] == pytest.approx(1 / 9)

    def test_manifest_records_rdkit_version(self, tmp_path: Path) -> None:
        manifest = self._manifest(_run(tmp_path, FIXTURE_ROWS))

        assert manifest["rdkit_version"] == rdBase.rdkitVersion

    def test_manifest_schema_is_canon_dedup_v1(self, tmp_path: Path) -> None:
        manifest = self._manifest(_run(tmp_path, FIXTURE_ROWS))

        assert manifest["schema"] == "canon_dedup_v1"

    def test_manifest_lists_excluded_steps(self, tmp_path: Path) -> None:
        manifest = self._manifest(_run(tmp_path, FIXTURE_ROWS))

        assert manifest["excluded_steps"] == [
            "salt_strip",
            "neutralize",
            "heavy_atom_cap",
            "descriptors",
            "decontaminate",
        ]

    def test_manifest_records_canonicalization_flags(self, tmp_path: Path) -> None:
        manifest = self._manifest(_run(tmp_path, FIXTURE_ROWS))

        assert manifest["canonicalization"] == {
            "isomeric_smiles": True,
            "kekule_smiles": False,
        }

    def test_manifest_chains_source_provenance(self, tmp_path: Path) -> None:
        raw_dir = _write_raw_v1_dir(tmp_path, FIXTURE_ROWS)
        cfg = _canon_config(raw_dir, tmp_path / "out")
        expected = hashlib.sha256((raw_dir / "MANIFEST.yaml").read_bytes()).hexdigest()

        canon_dedup(cfg, verify_input_sha=False)

        manifest = yaml.safe_load((cfg.output_dir / "MANIFEST.yaml").read_text())
        assert manifest["source_manifest_sha256"] == expected

    def test_manifest_shard_checksums_match_on_disk(self, tmp_path: Path) -> None:
        result = _run(tmp_path, FIXTURE_ROWS)

        manifest = self._manifest(result)
        for shard_meta in manifest["shards"]:
            on_disk = result.output_dir / shard_meta["file"]
            assert (
                hashlib.sha256(on_disk.read_bytes()).hexdigest() == shard_meta["sha256"]
            )

    def test_manifest_records_bucket_metadata_in_bucket_mode(
        self, tmp_path: Path
    ) -> None:
        manifest = self._manifest(_run(tmp_path, FIXTURE_ROWS, mode="bucket"))

        assert manifest["mode"] == "bucket"
        assert manifest["bucket_key"] == "smiles_prefix_2"
        assert manifest["n_buckets"] >= 1


class TestStructuralGuarantee:
    """The 'no charge/salt/heavy-atom step runs' guarantee is structural."""

    def test_pipeline_source_excludes_policy_machinery(self) -> None:
        src = inspect.getsource(canon_dedup_mod) + inspect.getsource(
            canonicalize_minimal_mod
        )

        forbidden = [
            "SaltRemover",
            "MolStandardize",
            "Uncharger",
            "LargestFragmentChooser",
            "GetNumHeavyAtoms",
            "apply_policy",
            "try_apply_policy",
        ]
        assert [sym for sym in forbidden if sym in src] == []

    def test_config_rejects_unknown_policy_knob(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError):
            CanonDedupConfig.model_validate(
                {
                    "name": "fx",
                    "input_dir": tmp_path / "raw_v1",
                    "output_dir": tmp_path / "out",
                    "neutralize": True,
                }
            )


class TestDeterminism:
    """Re-runs are byte-identical; the two dedup modes agree on content."""

    @pytest.mark.parametrize("mode", ["single_pass", "bucket"])
    def test_byte_identical_when_rerun_on_same_input(
        self, tmp_path: Path, mode: str
    ) -> None:
        # Both modes must be byte-deterministic across reruns — bucket mode is
        # the 50M+ path and its shard SHAs are recorded in the manifest.
        raw_dir = _write_raw_v1_dir(tmp_path, FIXTURE_ROWS)
        cfg_a = _canon_config(raw_dir, tmp_path / "out_a", mode=mode)
        cfg_b = _canon_config(raw_dir, tmp_path / "out_b", mode=mode)

        res_a = canon_dedup(cfg_a, verify_input_sha=False)
        res_b = canon_dedup(cfg_b, verify_input_sha=False)

        assert {s.sha256 for s in res_a.shards} == {s.sha256 for s in res_b.shards}

    def test_single_pass_and_bucket_modes_are_content_equivalent(
        self, tmp_path: Path
    ) -> None:
        raw_dir = _write_raw_v1_dir(tmp_path, FIXTURE_ROWS)
        cfg_single = _canon_config(raw_dir, tmp_path / "out_single", mode="single_pass")
        cfg_bucket = _canon_config(raw_dir, tmp_path / "out_bucket", mode="bucket")

        canon_dedup(cfg_single, verify_input_sha=False)
        canon_dedup(cfg_bucket, verify_input_sha=False)

        assert _output_table(cfg_single.output_dir).equals(
            _output_table(cfg_bucket.output_dir)
        )

    def test_parallel_canonicalization_matches_serial(self, tmp_path: Path) -> None:
        cycle = ["CCO", "OCC", "c1ccccc1", "CC(=O)O", "not a smiles"]
        rows = [(f"r{i:04d}", cycle[i % len(cycle)]) for i in range(150)]
        raw_dir = _write_raw_v1_dir(tmp_path, rows)
        cfg_serial = _canon_config(
            raw_dir, tmp_path / "out_serial", n_workers=1, rows_per_batch=8
        )
        cfg_parallel = _canon_config(
            raw_dir, tmp_path / "out_parallel", n_workers=4, rows_per_batch=8
        )

        res_serial = canon_dedup(cfg_serial, verify_input_sha=False)
        res_parallel = canon_dedup(cfg_parallel, verify_input_sha=False)

        assert _output_table(cfg_serial.output_dir).equals(
            _output_table(cfg_parallel.output_dir)
        )
        assert res_serial.n_rdkit_rejected == res_parallel.n_rdkit_rejected


class TestIO:
    """Output schema, staging cleanup, sharding, and input verification."""

    def test_output_schema_is_raw_v1(self, tmp_path: Path) -> None:
        result = _run(tmp_path, FIXTURE_ROWS)

        for shard in sorted(result.output_dir.glob("canon_dedup_v1-*.parquet")):
            assert pq.ParquetFile(shard).schema_arrow.equals(RAW_V1_SCHEMA)

    def test_no_intermediate_dirs_remain_after_run(self, tmp_path: Path) -> None:
        result = _run(tmp_path, FIXTURE_ROWS, mode="bucket")

        assert not (result.output_dir / "_canonical").exists()
        assert not (result.output_dir / "_buckets").exists()

    def test_atomic_output_replaces_prior_dir(self, tmp_path: Path) -> None:
        raw_dir = _write_raw_v1_dir(tmp_path, FIXTURE_ROWS)
        cfg = _canon_config(raw_dir, tmp_path / "out")
        cfg.output_dir.mkdir(parents=True)
        (cfg.output_dir / "stale.parquet").write_bytes(b"junk")

        canon_dedup(cfg, verify_input_sha=False)

        assert not (cfg.output_dir / "stale.parquet").exists()

    def test_multiple_output_shards_when_size_threshold_low(
        self, tmp_path: Path
    ) -> None:
        result = _run(
            tmp_path, _unique_rows(60), rows_per_batch=4, shard_target_bytes=2048
        )

        shards = sorted(result.output_dir.glob("canon_dedup_v1-*.parquet"))
        assert len(shards) >= 2

    def test_rejects_when_input_shard_sha_mismatch(self, tmp_path: Path) -> None:
        raw_dir = _write_raw_v1_dir(tmp_path, [("r1", "CCO")])
        shard = raw_dir / "raw_v1-000.parquet"
        shard.write_bytes(shard.read_bytes() + b"\x00")
        cfg = _canon_config(raw_dir, tmp_path / "out")

        with pytest.raises(ValueError, match=r"sha256 mismatch"):
            canon_dedup(cfg, verify_input_sha=True)

    def test_rejects_when_duckdb_memory_limit_malformed(self, tmp_path: Path) -> None:
        raw_dir = _write_raw_v1_dir(tmp_path, [("r1", "CCO")])
        cfg = _canon_config(
            raw_dir, tmp_path / "out", duckdb_memory_limit="two gigabytes"
        )

        with pytest.raises(ValueError, match=r"malformed duckdb_memory_limit"):
            canon_dedup(cfg, verify_input_sha=False)
