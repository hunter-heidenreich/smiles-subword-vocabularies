"""Tests for ``smiles_subword.tokenize.build_tokenizer`` end-to-end dispatch."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from smiles_subword.config import TokenizerConfig
from smiles_subword.tokenize import (
    ATOMIC_VOCAB_SIZE,
    SmirkAdapter,
    UnigramSmirkAdapter,
    build_tokenizer,
)

_FIXTURE_SMILES: tuple[str, ...] = (
    "CCO",
    "CC(=O)O",
    "CC(=O)Oc1ccccc1C(=O)O",
    "Cc1ccccc1",
    "Cc1ccc(C)cc1",
    "c1ccncc1",
    "c1ccc2ccccc2c1",
    "ClC(Cl)(Cl)Cl",
    "BrCCBr",
    "FC(F)(F)C(=O)O",
)


def _write_synthetic_sub_v1(parquet_dir: Path, repeats: int = 50) -> None:
    """Write a tiny Parquet stage to ``parquet_dir`` with a sub_v1-shaped column."""
    parquet_dir.mkdir(parents=True, exist_ok=True)
    rows = list(_FIXTURE_SMILES) * repeats
    table = pa.table({"smiles": rows})
    shard_path = parquet_dir / "sub_v1-00000.parquet"
    pq.write_table(table, shard_path)
    payload = {
        "schema": "sub_v1",
        "name": "synthetic",
        "shards": [
            {
                "file": shard_path.name,
                "sha256": "0" * 64,
                "n_rows": len(rows),
                "n_bytes": shard_path.stat().st_size,
            }
        ],
    }
    with (parquet_dir / "MANIFEST.yaml").open("w") as fh:
        yaml.safe_dump(payload, fh, sort_keys=False)


@pytest.fixture
def synthetic_sub_v1(tmp_path: Path) -> Path:
    parquet_dir = tmp_path / "sub_v1"
    _write_synthetic_sub_v1(parquet_dir, repeats=50)
    return parquet_dir


class TestDispatchSmirkBase:
    """``build_tokenizer`` returns the atomic-baseline SmirkAdapter."""

    def test_atomic_vocab_size(self, tmp_path: Path) -> None:
        cfg = TokenizerConfig(
            name="smirk_base",
            kind="smirk_base",
            output_dir=tmp_path / "out",
        )

        tok = build_tokenizer(cfg)

        assert isinstance(tok, SmirkAdapter)
        assert tok.vocab_size == ATOMIC_VOCAB_SIZE


class TestDispatchSmirkGpe:
    """``build_tokenizer`` runs train_gpe and (optionally) chains via ref."""

    def test_trains_at_target_vocab_and_records_corpus_sha(
        self, synthetic_sub_v1: Path, tmp_path: Path
    ) -> None:
        cfg = TokenizerConfig(
            name="gpe_180",
            kind="smirk_gpe",
            vocab_size=180,
            min_frequency=2,
            training_input=synthetic_sub_v1,
            output_dir=tmp_path / "gpe_180",
        )

        tok = build_tokenizer(cfg)
        tok.save(cfg.output_dir)
        meta = yaml.safe_load((cfg.output_dir / "meta.yaml").read_text())

        assert isinstance(tok, SmirkAdapter)
        assert tok.base_kind == "smirk_gpe"
        assert tok.vocab_size > ATOMIC_VOCAB_SIZE
        assert tok.vocab_size <= 180
        assert meta["training_corpus_sha"] is not None
        assert len(meta["training_corpus_sha"]) == 32

    def test_ships_the_natural_untrimmed_len(
        self, synthetic_sub_v1: Path, tmp_path: Path
    ) -> None:
        """No post-train trim to ``V``: the artifact keeps its realized length
        (six tail specials above the WordLevel base), not forced down to ``V``."""
        cfg = TokenizerConfig(
            name="gpe_200",
            kind="smirk_gpe",
            vocab_size=200,
            min_frequency=2,
            training_input=synthetic_sub_v1,
            output_dir=tmp_path / "gpe_200",
        )

        tok = build_tokenizer(cfg)
        tok.save(cfg.output_dir)
        reloaded = SmirkAdapter.load(cfg.output_dir)

        assert len(reloaded) == reloaded.vocab_size + 6
        assert len(reloaded) > reloaded.vocab_size

    def test_ref_artifact_dir_chains_trajectory(
        self, synthetic_sub_v1: Path, tmp_path: Path
    ) -> None:
        cfg_v1 = TokenizerConfig(
            name="gpe_v1",
            kind="smirk_gpe",
            vocab_size=170,
            min_frequency=2,
            training_input=synthetic_sub_v1,
            output_dir=tmp_path / "gpe_v1",
        )
        v1 = build_tokenizer(cfg_v1)
        v1.save(cfg_v1.output_dir)
        cfg_v2 = TokenizerConfig(
            name="gpe_v2",
            kind="smirk_gpe",
            vocab_size=200,
            min_frequency=2,
            training_input=synthetic_sub_v1,
            output_dir=tmp_path / "gpe_v2",
            ref_artifact_dir=cfg_v1.output_dir,
        )
        v2 = build_tokenizer(cfg_v2)
        v2.save(cfg_v2.output_dir)
        v1_merges = (cfg_v1.output_dir / "merges.txt").read_text().splitlines()[1:]
        v2_merges = (cfg_v2.output_dir / "merges.txt").read_text().splitlines()[1:]

        assert v2.vocab_size > v1.vocab_size
        assert v2_merges[: len(v1_merges)] == v1_merges


class TestDispatchSmirkUnigram:
    """``build_tokenizer`` runs train_unigram for the Unigram-LM arm."""

    def test_trains_at_target_vocab_and_records_corpus_sha(
        self, synthetic_sub_v1: Path, tmp_path: Path
    ) -> None:
        cfg = TokenizerConfig(
            name="unigram_200",
            kind="smirk_unigram",
            vocab_size=200,
            training_input=synthetic_sub_v1,
            output_dir=tmp_path / "unigram_200",
        )

        tok = build_tokenizer(cfg)
        tok.save(cfg.output_dir)
        meta = yaml.safe_load((cfg.output_dir / "meta.yaml").read_text())

        assert isinstance(tok, UnigramSmirkAdapter)
        assert tok.base_kind == "smirk_unigram"
        assert tok.vocab_size > ATOMIC_VOCAB_SIZE
        assert meta["training_corpus_sha"] is not None
        assert meta["n_merges"] is None
        assert not (cfg.output_dir / "merges.txt").exists()

    def test_seed_size_and_max_piece_length_flow_into_meta(
        self, synthetic_sub_v1: Path, tmp_path: Path
    ) -> None:
        cfg = TokenizerConfig(
            name="unigram_knobs",
            kind="smirk_unigram",
            vocab_size=200,
            training_input=synthetic_sub_v1,
            output_dir=tmp_path / "unigram_knobs",
            seed_size=4096,
            max_piece_length=64,
        )

        tok = build_tokenizer(cfg)
        tok.save(cfg.output_dir)
        meta = yaml.safe_load((cfg.output_dir / "meta.yaml").read_text())

        assert meta["seed_size"] == 4096
        assert meta["max_piece_length"] == 64


class TestSmiCacheReuse:
    """The ``.smi`` cache is reused when the corpus fingerprint matches."""

    def test_smi_cache_is_written_alongside_parquet_dir(
        self, synthetic_sub_v1: Path, tmp_path: Path
    ) -> None:
        cfg = TokenizerConfig(
            name="gpe_cache_probe",
            kind="smirk_gpe",
            vocab_size=170,
            min_frequency=2,
            training_input=synthetic_sub_v1,
            output_dir=tmp_path / "cache_probe",
        )
        build_tokenizer(cfg)

        cache_smi = synthetic_sub_v1.parent / f"{synthetic_sub_v1.name}.smi"
        cache_sha = synthetic_sub_v1.parent / f"{synthetic_sub_v1.name}.smi.sha"
        assert cache_smi.exists()
        assert cache_sha.exists()
        assert len(cache_sha.read_text().strip()) == 32

    def test_second_call_does_not_rewrite_cache(
        self, synthetic_sub_v1: Path, tmp_path: Path
    ) -> None:
        cfg_first = TokenizerConfig(
            name="gpe_first",
            kind="smirk_gpe",
            vocab_size=170,
            min_frequency=2,
            training_input=synthetic_sub_v1,
            output_dir=tmp_path / "first",
        )
        build_tokenizer(cfg_first)
        cache_smi = synthetic_sub_v1.parent / f"{synthetic_sub_v1.name}.smi"
        first_mtime = cache_smi.stat().st_mtime_ns
        cfg_second = TokenizerConfig(
            name="gpe_second",
            kind="smirk_gpe",
            vocab_size=170,
            min_frequency=2,
            training_input=synthetic_sub_v1,
            output_dir=tmp_path / "second",
        )

        build_tokenizer(cfg_second)

        assert cache_smi.stat().st_mtime_ns == first_mtime


class TestScaffoldLog:
    """``scaffold_log=True`` wires a per-merge-step log into the output dir."""

    def test_scaffold_log_writes_jsonl_in_output_dir(
        self, synthetic_sub_v1: Path, tmp_path: Path
    ) -> None:
        # Adapter-level scaffold logging is covered in test_smirk_adapter; this
        # pins the build_tokenizer wiring (cfg.scaffold_log -> output_dir mkdir
        # + scaffold.jsonl path -> train_gpe), which the adapter tests bypass.
        cfg = TokenizerConfig(
            name="gpe_scaffold",
            kind="smirk_gpe",
            vocab_size=180,
            min_frequency=2,
            training_input=synthetic_sub_v1,
            output_dir=tmp_path / "gpe_scaffold",
            scaffold_log=True,
        )

        build_tokenizer(cfg)

        log_path = cfg.output_dir / "scaffold.jsonl"
        assert log_path.exists()
        assert log_path.read_text().strip()  # per-merge-step records, not empty
