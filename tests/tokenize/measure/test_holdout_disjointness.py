"""No train/eval leakage at the measurement layer.

Every held-out-consuming runner (absorption/fertility/jaccard/distribution/
segmentation/transfer) evaluates on
``corpus_test_split_dir`` while tokenizers train on ``corpus_training_dir``.
This certifies (a) those two resolvers point at the disjoint ``test``/``train``
halves of one split, and (b) end-to-end, the SMILES a measurement streams from
the test split share nothing with the training split — so no measurement scores
a tokenizer on a molecule it was trained on.

The split's own partition/disjointness (by ``source_id``) is covered in
``tests/preprocess/test_holdout_split.py``; this ties that guarantee to the
exact reader (``iter_smiles_from_parquet``) the measurement runners use.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from smiles_subword.config import HoldoutSplitConfig
from smiles_subword.ingest._common import RAW_V1_SCHEMA
from smiles_subword.preprocess.holdout_split import split_train_test
from smiles_subword.tokenize._corpus import iter_smiles_from_parquet
from smiles_subword.tokenize.grid import corpus_training_dir
from smiles_subword.tokenize.measure._cells import corpus_test_split_dir

if TYPE_CHECKING:
    from pathlib import Path

_CORPORA = ["pubchem", "zinc22", "coconut", "real_space"]


class TestEvalAndTrainingResolveToDisjointHalves:
    """The eval resolver and training resolver point at sibling ``test``/``train``."""

    @pytest.mark.parametrize("corpus", _CORPORA)
    def test_test_and_train_dirs_are_distinct_siblings(self, corpus: str) -> None:
        train = corpus_training_dir(corpus)
        test = corpus_test_split_dir(corpus)
        assert train != test
        assert train.name == "train"
        assert test.name == "test"
        # same canon_dedup_v1 split parent → the two halves of one partition
        assert train.parent == test.parent
        assert train.parent.name == "canon_dedup_v1"


def _write_canon_dedup_corpus(input_dir: Path, rows: list[tuple[str, str]]) -> None:
    """Write a one-shard ``canon_dedup_v1`` dir (raw_v1 schema) + manifest."""
    input_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime(2026, 4, 26, 12, 0, 0, tzinfo=UTC)
    shard = input_dir / "canon_dedup_v1-00000.parquet"
    table = pa.table(
        {
            "source_id": [r[0] for r in rows],
            "smiles": [r[1] for r in rows],
            "source": ["pubchem"] * len(rows),
            "ingest_ts": [ts] * len(rows),
        },
        schema=RAW_V1_SCHEMA,
    )
    pq.write_table(table, shard, compression="zstd")
    manifest = {
        "schema": "canon_dedup_v1",
        "name": "fx",
        "shards": [
            {
                "file": shard.name,
                "sha256": hashlib.sha256(shard.read_bytes()).hexdigest(),
                "n_rows": len(rows),
                "n_bytes": shard.stat().st_size,
            }
        ],
    }
    with (input_dir / "MANIFEST.yaml").open("w") as fh:
        yaml.safe_dump(manifest, fh, sort_keys=False)


class TestMeasurementReaderSeesNoTrainingMolecules:
    """End-to-end: the held-out SMILES a measurement reads are disjoint from train."""

    def test_test_split_smiles_disjoint_from_train_split(self, tmp_path: Path) -> None:
        # Distinct SMILES per row (the post-canon-dedup case); a 50/50 split keeps
        # both halves comfortably non-empty regardless of the hash distribution.
        rows = [(f"id{i:06d}", f"C{'C' * (i % 30)}O{i}") for i in range(400)]
        input_dir = tmp_path / "canon_dedup_v1_in"
        _write_canon_dedup_corpus(input_dir, rows)
        cfg = HoldoutSplitConfig(
            name="fx",
            input_dir=input_dir,
            output_dir=tmp_path / "canon_dedup_v1",
            test_fraction=0.5,
            rows_per_batch=64,
        )

        result = split_train_test(cfg)

        # Read through the exact reader the measurement runners use.
        train_smiles = set(iter_smiles_from_parquet(result.output_dir / "train"))
        test_smiles = set(iter_smiles_from_parquet(result.output_dir / "test"))

        assert train_smiles, "training split must be non-empty"
        assert test_smiles, "held-out split must be non-empty"
        assert train_smiles.isdisjoint(test_smiles)
        # no molecule is silently dropped or duplicated across the partition
        assert len(train_smiles) + len(test_smiles) == 400
