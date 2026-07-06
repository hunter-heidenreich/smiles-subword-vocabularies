"""Tests for ``distribution_runner`` (held-out encode pass)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from smiles_subword.tokenize.measure import _cells
from smiles_subword.tokenize.measure.distribution import runner as distribution_runner

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


class _FakeAdapter:
    """Table-lookup adapter satisfying the encode + special-id interface.

    ``encode_batch`` is what :func:`iter_encoded_batches` calls; the special-id
    properties are read by ``collect_special_ids``. Encodings may include
    special ids to verify they are dropped from the distribution.
    """

    name = "fake"

    def __init__(
        self,
        *,
        vocab_size: int,
        encodings: dict[str, list[int]],
        bos_id: int = 0,
        eos_id: int = 1,
        pad_id: int = 2,
        unk_id: int | None = None,
    ) -> None:
        self.vocab_size = vocab_size
        self._encodings = encodings
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.unk_id = unk_id

    def encode(self, smiles: str, *, add_special_tokens: bool = False) -> list[int]:
        return list(self._encodings[smiles])

    def encode_batch(
        self, smiles: list[str], *, add_special_tokens: bool = False
    ) -> list[list[int]]:
        return [self.encode(s) for s in smiles]


_ENCODINGS = {
    "CCO": [0, 3, 4, 3, 1],
    "CC": [0, 3, 3, 1],
    "OO": [0, 4, 4, 1],
}


class TestBuildDistributionData:
    def test_drops_specials_and_counts_per_molecule(self) -> None:
        adapter = _FakeAdapter(vocab_size=10, encodings=_ENCODINGS)

        data = distribution_runner.build_distribution_data(
            adapter,
            ["CCO"],
            v_effective=7,
            special_ids=frozenset({0, 1, 2}),
        )

        assert data.n_molecules == 1
        assert data.total_tokens == 3
        assert set(data.local_token_ids) == {3, 4}

    def test_empty_only_specials_molecule_still_advances_index(self) -> None:
        adapter = _FakeAdapter(vocab_size=10, encodings={"x": [0, 1, 2]})

        data = distribution_runner.build_distribution_data(
            adapter,
            ["x"],
            v_effective=7,
            special_ids=frozenset({0, 1, 2}),
        )

        assert data.n_molecules == 1
        assert data.total_tokens == 0
        assert data.local_token_ids == ()

    def test_aggregate_count_matches_corpus_frequencies(self) -> None:
        adapter = _FakeAdapter(vocab_size=10, encodings=_ENCODINGS)

        data = distribution_runner.build_distribution_data(
            adapter,
            ["CCO", "CC", "OO"],
            v_effective=7,
            special_ids=frozenset({0, 1, 2}),
        )

        assert data.n_molecules == 3
        assert data.total_tokens == 3 + 2 + 2


class TestRunArmDistribution:
    def test_aggregates_held_out_stream(self, monkeypatch: pytest.MonkeyPatch) -> None:
        adapter = _FakeAdapter(vocab_size=10, encodings=_ENCODINGS)
        monkeypatch.setattr(
            _cells,
            "iter_smiles_from_parquet",
            lambda _dir: ["CCO", "CC", "OO"],
        )
        monkeypatch.setattr(
            distribution_runner, "eval_split_sha", lambda _corpus: "eval-X"
        )

        arm = distribution_runner.run_arm_distribution(
            adapter,
            cell_id="pubchem__smirk_gpe_v256_nmb",
            corpus="pubchem",
            arm="bpe",
            boundary="nmb",
            v_effective=256,
            special_ids=frozenset({0, 1, 2}),
            training_corpus_sha="sha-A",
        )

        assert arm.n_molecules == 3
        assert arm.vocab_size == 10
        assert arm.v_effective == 256
        assert arm.live_token_count == 2
        assert arm.live_token_count <= arm.v_effective
        assert arm.eval_split_sha == "eval-X"
        assert arm.total_tokens == 7

    def test_limit_molecules_truncates_stream(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        adapter = _FakeAdapter(vocab_size=10, encodings=_ENCODINGS)
        monkeypatch.setattr(
            _cells,
            "iter_smiles_from_parquet",
            lambda _dir: iter(["CCO", "CC", "OO"]),
        )

        arm = distribution_runner.run_arm_distribution(
            adapter,
            cell_id="x",
            corpus="pubchem",
            arm="bpe",
            boundary="nmb",
            v_effective=256,
            special_ids=frozenset({0, 1, 2}),
            training_corpus_sha="sha-A",
            eval_split_sha_value="eval-X",
            limit_molecules=1,
        )

        assert arm.n_molecules == 1


class TestCollectAllSpecialIds:
    def test_unions_protocol_specials_with_added_tokens(self, tmp_path: Path) -> None:
        adapter = _FakeAdapter(
            vocab_size=256, encodings={}, bos_id=250, eos_id=251, pad_id=253, unk_id=0
        )
        (tmp_path / "tokenizer.json").write_text(
            json.dumps(
                {
                    "added_tokens": [
                        {"id": 0, "content": "[UNK]", "special": True},
                        {"id": 250, "content": "[BOS]", "special": True},
                        {"id": 252, "content": "[SEP]", "special": True},
                        {"id": 254, "content": "[CLS]", "special": True},
                        {"id": 255, "content": "[MASK]", "special": True},
                        {"id": 7, "content": "cc", "special": False},
                    ]
                }
            )
        )

        ids = distribution_runner.collect_all_special_ids(adapter, tmp_path)

        assert ids == frozenset({0, 250, 251, 252, 253, 254, 255})

    def test_missing_tokenizer_json_falls_back_to_protocol_specials(
        self, tmp_path: Path
    ) -> None:
        adapter = _FakeAdapter(vocab_size=10, encodings={})

        ids = distribution_runner.collect_all_special_ids(adapter, tmp_path)

        assert ids == frozenset({0, 1, 2})
