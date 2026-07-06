"""Tests for ``segmentation_runner`` (held-out encode pass)."""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING

import pytest

from smiles_subword.tokenize.measure import _cells
from smiles_subword.tokenize.measure.segmentation import runner as segmentation_runner
from smiles_subword.tokenize.measure.segmentation.runner import (
    build_segmentation_data,
    build_unigram_piece_scores,
    run_arm_segmentation,
)

if TYPE_CHECKING:
    from pathlib import Path

_PIECE_SCORES = {
    ("C",): math.log(0.5),
    ("C", "C"): math.log(0.25),
    ("O",): -1.0,
}
_GLYPHS_BY_ID = {0: ("C",), 1: ("C", "C"), 2: ("O",)}


class _FakeAdapter:
    """Stub exposing only the surface :func:`build_segmentation_data` / runner touch."""

    def __init__(
        self,
        chunks_by_smi: dict[str, list[str]],
        ids_by_chunk: dict[str, list[int]],
    ) -> None:
        self._chunks = chunks_by_smi
        self._ids = ids_by_chunk
        self.encode_calls: list[str] = []
        self.vocab_size = 3

    def pretokenize_layer_b(self, smi: str) -> list[tuple[str, tuple[int, int]]]:
        return [(c, (0, 0)) for c in self._chunks[smi]]

    def encode(self, chunk: str, *, add_special_tokens: bool = False) -> list[int]:
        self.encode_calls.append(chunk)
        return self._ids[chunk]


def _write_unigram_json(path: Path) -> Path:
    payload = {
        "model": {
            "type": "Unigram",
            "vocab": [
                {"glyphs": ["C"], "score": math.log(0.5)},
                {"glyphs": ["C", "C"], "score": math.log(0.25)},
                {"glyphs": ["O"], "score": -1.0},
            ],
        }
    }
    path.write_text(json.dumps(payload))
    return path


class TestBuildUnigramPieceScores:
    def test_reads_glyph_tuple_scores(self, tmp_path: Path) -> None:
        tj = _write_unigram_json(tmp_path / "tokenizer.json")

        scores = build_unigram_piece_scores(tj)

        assert scores == pytest.approx(_PIECE_SCORES)

    def test_rejects_non_unigram(self, tmp_path: Path) -> None:
        tj = tmp_path / "tokenizer.json"
        tj.write_text(json.dumps({"model": {"type": "BPE", "vocab": {}}}))

        with pytest.raises(ValueError, match="not Unigram"):
            build_unigram_piece_scores(tj)


class TestBuildSegmentationData:
    def test_molecule_entropy_is_sum_of_chunk_entropies(self) -> None:
        adapter = _FakeAdapter(
            chunks_by_smi={"X": ["CC", "O"]},
            ids_by_chunk={"CC": [1], "O": [2]},
        )

        data = build_segmentation_data(
            adapter,  # type: ignore[arg-type]
            ["X"],
            piece_scores=_PIECE_SCORES,
            glyph_tuple_by_id=_GLYPHS_BY_ID,
            max_piece_len=2,
        )

        assert len(data) == 1
        assert data[0].entropy_nats == pytest.approx(math.log(2))
        assert data[0].n_glyphs == 3

    def test_distinct_chunks_encoded_once(self) -> None:
        adapter = _FakeAdapter(
            chunks_by_smi={"X": ["CC", "O"], "Y": ["CC", "CC"]},
            ids_by_chunk={"CC": [1], "O": [2]},
        )

        build_segmentation_data(
            adapter,  # type: ignore[arg-type]
            ["X", "Y"],
            piece_scores=_PIECE_SCORES,
            glyph_tuple_by_id=_GLYPHS_BY_ID,
            max_piece_len=2,
        )

        assert adapter.encode_calls.count("CC") == 1
        assert sorted(set(adapter.encode_calls)) == ["CC", "O"]

    def test_oov_glyph_is_forced_boundary_worth_one_glyph(self) -> None:
        adapter = _FakeAdapter(
            chunks_by_smi={"X": ["CCxCC"]},
            ids_by_chunk={"CCxCC": [1, 9, 1]},
        )

        data = build_segmentation_data(
            adapter,  # type: ignore[arg-type]
            ["X"],
            piece_scores=_PIECE_SCORES,
            glyph_tuple_by_id=_GLYPHS_BY_ID,
            max_piece_len=2,
        )

        assert data[0].n_glyphs == 5
        assert data[0].entropy_nats == pytest.approx(2 * math.log(2))

    def test_glyph_sequence_reconstructed_from_multi_token_segmentation(self) -> None:
        adapter = _FakeAdapter(
            chunks_by_smi={"X": ["CC"]},
            ids_by_chunk={"CC": [0, 0]},
        )

        data = build_segmentation_data(
            adapter,  # type: ignore[arg-type]
            ["X"],
            piece_scores=_PIECE_SCORES,
            glyph_tuple_by_id=_GLYPHS_BY_ID,
            max_piece_len=2,
        )

        assert data[0].n_glyphs == 2
        assert data[0].entropy_nats == pytest.approx(math.log(2))


class TestRunArmSegmentation:
    def test_bpe_arm_is_zero_without_encoding(self) -> None:
        adapter = _FakeAdapter(chunks_by_smi={}, ids_by_chunk={})

        arm = run_arm_segmentation(
            adapter,  # type: ignore[arg-type]
            cell_id="pubchem__smirk_gpe_v256_nmb",
            corpus="pubchem",
            arm="bpe",
            boundary="nmb",
            training_corpus_sha="sha-A",
        )

        assert arm.verified_by_construction is True
        assert arm.entropy_per_molecule_mean == 0.0
        assert adapter.encode_calls == []

    def test_unigram_arm_streams_and_aggregates(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tj = _write_unigram_json(tmp_path / "tokenizer.json")
        adapter = _FakeAdapter(
            chunks_by_smi={"X": ["CC", "O"]},
            ids_by_chunk={"CC": [1], "O": [2]},
        )
        monkeypatch.setattr(
            _cells, "iter_smiles_from_parquet", lambda _dir: iter(["X"])
        )
        monkeypatch.setattr(
            segmentation_runner, "eval_split_sha", lambda _corpus: "eval-sha"
        )

        arm = run_arm_segmentation(
            adapter,  # type: ignore[arg-type]
            cell_id="pubchem__smirk_unigram_v256_nmb",
            corpus="pubchem",
            arm="unigram",
            boundary="nmb",
            training_corpus_sha="sha-A",
            tokenizer_json=tj,
        )

        assert arm.verified_by_construction is False
        assert arm.n_molecules == 1
        assert arm.total_glyphs == 3
        assert arm.total_entropy_nats == pytest.approx(math.log(2))
        assert arm.entropy_per_glyph == pytest.approx(math.log(2) / 3)
        assert arm.eval_split_sha == "eval-sha"

    def test_unigram_arm_requires_tokenizer_json(self) -> None:
        adapter = _FakeAdapter(chunks_by_smi={}, ids_by_chunk={})

        with pytest.raises(ValueError, match="requires a tokenizer_json"):
            run_arm_segmentation(
                adapter,  # type: ignore[arg-type]
                cell_id="c",
                corpus="pubchem",
                arm="unigram",
                boundary="nmb",
                training_corpus_sha="sha-A",
            )
