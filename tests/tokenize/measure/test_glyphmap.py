"""Tests for ``measure._glyphmap`` (token-id glyph tuples + their lengths)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from smiles_subword.tokenize.measure._glyphmap import (
    build_bpe_glyph_tuples,
    build_unigram_glyph_tuples,
    glyph_count_map,
    glyph_tuple_map,
)

if TYPE_CHECKING:
    from pathlib import Path


def _write_bpe_tokenizer_json(path: Path) -> None:
    """A Smirk-GPE tokenizer.json.

    ``model.vocab`` holds only the base alphabet (``Cl`` is a two-char base
    glyph, one glyph); merge ``k`` mints id ``base_size + k`` referencing grown
    ids, so merge 0 ``C+O`` is id 4 (``CO``) and merge 1 ``CO+Cl`` is id 5.
    """
    payload = {
        "model": {
            "vocab": {"[UNK]": 0, "C": 1, "O": 2, "Cl": 3},
            "merges": [[1, 2], [4, 3]],
        }
    }
    path.write_text(json.dumps(payload))


def _write_unigram_tokenizer_json(path: Path) -> None:
    payload = {
        "model": {
            "type": "Unigram",
            "vocab": [
                {"glyphs": ["[UNK]"], "score": 0.0},
                {"glyphs": ["C"], "score": -1.0},
                {"glyphs": ["Cl"], "score": -2.0},
                {"glyphs": ["C", "O"], "score": -3.0},
                {"glyphs": ["C", "O", "Cl"], "score": -4.0},
            ],
        }
    }
    path.write_text(json.dumps(payload))


class TestBuildBpeGlyphTuples:
    def test_base_glyphs_are_single_element_tuples(self, tmp_path: Path) -> None:
        tok_json = tmp_path / "tokenizer.json"
        _write_bpe_tokenizer_json(tok_json)

        tuples = build_bpe_glyph_tuples(tok_json)

        assert tuples[1] == ("C",)
        assert tuples[3] == ("Cl",)  # multi-character atom is still one glyph

    def test_merges_concatenate_operand_tuples(self, tmp_path: Path) -> None:
        tok_json = tmp_path / "tokenizer.json"
        _write_bpe_tokenizer_json(tok_json)

        tuples = build_bpe_glyph_tuples(tok_json)

        assert tuples[4] == ("C", "O")
        assert tuples[5] == ("C", "O", "Cl")


class TestBuildUnigramGlyphTuples:
    def test_tuples_from_stored_glyph_lists(self, tmp_path: Path) -> None:
        tok_json = tmp_path / "tokenizer.json"
        _write_unigram_tokenizer_json(tok_json)

        tuples = build_unigram_glyph_tuples(tok_json)

        assert tuples == {
            0: ("[UNK]",),
            1: ("C",),
            2: ("Cl",),
            3: ("C", "O"),
            4: ("C", "O", "Cl"),
        }

    def test_non_unigram_model_raises(self, tmp_path: Path) -> None:
        tok_json = tmp_path / "tokenizer.json"
        _write_bpe_tokenizer_json(tok_json)

        with pytest.raises(ValueError, match="not Unigram"):
            build_unigram_glyph_tuples(tok_json)


class TestCrossArmGlyphTupleAgreement:
    def test_shared_subword_has_identical_tuple_across_arms(
        self, tmp_path: Path
    ) -> None:
        bpe_json = tmp_path / "bpe.json"
        ul_json = tmp_path / "ul.json"
        _write_bpe_tokenizer_json(bpe_json)
        _write_unigram_tokenizer_json(ul_json)

        bpe_tuples = set(build_bpe_glyph_tuples(bpe_json).values())
        ul_tuples = set(build_unigram_glyph_tuples(ul_json).values())

        assert ("C", "O") in bpe_tuples & ul_tuples
        assert ("C", "O", "Cl") in bpe_tuples & ul_tuples


class TestGlyphTupleMap:
    def test_dispatches_to_bpe(self, tmp_path: Path) -> None:
        _write_bpe_tokenizer_json(tmp_path / "tokenizer.json")
        assert glyph_tuple_map(tmp_path, "bpe")[5] == ("C", "O", "Cl")

    def test_dispatches_to_unigram(self, tmp_path: Path) -> None:
        _write_unigram_tokenizer_json(tmp_path / "tokenizer.json")
        assert glyph_tuple_map(tmp_path, "unigram")[4] == ("C", "O", "Cl")


class TestGlyphCountMap:
    def test_bpe_counts_are_tuple_lengths(self, tmp_path: Path) -> None:
        _write_bpe_tokenizer_json(tmp_path / "tokenizer.json")

        counts = glyph_count_map(tmp_path, "bpe")

        assert counts[1] == 1  # base glyph
        assert counts[3] == 1  # multi-character atom Cl is one glyph
        assert counts[4] == 2  # merged CO
        assert counts[5] == 3  # merged COCl

    def test_unigram_counts_are_tuple_lengths(self, tmp_path: Path) -> None:
        _write_unigram_tokenizer_json(tmp_path / "tokenizer.json")

        counts = glyph_count_map(tmp_path, "unigram")

        assert counts == {0: 1, 1: 1, 2: 1, 3: 2, 4: 3}

    def test_count_map_is_lengths_of_tuple_map(self, tmp_path: Path) -> None:
        """The count map is exactly the per-id tuple length (one merge-tree walk)."""
        _write_bpe_tokenizer_json(tmp_path / "tokenizer.json")

        tuples = glyph_tuple_map(tmp_path, "bpe")
        counts = glyph_count_map(tmp_path, "bpe")

        assert counts == {tid: len(t) for tid, t in tuples.items()}
