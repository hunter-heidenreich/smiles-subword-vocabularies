"""Tests for ``smiles_subword.tokenize.measure.jaccard.inventory`` (chunk inventory)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from smiles_subword.tokenize.adapters.smirk import SmirkAdapter
from smiles_subword.tokenize.measure.jaccard import inventory
from smiles_subword.tokenize.measure.jaccard.inventory import (
    ChunkInventory,
    build_chunk_inventory,
    classify_subwords,
    get_or_build_inventory,
    read_inventory,
    write_inventory,
)

if TYPE_CHECKING:
    from pathlib import Path


class _FakeAdapter:
    """Minimal adapter exposing the two surfaces the inventory code uses."""

    def __init__(
        self,
        pretok: dict[str, list[tuple[str, tuple[int, int]]]],
        enc: dict[str, list[int]],
    ) -> None:
        self._pretok = pretok
        self._enc = enc

    def pretokenize_layer_b(self, smi: str) -> list[tuple[str, tuple[int, int]]]:
        return self._pretok[smi]

    def encode_batch(
        self, batch: list[str], *, add_special_tokens: bool = False
    ) -> list[list[int]]:
        return [self._enc[s] for s in batch]


class TestBuildChunkInventory:
    def _adapter(self) -> _FakeAdapter:
        return _FakeAdapter(
            pretok={
                "AB": [("AB", (0, 2))],
                "X[YZ]": [("X", (0, 1)), ("[YZ]", (1, 4))],
            },
            enc={},
        )

    def test_splits_bracket_from_nonbracket(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            inventory, "iter_smiles_from_parquet", lambda _d: ["AB", "X[YZ]"]
        )

        inv = build_chunk_inventory(
            self._adapter(), tmp_path, training_corpus_sha="sha-1"
        )

        assert inv.bracket_chunks == ("[YZ]",)
        assert set(inv.nonbracket_chunks) == {"AB", "X"}
        assert inv.n_molecules_scanned == 2
        assert inv.nonbracket_cap_bound is False

    def test_distinct_chunks_deduplicated(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            inventory, "iter_smiles_from_parquet", lambda _d: ["AB", "AB", "AB"]
        )

        inv = build_chunk_inventory(
            self._adapter(), tmp_path, training_corpus_sha="sha-1"
        )

        assert inv.nonbracket_chunks == ("AB",)
        assert inv.n_molecules_scanned == 3

    def test_nonbracket_cap_binds(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        adapter = _FakeAdapter(
            pretok={"P": [("A", (0, 1)), ("B", (1, 2)), ("C", (2, 3))]}, enc={}
        )
        monkeypatch.setattr(inventory, "iter_smiles_from_parquet", lambda _d: ["P"])

        inv = build_chunk_inventory(
            adapter, tmp_path, training_corpus_sha="sha-1", nonbracket_cap=2
        )

        assert len(inv.nonbracket_chunks) == 2
        assert inv.nonbracket_cap_bound is True

    def test_limit_molecules_truncates_scan(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            inventory, "iter_smiles_from_parquet", lambda _d: ["AB", "X[YZ]"]
        )

        inv = build_chunk_inventory(
            self._adapter(), tmp_path, training_corpus_sha="s", limit_molecules=1
        )

        assert inv.n_molecules_scanned == 1


class TestInventoryRoundTrip:
    def _inv(self) -> ChunkInventory:
        return ChunkInventory(
            training_corpus_sha="sha-1",
            bracket_chunks=("[YZ]",),
            nonbracket_chunks=("AB", "X"),
            n_molecules_scanned=2,
            nonbracket_cap_bound=False,
        )

    def test_write_then_read_round_trips(self, tmp_path: Path) -> None:
        path = tmp_path / "inv.json"
        write_inventory(path, self._inv())

        out = read_inventory(path, training_corpus_sha="sha-1")

        assert out == self._inv()

    def test_read_returns_none_on_sha_mismatch(self, tmp_path: Path) -> None:
        path = tmp_path / "inv.json"
        write_inventory(path, self._inv())

        assert read_inventory(path, training_corpus_sha="other-sha") is None

    def test_read_missing_file_returns_none(self, tmp_path: Path) -> None:
        assert read_inventory(tmp_path / "absent.json", training_corpus_sha="x") is None


class TestGetOrBuildInventory:
    def test_builds_and_caches_on_first_call(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        adapter = _FakeAdapter(pretok={"AB": [("AB", (0, 2))]}, enc={})
        monkeypatch.setattr(inventory, "iter_smiles_from_parquet", lambda _d: ["AB"])
        cache = tmp_path / "cache.json"

        inv = get_or_build_inventory(
            adapter, tmp_path, cache, training_corpus_sha="sha-1"
        )

        assert cache.is_file()
        assert inv.nonbracket_chunks == ("AB",)

    def test_second_call_reads_cache_without_streaming(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        adapter = _FakeAdapter(pretok={"AB": [("AB", (0, 2))]}, enc={})
        monkeypatch.setattr(inventory, "iter_smiles_from_parquet", lambda _d: ["AB"])
        cache = tmp_path / "cache.json"
        get_or_build_inventory(adapter, tmp_path, cache, training_corpus_sha="sha-1")

        def _boom(_d: object) -> list[str]:
            raise AssertionError("should not re-stream a cached inventory")

        monkeypatch.setattr(inventory, "iter_smiles_from_parquet", _boom)
        inv = get_or_build_inventory(
            adapter, tmp_path, cache, training_corpus_sha="sha-1"
        )

        assert inv.nonbracket_chunks == ("AB",)


class TestClassifySubwords:
    def test_partitions_structural_bracket_unseen(self) -> None:
        glyph_tuple_by_id = {
            1: ("A",),
            2: ("B",),
            3: ("A", "B"),
            4: ("X", "Y"),
            5: ("Z", "Z"),
        }
        multi = frozenset({("A", "B"), ("X", "Y"), ("Z", "Z")})
        adapter = _FakeAdapter(pretok={}, enc={"AB": [3], "[XY]": [4]})
        inventory = ChunkInventory(
            training_corpus_sha="s",
            bracket_chunks=("[XY]",),
            nonbracket_chunks=("AB",),
            n_molecules_scanned=1,
            nonbracket_cap_bound=False,
        )

        split = classify_subwords(adapter, glyph_tuple_by_id, multi, inventory)

        assert split.structural == frozenset({("A", "B")})
        assert split.bracket_internal == frozenset({("X", "Y")})
        assert split.unseen == frozenset({("Z", "Z")})

    def test_piece_in_both_contexts_is_structural(self) -> None:
        glyph_tuple_by_id = {1: ("A",), 2: ("B",), 3: ("A", "B")}
        multi = frozenset({("A", "B")})
        adapter = _FakeAdapter(pretok={}, enc={"AB": [3], "[AB]": [3]})
        inventory = ChunkInventory(
            training_corpus_sha="s",
            bracket_chunks=("[AB]",),
            nonbracket_chunks=("AB",),
            n_molecules_scanned=1,
            nonbracket_cap_bound=False,
        )

        split = classify_subwords(adapter, glyph_tuple_by_id, multi, inventory)

        assert split.structural == frozenset({("A", "B")})
        assert split.bracket_internal == frozenset()


class TestChunkLocality:
    """The dedup-equivalence guard: tokens never span Layer-B chunks."""

    @pytest.mark.parametrize(
        "smi",
        ["CCO", "c1cc[nH]c1", "[O-]C(=O)C", "C[C@@H](N)O", "Clc1ccccc1"],
    )
    def test_encode_molecule_equals_concat_of_chunk_encodes(self, smi: str) -> None:
        adapter = SmirkAdapter.atomic()

        whole = adapter.encode(smi)
        chunks = [c for c, _span in adapter.pretokenize_layer_b(smi)]
        concat = [t for c in chunks for t in adapter.encode(c)]

        assert whole == concat
