"""Encode-batch contract + the shared ``_batched`` chunking helper.

Pins the contract that :meth:`Tokenizer.encode_batch` returns id-lists
byte-identical to a list comprehension over :meth:`Tokenizer.encode` per
element, and that :func:`iter_encoded_batches` preserves per-SMILES
correspondence across chunk boundaries. A pure-Python fake stands in for the
trivial list-comprehension fallback so the chunking helpers are exercised
with no artifact on disk. (The real smirk rust batch path is covered by
``test_smirk_adapter.py``, which trains synthetically and always runs.)
"""

from __future__ import annotations

import pytest

_PROBE_SMILES = (
    "CCO",
    "c1ccccc1",
    "CC(=O)O",
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "OC[C@@H](O)[C@H](O)[C@H](O)CO",
    "[Cl].CC(C)NCC(O)COc1cccc2ccccc12",
    "C(C(=O)O)N",
    "c1ccncc1",
)


class _FakeTokenizer:
    """Minimal pure-Python ``Tokenizer`` whose ``encode_batch`` is the
    trivial per-element fallback, used to exercise the shared
    ``_batched`` helpers independently of any rust path."""

    name = "fake"
    bos_id = 0
    eos_id = 1
    pad_id = 2
    unk_id: int | None = None
    vocab_size = 128

    def encode(self, smiles: str, *, add_special_tokens: bool = False) -> list[int]:
        ids = [3 + (ord(c) % 64) for c in smiles]
        if add_special_tokens:
            return [self.bos_id, *ids, self.eos_id]
        return ids

    def encode_batch(
        self, smiles: list[str], *, add_special_tokens: bool = False
    ) -> list[list[int]]:
        return [self.encode(s, add_special_tokens=add_special_tokens) for s in smiles]


@pytest.fixture
def fake_tokenizer() -> _FakeTokenizer:
    return _FakeTokenizer()


class TestFakeEncodeBatchMatchesPerElement:
    """Pure-Python list-comprehension fallback matches the per-element path."""

    def test_matches_for_canonical_probe_set(
        self, fake_tokenizer: _FakeTokenizer
    ) -> None:
        single = [fake_tokenizer.encode(s) for s in _PROBE_SMILES]

        batched = fake_tokenizer.encode_batch(list(_PROBE_SMILES))

        assert batched == single

    def test_add_special_tokens_propagates_per_element(
        self, fake_tokenizer: _FakeTokenizer
    ) -> None:
        single = [
            fake_tokenizer.encode(s, add_special_tokens=True) for s in _PROBE_SMILES
        ]

        batched = fake_tokenizer.encode_batch(
            list(_PROBE_SMILES), add_special_tokens=True
        )

        assert batched == single

    def test_empty_list_returns_empty_list(
        self, fake_tokenizer: _FakeTokenizer
    ) -> None:
        assert fake_tokenizer.encode_batch([]) == []


class TestIterEncodedBatchesMatchesPerElement:
    """The chunked helper preserves per-SMILES correspondence."""

    def test_ids_match_per_element_at_small_batch_size(
        self, fake_tokenizer: _FakeTokenizer
    ) -> None:
        from smiles_subword.tokenize._batched import iter_encoded_batches

        expected = [fake_tokenizer.encode(s) for s in _PROBE_SMILES]

        actual = list(
            iter_encoded_batches(fake_tokenizer, iter(_PROBE_SMILES), batch_size=3)
        )

        assert actual == expected

    def test_rejects_zero_batch_size(self, fake_tokenizer: _FakeTokenizer) -> None:
        from smiles_subword.tokenize._batched import iter_encoded_batches

        with pytest.raises(ValueError, match=r"batch_size must be"):
            list(
                iter_encoded_batches(fake_tokenizer, iter(_PROBE_SMILES), batch_size=0)
            )
