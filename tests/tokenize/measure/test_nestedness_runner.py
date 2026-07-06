"""Tests for ``nestedness_runner`` (dual encode-and-compare plumbing)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from smiles_subword.tokenize.adapters.smirk import SmirkAdapter
from smiles_subword.tokenize.measure.nestedness.runner import compare_held_out

if TYPE_CHECKING:
    from smiles_subword.tokenize.measure.nestedness import GlyphTuple

_FIXTURE_SMILES: tuple[str, ...] = (
    "CCO",
    "CC(=O)O",
    "c1ccccc1",
    "CC(=O)Oc1ccccc1C(=O)O",
)


@pytest.fixture
def atomic_adapter() -> SmirkAdapter:
    return SmirkAdapter.atomic()


def _all_ids(adapter: SmirkAdapter, smiles: tuple[str, ...]) -> set[int]:
    return {tid for ids in adapter.encode_batch(list(smiles)) for tid in ids}


class TestCompareHeldOut:
    def test_two_atomic_arms_agree_everywhere(
        self, atomic_adapter: SmirkAdapter
    ) -> None:
        # Both arms are the same atomic tokenizer: every token is one glyph, so
        # every boundary agrees and no molecule has a conflict.
        per_mol, mismatch = compare_held_out(
            atomic_adapter,
            atomic_adapter,
            _FIXTURE_SMILES,
            {},  # default glyph count 1 per token
            {},  # default glyph tuple ("?",) per token
            batch_size=2,
        )

        assert mismatch == 0
        assert len(per_mol) == len(_FIXTURE_SMILES)
        for pm in per_mol:
            assert pm.n_conflict == 0
            assert pm.n_nest == 0
            assert pm.is_nested is True

    def test_length_mismatch_is_counted_not_raised(
        self, atomic_adapter: SmirkAdapter
    ) -> None:
        # Give the UL map a 2-glyph tuple for every emitted id while the BPE map
        # stays at the default 1 — every molecule's totals then disagree.
        ul_tuples: dict[int, GlyphTuple] = dict.fromkeys(
            _all_ids(atomic_adapter, _FIXTURE_SMILES), ("C", "C")
        )

        per_mol, mismatch = compare_held_out(
            atomic_adapter, atomic_adapter, _FIXTURE_SMILES, {}, ul_tuples
        )

        assert per_mol == []
        assert mismatch == len(_FIXTURE_SMILES)

    def test_empty_stream_yields_no_records(self, atomic_adapter: SmirkAdapter) -> None:
        per_mol, mismatch = compare_held_out(atomic_adapter, atomic_adapter, [], {}, {})

        assert per_mol == []
        assert mismatch == 0
