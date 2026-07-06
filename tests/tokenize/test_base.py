"""Tests for ``smiles_subword.tokenize.base`` (Tokenizer-protocol helpers)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

from smiles_subword.tokenize.base import Tokenizer, collect_special_ids


def _tok(
    *, bos_id: int = 0, eos_id: int = 1, pad_id: int = 2, unk_id: int | None = None
) -> Tokenizer:
    """A stand-in exposing only the special-id members the helper reads."""
    return cast(
        "Tokenizer",
        SimpleNamespace(bos_id=bos_id, eos_id=eos_id, pad_id=pad_id, unk_id=unk_id),
    )


class TestCollectSpecialIds:
    """Specials dedup; ``unk_id=None`` is dropped."""

    def test_all_four_specials_when_unk_set(self) -> None:
        assert collect_special_ids(_tok(unk_id=4)) == frozenset({0, 1, 2, 4})

    def test_unk_none_excluded(self) -> None:
        assert collect_special_ids(_tok(unk_id=None)) == frozenset({0, 1, 2})

    def test_repeats_dedup(self) -> None:
        tok = _tok(bos_id=0, eos_id=0, pad_id=0, unk_id=None)

        assert collect_special_ids(tok) == frozenset({0})
