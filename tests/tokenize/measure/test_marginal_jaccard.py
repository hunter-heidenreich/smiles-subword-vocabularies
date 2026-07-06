"""Unit tests for the marginal cross-arm Jaccard pure computation."""

from __future__ import annotations

import math

from smiles_subword.tokenize.measure.supplementary.marginal_jaccard import (
    build_step,
    fresh_pieces,
)

# Glyph-tuples standing in for multi-glyph pieces; identity is the exact sequence.
_CC = ("C", "C")
_CCC = ("C", "C", "C")
_CCO = ("C", "C", "O")
_NCC = ("N", "C", "C")


def test_fresh_pieces_is_set_difference() -> None:
    lower = frozenset({_CC})
    upper = frozenset({_CC, _CCC, _CCO})
    assert fresh_pieces(lower, upper) == frozenset({_CCC, _CCO})


def test_fresh_pieces_handles_non_nesting() -> None:
    # Unigram need not nest: a piece in lower can be absent from upper.
    lower = frozenset({_CC, _NCC})
    upper = frozenset({_CC, _CCC})
    assert fresh_pieces(lower, upper) == frozenset({_CCC})


def test_build_step_counts_and_jaccard() -> None:
    # BPE adds {CCC, CCO}; Unigram adds {CCC, NCC}; shared fresh = {CCC}.
    step = build_step(
        corpus="pubchem",
        boundary="nmb",
        v_lower=256,
        v_upper=512,
        bpe_lower=frozenset({_CC}),
        bpe_upper=frozenset({_CC, _CCC, _CCO}),
        unigram_lower=frozenset({_CC}),
        unigram_upper=frozenset({_CC, _CCC, _NCC}),
    )
    assert step.n_fresh_bpe == 2
    assert step.n_fresh_unigram == 2
    assert step.n_fresh_shared == 1
    # union of fresh sets = {CCC, CCO, NCC} = 3, intersection = 1
    assert step.marginal_jaccard == 1 / 3
    assert step.v_lower == 256
    assert step.v_upper == 512


def test_build_step_nan_when_no_fresh_pieces() -> None:
    same = frozenset({_CC, _CCC})
    step = build_step(
        corpus="zinc22",
        boundary="mb",
        v_lower=512,
        v_upper=1024,
        bpe_lower=same,
        bpe_upper=same,
        unigram_lower=same,
        unigram_upper=same,
    )
    assert step.n_fresh_bpe == 0
    assert step.n_fresh_unigram == 0
    assert math.isnan(step.marginal_jaccard)


def test_marginal_step_as_dict_roundtrip() -> None:
    step = build_step(
        corpus="coconut",
        boundary="nmb",
        v_lower=256,
        v_upper=512,
        bpe_lower=frozenset(),
        bpe_upper=frozenset({_CCC}),
        unigram_lower=frozenset(),
        unigram_upper=frozenset({_CCO}),
    )
    d = step.as_dict()
    assert d["corpus"] == "coconut"
    assert d["n_fresh_shared"] == 0
    assert d["marginal_jaccard"] == 0.0
