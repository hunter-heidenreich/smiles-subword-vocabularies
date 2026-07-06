"""Tests for per-pair vocabulary characterization primitives."""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING

from smiles_subword.tokenize.measure.jaccard import jaccard
from smiles_subword.tokenize.measure.supplementary import vocab_characterization as vc

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

CC = ("C", "C")
CCC = ("C", "C", "C")
ccc = ("c", "c", "c")
NS = ("N", "S")


def test_partition_splits_into_shared_and_unique() -> None:
    part = vc.partition(frozenset({CC, CCC, NS}), frozenset({CC, ccc}))
    assert part.shared == frozenset({CC})
    assert part.bpe_only == frozenset({CCC, NS})
    assert part.unigram_only == frozenset({ccc})


def test_partition_jaccard_matches_jaccard() -> None:
    a, b = frozenset({CC, CCC, NS}), frozenset({CC, ccc})
    assert vc.partition(a, b).jaccard == jaccard(a, b)


def test_partition_jaccard_is_one_when_identical() -> None:
    a = frozenset({CC, CCC})
    assert vc.partition(a, a).jaccard == 1.0


def test_partition_jaccard_is_zero_when_disjoint() -> None:
    assert vc.partition(frozenset({CC}), frozenset({ccc})).jaccard == 0.0


def test_partition_jaccard_is_nan_when_both_empty() -> None:
    assert math.isnan(vc.partition(frozenset(), frozenset()).jaccard)


def test_length_profile_summarizes_distribution() -> None:
    prof = vc.length_profile(frozenset({CC, CCC, ccc, NS}))
    assert prof.histogram == {2: 2, 3: 2}
    assert prof.mean == 2.5
    assert prof.median == 2.5
    assert prof.max_length == 3
    assert prof.n_pieces == 4


def test_length_profile_is_empty_when_no_pieces() -> None:
    prof = vc.length_profile(frozenset())
    assert prof.histogram == {}
    assert prof.max_length == 0
    assert prof.n_pieces == 0


def test_rank_frequency_orders_by_descending_count() -> None:
    ranked = vc.rank_frequency({CC: 10, CCC: 30, NS: 20})
    assert [piece for _, piece, _ in ranked] == [CCC, NS, CC]
    assert [rank for rank, _, _ in ranked] == [1, 2, 3]


def test_rank_frequency_breaks_ties_by_piece() -> None:
    ranked = vc.rank_frequency({CCC: 5, CC: 5})
    assert [piece for _, piece, _ in ranked] == [CC, CCC]


def test_rank_frequency_keeps_zero_count_pieces_in_tail() -> None:
    ranked = vc.rank_frequency({CC: 4, NS: 0})
    assert ranked[-1] == (2, NS, 0)


def test_characterize_pair_payload_is_json_serializable() -> None:
    payload = vc.characterize_pair(
        "pubchem__v512_nmb",
        frozenset({CC, CCC}),
        frozenset({CC, ccc}),
        bpe_holdout_counts={CC: 9, CCC: 3},
    )
    dumped = json.loads(json.dumps(payload))
    assert dumped["partition"]["shared"] == ["CC"]
    assert dumped["partition"]["bpe_only"] == ["CCC"]
    assert dumped["partition"]["unigram_only"] == ["ccc"]
    assert dumped["rank_frequency"]["bpe_holdout"][0] == {
        "rank": 1,
        "piece": "CC",
        "freq": 9,
    }
    assert dumped["rank_frequency"]["bpe_train"] is None


def test_write_vocab_characterization_round_trips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(vc, "VOCAB_CHAR_DIR", tmp_path)
    payload = vc.characterize_pair(
        "coconut__v256_mb", frozenset({CC}), frozenset({ccc})
    )
    path = vc.write_vocab_characterization(payload)
    assert json.loads(path.read_text()) == payload
