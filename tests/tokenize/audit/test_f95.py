"""Tests for ``smiles_subword.tokenize.audit.f95`` (F_{p,n} confirmation)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from smiles_subword.tokenize.adapters.smirk import SmirkAdapter
from smiles_subword.tokenize.adapters.smirk_unigram import UnigramSmirkAdapter
from smiles_subword.tokenize.audit.f95 import (
    F95_GRID,
    HEADLINE_N,
    _non_atomic_token_ids,
    compute_f95,
    compute_f95_from_encoded,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
SAMPLE_CORPUS = REPO_ROOT / "tests" / "data" / "_pubchem_sample_1k.smi"

_ATOMIC = frozenset({"C", "O"})


class _FakeTokenizer:
    """Minimal Tokenizer-protocol fake with an explicit id->token map.

    Ids 0-2 are specials, 3-4 are atomic glyphs (``C`` / ``O``), 5-14 are ten
    learned (non-atomic) tokens ``m5``..``m14``.
    """

    name = "fake"

    def __init__(
        self,
        *,
        encodings: dict[str, list[int]] | None = None,
        bos_id: int = 0,
        eos_id: int = 1,
        pad_id: int = 2,
        unk_id: int | None = None,
        id_to_token: dict[int, str] | None = None,
        vocab_size: int = 15,
    ) -> None:
        self.vocab_size = vocab_size
        self._encodings = encodings or {}
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.unk_id = unk_id
        default = {0: "<bos>", 1: "<eos>", 2: "<pad>", 3: "C", 4: "O"}
        default.update({tid: f"m{tid}" for tid in range(5, vocab_size)})
        self._id_to_token = id_to_token if id_to_token is not None else default

    def __len__(self) -> int:
        return self.vocab_size

    def encode(self, smiles: str, *, add_special_tokens: bool = False) -> list[int]:
        return list(self._encodings.get(smiles, []))

    def encode_batch(
        self, smiles: list[str], *, add_special_tokens: bool = False
    ) -> list[list[int]]:
        return [self.encode(s) for s in smiles]

    def decode(self, ids: list[int], *, skip_special_tokens: bool = True) -> str:
        return ""

    def token_to_id(self, tok: str) -> int | None:
        return None

    def id_to_token(self, idx: int) -> str:
        return self._id_to_token.get(idx, "")

    def save(self, path: Path) -> None:
        return None

    @classmethod
    def load(cls, path: Path) -> _FakeTokenizer:
        raise NotImplementedError


def _flat_molecule(firings: dict[int, int]) -> list[int]:
    """One molecule carrying each token id ``firings[id]`` times."""
    return [tid for tid, count in firings.items() for _ in range(count)]


_GRADED_FIRINGS = {
    5: 200, 6: 200, 7: 200, 8: 200, 9: 200,
    10: 100, 11: 100,
    12: 75, 13: 75,
    14: 10,
}  # fmt: skip
"""Ten learned tokens: 9 fire >=50, 7 fire >=100, 5 fire >=200."""


class TestComputeF95Grid:
    """The full 3x3 (p, n) grid and the headline gate."""

    def test_evaluates_all_nine_grid_points(self) -> None:
        tok = _FakeTokenizer()

        result = compute_f95_from_encoded(
            tok, [_flat_molecule(_GRADED_FIRINGS)], arm="bpe", atomic_tokens=_ATOMIC
        )

        assert len(result.fp_thresholds) == 9
        assert [(t.p, t.n_min) for t in result.fp_thresholds] == list(F95_GRID)

    def test_clearance_depends_only_on_n_not_p(self) -> None:
        tok = _FakeTokenizer()

        result = compute_f95_from_encoded(
            tok, [_flat_molecule(_GRADED_FIRINGS)], arm="bpe", atomic_tokens=_ATOMIC
        )

        for n in (50, 100, 200):
            clearing = {
                t.n_merges_clearing for t in result.fp_thresholds if t.n_min == n
            }
            assert len(clearing) == 1

    def test_clearance_by_n_matches_the_firings(self) -> None:
        tok = _FakeTokenizer()

        result = compute_f95_from_encoded(
            tok, [_flat_molecule(_GRADED_FIRINGS)], arm="bpe", atomic_tokens=_ATOMIC
        )

        assert result.clearance_by_n == {50: 0.9, 100: 0.7, 200: 0.5}

    def test_headline_clearance_is_the_n100_fraction(self) -> None:
        tok = _FakeTokenizer()

        result = compute_f95_from_encoded(
            tok, [_flat_molecule(_GRADED_FIRINGS)], arm="bpe", atomic_tokens=_ATOMIC
        )

        assert result.headline_clearance == result.clearance_by_n[HEADLINE_N]


class TestFpThresholdDiagnostics:
    """``crossed`` / ``crossed_at_rank`` — the per-(p, n) diagnostic fields.

    Over ``_GRADED_FIRINGS`` the clearing fraction is 0.9 at n=50, 0.7 at
    n=100, 0.5 at n=200; ``crossed_at_rank`` is the first id-ordered rank
    falling below ``n_min`` when the ``p`` bar is not met.
    """

    def test_crossed_and_crossed_at_rank_per_grid_point(self) -> None:
        tok = _FakeTokenizer()

        result = compute_f95_from_encoded(
            tok, [_flat_molecule(_GRADED_FIRINGS)], arm="bpe", atomic_tokens=_ATOMIC
        )
        by_pn = {(t.p, t.n_min): t for t in result.fp_thresholds}

        # 0.9 clears the 0.90 bar -> crossed, no diagnostic rank.
        assert by_pn[(0.90, 50)].crossed is True
        assert by_pn[(0.90, 50)].crossed_at_rank is None
        # 0.9 misses the 0.95 bar; first rank below 50 firings is the n=10 token.
        assert by_pn[(0.95, 50)].crossed is False
        assert by_pn[(0.95, 50)].crossed_at_rank == 9
        # 0.7 misses every bar at n=100; first rank below 100 is the first n=75.
        assert by_pn[(0.95, 100)].crossed is False
        assert by_pn[(0.95, 100)].crossed_at_rank == 7
        # 0.5 misses at n=200; first rank below 200 is the first n=100.
        assert by_pn[(0.99, 200)].crossed is False
        assert by_pn[(0.99, 200)].crossed_at_rank == 5


class TestTrainingCountsByID:
    """Per-piece training counts deposited for the vocabulary-characterization Zipf."""

    def test_holds_non_atomic_firings(self) -> None:
        tok = _FakeTokenizer()

        result = compute_f95_from_encoded(
            tok, [_flat_molecule(_GRADED_FIRINGS)], arm="bpe", atomic_tokens=_ATOMIC
        )

        assert result.training_counts_by_id == _GRADED_FIRINGS

    def test_excludes_atomic_and_special_ids(self) -> None:
        tok = _FakeTokenizer()
        firings = {**_GRADED_FIRINGS, 3: 999, 0: 5}

        result = compute_f95_from_encoded(
            tok, [_flat_molecule(firings)], arm="bpe", atomic_tokens=_ATOMIC
        )

        assert 3 not in result.training_counts_by_id
        assert 0 not in result.training_counts_by_id


class TestEmbeddingTailFlag:
    """The ``embedding-tail-unsafe`` flag is F_{0.95,100} failing."""

    def test_unsafe_when_headline_clearance_below_95_percent(self) -> None:
        tok = _FakeTokenizer()

        result = compute_f95_from_encoded(
            tok, [_flat_molecule(_GRADED_FIRINGS)], arm="bpe", atomic_tokens=_ATOMIC
        )

        assert result.headline_clearance == 0.7
        assert result.embedding_tail_unsafe is True

    def test_safe_when_every_learned_token_clears(self) -> None:
        tok = _FakeTokenizer()
        firings = dict.fromkeys(range(5, 15), 200)

        result = compute_f95_from_encoded(
            tok, [_flat_molecule(firings)], arm="bpe", atomic_tokens=_ATOMIC
        )

        assert result.headline_clearance == 1.0
        assert result.embedding_tail_unsafe is False


class TestVocabularyExclusions:
    """Atomic glyphs and specials never enter the metric."""

    def test_atomic_and_special_tokens_are_excluded(self) -> None:
        tok = _FakeTokenizer()
        firings = {**_GRADED_FIRINGS, 0: 999, 1: 999, 2: 999, 3: 999, 4: 999}

        result = compute_f95_from_encoded(
            tok, [_flat_molecule(firings)], arm="bpe", atomic_tokens=_ATOMIC
        )

        assert result.n_non_atomic == 10
        assert result.clearance_by_n == {50: 0.9, 100: 0.7, 200: 0.5}

    def test_non_atomic_ids_excludes_atomic_and_specials(self) -> None:
        tok = _FakeTokenizer()

        ids = _non_atomic_token_ids(tok, _ATOMIC)

        assert ids == list(range(5, 15))

    def test_non_atomic_ids_skips_empty_token_strings(self) -> None:
        tok = _FakeTokenizer(
            id_to_token={3: "C", 5: "m5", 6: ""}, vocab_size=7, unk_id=None
        )

        ids = _non_atomic_token_ids(tok, frozenset({"C"}))

        assert ids == [5]


class TestCorpusCounts:
    """Molecule and token tallies."""

    def test_counts_molecules_and_tokens(self) -> None:
        tok = _FakeTokenizer()
        encoded = [[5, 6], [7], [8, 9, 10]]

        result = compute_f95_from_encoded(
            tok, encoded, arm="bpe", atomic_tokens=_ATOMIC
        )

        assert result.n_corpus_molecules == 3
        assert result.n_corpus_tokens == 6


class TestComputeF95Raises:
    """Boundary and interface failures."""

    def test_empty_corpus_raises(self) -> None:
        tok = _FakeTokenizer()

        with pytest.raises(ValueError, match=r"zero molecules"):
            compute_f95_from_encoded(tok, [], arm="bpe", atomic_tokens=_ATOMIC)

    def test_no_non_atomic_vocabulary_raises(self) -> None:
        tok = _FakeTokenizer()
        every_token = frozenset(tok.id_to_token(i) for i in range(len(tok)))

        with pytest.raises(ValueError, match=r"no non-atomic vocabulary"):
            compute_f95_from_encoded(
                tok, [[5, 6]], arm="bpe", atomic_tokens=every_token
            )

    def test_unknown_arm_raises(self) -> None:
        tok = _FakeTokenizer()

        with pytest.raises(ValueError, match=r"arm must be"):
            compute_f95_from_encoded(tok, [[5]], arm="selfies", atomic_tokens=_ATOMIC)

    def test_invalid_p_propagates_from_evaluate_fp(self) -> None:
        tok = _FakeTokenizer()

        with pytest.raises(ValueError, match=r"p must be in"):
            compute_f95_from_encoded(
                tok, [[5]], arm="bpe", atomic_tokens=_ATOMIC, fp_thresholds=((1.5, 50),)
            )


class TestComputeF95SmilesPath:
    """``compute_f95`` (SMILES iterator) matches the pre-encoded path."""

    def test_matches_compute_f95_from_encoded(self) -> None:
        encodings = {"a": [5, 5, 6], "b": [7, 8, 9, 14], "c": [10, 11]}
        tok = _FakeTokenizer(encodings=encodings)
        smis = ["a", "b", "c", "a"]

        from_smiles = compute_f95(tok, smis, arm="bpe", atomic_tokens=_ATOMIC)
        from_encoded = compute_f95_from_encoded(
            tok, [tok.encode(s) for s in smis], arm="bpe", atomic_tokens=_ATOMIC
        )

        assert from_smiles == from_encoded


class TestF95ResultSerialization:
    """``as_dict`` is JSON-ready."""

    def test_as_dict_round_trips_through_json(self) -> None:
        tok = _FakeTokenizer()
        result = compute_f95_from_encoded(
            tok, [_flat_molecule(_GRADED_FIRINGS)], arm="bpe", atomic_tokens=_ATOMIC
        )

        payload = json.loads(json.dumps(result.as_dict()))

        assert payload["clearance_by_n"] == {"50": 0.9, "100": 0.7, "200": 0.5}
        assert payload["embedding_tail_unsafe"] is True
        assert len(payload["fp_thresholds"]) == 9


@pytest.fixture(scope="module")
def tiny_tokenizers() -> tuple[SmirkAdapter, UnigramSmirkAdapter, list[str]]:
    corpus = str(SAMPLE_CORPUS)
    gpe = SmirkAdapter.train_gpe([corpus], name="t_gpe_f95", vocab_size=400)
    uni = UnigramSmirkAdapter.train_unigram([corpus], name="t_uni_f95", vocab_size=400)
    smiles = [ln.strip() for ln in SAMPLE_CORPUS.read_text().splitlines() if ln.strip()]
    return gpe, uni, smiles


class TestComputeF95RealTokenizers:
    """End-to-end against freshly trained Smirk-GPE and Unigram tokenizers."""

    def test_bpe_arm_confirms(
        self, tiny_tokenizers: tuple[SmirkAdapter, UnigramSmirkAdapter, list[str]]
    ) -> None:
        gpe, _uni, smiles = tiny_tokenizers
        atomic = frozenset(gpe.hf_tokenizer.get_vocab())

        result = compute_f95(gpe, smiles, arm="bpe", atomic_tokens=atomic)

        assert result.arm == "bpe"
        assert len(result.fp_thresholds) == 9
        assert 0 < result.n_non_atomic <= len(gpe)
        assert 0.0 <= result.headline_clearance <= 1.0

    def test_unigram_arm_confirms_with_matched_alphabet(
        self, tiny_tokenizers: tuple[SmirkAdapter, UnigramSmirkAdapter, list[str]]
    ) -> None:
        gpe, uni, smiles = tiny_tokenizers
        atomic = frozenset(gpe.hf_tokenizer.get_vocab())

        result = compute_f95(uni, smiles, arm="unigram", atomic_tokens=atomic)

        assert result.arm == "unigram"
        assert 0 < result.n_non_atomic <= len(uni)
        assert 0.0 <= result.headline_clearance <= 1.0
