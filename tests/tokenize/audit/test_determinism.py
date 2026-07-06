"""Tests for ``smiles_subword.tokenize.audit.determinism`` (compute layer)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from smiles_subword.tokenize.adapters.smirk import SmirkAdapter
from smiles_subword.tokenize.adapters.smirk_unigram import UnigramSmirkAdapter
from smiles_subword.tokenize.audit.determinism import (
    ArtifactDigest,
    DeterminismResult,
    compare_artifacts,
    digest_artifact,
    unigram_glyph_set,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
SAMPLE_CORPUS = REPO_ROOT / "tests" / "data" / "_pubchem_sample_1k.smi"


def _write_bpe(
    art_dir: Path, *, tokenizer_json: str = "{}", merges: str = "a b\n"
) -> None:
    """Write the two files the BPE digest reads — raw content, need not parse."""
    art_dir.mkdir(parents=True, exist_ok=True)
    (art_dir / "tokenizer.json").write_text(tokenizer_json)
    (art_dir / "merges.txt").write_text(merges)


def _write_unigram(art_dir: Path, pieces: list[tuple[list[str], float]]) -> None:
    """Write a minimal Unigram ``tokenizer.json`` with the given vocab pieces."""
    art_dir.mkdir(parents=True, exist_ok=True)
    model = {
        "type": "Unigram",
        "vocab": [{"glyphs": glyphs, "score": score} for glyphs, score in pieces],
    }
    (art_dir / "tokenizer.json").write_text(json.dumps({"model": model}))


_UNI_PIECES: list[tuple[list[str], float]] = [
    (["C"], -1.0),
    (["O"], -2.0),
    (["N"], -3.0),
]


class TestDigestArtifact:
    """``digest_artifact`` fingerprints one artifact directory per arm."""

    def test_unknown_algo_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match=r"algo must be"):
            digest_artifact(tmp_path, algo="selfies")

    def test_bpe_digests_the_two_artifact_files(self, tmp_path: Path) -> None:
        _write_bpe(tmp_path)

        digest = digest_artifact(tmp_path, algo="bpe")

        assert digest.algo == "bpe"
        assert digest.tokenizer_json_sha
        assert digest.merges_txt_sha
        assert digest.piece_set_sha is None
        assert digest.vocab_order_sha is None

    def test_unigram_digests_the_glyph_vocab(self, tmp_path: Path) -> None:
        _write_unigram(tmp_path, _UNI_PIECES)

        digest = digest_artifact(tmp_path, algo="unigram")

        assert digest.algo == "unigram"
        assert digest.merges_txt_sha is None
        assert digest.piece_set_sha
        assert digest.vocab_order_sha
        assert digest.log_probs_sha

    def test_unigram_digest_rejects_a_non_unigram_model(self, tmp_path: Path) -> None:
        (tmp_path / "tokenizer.json").write_text(json.dumps({"model": {"type": "BPE"}}))

        with pytest.raises(ValueError, match=r"is not Unigram"):
            digest_artifact(tmp_path, algo="unigram")

    def test_identical_bpe_content_yields_an_identical_digest(
        self, tmp_path: Path
    ) -> None:
        _write_bpe(tmp_path / "a")
        _write_bpe(tmp_path / "b")

        assert digest_artifact(tmp_path / "a", algo="bpe") == digest_artifact(
            tmp_path / "b", algo="bpe"
        )

    def test_a_one_line_merges_change_shifts_the_digest(self, tmp_path: Path) -> None:
        _write_bpe(tmp_path / "a", merges="a b\n")
        _write_bpe(tmp_path / "b", merges="a b\nc d\n")

        digest_a = digest_artifact(tmp_path / "a", algo="bpe")
        digest_b = digest_artifact(tmp_path / "b", algo="bpe")

        assert digest_a.tokenizer_json_sha == digest_b.tokenizer_json_sha
        assert digest_a.merges_txt_sha != digest_b.merges_txt_sha


class TestUnigramGlyphSet:
    def test_returns_the_piece_set_as_glyph_tuples(self, tmp_path: Path) -> None:
        _write_unigram(tmp_path, _UNI_PIECES)

        pieces = unigram_glyph_set(tmp_path)

        assert pieces == frozenset({("C",), ("O",), ("N",)})


class TestCompareArtifactsBpe:
    """BPE: deterministic iff both artifact files are byte-identical."""

    def test_identical_artifacts_are_deterministic(self, tmp_path: Path) -> None:
        _write_bpe(tmp_path / "a")
        _write_bpe(tmp_path / "b")
        digest_a = digest_artifact(tmp_path / "a", algo="bpe")
        digest_b = digest_artifact(tmp_path / "b", algo="bpe")

        result = compare_artifacts(digest_a, digest_b)

        assert result.deterministic is True
        assert result.mismatch_kind is None
        assert result.rerun_spread == 0

    def test_a_tokenizer_json_diff_is_a_bpe_byte_mismatch(self, tmp_path: Path) -> None:
        _write_bpe(tmp_path / "a", tokenizer_json='{"v": 1}')
        _write_bpe(tmp_path / "b", tokenizer_json='{"v": 2}')

        result = compare_artifacts(
            digest_artifact(tmp_path / "a", algo="bpe"),
            digest_artifact(tmp_path / "b", algo="bpe"),
        )

        assert result.deterministic is False
        assert result.mismatch_kind == "bpe_byte"

    def test_a_merges_diff_is_a_bpe_byte_mismatch(self, tmp_path: Path) -> None:
        _write_bpe(tmp_path / "a", merges="a b\n")
        _write_bpe(tmp_path / "b", merges="c d\n")

        result = compare_artifacts(
            digest_artifact(tmp_path / "a", algo="bpe"),
            digest_artifact(tmp_path / "b", algo="bpe"),
        )

        assert result.deterministic is False
        assert result.mismatch_kind == "bpe_byte"


class TestCompareArtifactsUnigram:
    """Unigram: deterministic iff the piece set is identical."""

    def test_identical_piece_sets_are_deterministic(self, tmp_path: Path) -> None:
        _write_unigram(tmp_path / "a", _UNI_PIECES)
        _write_unigram(tmp_path / "b", _UNI_PIECES)

        result = compare_artifacts(
            digest_artifact(tmp_path / "a", algo="unigram"),
            digest_artifact(tmp_path / "b", algo="unigram"),
            pieces=(
                unigram_glyph_set(tmp_path / "a"),
                unigram_glyph_set(tmp_path / "b"),
            ),
        )

        assert result.deterministic is True
        assert result.rerun_spread == 0

    def test_one_swapped_piece_is_a_rerun_spread_of_two(self, tmp_path: Path) -> None:
        swapped = [(["C"], -1.0), (["O"], -2.0), (["S"], -3.0)]
        _write_unigram(tmp_path / "a", _UNI_PIECES)
        _write_unigram(tmp_path / "b", swapped)

        result = compare_artifacts(
            digest_artifact(tmp_path / "a", algo="unigram"),
            digest_artifact(tmp_path / "b", algo="unigram"),
            pieces=(
                unigram_glyph_set(tmp_path / "a"),
                unigram_glyph_set(tmp_path / "b"),
            ),
        )

        assert result.deterministic is False
        assert result.mismatch_kind == "unigram_piece_set"
        assert result.rerun_spread == 2

    def test_unigram_comparison_without_pieces_raises(self, tmp_path: Path) -> None:
        _write_unigram(tmp_path, _UNI_PIECES)
        digest = digest_artifact(tmp_path, algo="unigram")

        with pytest.raises(ValueError, match=r"requires the canonical/rerun piece"):
            compare_artifacts(digest, digest)


class TestCompareArtifactsGuards:
    def test_comparing_two_different_arms_raises(self, tmp_path: Path) -> None:
        _write_bpe(tmp_path / "bpe")
        _write_unigram(tmp_path / "uni", _UNI_PIECES)

        with pytest.raises(ValueError, match=r"cannot compare"):
            compare_artifacts(
                digest_artifact(tmp_path / "bpe", algo="bpe"),
                digest_artifact(tmp_path / "uni", algo="unigram"),
            )


class TestDeterminismResultSerialization:
    def test_as_dict_round_trips_through_json(self, tmp_path: Path) -> None:
        _write_bpe(tmp_path / "a")
        _write_bpe(tmp_path / "b", merges="x y\n")
        result = compare_artifacts(
            digest_artifact(tmp_path / "a", algo="bpe"),
            digest_artifact(tmp_path / "b", algo="bpe"),
        )

        payload = json.loads(json.dumps(result.as_dict()))

        assert payload["deterministic"] is False
        assert payload["mismatch_kind"] == "bpe_byte"
        assert "tokenizer_json_sha" in payload["canonical_digest"]
        assert "tokenizer_json_sha" in payload["rerun_digest"]


class TestRealBpeDeterminism:
    """The load-bearing assertion: a real BPE retrain is byte-identical."""

    def test_two_gpe_trains_produce_a_byte_identical_artifact(
        self, tmp_path: Path
    ) -> None:
        corpus = str(SAMPLE_CORPUS)
        first = SmirkAdapter.train_gpe([corpus], name="t_gpe_det", vocab_size=300)
        second = SmirkAdapter.train_gpe([corpus], name="t_gpe_det", vocab_size=300)
        first.save(tmp_path / "a")
        second.save(tmp_path / "b")

        result = compare_artifacts(
            digest_artifact(tmp_path / "a", algo="bpe"),
            digest_artifact(tmp_path / "b", algo="bpe"),
        )

        assert result.deterministic is True


class TestRealUnigramDeterminism:
    """The Unigram counterpart: a real retrain keeps an identical piece set.

    BPE is byte-deterministic (``TestRealBpeDeterminism``); Unigram is not —
    only its *piece set* is the load-bearing invariant, and only in the
    stable regime (small ``V``, NMB), away from the known V=1024 NMB jitter
    cell. This pins that the piece set is reproducible there, the claim the
    synthetic-JSON ``compare_artifacts`` tests above can only assume.
    """

    def test_two_unigram_trains_produce_an_identical_piece_set(
        self, tmp_path: Path
    ) -> None:
        corpus = str(SAMPLE_CORPUS)
        first = UnigramSmirkAdapter.train_unigram(
            [corpus], name="t_uni_det", vocab_size=400
        )
        second = UnigramSmirkAdapter.train_unigram(
            [corpus], name="t_uni_det", vocab_size=400
        )
        first.save(tmp_path / "a")
        second.save(tmp_path / "b")

        result = compare_artifacts(
            digest_artifact(tmp_path / "a", algo="unigram"),
            digest_artifact(tmp_path / "b", algo="unigram"),
            pieces=(
                unigram_glyph_set(tmp_path / "a"),
                unigram_glyph_set(tmp_path / "b"),
            ),
        )

        assert result.deterministic is True
        assert result.mismatch_kind is None
        assert result.rerun_spread == 0


def _bpe_digest() -> ArtifactDigest:
    return ArtifactDigest(
        algo="bpe",
        tokenizer_json_sha="t",
        merges_txt_sha="m",
        piece_set_sha=None,
        vocab_order_sha=None,
        log_probs_sha=None,
    )


class TestDeterminismResultShape:
    def test_result_carries_both_digests(self) -> None:
        digest = _bpe_digest()

        result = DeterminismResult(
            arm="bpe",
            deterministic=True,
            mismatch_kind=None,
            rerun_spread=0,
            canonical=digest,
            rerun=digest,
        )

        assert result.canonical is digest
        assert result.rerun is digest
