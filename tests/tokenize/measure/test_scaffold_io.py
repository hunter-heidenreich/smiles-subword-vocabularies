"""Tests for ``smiles_subword.tokenize.measure.scaffold.io`` (deposition + aggregator).

Validate write / read round-trip, freshness checks tied to corpus +
scaffold-log SHAs, deposit dispatch, and aggregator behavior. The
runtime scaffold-log reading is exercised in ``test_scaffold_runner.py``;
these tests stub it via monkeypatch so the IO contract is testable in
isolation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from smiles_subword.tokenize.measure._pairing import MatchedPair, PairKey, UnpairedCell
from smiles_subword.tokenize.measure.scaffold import (
    ArmScaffold,
    MatchedPairScaffold,
    UnpairedScaffold,
    compute_matched_pair_scaffold,
    compute_unigram_arm_scaffold,
    compute_unpaired_scaffold,
    empty_surface_breakdown,
)
from smiles_subword.tokenize.measure.scaffold import io as scaffold_io
from smiles_subword.tokenize.measure.scaffold import runner as scaffold_runner


def _bpe_arm(
    *,
    cell_id: str = "pubchem__smirk_gpe_v256_nmb",
    boundary: str = "nmb",
    scaffold_count: int = 20,
    vocab_size: int = 256,
    sha: str = "corpus-A",
    log_sha: str = "log-A",
) -> ArmScaffold:
    return ArmScaffold(
        cell_id=cell_id,
        arm="bpe",
        boundary=boundary,  # type: ignore[arg-type]
        vocab_size=vocab_size,
        n_merges=vocab_size - 159,
        scaffold_count=scaffold_count,
        scaffold_fraction_of_v=scaffold_count / vocab_size,
        surface_form_breakdown=empty_surface_breakdown(),
        threshold=3,
        verified_by_construction=False,
        training_corpus_sha=sha,
        scaffold_log_sha=log_sha,
    )


def _ul_arm(
    *,
    cell_id: str = "pubchem__smirk_unigram_v256_nmb",
    boundary: str = "nmb",
    vocab_size: int = 256,
    sha: str = "corpus-B",
) -> ArmScaffold:
    return compute_unigram_arm_scaffold(
        cell_id=cell_id,
        boundary=boundary,  # type: ignore[arg-type]
        vocab_size=vocab_size,
        training_corpus_sha=sha,
    )


def _matched_record(
    pair_key: str = "pubchem__v256_nmb",
    *,
    bpe_sha: str = "corpus-A",
    bpe_log_sha: str = "log-A",
    ul_sha: str = "corpus-B",
) -> MatchedPairScaffold:
    return compute_matched_pair_scaffold(
        _bpe_arm(sha=bpe_sha, log_sha=bpe_log_sha),
        _ul_arm(sha=ul_sha),
        pair_key=pair_key,
        tier="headline",
        corpus="pubchem",
        vocab_size=256,
        boundary="nmb",
    )


def _unpaired_record() -> UnpairedScaffold:
    arm = _bpe_arm(
        cell_id="zinc22__smirk_gpe_v2048_nmb",
        vocab_size=2048,
        scaffold_count=50,
        sha="corpus-C",
        log_sha="log-C",
    )
    return compute_unpaired_scaffold(
        arm,
        pair_key="zinc22__v2048_nmb",
        tier="conditional",
        corpus="zinc22",
        vocab_size=2048,
        boundary="nmb",
        extras_kind=None,
        extras_label=None,
        missing_arm="unigram",
        unpaired_reason="conditional_negative_branch",
    )


def _write_cell(
    artifacts_root: Path,
    *,
    corpus: str,
    name: str,
    base_kind: str,
    vocab_size: int,
    n_merges: int | None,
    sha: str,
    merge_brackets: bool = False,
    scaffold_jsonl: str | None = None,
) -> Path:
    cell_dir = artifacts_root / corpus / name
    cell_dir.mkdir(parents=True, exist_ok=True)
    meta: dict[str, Any] = {
        "name": name,
        "base_kind": base_kind,
        "vocab_size": vocab_size,
        "training_corpus_sha": sha,
        "merge_brackets": merge_brackets,
        "split_structure": True,
    }
    if n_merges is not None:
        meta["n_merges"] = n_merges
    (cell_dir / "meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
    if scaffold_jsonl is not None:
        (cell_dir / "scaffold.jsonl").write_text(scaffold_jsonl)
    return cell_dir


@pytest.fixture(autouse=True)
def _redirect_data_dirs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(scaffold_io, "SCAFFOLD_DATA_DIR", tmp_path / "scaffold_root")
    monkeypatch.setattr(
        scaffold_io, "SCAFFOLD_CELL_DIR", tmp_path / "scaffold_root" / "scaffold"
    )
    monkeypatch.setattr(
        scaffold_io,
        "SCAFFOLD_TABLE_JSON",
        tmp_path / "scaffold_root" / "scaffold_table.json",
    )
    monkeypatch.setattr(
        scaffold_io,
        "SCAFFOLD_TABLE_MD",
        tmp_path / "scaffold_root" / "scaffold_table.md",
    )


class TestJsonPath:
    def test_path_is_under_the_scaffold_cell_dir(self, tmp_path: Path) -> None:
        assert scaffold_io.scaffold_json_path("pubchem__v256_nmb") == (
            tmp_path / "scaffold_root" / "scaffold" / "pubchem__v256_nmb.json"
        )


class TestWriteReadRoundTrip:
    def test_matched_record_round_trip(self) -> None:
        record = _matched_record()

        scaffold_io.write_scaffold_json(record)
        payload = scaffold_io.read_scaffold_json("pubchem__v256_nmb")

        assert payload is not None
        assert payload["schema_version"] == scaffold_io.SCHEMA_VERSION
        assert payload["pair_status"] == "matched"
        assert payload["delta_scaffold_fraction"] == pytest.approx(20 / 256)
        assert payload["bpe"]["scaffold_log_sha"] == "log-A"
        assert payload["unigram"]["verified_by_construction"] is True

    def test_unpaired_record_round_trip(self) -> None:
        scaffold_io.write_scaffold_json(_unpaired_record())

        payload = scaffold_io.read_scaffold_json("zinc22__v2048_nmb")

        assert payload is not None
        assert payload["pair_status"] == "single_arm"
        assert payload["missing_arm"] == "unigram"
        assert payload["unigram"] is None
        assert payload["bpe"] is not None

    def test_write_leaves_no_tmp_file(self) -> None:
        path = scaffold_io.write_scaffold_json(_matched_record())

        siblings = list(path.parent.iterdir())

        assert siblings == [path]

    def test_read_returns_none_when_absent(self) -> None:
        assert scaffold_io.read_scaffold_json("nothing__here") is None

    def test_read_returns_none_for_corrupt_json(self) -> None:
        path = scaffold_io.scaffold_json_path("bad__pair")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("not json {")

        assert scaffold_io.read_scaffold_json("bad__pair") is None


def _patch_artifact_dir(monkeypatch: pytest.MonkeyPatch, artifacts_root: Path) -> None:
    def _redirect(corpus: str, name: str) -> Path:
        return artifacts_root / corpus / name

    monkeypatch.setattr(
        "smiles_subword.tokenize.measure._cellmeta.tokenizer_artifact_dir", _redirect
    )
    monkeypatch.setattr(scaffold_runner, "tokenizer_artifact_dir", _redirect)


class TestIsScaffoldDone:
    def test_true_when_meta_and_log_sha_match(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        artifacts_root = tmp_path / "artifacts"
        _patch_artifact_dir(monkeypatch, artifacts_root)
        scaffold_body = "header\nrecord\n"
        _write_cell(
            artifacts_root,
            corpus="pubchem",
            name="smirk_gpe_v256_nmb",
            base_kind="smirk_gpe",
            vocab_size=256,
            n_merges=97,
            sha="corpus-A",
            scaffold_jsonl=scaffold_body,
        )
        _write_cell(
            artifacts_root,
            corpus="pubchem",
            name="smirk_unigram_v256_nmb",
            base_kind="smirk_unigram",
            vocab_size=256,
            n_merges=None,
            sha="corpus-B",
        )
        log_sha = hashlib.sha256(scaffold_body.encode("utf-8")).hexdigest()
        scaffold_io.write_scaffold_json(_matched_record(bpe_log_sha=log_sha))

        assert scaffold_io.is_scaffold_done("pubchem__v256_nmb") is True

    def test_false_when_log_sha_drifted(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        artifacts_root = tmp_path / "artifacts"
        _patch_artifact_dir(monkeypatch, artifacts_root)
        _write_cell(
            artifacts_root,
            corpus="pubchem",
            name="smirk_gpe_v256_nmb",
            base_kind="smirk_gpe",
            vocab_size=256,
            n_merges=97,
            sha="corpus-A",
            scaffold_jsonl="now-different-bytes\n",
        )
        _write_cell(
            artifacts_root,
            corpus="pubchem",
            name="smirk_unigram_v256_nmb",
            base_kind="smirk_unigram",
            vocab_size=256,
            n_merges=None,
            sha="corpus-B",
        )
        scaffold_io.write_scaffold_json(_matched_record(bpe_log_sha="stale-sha"))

        assert scaffold_io.is_scaffold_done("pubchem__v256_nmb") is False

    def test_false_when_corpus_sha_drifted(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        artifacts_root = tmp_path / "artifacts"
        _patch_artifact_dir(monkeypatch, artifacts_root)
        scaffold_body = "x\n"
        _write_cell(
            artifacts_root,
            corpus="pubchem",
            name="smirk_gpe_v256_nmb",
            base_kind="smirk_gpe",
            vocab_size=256,
            n_merges=97,
            sha="DIFFERENT",
            scaffold_jsonl=scaffold_body,
        )
        _write_cell(
            artifacts_root,
            corpus="pubchem",
            name="smirk_unigram_v256_nmb",
            base_kind="smirk_unigram",
            vocab_size=256,
            n_merges=None,
            sha="corpus-B",
        )
        log_sha = hashlib.sha256(scaffold_body.encode("utf-8")).hexdigest()
        scaffold_io.write_scaffold_json(_matched_record(bpe_log_sha=log_sha))

        assert scaffold_io.is_scaffold_done("pubchem__v256_nmb") is False

    def test_false_when_no_payload(self) -> None:
        assert scaffold_io.is_scaffold_done("never_deposited") is False


class TestDepositPair:
    def test_returns_pending_reason_when_cell_lacks_meta(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        artifacts_root = tmp_path / "artifacts"
        _patch_artifact_dir(monkeypatch, artifacts_root)
        pair = MatchedPair(
            key=PairKey(corpus="pubchem", vocab_size=256, boundary="nmb"),
            tier="headline",
            bpe_cell_id="pubchem__smirk_gpe_v256_nmb",
            unigram_cell_id="pubchem__smirk_unigram_v256_nmb",
        )

        path, reason = scaffold_io.deposit_pair(pair)

        assert path is None
        assert reason is not None
        assert "meta.yaml" in reason

    def test_unpaired_returns_pending_when_log_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        artifacts_root = tmp_path / "artifacts"
        _patch_artifact_dir(monkeypatch, artifacts_root)
        _write_cell(
            artifacts_root,
            corpus="zinc22",
            name="smirk_gpe_v2048_nmb",
            base_kind="smirk_gpe",
            vocab_size=2048,
            n_merges=1889,
            sha="corpus-C",
        )
        unpaired = UnpairedCell(
            key=PairKey(corpus="zinc22", vocab_size=2048, boundary="nmb"),
            tier="conditional",
            cell_id="zinc22__smirk_gpe_v2048_nmb",
            arm="bpe",
            reason="conditional_negative_branch",
        )

        path, reason = scaffold_io.deposit_unpaired(unpaired)

        assert path is None
        assert reason is not None
        assert "scaffold.jsonl" in reason


class TestBuildScaffoldTable:
    def test_aggregates_present_and_pending(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        matched = MatchedPair(
            key=PairKey(corpus="pubchem", vocab_size=256, boundary="nmb"),
            tier="headline",
            bpe_cell_id="pubchem__smirk_gpe_v256_nmb",
            unigram_cell_id="pubchem__smirk_unigram_v256_nmb",
        )
        unpaired = UnpairedCell(
            key=PairKey(corpus="zinc22", vocab_size=2048, boundary="nmb"),
            tier="conditional",
            cell_id="zinc22__smirk_gpe_v2048_nmb",
            arm="bpe",
            reason="conditional_negative_branch",
        )
        monkeypatch.setattr(
            "smiles_subword.tokenize.measure.scaffold.io.pair_all_cells",
            lambda: ([matched], [unpaired]),
        )
        scaffold_io.write_scaffold_json(_matched_record())

        json_path, md_path = scaffold_io.build_scaffold_table()
        payload = json.loads(json_path.read_text())

        assert payload["n_pairs"] == 2
        assert payload["n_present"] == 1
        assert payload["pending"] == ["zinc22__v2048_nmb"]
        assert md_path.exists()
        assert "pubchem__v256_nmb" in md_path.read_text()


class TestRecordBuilders:
    """Deposit-vs-pend builder branches not reached through the real-cell tests."""

    def _pair(self) -> MatchedPair:
        return MatchedPair(
            key=PairKey(corpus="pubchem", vocab_size=256, boundary="nmb"),
            tier="headline",
            bpe_cell_id="pubchem__smirk_gpe_v256_nmb",
            unigram_cell_id="pubchem__smirk_unigram_v256_nmb",
        )

    def test_matched_propagates_unigram_reason(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake(cell_id: str, arm: str) -> ArmScaffold | str:
            return _bpe_arm() if arm == "bpe" else "no meta.yaml for unigram"

        monkeypatch.setattr(scaffold_io, "_arm_from_cell", fake)

        result = scaffold_io._matched_pair_record(self._pair())
        assert result == "no meta.yaml for unigram"

    def test_matched_boundary_mismatch_pends(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake(cell_id: str, arm: str) -> ArmScaffold:
            return _bpe_arm() if arm == "bpe" else _ul_arm(boundary="mb")

        monkeypatch.setattr(scaffold_io, "_arm_from_cell", fake)

        result = scaffold_io._matched_pair_record(self._pair())
        assert isinstance(result, str)
        assert "boundary mismatch" in result

    def test_matched_success_builds_record(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            scaffold_io,
            "_arm_from_cell",
            lambda _cid, arm: _bpe_arm() if arm == "bpe" else _ul_arm(),
        )

        record = scaffold_io._matched_pair_record(self._pair())

        assert isinstance(record, MatchedPairScaffold)
        assert record.pair_key == "pubchem__v256_nmb"

    def test_unpaired_success_wraps_with_complement(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            scaffold_io, "_arm_from_cell", lambda _cid, _arm: _bpe_arm()
        )
        unpaired = UnpairedCell(
            key=PairKey(corpus="zinc22", vocab_size=2048, boundary="nmb"),
            tier="conditional",
            cell_id="zinc22__smirk_gpe_v2048_nmb",
            arm="bpe",
            reason="conditional_negative_branch",
        )

        record = scaffold_io._unpaired_record(unpaired)

        assert isinstance(record, UnpairedScaffold)
        assert record.missing_arm == "unigram"


class TestBuildTableUnpairedRow:
    def test_present_unpaired_record_projects_a_row(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        unpaired = UnpairedCell(
            key=PairKey(corpus="zinc22", vocab_size=2048, boundary="nmb"),
            tier="conditional",
            cell_id="zinc22__smirk_gpe_v2048_nmb",
            arm="bpe",
            reason="conditional_negative_branch",
        )
        monkeypatch.setattr(
            "smiles_subword.tokenize.measure.scaffold.io.pair_all_cells",
            lambda: ([], [unpaired]),
        )
        scaffold_io.write_scaffold_json(_unpaired_record())

        json_path, _md = scaffold_io.build_scaffold_table()
        payload = json.loads(json_path.read_text())

        assert payload["n_unpaired_present"] == 1
        assert payload["unpaired"][0]["pair_key"] == "zinc22__v2048_nmb"
        assert payload["unpaired"][0]["present_arm"] == "bpe"


class TestDepositAll:
    def test_filters_by_only_pair_keys(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        match_a = MatchedPair(
            key=PairKey(corpus="pubchem", vocab_size=256, boundary="nmb"),
            tier="headline",
            bpe_cell_id="pubchem__smirk_gpe_v256_nmb",
            unigram_cell_id="pubchem__smirk_unigram_v256_nmb",
        )
        match_b = MatchedPair(
            key=PairKey(corpus="pubchem", vocab_size=512, boundary="nmb"),
            tier="headline",
            bpe_cell_id="pubchem__smirk_gpe_v512_nmb",
            unigram_cell_id="pubchem__smirk_unigram_v512_nmb",
        )
        monkeypatch.setattr(
            "smiles_subword.tokenize.measure.scaffold.io.pair_all_cells",
            lambda: ([match_a, match_b], []),
        )
        artifacts_root = tmp_path / "artifacts"
        _patch_artifact_dir(monkeypatch, artifacts_root)

        deposited, pending = scaffold_io.deposit_all(
            only_pair_keys=frozenset({"pubchem__v512_nmb"})
        )

        slugs = [pk for pk, _ in pending]
        assert "pubchem__v512_nmb" in deposited or "pubchem__v512_nmb" in slugs
        assert "pubchem__v256_nmb" not in deposited
        assert "pubchem__v256_nmb" not in slugs
