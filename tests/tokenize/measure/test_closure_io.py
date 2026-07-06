"""Tests for ``closure_io`` (deposition + aggregator).

Closure is vocabulary-only, so freshness keys on ``training_corpus_sha`` alone
(no held-out split SHA). These tests cover the write/read round-trip, that
SHA-drift check, a stubbed ``deposit_pair``, and aggregator behavior.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from smiles_subword.tokenize.measure._pairing import (
    MatchedPair,
    PairKey,
    UnpairedCell,
)
from smiles_subword.tokenize.measure.closure import (
    MatchedPairClosure,
    UnpairedClosure,
    compute_arm_closure,
    compute_matched_pair_closure,
    compute_unpaired_closure,
)
from smiles_subword.tokenize.measure.closure import io as closure_io

C = ("C",)
CC = ("C", "C")
CCC = ("C", "C", "C")
CO = ("C", "O")
CCCC = ("C", "C", "C", "C")


def _bpe_arm(sha: str = "sha-bpe"):  # noqa: ANN202 - test helper
    return compute_arm_closure(
        [C, ("O",), CC, CCC],
        cell_id="pubchem__smirk_gpe_v256_nmb",
        arm="bpe",
        boundary="nmb",
        vocab_size=256,
        training_corpus_sha=sha,
    )


def _ul_arm(sha: str = "sha-ul"):  # noqa: ANN202 - test helper
    return compute_arm_closure(
        [C, ("O",), CCCC, CO],
        cell_id="pubchem__smirk_unigram_v256_nmb",
        arm="unigram",
        boundary="nmb",
        vocab_size=256,
        training_corpus_sha=sha,
    )


def _matched_record(
    *, bpe_sha: str = "sha-bpe", ul_sha: str = "sha-ul"
) -> MatchedPairClosure:
    return compute_matched_pair_closure(
        _bpe_arm(bpe_sha),
        _ul_arm(ul_sha),
        pair_key="pubchem__v256_nmb",
        tier="headline",
        corpus="pubchem",
        vocab_size=256,
        boundary="nmb",
    )


def _unpaired_record() -> UnpairedClosure:
    present = compute_arm_closure(
        [C, ("O",), CC, CCC],
        cell_id="zinc22__smirk_gpe_v2048_nmb",
        arm="bpe",
        boundary="nmb",
        vocab_size=2048,
        training_corpus_sha="sha-cond",
    )
    return compute_unpaired_closure(
        present,
        pair_key="zinc22__v2048_nmb",
        tier="conditional",
        corpus="zinc22",
        vocab_size=2048,
        boundary="nmb",
        extras_kind=None,
        extras_label=None,
        present_arm="bpe",
        missing_arm="unigram",
        unpaired_reason="conditional_negative_branch",
    )


def _write_meta(artifact_dir: Path, *, base_kind: str, sha: str) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": artifact_dir.name,
        "base_kind": base_kind,
        "vocab_size": 256,
        "training_corpus_sha": sha,
        "merge_brackets": False,
    }
    (artifact_dir / "meta.yaml").write_text(yaml.safe_dump(payload, sort_keys=False))


@pytest.fixture(autouse=True)
def _redirect_data_dirs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    root = tmp_path / "closure_root"
    monkeypatch.setattr(closure_io, "CLOSURE_DATA_DIR", root)
    monkeypatch.setattr(closure_io, "CLOSURE_CELL_DIR", root / "closure")
    monkeypatch.setattr(closure_io, "CLOSURE_TABLE_JSON", root / "closure_table.json")
    monkeypatch.setattr(closure_io, "CLOSURE_TABLE_MD", root / "closure_table.md")
    return tmp_path


def _patch_artifacts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    artifacts_root = tmp_path / "artifacts"
    monkeypatch.setattr(
        "smiles_subword.tokenize.measure._cellmeta.tokenizer_artifact_dir",
        lambda corpus, name: artifacts_root / corpus / name,
    )
    return artifacts_root


class TestWriteReadRoundTrip:
    def test_matched_record_round_trip(self) -> None:
        closure_io.write_closure_json(_matched_record())
        payload = closure_io.read_closure_json("pubchem__v256_nmb")

        assert payload is not None
        assert payload["schema_version"] == closure_io.SCHEMA_VERSION
        assert payload["pair_status"] == "matched"
        assert payload["bpe"]["c_bin"] == 1.0
        assert payload["unigram"]["c_bin"] == pytest.approx(0.5)
        assert payload["delta_c_bin"] == pytest.approx(0.5)

    def test_unpaired_record_round_trip(self) -> None:
        closure_io.write_closure_json(_unpaired_record())
        payload = closure_io.read_closure_json("zinc22__v2048_nmb")

        assert payload is not None
        assert payload["pair_status"] == "single_arm"
        assert payload["present_arm"] == "bpe"
        assert payload["bpe"]["c_bin"] == 1.0  # present-arm block keyed by arm

    def test_read_missing_returns_none(self) -> None:
        assert closure_io.read_closure_json("absent__v1_nmb") is None

    def test_read_malformed_returns_none(self) -> None:
        path = closure_io.closure_json_path("pubchem__v256_nmb")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not valid json")
        assert closure_io.read_closure_json("pubchem__v256_nmb") is None


class TestIsClosureDone:
    def test_true_when_both_arm_shas_match(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        artifacts = _patch_artifacts(monkeypatch, tmp_path)
        _write_meta(
            artifacts / "pubchem" / "smirk_gpe_v256_nmb",
            base_kind="smirk_gpe",
            sha="sha-bpe",
        )
        _write_meta(
            artifacts / "pubchem" / "smirk_unigram_v256_nmb",
            base_kind="smirk_unigram",
            sha="sha-ul",
        )
        closure_io.write_closure_json(_matched_record())

        assert closure_io.is_closure_done("pubchem__v256_nmb") is True

    def test_false_when_arm_sha_drifted(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        artifacts = _patch_artifacts(monkeypatch, tmp_path)
        _write_meta(
            artifacts / "pubchem" / "smirk_gpe_v256_nmb",
            base_kind="smirk_gpe",
            sha="sha-bpe-NEW",  # drifted from deposited sha-bpe
        )
        _write_meta(
            artifacts / "pubchem" / "smirk_unigram_v256_nmb",
            base_kind="smirk_unigram",
            sha="sha-ul",
        )
        closure_io.write_closure_json(_matched_record())

        assert closure_io.is_closure_done("pubchem__v256_nmb") is False

    def test_false_when_record_absent(self) -> None:
        assert closure_io.is_closure_done("pubchem__v256_nmb") is False


class TestDepositPair:
    def test_deposit_pair_writes_record(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        artifacts = _patch_artifacts(monkeypatch, tmp_path)
        _write_meta(
            artifacts / "pubchem" / "smirk_gpe_v256_nmb",
            base_kind="smirk_gpe",
            sha="sha-bpe",
        )
        _write_meta(
            artifacts / "pubchem" / "smirk_unigram_v256_nmb",
            base_kind="smirk_unigram",
            sha="sha-ul",
        )
        # Stub the vocab read so no real tokenizer.json is needed.
        monkeypatch.setattr(
            closure_io,
            "run_arm_closure",
            lambda *, arm, **_kw: _bpe_arm() if arm == "bpe" else _ul_arm(),
        )
        pair = MatchedPair(
            key=PairKey(corpus="pubchem", vocab_size=256, boundary="nmb"),
            tier="headline",
            bpe_cell_id="pubchem__smirk_gpe_v256_nmb",
            unigram_cell_id="pubchem__smirk_unigram_v256_nmb",
        )

        path, reason = closure_io.deposit_pair(pair)

        assert reason is None
        assert path is not None
        assert path.is_file()
        assert closure_io.read_closure_json("pubchem__v256_nmb") is not None

    def test_deposit_pair_pending_when_meta_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_artifacts(monkeypatch, tmp_path)  # no meta written
        pair = MatchedPair(
            key=PairKey(corpus="pubchem", vocab_size=256, boundary="nmb"),
            tier="headline",
            bpe_cell_id="pubchem__smirk_gpe_v256_nmb",
            unigram_cell_id="pubchem__smirk_unigram_v256_nmb",
        )

        path, reason = closure_io.deposit_pair(pair)

        assert path is None
        assert reason is not None
        assert "meta.yaml" in reason


def _matched_pair_obj() -> MatchedPair:
    return MatchedPair(
        key=PairKey(corpus="pubchem", vocab_size=256, boundary="nmb"),
        tier="headline",
        bpe_cell_id="pubchem__smirk_gpe_v256_nmb",
        unigram_cell_id="pubchem__smirk_unigram_v256_nmb",
    )


def _unpaired_cell_obj() -> UnpairedCell:
    return UnpairedCell(
        key=PairKey(corpus="zinc22", vocab_size=2048, boundary="nmb"),
        tier="conditional",
        cell_id="zinc22__smirk_gpe_v2048_nmb",
        arm="bpe",
        reason="conditional_negative_branch",
    )


def _stub_run_arm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Echo cell_id/sha so deposited blocks pass the later freshness check."""

    def _run(*, cell_id, arm, boundary, vocab_size, training_corpus_sha, **_):  # noqa: ANN001, ANN003, ANN202
        tuples = [C, ("O",), CC, CCC] if arm == "bpe" else [C, ("O",), CCCC, CO]
        return compute_arm_closure(
            tuples,
            cell_id=cell_id,
            arm=arm,
            boundary=boundary,
            vocab_size=vocab_size,
            training_corpus_sha=training_corpus_sha,
        )

    monkeypatch.setattr(closure_io, "run_arm_closure", _run)


def _write_all_meta(artifacts: Path) -> None:
    _write_meta(
        artifacts / "pubchem" / "smirk_gpe_v256_nmb",
        base_kind="smirk_gpe",
        sha="sha-bpe",
    )
    _write_meta(
        artifacts / "pubchem" / "smirk_unigram_v256_nmb",
        base_kind="smirk_unigram",
        sha="sha-ul",
    )
    _write_meta(
        artifacts / "zinc22" / "smirk_gpe_v2048_nmb",
        base_kind="smirk_gpe",
        sha="sha-cond",
    )


class TestDepositUnpaired:
    def test_writes_present_arm_only(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        artifacts = _patch_artifacts(monkeypatch, tmp_path)
        _write_all_meta(artifacts)
        _stub_run_arm(monkeypatch)

        path, reason = closure_io.deposit_unpaired(_unpaired_cell_obj())

        assert reason is None
        assert path is not None
        payload = closure_io.read_closure_json("zinc22__v2048_nmb")
        assert payload is not None
        assert payload["pair_status"] == "single_arm"
        assert payload["present_arm"] == "bpe"
        assert closure_io.is_closure_done("zinc22__v2048_nmb") is True


class TestDepositAll:
    def test_deposits_then_skips_idempotently(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        artifacts = _patch_artifacts(monkeypatch, tmp_path)
        _write_all_meta(artifacts)
        _stub_run_arm(monkeypatch)
        pairs = [_matched_pair_obj()]
        unpaired = [_unpaired_cell_obj()]

        deposited, pending = closure_io.deposit_all(pairs, unpaired)
        assert pending == []
        assert set(deposited) == {"pubchem__v256_nmb", "zinc22__v2048_nmb"}

        # Second sweep recomputes nothing (every record is fresh) but still
        # reports the pairs as done.
        again, pending2 = closure_io.deposit_all(pairs, unpaired)
        assert pending2 == []
        assert set(again) == {"pubchem__v256_nmb", "zinc22__v2048_nmb"}

    def test_only_pair_keys_filters(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        artifacts = _patch_artifacts(monkeypatch, tmp_path)
        _write_all_meta(artifacts)
        _stub_run_arm(monkeypatch)

        deposited, _pending = closure_io.deposit_all(
            [_matched_pair_obj()],
            [_unpaired_cell_obj()],
            only_pair_keys=frozenset({"pubchem__v256_nmb"}),
        )

        assert deposited == ["pubchem__v256_nmb"]

    def test_missing_meta_routes_both_to_pending(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_artifacts(monkeypatch, tmp_path)  # no meta.yaml on disk

        deposited, pending = closure_io.deposit_all(
            [_matched_pair_obj()], [_unpaired_cell_obj()]
        )

        assert deposited == []
        pending_keys = {pk for pk, _reason in pending}
        assert pending_keys == {"pubchem__v256_nmb", "zinc22__v2048_nmb"}
        assert all("meta.yaml" in reason for _pk, reason in pending)


class TestErrorPaths:
    def test_malformed_cell_id_pends(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_artifacts(monkeypatch, tmp_path)
        pair = MatchedPair(
            key=PairKey(corpus="pubchem", vocab_size=256, boundary="nmb"),
            tier="headline",
            bpe_cell_id="nodoubleunderscore",  # no '__' -> malformed
            unigram_cell_id="pubchem__smirk_unigram_v256_nmb",
        )
        _path, reason = closure_io.deposit_pair(pair)
        assert reason is not None
        assert "malformed cell_id" in reason

    def test_meta_without_sha_pends(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        artifacts = _patch_artifacts(monkeypatch, tmp_path)
        cell = artifacts / "pubchem" / "smirk_gpe_v256_nmb"
        cell.mkdir(parents=True, exist_ok=True)
        (cell / "meta.yaml").write_text("name: x\nbase_kind: smirk_gpe\n")

        _path, reason = closure_io.deposit_pair(_matched_pair_obj())
        assert reason is not None
        assert "missing training_corpus_sha" in reason

    def test_runner_exception_pends(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        artifacts = _patch_artifacts(monkeypatch, tmp_path)
        _write_all_meta(artifacts)

        def _boom(**_kw: object) -> object:
            raise FileNotFoundError("no tokenizer.json")

        monkeypatch.setattr(closure_io, "run_arm_closure", _boom)

        _path, reason = closure_io.deposit_pair(_matched_pair_obj())
        assert reason is not None
        assert "no tokenizer.json" in reason

    def test_unigram_arm_error_pends(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        artifacts = _patch_artifacts(monkeypatch, tmp_path)
        # BPE meta present (its arm succeeds) but Unigram meta is absent.
        _write_meta(
            artifacts / "pubchem" / "smirk_gpe_v256_nmb",
            base_kind="smirk_gpe",
            sha="sha-bpe",
        )
        _stub_run_arm(monkeypatch)

        _path, reason = closure_io.deposit_pair(_matched_pair_obj())
        assert reason is not None
        assert "no meta.yaml" in reason

    def test_default_args_walk_manifest_and_pend(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # No artifacts on disk: deposit_all() with no explicit lists falls back
        # to the manifest and routes every pair to pending.
        _patch_artifacts(monkeypatch, tmp_path)
        deposited, pending = closure_io.deposit_all()
        assert deposited == []
        assert len(pending) > 0

    @pytest.mark.parametrize(
        "payload",
        [
            {"pair_status": "matched", "bpe": "notadict", "unigram": {}},
            {
                "pair_status": "matched",
                "bpe": {"training_corpus_sha": "x"},  # no cell_id
                "unigram": {},
            },
            {
                "pair_status": "matched",
                "bpe": {"cell_id": "nounderscore", "training_corpus_sha": "x"},
                "unigram": {},
            },
            {"pair_status": "single_arm"},  # no present_arm
        ],
    )
    def test_is_done_false_on_bad_payload(self, payload: dict[str, object]) -> None:
        path = closure_io.closure_json_path("pubchem__v256_nmb")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))
        assert closure_io.is_closure_done("pubchem__v256_nmb") is False

    def test_is_done_false_when_meta_absent_for_block(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_artifacts(monkeypatch, tmp_path)  # no meta on disk
        path = closure_io.closure_json_path("pubchem__v256_nmb")
        path.parent.mkdir(parents=True, exist_ok=True)
        block = {
            "cell_id": "pubchem__smirk_gpe_v256_nmb",
            "training_corpus_sha": "x",
        }
        path.write_text(
            json.dumps({"pair_status": "matched", "bpe": block, "unigram": block})
        )
        assert closure_io.is_closure_done("pubchem__v256_nmb") is False


class TestBuildTable:
    def test_unpaired_row_renders_in_aggregate(self) -> None:
        closure_io.write_closure_json(_unpaired_record())

        json_path, md_path = closure_io.build_closure_table()
        table = json.loads(json_path.read_text())

        present = {r["pair_key"] for r in table["unpaired"]}
        assert "zinc22__v2048_nmb" in present
        assert table["n_unpaired_present"] == len(table["unpaired"])
        assert "Single-arm" in md_path.read_text()

    def test_aggregates_present_and_lists_pending(self) -> None:
        # Only deposit one of the manifest's pairs; the rest become pending.
        closure_io.write_closure_json(_matched_record())

        json_path, md_path = closure_io.build_closure_table()
        table = json.loads(json_path.read_text())

        present = {r["pair_key"] for r in table["matched"]}
        assert "pubchem__v256_nmb" in present
        assert table["n_matched_present"] == len(table["matched"])
        assert table["pending"]  # the rest of the manifest is undeposited
        assert md_path.is_file()
