"""Tests for ``nestedness_io`` (deposition + aggregator).

Validate write / read round-trip, freshness checks, and aggregator behavior.
The runtime dual-encode path (``deposit_pair`` end-to-end) is exercised by the
smoke run; these tests stub it via :func:`monkeypatch` so they exercise the IO
contract without loading real artifacts.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from smiles_subword.tokenize.measure._cellmeta import ArmInfo
from smiles_subword.tokenize.measure._pairing import MatchedPair, PairKey, UnpairedCell
from smiles_subword.tokenize.measure.nestedness import (
    MatchedPairNestedness,
    PerMoleculeNestedness,
    UnpairedNestedness,
    compute_pair_nestedness,
    make_unpaired_nestedness,
)
from smiles_subword.tokenize.measure.nestedness import io as nestedness_io


def _pm() -> PerMoleculeNestedness:
    return PerMoleculeNestedness(
        n_positions=10,
        n_agree_cut=6,
        n_nest=3,
        n_conflict=1,
        n_agree_merge=0,
        emitted_by_class=(2, 0, 0, 5, 0),
        cut_through_by_class=(0, 0, 0, 2, 0),
    )


def _matched_record(
    *,
    pair_key: str = "pubchem__v256_nmb",
    bpe_sha: str = "sha-bpe",
    ul_sha: str = "sha-ul",
    eval_sha: str = "eval-A",
) -> MatchedPairNestedness:
    return compute_pair_nestedness(
        [_pm()],
        pair_key=pair_key,
        tier="headline",
        corpus="pubchem",
        vocab_size=256,
        boundary="nmb",
        bpe_cell_id="pubchem__smirk_gpe_v256_nmb",
        unigram_cell_id="pubchem__smirk_unigram_v256_nmb",
        bpe_training_corpus_sha=bpe_sha,
        unigram_training_corpus_sha=ul_sha,
        eval_split_sha=eval_sha,
    )


def _unpaired_record() -> UnpairedNestedness:
    return make_unpaired_nestedness(
        pair_key="zinc22__v2048_nmb",
        tier="conditional",
        corpus="zinc22",
        vocab_size=2048,
        boundary="nmb",
        extras_kind=None,
        extras_label=None,
        present_arm="bpe",
        present_cell_id="zinc22__smirk_gpe_v2048_nmb",
        present_training_corpus_sha="sha-cond",
        eval_split_sha="eval-A",
        missing_arm="unigram",
        unpaired_reason="conditional_negative_branch",
    )


def _write_meta(artifact_dir: Path, *, base_kind: str, sha: str) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": artifact_dir.name,
        "base_kind": base_kind,
        "vocab_size": 256,  # dummy — freshness keys only off training_corpus_sha
        "training_corpus_sha": sha,
        "merge_brackets": False,
        "split_structure": True,
    }
    (artifact_dir / "meta.yaml").write_text(yaml.safe_dump(payload, sort_keys=False))


def _write_test_split_manifest(corpus_dir: Path, shas: list[str]) -> None:
    test_dir = corpus_dir / "canon_dedup_v1" / "test"
    test_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "canon_dedup_v1",
        "shards": [
            {"file": f"s-{i}.parquet", "sha256": sha, "n_rows": 1, "n_bytes": 1}
            for i, sha in enumerate(shas)
        ],
    }
    (test_dir / "MANIFEST.yaml").write_text(yaml.safe_dump(payload, sort_keys=False))


@pytest.fixture(autouse=True)
def _redirect_data_dirs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root = tmp_path / "nestedness_root"
    monkeypatch.setattr(nestedness_io, "NESTEDNESS_DATA_DIR", root)
    monkeypatch.setattr(nestedness_io, "NESTEDNESS_CELL_DIR", root / "nestedness")
    monkeypatch.setattr(
        nestedness_io, "NESTEDNESS_TABLE_JSON", root / "nestedness_table.json"
    )
    monkeypatch.setattr(
        nestedness_io, "NESTEDNESS_TABLE_MD", root / "nestedness_table.md"
    )


def _patch_artifacts_and_data(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[Path, Path]:
    artifacts_root = tmp_path / "artifacts"
    data_root = tmp_path / "data"
    monkeypatch.setattr(
        "smiles_subword.tokenize.measure._cellmeta.tokenizer_artifact_dir",
        lambda corpus, name: artifacts_root / corpus / name,
    )
    monkeypatch.setattr("smiles_subword.paths.DATA_DIR", data_root)
    return artifacts_root, data_root


class TestJsonPath:
    def test_path_is_under_the_cell_dir(self, tmp_path: Path) -> None:
        assert nestedness_io.nestedness_json_path("pubchem__v256_nmb") == (
            tmp_path / "nestedness_root" / "nestedness" / "pubchem__v256_nmb.json"
        )


class TestWriteReadRoundTrip:
    def test_matched_record_round_trip(self) -> None:
        nestedness_io.write_nestedness_json(_matched_record())
        payload = nestedness_io.read_nestedness_json("pubchem__v256_nmb")

        assert payload is not None
        assert payload["schema_version"] == nestedness_io.SCHEMA_VERSION
        assert payload["pair_status"] == "matched"
        assert payload["boundary_jaccard"] == pytest.approx(6 / 10)
        assert payload["conflict_rate"] == pytest.approx(1 / 10)
        assert payload["cut_rate_by_class"]["heteroatom"] == pytest.approx(2 / 5)

    def test_unpaired_record_round_trip(self) -> None:
        nestedness_io.write_nestedness_json(_unpaired_record())
        payload = nestedness_io.read_nestedness_json("zinc22__v2048_nmb")

        assert payload is not None
        assert payload["pair_status"] == "single_arm"
        assert payload["missing_arm"] == "unigram"
        assert payload["present_arm"] == "bpe"

    def test_read_missing_returns_none(self) -> None:
        assert nestedness_io.read_nestedness_json("absent__v1_nmb") is None


class TestIsNestednessDone:
    def test_true_when_meta_and_eval_split_match(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        artifacts_root, data_root = _patch_artifacts_and_data(monkeypatch, tmp_path)
        _write_meta(
            artifacts_root / "pubchem" / "smirk_gpe_v256_nmb",
            base_kind="smirk_gpe",
            sha="sha-bpe",
        )
        _write_meta(
            artifacts_root / "pubchem" / "smirk_unigram_v256_nmb",
            base_kind="smirk_unigram",
            sha="sha-ul",
        )
        _write_test_split_manifest(data_root / "processed" / "pubchem", ["aa", "bb"])
        eval_sha = nestedness_io.eval_split_sha("pubchem")
        nestedness_io.write_nestedness_json(_matched_record(eval_sha=eval_sha))

        assert nestedness_io.is_nestedness_done("pubchem__v256_nmb") is True

    def test_false_when_eval_split_drifted(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        artifacts_root, data_root = _patch_artifacts_and_data(monkeypatch, tmp_path)
        _write_meta(
            artifacts_root / "pubchem" / "smirk_gpe_v256_nmb",
            base_kind="smirk_gpe",
            sha="sha-bpe",
        )
        _write_meta(
            artifacts_root / "pubchem" / "smirk_unigram_v256_nmb",
            base_kind="smirk_unigram",
            sha="sha-ul",
        )
        _write_test_split_manifest(data_root / "processed" / "pubchem", ["aa", "bb"])
        nestedness_io.write_nestedness_json(_matched_record(eval_sha="eval-DIFFERENT"))

        assert nestedness_io.is_nestedness_done("pubchem__v256_nmb") is False

    def test_false_when_meta_sha_drifted(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        artifacts_root, data_root = _patch_artifacts_and_data(monkeypatch, tmp_path)
        _write_meta(
            artifacts_root / "pubchem" / "smirk_gpe_v256_nmb",
            base_kind="smirk_gpe",
            sha="sha-bpe-NEW",
        )
        _write_meta(
            artifacts_root / "pubchem" / "smirk_unigram_v256_nmb",
            base_kind="smirk_unigram",
            sha="sha-ul",
        )
        _write_test_split_manifest(data_root / "processed" / "pubchem", ["aa", "bb"])
        eval_sha = nestedness_io.eval_split_sha("pubchem")
        nestedness_io.write_nestedness_json(_matched_record(eval_sha=eval_sha))

        assert nestedness_io.is_nestedness_done("pubchem__v256_nmb") is False

    def test_false_when_meta_missing(self) -> None:
        nestedness_io.write_nestedness_json(_matched_record())

        assert nestedness_io.is_nestedness_done("pubchem__v256_nmb") is False

    def test_unpaired_freshness_checks_present_arm(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        artifacts_root, data_root = _patch_artifacts_and_data(monkeypatch, tmp_path)
        _write_meta(
            artifacts_root / "zinc22" / "smirk_gpe_v2048_nmb",
            base_kind="smirk_gpe",
            sha="sha-cond",
        )
        _write_test_split_manifest(data_root / "processed" / "zinc22", ["aa"])
        eval_sha = nestedness_io.eval_split_sha("zinc22")
        rec = make_unpaired_nestedness(
            pair_key="zinc22__v2048_nmb",
            tier="conditional",
            corpus="zinc22",
            vocab_size=2048,
            boundary="nmb",
            extras_kind=None,
            extras_label=None,
            present_arm="bpe",
            present_cell_id="zinc22__smirk_gpe_v2048_nmb",
            present_training_corpus_sha="sha-cond",
            eval_split_sha=eval_sha,
            missing_arm="unigram",
            unpaired_reason="conditional_negative_branch",
        )
        nestedness_io.write_nestedness_json(rec)

        assert nestedness_io.is_nestedness_done("zinc22__v2048_nmb") is True


class TestUnpairedDeposit:
    def test_unpaired_dispatches_through_helper(
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
            nestedness_io, "_unpaired_record", lambda _u: _unpaired_record()
        )

        path, reason = nestedness_io.deposit_unpaired(unpaired)

        assert reason is None
        assert path is not None
        assert nestedness_io.read_nestedness_json("zinc22__v2048_nmb") is not None


class TestRecordBuilders:
    """The deposit-vs-pend branches of the matched / unpaired record builders."""

    def _pair(self) -> MatchedPair:
        return MatchedPair(
            key=PairKey(corpus="pubchem", vocab_size=256, boundary="nmb"),
            tier="headline",
            bpe_cell_id="pubchem__smirk_gpe_v256_nmb",
            unigram_cell_id="pubchem__smirk_unigram_v256_nmb",
        )

    def _unpaired(self) -> UnpairedCell:
        return UnpairedCell(
            key=PairKey(corpus="zinc22", vocab_size=2048, boundary="nmb"),
            tier="conditional",
            cell_id="zinc22__smirk_gpe_v2048_nmb",
            arm="bpe",
            reason="conditional_negative_branch",
        )

    def test_matched_propagates_bpe_info_reason(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(nestedness_io, "arm_info", lambda _cid: "no meta.yaml")

        assert nestedness_io._matched_pair_record(self._pair()) == "no meta.yaml"

    def test_matched_propagates_unigram_info_reason(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake(cell_id: str) -> ArmInfo | str:
            return (
                ArmInfo(name="smirk_gpe_v256_nmb", training_corpus_sha="T")
                if "gpe" in cell_id
                else "no meta.yaml for unigram"
            )

        monkeypatch.setattr(nestedness_io, "arm_info", fake)

        result = nestedness_io._matched_pair_record(self._pair())
        assert result == "no meta.yaml for unigram"

    def test_matched_missing_adapter_pends(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            nestedness_io,
            "arm_info",
            lambda cid: ArmInfo(name=cid.split("__")[1], training_corpus_sha="T"),
        )

        def boom(_c: str, _n: str) -> object:
            raise FileNotFoundError("no tokenizer.json")

        monkeypatch.setattr(nestedness_io, "load_cell_adapter", boom)

        assert "no tokenizer.json" in nestedness_io._matched_pair_record(self._pair())

    def test_matched_success_dispatches_to_runner(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            nestedness_io,
            "arm_info",
            lambda cid: ArmInfo(name=cid.split("__")[1], training_corpus_sha="T"),
        )
        monkeypatch.setattr(nestedness_io, "load_cell_adapter", lambda _c, _n: object())
        sentinel = _matched_record()
        monkeypatch.setattr(
            nestedness_io, "run_pair_nestedness", lambda *a, **k: sentinel
        )

        assert nestedness_io._matched_pair_record(self._pair()) is sentinel

    def test_unpaired_propagates_info_reason(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(nestedness_io, "arm_info", lambda _cid: "no meta.yaml")

        assert nestedness_io._unpaired_record(self._unpaired()) == "no meta.yaml"

    def test_unpaired_pends_when_eval_split_unresolvable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            nestedness_io,
            "arm_info",
            lambda _cid: ArmInfo(name="smirk_gpe_v2048_nmb", training_corpus_sha="T"),
        )

        def boom(_c: str) -> str:
            raise FileNotFoundError("no test split MANIFEST")

        monkeypatch.setattr(nestedness_io, "eval_split_sha", boom)

        assert "no test split MANIFEST" in nestedness_io._unpaired_record(
            self._unpaired()
        )

    def test_unpaired_success_builds_record(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            nestedness_io,
            "arm_info",
            lambda _cid: ArmInfo(name="smirk_gpe_v2048_nmb", training_corpus_sha="T"),
        )
        monkeypatch.setattr(nestedness_io, "eval_split_sha", lambda _c: "eval-X")
        sentinel = _unpaired_record()
        monkeypatch.setattr(
            nestedness_io, "make_unpaired_nestedness", lambda **_k: sentinel
        )

        assert nestedness_io._unpaired_record(self._unpaired()) is sentinel


class TestBuildTable:
    def test_present_and_pending_split(self, monkeypatch: pytest.MonkeyPatch) -> None:
        present = MatchedPair(
            key=PairKey(corpus="pubchem", vocab_size=256, boundary="nmb"),
            tier="headline",
            bpe_cell_id="pubchem__smirk_gpe_v256_nmb",
            unigram_cell_id="pubchem__smirk_unigram_v256_nmb",
        )
        missing = MatchedPair(
            key=PairKey(corpus="coconut", vocab_size=512, boundary="mb"),
            tier="headline",
            bpe_cell_id="coconut__smirk_gpe_v512_mb",
            unigram_cell_id="coconut__smirk_unigram_v512_mb",
        )
        monkeypatch.setattr(
            nestedness_io, "pair_all_cells", lambda: ([present, missing], [])
        )
        nestedness_io.write_nestedness_json(_matched_record())

        table_json, _table_md = nestedness_io.build_nestedness_table()
        import json

        payload = json.loads(table_json.read_text())

        assert payload["n_matched_present"] == 1
        assert payload["pending"] == ["coconut__v512_mb"]
        assert payload["matched"][0]["pair_key"] == "pubchem__v256_nmb"
