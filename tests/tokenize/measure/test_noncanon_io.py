"""Tests for ``noncanon_io`` (deposition + aggregator).

Non-canonicity is held-out-evaluated and within-arm, so freshness keys on both
``training_corpus_sha`` and ``eval_split_sha`` and a single-arm coordinate
deposits a real reading. These stub the orbit pass and adapter load and pin
``eval_split_sha``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from smiles_subword.tokenize.measure import _deposit
from smiles_subword.tokenize.measure._cellmeta import ArmInfo
from smiles_subword.tokenize.measure._pairing import MatchedPair, PairKey, UnpairedCell
from smiles_subword.tokenize.measure.noncanon import (
    ArmNoncanon,
    MatchedPairNoncanon,
    PerMoleculeNoncanon,
    UnpairedNoncanon,
    compute_arm_noncanon,
    compute_matched_pair_noncanon,
    compute_unpaired_noncanon,
)
from smiles_subword.tokenize.measure.noncanon import io as noncanon_io

EVAL_SHA = "sha-eval"


def _arm(*, arm: str, cell_id: str, train_sha: str, bag: float) -> ArmNoncanon:
    return compute_arm_noncanon(
        [
            PerMoleculeNoncanon(
                canon_fert=10,
                rand_fert_mean=12.0,
                axis_dfert={
                    "random": 0.2,
                    "kekule": 0.1,
                    "explicitH": 2.0,
                    "obcanon": 0.15,
                },
                axis_bag={
                    "random": bag,
                    "kekule": 0.5,
                    "explicitH": 0.8,
                    "obcanon": 0.3,
                },
            )
        ],
        cell_id=cell_id,
        arm=arm,  # type: ignore[arg-type]
        boundary="nmb",
        training_corpus_sha=train_sha,
        eval_split_sha=EVAL_SHA,
        n_resamples=16,
    )


def _matched_record(
    *, bpe_sha: str = "sha-bpe", ul_sha: str = "sha-ul"
) -> MatchedPairNoncanon:
    return compute_matched_pair_noncanon(
        _arm(
            arm="bpe",
            cell_id="pubchem__smirk_gpe_v1024_nmb",
            train_sha=bpe_sha,
            bag=0.40,
        ),
        _arm(
            arm="unigram",
            cell_id="pubchem__smirk_unigram_v1024_nmb",
            train_sha=ul_sha,
            bag=0.17,
        ),
        pair_key="pubchem__v1024_nmb",
        tier="headline",
        corpus="pubchem",
        vocab_size=1024,
        boundary="nmb",
    )


def _unpaired_record() -> UnpairedNoncanon:
    present = _arm(
        arm="bpe", cell_id="zinc22__smirk_gpe_v2048_nmb", train_sha="sha-cond", bag=0.33
    )
    return compute_unpaired_noncanon(
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


def _write_meta(artifact_dir: Path, *, sha: str) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "meta.yaml").write_text(
        yaml.safe_dump({"name": artifact_dir.name, "training_corpus_sha": sha})
    )


@pytest.fixture(autouse=True)
def _redirect(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root = tmp_path / "nc_root"
    monkeypatch.setattr(noncanon_io, "NONCANON_DATA_DIR", root)
    monkeypatch.setattr(noncanon_io, "NONCANON_CELL_DIR", root / "noncanon")
    monkeypatch.setattr(
        noncanon_io, "NONCANON_TABLE_JSON", root / "noncanon_table.json"
    )
    monkeypatch.setattr(noncanon_io, "NONCANON_TABLE_MD", root / "noncanon_table.md")
    monkeypatch.setattr(_deposit, "eval_split_sha", lambda _corpus: EVAL_SHA)


def _patch_artifacts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    root = tmp_path / "artifacts"
    monkeypatch.setattr(
        "smiles_subword.tokenize.measure._cellmeta.tokenizer_artifact_dir",
        lambda corpus, name: root / corpus / name,
    )
    return root


def _matched_pair_obj() -> MatchedPair:
    return MatchedPair(
        key=PairKey(corpus="pubchem", vocab_size=1024, boundary="nmb"),
        tier="headline",
        bpe_cell_id="pubchem__smirk_gpe_v1024_nmb",
        unigram_cell_id="pubchem__smirk_unigram_v1024_nmb",
    )


def _unpaired_cell_obj() -> UnpairedCell:
    return UnpairedCell(
        key=PairKey(corpus="zinc22", vocab_size=2048, boundary="nmb"),
        tier="conditional",
        cell_id="zinc22__smirk_gpe_v2048_nmb",
        arm="bpe",
        reason="conditional_negative_branch",
    )


def _stub_runners(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(noncanon_io, "load_cell_adapter", lambda *a, **k: object())

    def _run_pair(*_a: object, **kw: object) -> MatchedPairNoncanon:
        return _matched_record(
            bpe_sha=str(kw["bpe_training_corpus_sha"]),
            ul_sha=str(kw["unigram_training_corpus_sha"]),
        )

    def _run_single(*_a: object, **kw: object) -> ArmNoncanon:
        return _arm(
            arm=str(kw["arm"]),
            cell_id=str(kw["cell_id"]),
            train_sha=str(kw["training_corpus_sha"]),
            bag=0.33,
        )

    monkeypatch.setattr(noncanon_io, "run_pair_noncanon", _run_pair)
    monkeypatch.setattr(noncanon_io, "run_single_arm_noncanon", _run_single)


def _write_all_meta(artifacts: Path) -> None:
    _write_meta(artifacts / "pubchem" / "smirk_gpe_v1024_nmb", sha="sha-bpe")
    _write_meta(artifacts / "pubchem" / "smirk_unigram_v1024_nmb", sha="sha-ul")
    _write_meta(artifacts / "zinc22" / "smirk_gpe_v2048_nmb", sha="sha-cond")


class TestRoundTrip:
    def test_matched(self) -> None:
        noncanon_io.write_noncanon_json(_matched_record())
        payload = noncanon_io.read_noncanon_json("pubchem__v1024_nmb")
        assert payload is not None
        assert payload["pair_status"] == "matched"
        assert payload["delta_bag_instab"]["random"] == pytest.approx(0.23)
        assert payload["bpe"]["axes"]["random"]["bag_instab"] == pytest.approx(0.40)

    def test_unpaired(self) -> None:
        noncanon_io.write_noncanon_json(_unpaired_record())
        payload = noncanon_io.read_noncanon_json("zinc22__v2048_nmb")
        assert payload is not None
        assert payload["pair_status"] == "single_arm"
        assert "unigram" not in payload

    def test_read_malformed(self) -> None:
        path = noncanon_io.noncanon_json_path("pubchem__v1024_nmb")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{nope")
        assert noncanon_io.read_noncanon_json("pubchem__v1024_nmb") is None


class TestIsDone:
    def test_true_when_shas_match(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        artifacts = _patch_artifacts(monkeypatch, tmp_path)
        _write_all_meta(artifacts)
        noncanon_io.write_noncanon_json(_matched_record())
        assert noncanon_io.is_noncanon_done("pubchem__v1024_nmb") is True

    def test_false_when_eval_sha_drifts(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        artifacts = _patch_artifacts(monkeypatch, tmp_path)
        _write_all_meta(artifacts)
        noncanon_io.write_noncanon_json(_matched_record())
        monkeypatch.setattr(_deposit, "eval_split_sha", lambda _c: "sha-eval-NEW")
        assert noncanon_io.is_noncanon_done("pubchem__v1024_nmb") is False

    def test_false_when_train_sha_drifts(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        artifacts = _patch_artifacts(monkeypatch, tmp_path)
        _write_meta(artifacts / "pubchem" / "smirk_gpe_v1024_nmb", sha="sha-bpe-NEW")
        _write_meta(artifacts / "pubchem" / "smirk_unigram_v1024_nmb", sha="sha-ul")
        noncanon_io.write_noncanon_json(_matched_record())
        assert noncanon_io.is_noncanon_done("pubchem__v1024_nmb") is False

    def test_false_when_absent(self) -> None:
        assert noncanon_io.is_noncanon_done("pubchem__v1024_nmb") is False


class TestDeposit:
    def test_pair_and_idempotent(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        artifacts = _patch_artifacts(monkeypatch, tmp_path)
        _write_all_meta(artifacts)
        _stub_runners(monkeypatch)
        pairs = [_matched_pair_obj()]
        unpaired = [_unpaired_cell_obj()]
        deposited, pending = noncanon_io.deposit_all(pairs, unpaired)
        assert pending == []
        assert set(deposited) == {"pubchem__v1024_nmb", "zinc22__v2048_nmb"}
        again, pending2 = noncanon_io.deposit_all(pairs, unpaired)
        assert pending2 == []
        assert set(again) == {"pubchem__v1024_nmb", "zinc22__v2048_nmb"}

    def test_unpaired_present_only(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        artifacts = _patch_artifacts(monkeypatch, tmp_path)
        _write_all_meta(artifacts)
        _stub_runners(monkeypatch)
        path, reason = noncanon_io.deposit_unpaired(_unpaired_cell_obj())
        assert reason is None
        assert path is not None
        payload = noncanon_io.read_noncanon_json("zinc22__v2048_nmb")
        assert payload is not None
        assert payload["present_arm"] == "bpe"

    def test_pending_when_meta_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_artifacts(monkeypatch, tmp_path)
        path, reason = noncanon_io.deposit_pair(_matched_pair_obj())
        assert path is None
        assert reason is not None
        assert "meta.yaml" in reason

    def test_adapter_missing_pends(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        artifacts = _patch_artifacts(monkeypatch, tmp_path)
        _write_all_meta(artifacts)

        def _boom(*_a: object, **_k: object) -> object:
            raise FileNotFoundError("no tokenizer.json")

        monkeypatch.setattr(noncanon_io, "load_cell_adapter", _boom)
        _path, reason = noncanon_io.deposit_pair(_matched_pair_obj())
        assert reason is not None
        assert "no tokenizer.json" in reason

    def test_pair_pends_on_unigram_info_reason(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # bpe resolves but unigram does not; the unigram reason propagates
        # (the sibling of the bpe-info branch the meta-missing test hits first).
        def fake(cid: str) -> ArmInfo | str:
            return (
                ArmInfo(name=cid.split("__")[1], training_corpus_sha="T")
                if "gpe" in cid
                else "no meta.yaml for unigram"
            )

        monkeypatch.setattr(noncanon_io, "arm_info", fake)

        path, reason = noncanon_io.deposit_pair(_matched_pair_obj())
        assert path is None
        assert reason == "no meta.yaml for unigram"

    def test_unpaired_pends_when_meta_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_artifacts(monkeypatch, tmp_path)  # no meta written
        path, reason = noncanon_io.deposit_unpaired(_unpaired_cell_obj())
        assert path is None
        assert reason is not None
        assert "meta.yaml" in reason

    def test_unpaired_adapter_missing_pends(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        artifacts = _patch_artifacts(monkeypatch, tmp_path)
        _write_all_meta(artifacts)

        def _boom(*_a: object, **_k: object) -> object:
            raise FileNotFoundError("no tokenizer.json")

        monkeypatch.setattr(noncanon_io, "load_cell_adapter", _boom)
        _path, reason = noncanon_io.deposit_unpaired(_unpaired_cell_obj())
        assert reason is not None
        assert "no tokenizer.json" in reason


class TestBuildTable:
    def test_aggregate(self) -> None:
        noncanon_io.write_noncanon_json(_matched_record())
        noncanon_io.write_noncanon_json(_unpaired_record())
        json_path, md_path = noncanon_io.build_noncanon_table()
        table = json.loads(json_path.read_text())
        assert "pubchem__v1024_nmb" in {r["pair_key"] for r in table["matched"]}
        assert "zinc22__v2048_nmb" in {r["pair_key"] for r in table["unpaired"]}
        assert table["pending"]
        assert "bag-instability" in md_path.read_text()

    @pytest.mark.parametrize(
        "payload",
        [
            {"pair_status": "matched", "bpe": "x", "unigram": {}},
            {"pair_status": "single_arm"},
        ],
    )
    def test_is_done_bad_payload(self, payload: dict[str, object]) -> None:
        path = noncanon_io.noncanon_json_path("pubchem__v1024_nmb")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))
        assert noncanon_io.is_noncanon_done("pubchem__v1024_nmb") is False
