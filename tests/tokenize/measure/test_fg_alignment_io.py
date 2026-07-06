"""Tests for ``fg_alignment_io`` (deposition + aggregator).

Locality is held-out-evaluated, so freshness keys on both
``training_corpus_sha`` and ``eval_split_sha``; it is within-arm, so a single-arm
coordinate deposits a real reading. These tests stub the held-out encode pass
(``run_pair_fg_alignment`` / ``run_single_arm_fg_alignment``) and the adapter
load, and monkeypatch ``eval_split_sha`` to a fixed value.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from smiles_subword.tokenize.measure import _deposit
from smiles_subword.tokenize.measure._pairing import (
    MatchedPair,
    PairKey,
    UnpairedCell,
)
from smiles_subword.tokenize.measure.fg_alignment import (
    ArmFgAlignment,
    MatchedPairFgAlignment,
    PerMoleculeFgLocality,
    UnpairedFgAlignment,
    compute_arm_fg_alignment,
    compute_matched_pair_fg_alignment,
    compute_unpaired_fg_alignment,
)
from smiles_subword.tokenize.measure.fg_alignment import io as fg_alignment_io

EVAL_SHA = "sha-eval"


def _arm(
    *,
    arm: str,
    cell_id: str,
    train_sha: str,
    local: int,
) -> ArmFgAlignment:
    return compute_arm_fg_alignment(
        [
            PerMoleculeFgLocality(
                n_bonds=2,
                n_local=local,
                class_bonds={"C=O": 2},
                class_local={"C=O": local},
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
) -> MatchedPairFgAlignment:
    return compute_matched_pair_fg_alignment(
        _arm(
            arm="bpe", cell_id="pubchem__smirk_gpe_v256_nmb", train_sha=bpe_sha, local=2
        ),
        _arm(
            arm="unigram",
            cell_id="pubchem__smirk_unigram_v256_nmb",
            train_sha=ul_sha,
            local=0,
        ),
        pair_key="pubchem__v256_nmb",
        tier="headline",
        corpus="pubchem",
        vocab_size=256,
        boundary="nmb",
    )


def _unpaired_record() -> UnpairedFgAlignment:
    present = _arm(
        arm="bpe",
        cell_id="zinc22__smirk_gpe_v2048_nmb",
        train_sha="sha-cond",
        local=2,
    )
    return compute_unpaired_fg_alignment(
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
    payload = {"name": artifact_dir.name, "training_corpus_sha": sha}
    (artifact_dir / "meta.yaml").write_text(yaml.safe_dump(payload, sort_keys=False))


@pytest.fixture(autouse=True)
def _redirect(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    root = tmp_path / "fg_root"
    monkeypatch.setattr(fg_alignment_io, "FG_ALIGNMENT_DATA_DIR", root)
    monkeypatch.setattr(fg_alignment_io, "FG_ALIGNMENT_CELL_DIR", root / "fg_alignment")
    monkeypatch.setattr(
        fg_alignment_io, "FG_ALIGNMENT_TABLE_JSON", root / "fg_alignment_table.json"
    )
    monkeypatch.setattr(
        fg_alignment_io, "FG_ALIGNMENT_TABLE_MD", root / "fg_alignment_table.md"
    )
    # Freshness consults the corpus eval-split SHA; pin it.
    monkeypatch.setattr(_deposit, "eval_split_sha", lambda _corpus: EVAL_SHA)
    return tmp_path


def _patch_artifacts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    artifacts_root = tmp_path / "artifacts"
    monkeypatch.setattr(
        "smiles_subword.tokenize.measure._cellmeta.tokenizer_artifact_dir",
        lambda corpus, name: artifacts_root / corpus / name,
    )
    return artifacts_root


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


def _stub_runners(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the encode-pass runners + adapter load with fixtures."""
    monkeypatch.setattr(fg_alignment_io, "load_cell_adapter", lambda *a, **k: object())

    def _run_pair(*_a: object, **kw: object) -> MatchedPairFgAlignment:
        return _matched_record(
            bpe_sha=str(kw["bpe_training_corpus_sha"]),
            ul_sha=str(kw["unigram_training_corpus_sha"]),
        )

    def _run_single(*_a: object, **kw: object) -> ArmFgAlignment:
        return _arm(
            arm=str(kw["arm"]),
            cell_id=str(kw["cell_id"]),
            train_sha=str(kw["training_corpus_sha"]),
            local=2,
        )

    monkeypatch.setattr(fg_alignment_io, "run_pair_fg_alignment", _run_pair)
    monkeypatch.setattr(fg_alignment_io, "run_single_arm_fg_alignment", _run_single)


def _write_all_meta(artifacts: Path) -> None:
    _write_meta(artifacts / "pubchem" / "smirk_gpe_v256_nmb", sha="sha-bpe")
    _write_meta(artifacts / "pubchem" / "smirk_unigram_v256_nmb", sha="sha-ul")
    _write_meta(artifacts / "zinc22" / "smirk_gpe_v2048_nmb", sha="sha-cond")


class TestWriteReadRoundTrip:
    def test_matched_record_round_trip(self) -> None:
        fg_alignment_io.write_fg_alignment_json(_matched_record())
        payload = fg_alignment_io.read_fg_alignment_json("pubchem__v256_nmb")
        assert payload is not None
        assert payload["pair_status"] == "matched"
        assert payload["bpe"]["locality"] == pytest.approx(1.0)
        assert payload["unigram"]["locality"] == pytest.approx(0.0)
        assert payload["delta_locality"] == pytest.approx(1.0)

    def test_unpaired_round_trip(self) -> None:
        fg_alignment_io.write_fg_alignment_json(_unpaired_record())
        payload = fg_alignment_io.read_fg_alignment_json("zinc22__v2048_nmb")
        assert payload is not None
        assert payload["pair_status"] == "single_arm"
        assert payload["bpe"]["locality"] == pytest.approx(1.0)

    def test_read_missing_returns_none(self) -> None:
        assert fg_alignment_io.read_fg_alignment_json("absent__v1_nmb") is None

    def test_read_malformed_returns_none(self) -> None:
        path = fg_alignment_io.fg_alignment_json_path("pubchem__v256_nmb")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not valid json")
        assert fg_alignment_io.read_fg_alignment_json("pubchem__v256_nmb") is None


class TestIsDone:
    def test_true_when_shas_match(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        artifacts = _patch_artifacts(monkeypatch, tmp_path)
        _write_all_meta(artifacts)
        fg_alignment_io.write_fg_alignment_json(_matched_record())
        assert fg_alignment_io.is_fg_alignment_done("pubchem__v256_nmb") is True

    def test_false_when_train_sha_drifts(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        artifacts = _patch_artifacts(monkeypatch, tmp_path)
        _write_meta(artifacts / "pubchem" / "smirk_gpe_v256_nmb", sha="sha-bpe-NEW")
        _write_meta(artifacts / "pubchem" / "smirk_unigram_v256_nmb", sha="sha-ul")
        fg_alignment_io.write_fg_alignment_json(_matched_record())
        assert fg_alignment_io.is_fg_alignment_done("pubchem__v256_nmb") is False

    def test_false_when_eval_sha_drifts(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        artifacts = _patch_artifacts(monkeypatch, tmp_path)
        _write_all_meta(artifacts)
        fg_alignment_io.write_fg_alignment_json(_matched_record())
        # The corpus split was redrawn since deposit.
        monkeypatch.setattr(_deposit, "eval_split_sha", lambda _corpus: "sha-eval-NEW")
        assert fg_alignment_io.is_fg_alignment_done("pubchem__v256_nmb") is False

    def test_false_when_record_absent(self) -> None:
        assert fg_alignment_io.is_fg_alignment_done("pubchem__v256_nmb") is False


class TestDeposit:
    def test_deposit_pair_writes_record(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        artifacts = _patch_artifacts(monkeypatch, tmp_path)
        _write_all_meta(artifacts)
        _stub_runners(monkeypatch)
        path, reason = fg_alignment_io.deposit_pair(_matched_pair_obj())
        assert reason is None
        assert path is not None
        assert path.is_file()
        assert fg_alignment_io.is_fg_alignment_done("pubchem__v256_nmb") is True

    def test_deposit_unpaired_present_only(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        artifacts = _patch_artifacts(monkeypatch, tmp_path)
        _write_all_meta(artifacts)
        _stub_runners(monkeypatch)
        path, reason = fg_alignment_io.deposit_unpaired(_unpaired_cell_obj())
        assert reason is None
        assert path is not None
        payload = fg_alignment_io.read_fg_alignment_json("zinc22__v2048_nmb")
        assert payload is not None
        assert payload["present_arm"] == "bpe"

    def test_deposit_pair_pending_when_meta_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_artifacts(monkeypatch, tmp_path)
        path, reason = fg_alignment_io.deposit_pair(_matched_pair_obj())
        assert path is None
        assert reason is not None
        assert "meta.yaml" in reason

    def test_deposits_then_skips_idempotently(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        artifacts = _patch_artifacts(monkeypatch, tmp_path)
        _write_all_meta(artifacts)
        _stub_runners(monkeypatch)
        pairs = [_matched_pair_obj()]
        unpaired = [_unpaired_cell_obj()]
        deposited, pending = fg_alignment_io.deposit_all(pairs, unpaired)
        assert pending == []
        assert set(deposited) == {"pubchem__v256_nmb", "zinc22__v2048_nmb"}
        again, pending2 = fg_alignment_io.deposit_all(pairs, unpaired)
        assert pending2 == []
        assert set(again) == {"pubchem__v256_nmb", "zinc22__v2048_nmb"}

    def test_only_pair_keys_filters(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        artifacts = _patch_artifacts(monkeypatch, tmp_path)
        _write_all_meta(artifacts)
        _stub_runners(monkeypatch)
        deposited, _pending = fg_alignment_io.deposit_all(
            [_matched_pair_obj()],
            [_unpaired_cell_obj()],
            only_pair_keys=frozenset({"pubchem__v256_nmb"}),
        )
        assert deposited == ["pubchem__v256_nmb"]


class TestErrorPaths:
    def test_malformed_cell_id_pends(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_artifacts(monkeypatch, tmp_path)
        pair = MatchedPair(
            key=PairKey(corpus="pubchem", vocab_size=256, boundary="nmb"),
            tier="headline",
            bpe_cell_id="nodoubleunderscore",
            unigram_cell_id="pubchem__smirk_unigram_v256_nmb",
        )
        _path, reason = fg_alignment_io.deposit_pair(pair)
        assert reason is not None
        assert "malformed cell_id" in reason

    def test_adapter_missing_pends(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        artifacts = _patch_artifacts(monkeypatch, tmp_path)
        _write_all_meta(artifacts)

        def _boom(*_a: object, **_k: object) -> object:
            raise FileNotFoundError("no tokenizer.json")

        monkeypatch.setattr(fg_alignment_io, "load_cell_adapter", _boom)
        _path, reason = fg_alignment_io.deposit_pair(_matched_pair_obj())
        assert reason is not None
        assert "no tokenizer.json" in reason

    @pytest.mark.parametrize(
        "payload",
        [
            {"pair_status": "matched", "bpe": "notadict", "unigram": {}},
            {
                "pair_status": "matched",
                "bpe": {"training_corpus_sha": "x"},
                "unigram": {},
            },
            {"pair_status": "single_arm"},
        ],
    )
    def test_is_done_false_on_bad_payload(self, payload: dict[str, object]) -> None:
        path = fg_alignment_io.fg_alignment_json_path("pubchem__v256_nmb")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))
        assert fg_alignment_io.is_fg_alignment_done("pubchem__v256_nmb") is False

    def test_default_args_walk_manifest_and_pend(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_artifacts(monkeypatch, tmp_path)
        deposited, pending = fg_alignment_io.deposit_all()
        assert deposited == []
        assert len(pending) > 0


class TestBuildTable:
    def test_matched_and_unpaired_aggregate(self) -> None:
        fg_alignment_io.write_fg_alignment_json(_matched_record())
        fg_alignment_io.write_fg_alignment_json(_unpaired_record())
        json_path, md_path = fg_alignment_io.build_fg_alignment_table()
        table = json.loads(json_path.read_text())
        assert "pubchem__v256_nmb" in {r["pair_key"] for r in table["matched"]}
        assert "zinc22__v2048_nmb" in {r["pair_key"] for r in table["unpaired"]}
        assert table["pending"]  # rest of the manifest undeposited
        assert "functional-bond locality" in md_path.read_text()
