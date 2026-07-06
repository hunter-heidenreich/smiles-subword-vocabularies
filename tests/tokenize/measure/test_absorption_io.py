"""Tests for ``absorption_io`` (deposition + aggregator).

Validate write / read round-trip, freshness checks, and aggregator
behavior. The runtime encoding path (``deposit_pair`` end-to-end) is
exercised separately by the smoke / sentinel runs — these tests stub it
via :func:`monkeypatch` so they exercise the IO contract without
training or loading real artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from smiles_subword.tokenize.measure._cellmeta import CellMetaFields
from smiles_subword.tokenize.measure._cells import eval_split_sha
from smiles_subword.tokenize.measure._pairing import MatchedPair, PairKey, UnpairedCell
from smiles_subword.tokenize.measure.absorption import (
    ArmAbsorption,
    MatchedPairAbsorption,
    UnpairedAbsorption,
    compute_matched_pair_absorption,
    compute_unpaired_absorption,
)
from smiles_subword.tokenize.measure.absorption import io as absorption_io


def _arm(
    *,
    arm: str,
    cell_id: str,
    boundary: str = "nmb",
    absorbed: float = 0.80,
    cross: float | None = None,
    sha: str = "sha-A",
    eval_sha: str = "eval-A",
) -> ArmAbsorption:
    return ArmAbsorption(
        cell_id=cell_id,
        arm=arm,  # type: ignore[arg-type]
        boundary=boundary,  # type: ignore[arg-type]
        n_molecules=100,
        n_chunks=1000,
        n_absorbed=int(absorbed * 1000),
        n_cross_chunk_total=int((cross or 0.0) * 1000) if cross is not None else None,
        absorbed_fraction=absorbed,
        absorbed_ci=(absorbed - 0.01, absorbed + 0.01),
        cross_chunk_fraction=cross,
        cross_chunk_ci=(cross - 0.01, cross + 0.01) if cross is not None else None,
        training_corpus_sha=sha,
        eval_split_sha=eval_sha,
        bootstrap_seed=42,
        n_resamples=1000,
    )


def _matched_record(
    pair_key: str = "pubchem__v256_nmb",
    *,
    bpe_sha: str = "sha-bpe",
    ul_sha: str = "sha-ul",
    eval_sha: str = "eval-A",
) -> MatchedPairAbsorption:
    bpe = _arm(
        arm="bpe",
        cell_id="pubchem__smirk_gpe_v256_nmb",
        absorbed=0.80,
        sha=bpe_sha,
        eval_sha=eval_sha,
    )
    ul = _arm(
        arm="unigram",
        cell_id="pubchem__smirk_unigram_v256_nmb",
        absorbed=0.65,
        sha=ul_sha,
        eval_sha=eval_sha,
    )
    return compute_matched_pair_absorption(
        bpe,
        ul,
        pair_key=pair_key,
        tier="headline",
        corpus="pubchem",
        vocab_size=256,
        boundary="nmb",
    )


def _unpaired_record() -> UnpairedAbsorption:
    arm = _arm(
        arm="bpe",
        cell_id="zinc22__smirk_gpe_v2048_nmb",
        absorbed=0.70,
        sha="sha-cond",
    )
    return compute_unpaired_absorption(
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


def _write_meta(
    artifact_dir: Path, *, base_kind: str, merge_brackets: bool, sha: str
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": artifact_dir.name,
        "base_kind": base_kind,
        "vocab_size": 256,
        "training_corpus_sha": sha,
        "merge_brackets": merge_brackets,
        "split_structure": True,
    }
    (artifact_dir / "meta.yaml").write_text(yaml.safe_dump(payload, sort_keys=False))


def _write_test_split_manifest(corpus_dir: Path, shas: list[str]) -> None:
    test_dir = corpus_dir / "canon_dedup_v1" / "test"
    test_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "canon_dedup_v1",
        "shards": [
            {
                "file": f"canon_dedup_v1-{i:05d}.parquet",
                "sha256": sha,
                "n_rows": 1,
                "n_bytes": 1,
            }
            for i, sha in enumerate(shas)
        ],
    }
    (test_dir / "MANIFEST.yaml").write_text(yaml.safe_dump(payload, sort_keys=False))


@pytest.fixture(autouse=True)
def _redirect_data_dirs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setattr(
        absorption_io, "ABSORPTION_DATA_DIR", tmp_path / "absorption_root"
    )
    monkeypatch.setattr(
        absorption_io,
        "ABSORPTION_CELL_DIR",
        tmp_path / "absorption_root" / "absorption",
    )
    monkeypatch.setattr(
        absorption_io,
        "ABSORPTION_TABLE_JSON",
        tmp_path / "absorption_root" / "absorption_table.json",
    )
    monkeypatch.setattr(
        absorption_io,
        "ABSORPTION_TABLE_MD",
        tmp_path / "absorption_root" / "absorption_table.md",
    )
    return tmp_path


class TestJsonPath:
    def test_path_is_under_the_absorption_cell_dir(self, tmp_path: Path) -> None:
        assert absorption_io.absorption_json_path("pubchem__v256_nmb") == (
            tmp_path / "absorption_root" / "absorption" / "pubchem__v256_nmb.json"
        )


class TestWriteReadRoundTrip:
    def test_matched_record_round_trip(self) -> None:
        record = _matched_record()

        absorption_io.write_absorption_json(record)
        payload = absorption_io.read_absorption_json("pubchem__v256_nmb")

        assert payload is not None
        assert payload["schema_version"] == absorption_io.SCHEMA_VERSION
        assert payload["pair_status"] == "matched"
        assert payload["delta_absorbed"] == pytest.approx(0.15)
        assert payload["delta_cross_chunk"] is None

    def test_unpaired_record_round_trip(self) -> None:
        absorption_io.write_absorption_json(_unpaired_record())

        payload = absorption_io.read_absorption_json("zinc22__v2048_nmb")

        assert payload is not None
        assert payload["pair_status"] == "single_arm"
        assert payload["missing_arm"] == "unigram"
        assert payload["unigram"] is None
        assert payload["bpe"] is not None

    def test_write_leaves_no_tmp_file(self) -> None:
        path = absorption_io.write_absorption_json(_matched_record())

        siblings = list(path.parent.iterdir())

        assert siblings == [path]

    def test_read_returns_none_when_absent(self) -> None:
        assert absorption_io.read_absorption_json("nothing__here") is None

    def test_read_returns_none_for_corrupt_json(self, tmp_path: Path) -> None:
        path = absorption_io.absorption_json_path("bad__pair")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("not json {")

        assert absorption_io.read_absorption_json("bad__pair") is None


class TestIsAbsorptionDone:
    def test_true_when_meta_and_eval_split_match(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        artifacts_root = tmp_path / "artifacts"
        data_root = tmp_path / "data"
        monkeypatch.setattr(
            "smiles_subword.tokenize.measure._cellmeta.tokenizer_artifact_dir",
            lambda corpus, name: artifacts_root / corpus / name,
        )
        monkeypatch.setattr("smiles_subword.paths.DATA_DIR", data_root)
        _write_meta(
            artifacts_root / "pubchem" / "smirk_gpe_v256_nmb",
            base_kind="smirk_gpe",
            merge_brackets=False,
            sha="sha-bpe",
        )
        _write_meta(
            artifacts_root / "pubchem" / "smirk_unigram_v256_nmb",
            base_kind="smirk_unigram",
            merge_brackets=False,
            sha="sha-ul",
        )
        _write_test_split_manifest(data_root / "processed" / "pubchem", ["aa", "bb"])
        eval_sha = eval_split_sha("pubchem")
        absorption_io.write_absorption_json(_matched_record(eval_sha=eval_sha))

        assert absorption_io.is_absorption_done("pubchem__v256_nmb") is True

    def test_false_when_eval_split_sha_drifted(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        artifacts_root = tmp_path / "artifacts"
        data_root = tmp_path / "data"
        monkeypatch.setattr(
            "smiles_subword.tokenize.measure._cellmeta.tokenizer_artifact_dir",
            lambda corpus, name: artifacts_root / corpus / name,
        )
        monkeypatch.setattr("smiles_subword.paths.DATA_DIR", data_root)
        _write_meta(
            artifacts_root / "pubchem" / "smirk_gpe_v256_nmb",
            base_kind="smirk_gpe",
            merge_brackets=False,
            sha="sha-bpe",
        )
        _write_meta(
            artifacts_root / "pubchem" / "smirk_unigram_v256_nmb",
            base_kind="smirk_unigram",
            merge_brackets=False,
            sha="sha-ul",
        )
        _write_test_split_manifest(data_root / "processed" / "pubchem", ["aa", "bb"])
        absorption_io.write_absorption_json(_matched_record(eval_sha="eval-DIFFERENT"))

        assert absorption_io.is_absorption_done("pubchem__v256_nmb") is False

    def test_false_when_meta_sha_drifted(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        artifacts_root = tmp_path / "artifacts"
        data_root = tmp_path / "data"
        monkeypatch.setattr(
            "smiles_subword.tokenize.measure._cellmeta.tokenizer_artifact_dir",
            lambda corpus, name: artifacts_root / corpus / name,
        )
        monkeypatch.setattr("smiles_subword.paths.DATA_DIR", data_root)
        _write_meta(
            artifacts_root / "pubchem" / "smirk_gpe_v256_nmb",
            base_kind="smirk_gpe",
            merge_brackets=False,
            sha="sha-bpe-NEW",
        )
        _write_meta(
            artifacts_root / "pubchem" / "smirk_unigram_v256_nmb",
            base_kind="smirk_unigram",
            merge_brackets=False,
            sha="sha-ul",
        )
        _write_test_split_manifest(data_root / "processed" / "pubchem", ["aa", "bb"])
        eval_sha = eval_split_sha("pubchem")
        absorption_io.write_absorption_json(_matched_record(eval_sha=eval_sha))

        assert absorption_io.is_absorption_done("pubchem__v256_nmb") is False

    def test_false_when_meta_missing(self) -> None:
        absorption_io.write_absorption_json(_matched_record())

        assert absorption_io.is_absorption_done("pubchem__v256_nmb") is False


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

        def fake_unpaired_record(u: UnpairedCell) -> UnpairedAbsorption:
            del u
            return _unpaired_record()

        monkeypatch.setattr(absorption_io, "_unpaired_record", fake_unpaired_record)

        path, reason = absorption_io.deposit_unpaired(unpaired)

        assert reason is None
        assert path is not None
        payload = absorption_io.read_absorption_json("zinc22__v2048_nmb")
        assert payload is not None
        assert payload["pair_status"] == "single_arm"


class TestBuildTable:
    def test_aggregates_matched_and_unpaired_records(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from smiles_subword.tokenize.measure._pairing import MatchedPair

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
            absorption_io, "pair_all_cells", lambda: ([matched], [unpaired])
        )

        absorption_io.write_absorption_json(_matched_record())
        absorption_io.write_absorption_json(_unpaired_record())

        table_json_path, table_md_path = absorption_io.build_absorption_table()
        table = json.loads(table_json_path.read_text())

        assert table["n_pairs"] == 2
        assert table["n_matched_present"] == 1
        assert table["n_unpaired_present"] == 1
        assert table["pending"] == []
        assert "Matched pairs" in table_md_path.read_text()

    def test_lists_pending_when_payload_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from smiles_subword.tokenize.measure._pairing import MatchedPair

        matched = MatchedPair(
            key=PairKey(corpus="pubchem", vocab_size=512, boundary="nmb"),
            tier="headline",
            bpe_cell_id="pubchem__smirk_gpe_v512_nmb",
            unigram_cell_id="pubchem__smirk_unigram_v512_nmb",
        )
        monkeypatch.setattr(absorption_io, "pair_all_cells", lambda: ([matched], []))

        table_json_path, _ = absorption_io.build_absorption_table()
        table = json.loads(table_json_path.read_text())

        assert table["pending"] == ["pubchem__v512_nmb"]


def _meta(corpus: str = "pubchem", name: str = "smirk_gpe_v256_nmb") -> CellMetaFields:
    return CellMetaFields(
        corpus=corpus,
        name=name,
        artifact_dir=Path("/art") / corpus / name,
        boundary="nmb",
        training_corpus_sha="T",
    )


class TestArmFromCell:
    """The cell_id -> ArmAbsorption bridge: pends (string) on unresolved cells."""

    def test_returns_arm_on_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(absorption_io, "resolve_cell_meta", lambda _cid: _meta())
        monkeypatch.setattr(absorption_io, "load_cell_adapter", lambda _c, _n: object())
        sentinel = _arm(arm="bpe", cell_id="pubchem__x")
        monkeypatch.setattr(
            absorption_io, "run_arm_absorption", lambda *a, **k: sentinel
        )

        assert absorption_io._arm_from_cell("pubchem__x", "bpe") is sentinel

    def test_meta_error_propagates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            absorption_io, "resolve_cell_meta", lambda _cid: "no meta.yaml for x"
        )

        assert absorption_io._arm_from_cell("x", "bpe") == "no meta.yaml for x"

    def test_missing_adapter_returns_reason(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(absorption_io, "resolve_cell_meta", lambda _cid: _meta())

        def boom(_c: str, _n: str) -> object:
            raise FileNotFoundError("no tokenizer.json")

        monkeypatch.setattr(absorption_io, "load_cell_adapter", boom)

        assert "no tokenizer.json" in absorption_io._arm_from_cell("x", "bpe")


class TestRecordBuilders:
    """Deposit-vs-pend: error propagation, the boundary guard, success."""

    def _pair(self) -> MatchedPair:
        return MatchedPair(
            key=PairKey(corpus="pubchem", vocab_size=256, boundary="nmb"),
            tier="headline",
            bpe_cell_id="pubchem__smirk_gpe_v256_nmb",
            unigram_cell_id="pubchem__smirk_unigram_v256_nmb",
        )

    def test_matched_propagates_bpe_reason(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            absorption_io, "_arm_from_cell", lambda *_a, **_k: "no meta.yaml"
        )

        assert absorption_io._matched_pair_record(self._pair()) == "no meta.yaml"

    def test_matched_propagates_unigram_reason(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake(cell_id: str, arm: str) -> ArmAbsorption | str:
            return (
                _arm(arm="bpe", cell_id=cell_id)
                if arm == "bpe"
                else "no meta.yaml for unigram"
            )

        monkeypatch.setattr(absorption_io, "_arm_from_cell", fake)

        result = absorption_io._matched_pair_record(self._pair())
        assert result == "no meta.yaml for unigram"

    def test_matched_boundary_mismatch_pends(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake(cell_id: str, arm: str) -> ArmAbsorption:
            return _arm(
                arm=arm, cell_id=cell_id, boundary="nmb" if arm == "bpe" else "mb"
            )

        monkeypatch.setattr(absorption_io, "_arm_from_cell", fake)

        result = absorption_io._matched_pair_record(self._pair())
        assert isinstance(result, str)
        assert "boundary mismatch" in result

    def test_matched_success_builds_record(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            absorption_io,
            "_arm_from_cell",
            lambda _cid, arm: _arm(arm=arm, cell_id=arm),
        )

        record = absorption_io._matched_pair_record(self._pair())

        assert isinstance(record, MatchedPairAbsorption)
        assert record.pair_key == "pubchem__v256_nmb"

    def test_unpaired_propagates_reason(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            absorption_io, "_arm_from_cell", lambda *_a, **_k: "no meta.yaml"
        )
        unpaired = UnpairedCell(
            key=PairKey(corpus="zinc22", vocab_size=2048, boundary="nmb"),
            tier="conditional",
            cell_id="zinc22__smirk_gpe_v2048_nmb",
            arm="bpe",
            reason="conditional_negative_branch",
        )

        assert absorption_io._unpaired_record(unpaired) == "no meta.yaml"

    def test_unpaired_success_wraps_with_complement(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            absorption_io,
            "_arm_from_cell",
            lambda _cid, arm: _arm(arm=arm, cell_id=arm),
        )
        unpaired = UnpairedCell(
            key=PairKey(corpus="zinc22", vocab_size=2048, boundary="nmb"),
            tier="conditional",
            cell_id="zinc22__smirk_gpe_v2048_nmb",
            arm="bpe",
            reason="conditional_negative_branch",
        )

        record = absorption_io._unpaired_record(unpaired)

        assert isinstance(record, UnpairedAbsorption)
        assert record.missing_arm == "unigram"
        assert record.unpaired_reason == "conditional_negative_branch"
