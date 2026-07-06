"""Tests for ``smiles_subword.tokenize.measure.fertility.io`` (deposition + aggregator).

Validate write / read round-trip, freshness checks, and aggregator behavior.
The runtime encoding path (``deposit_pair`` end-to-end) is exercised
separately by the smoke / sentinel runs — these tests stub it via
:func:`monkeypatch` so they exercise the IO contract without training or
loading real artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from smiles_subword.tokenize.measure._cellmeta import CellMetaFields
from smiles_subword.tokenize.measure._cells import eval_split_sha
from smiles_subword.tokenize.measure._pairing import MatchedPair, PairKey, UnpairedCell
from smiles_subword.tokenize.measure.fertility import (
    ArmFertility,
    MatchedPairFertility,
    UnpairedFertility,
    compute_matched_pair_fertility,
    compute_unpaired_fertility,
)
from smiles_subword.tokenize.measure.fertility import io as fertility_io


def _arm(
    *,
    arm: str,
    cell_id: str,
    boundary: str = "nmb",
    fertility: float = 5.0,
    glyphs_per_token: float = 2.0,
    total_glyphs: int = 1000,
    sha: str = "sha-A",
    eval_sha: str = "eval-A",
) -> ArmFertility:
    return ArmFertility(
        cell_id=cell_id,
        arm=arm,  # type: ignore[arg-type]
        boundary=boundary,  # type: ignore[arg-type]
        n_molecules=100,
        total_tokens=int(fertility * 100),
        total_glyphs=total_glyphs,
        fertility_mean=fertility,
        fertility_ci=(fertility - 0.1, fertility + 0.1),
        glyphs_per_token_mean=glyphs_per_token,
        glyphs_per_token_ci=(glyphs_per_token - 0.05, glyphs_per_token + 0.05),
        tokens_per_molecule_variance=1.5,
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
    bpe_total_glyphs: int = 1000,
    ul_total_glyphs: int = 1000,
) -> MatchedPairFertility:
    bpe = _arm(
        arm="bpe",
        cell_id="pubchem__smirk_gpe_v256_nmb",
        fertility=4.0,
        sha=bpe_sha,
        eval_sha=eval_sha,
        total_glyphs=bpe_total_glyphs,
    )
    ul = _arm(
        arm="unigram",
        cell_id="pubchem__smirk_unigram_v256_nmb",
        fertility=5.0,
        sha=ul_sha,
        eval_sha=eval_sha,
        total_glyphs=ul_total_glyphs,
    )
    return compute_matched_pair_fertility(
        bpe,
        ul,
        pair_key=pair_key,
        tier="headline",
        corpus="pubchem",
        vocab_size=256,
        boundary="nmb",
    )


def _unpaired_record() -> UnpairedFertility:
    arm = _arm(arm="bpe", cell_id="zinc22__smirk_gpe_v2048_nmb", sha="sha-cond")
    return compute_unpaired_fertility(
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
    monkeypatch.setattr(fertility_io, "FERTILITY_DATA_DIR", tmp_path / "fertility_root")
    monkeypatch.setattr(
        fertility_io, "FERTILITY_CELL_DIR", tmp_path / "fertility_root" / "fertility"
    )
    monkeypatch.setattr(
        fertility_io,
        "FERTILITY_TABLE_JSON",
        tmp_path / "fertility_root" / "fertility_table.json",
    )
    monkeypatch.setattr(
        fertility_io,
        "FERTILITY_TABLE_MD",
        tmp_path / "fertility_root" / "fertility_table.md",
    )
    return tmp_path


class TestJsonPath:
    def test_path_is_under_the_fertility_cell_dir(self, tmp_path: Path) -> None:
        assert fertility_io.fertility_json_path("pubchem__v256_nmb") == (
            tmp_path / "fertility_root" / "fertility" / "pubchem__v256_nmb.json"
        )


class TestWriteReadRoundTrip:
    def test_matched_record_round_trip(self) -> None:
        fertility_io.write_fertility_json(_matched_record())
        payload = fertility_io.read_fertility_json("pubchem__v256_nmb")

        assert payload is not None
        assert payload["schema_version"] == fertility_io.SCHEMA_VERSION
        assert payload["pair_status"] == "matched"
        assert payload["delta_fertility"] == pytest.approx(-1.0)
        assert payload["total_glyphs_consistent"] is True

    def test_unpaired_record_round_trip(self) -> None:
        fertility_io.write_fertility_json(_unpaired_record())
        payload = fertility_io.read_fertility_json("zinc22__v2048_nmb")

        assert payload is not None
        assert payload["pair_status"] == "single_arm"
        assert payload["missing_arm"] == "unigram"
        assert payload["unigram"] is None
        assert payload["bpe"] is not None

    def test_write_leaves_no_tmp_file(self) -> None:
        path = fertility_io.write_fertility_json(_matched_record())

        assert list(path.parent.iterdir()) == [path]

    def test_read_returns_none_when_absent(self) -> None:
        assert fertility_io.read_fertility_json("nothing__here") is None

    def test_read_returns_none_for_corrupt_json(self) -> None:
        path = fertility_io.fertility_json_path("bad__pair")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("not json {")

        assert fertility_io.read_fertility_json("bad__pair") is None


class TestIsFertilityDone:
    def _setup(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
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
        return data_root

    def test_true_when_meta_and_eval_split_match(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._setup(monkeypatch, tmp_path)
        eval_sha = eval_split_sha("pubchem")
        fertility_io.write_fertility_json(_matched_record(eval_sha=eval_sha))

        assert fertility_io.is_fertility_done("pubchem__v256_nmb") is True

    def test_false_when_eval_split_sha_drifted(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._setup(monkeypatch, tmp_path)
        fertility_io.write_fertility_json(_matched_record(eval_sha="eval-DIFFERENT"))

        assert fertility_io.is_fertility_done("pubchem__v256_nmb") is False

    def test_false_when_meta_sha_drifted(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._setup(monkeypatch, tmp_path)
        eval_sha = eval_split_sha("pubchem")
        fertility_io.write_fertility_json(
            _matched_record(eval_sha=eval_sha, bpe_sha="sha-bpe-STALE")
        )

        assert fertility_io.is_fertility_done("pubchem__v256_nmb") is False

    def test_false_when_meta_missing(self) -> None:
        fertility_io.write_fertility_json(_matched_record())

        assert fertility_io.is_fertility_done("pubchem__v256_nmb") is False


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
            fertility_io, "_unpaired_record", lambda _u: _unpaired_record()
        )

        path, reason = fertility_io.deposit_unpaired(unpaired)

        assert reason is None
        assert path is not None
        payload = fertility_io.read_fertility_json("zinc22__v2048_nmb")
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
            fertility_io, "pair_all_cells", lambda **_kw: ([matched], [unpaired])
        )

        fertility_io.write_fertility_json(_matched_record())
        fertility_io.write_fertility_json(_unpaired_record())

        table_json_path, table_md_path = fertility_io.build_fertility_table()
        table = json.loads(table_json_path.read_text())

        assert table["n_pairs"] == 2
        assert table["n_matched_present"] == 1
        assert table["n_unpaired_present"] == 1
        assert table["pending"] == []
        assert table["glyph_invariant_violations"] == []
        assert "Matched pairs" in table_md_path.read_text()

    def test_flags_glyph_invariant_violation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from smiles_subword.tokenize.measure._pairing import MatchedPair

        matched = MatchedPair(
            key=PairKey(corpus="pubchem", vocab_size=256, boundary="nmb"),
            tier="headline",
            bpe_cell_id="pubchem__smirk_gpe_v256_nmb",
            unigram_cell_id="pubchem__smirk_unigram_v256_nmb",
        )
        monkeypatch.setattr(
            fertility_io, "pair_all_cells", lambda **_kw: ([matched], [])
        )

        fertility_io.write_fertility_json(
            _matched_record(bpe_total_glyphs=1001, ul_total_glyphs=1000)
        )

        table_json_path, _ = fertility_io.build_fertility_table()
        table = json.loads(table_json_path.read_text())

        assert table["glyph_invariant_violations"] == ["pubchem__v256_nmb"]

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
        monkeypatch.setattr(
            fertility_io, "pair_all_cells", lambda **_kw: ([matched], [])
        )

        table_json_path, _ = fertility_io.build_fertility_table()
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
    """``_arm_from_cell`` bridges a cell_id to ArmFertility, returning a
    pending-reason string (not raising) when the cell is unresolved."""

    def test_returns_arm_on_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(fertility_io, "resolve_cell_meta", lambda _cid: _meta())
        monkeypatch.setattr(fertility_io, "load_cell_adapter", lambda _c, _n: object())
        sentinel = _arm(arm="bpe", cell_id="pubchem__x")
        monkeypatch.setattr(fertility_io, "run_arm_fertility", lambda *a, **k: sentinel)

        assert fertility_io._arm_from_cell("pubchem__x", "bpe") is sentinel

    def test_propagates_meta_error_reason(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            fertility_io, "resolve_cell_meta", lambda _cid: "no meta.yaml for x"
        )

        assert fertility_io._arm_from_cell("x", "bpe") == "no meta.yaml for x"

    def test_returns_reason_when_adapter_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(fertility_io, "resolve_cell_meta", lambda _cid: _meta())

        def boom(_c: str, _n: str) -> object:
            raise FileNotFoundError("no tokenizer.json")

        monkeypatch.setattr(fertility_io, "load_cell_adapter", boom)

        assert "no tokenizer.json" in fertility_io._arm_from_cell("x", "bpe")


class TestMatchedPairRecord:
    """The matched record builder: error propagation, the boundary guard, success."""

    def _pair(self) -> MatchedPair:
        return MatchedPair(
            key=PairKey(corpus="pubchem", vocab_size=256, boundary="nmb"),
            tier="headline",
            bpe_cell_id="pubchem__smirk_gpe_v256_nmb",
            unigram_cell_id="pubchem__smirk_unigram_v256_nmb",
        )

    def test_propagates_a_bpe_arm_pending_reason(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            fertility_io, "_arm_from_cell", lambda _cid, _arm: "no meta.yaml"
        )

        assert fertility_io._matched_pair_record(self._pair()) == "no meta.yaml"

    def test_propagates_a_unigram_arm_pending_reason(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake(_cid: str, arm: str) -> ArmFertility | str:
            return (
                _arm(arm="bpe", cell_id="b")
                if arm == "bpe"
                else "no meta.yaml for unigram"
            )

        monkeypatch.setattr(fertility_io, "_arm_from_cell", fake)

        result = fertility_io._matched_pair_record(self._pair())
        assert result == "no meta.yaml for unigram"

    def test_boundary_mismatch_pends_rather_than_computing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake(_cid: str, arm: str) -> ArmFertility:
            return _arm(
                arm=arm,
                cell_id=arm,
                boundary="nmb" if arm == "bpe" else "mb",
            )

        monkeypatch.setattr(fertility_io, "_arm_from_cell", fake)

        result = fertility_io._matched_pair_record(self._pair())
        assert isinstance(result, str)
        assert "boundary mismatch" in result

    def test_success_builds_record(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            fertility_io,
            "_arm_from_cell",
            lambda _cid, arm: _arm(arm=arm, cell_id=arm),
        )

        record = fertility_io._matched_pair_record(self._pair())

        assert isinstance(record, MatchedPairFertility)
        assert record.pair_key == "pubchem__v256_nmb"
        assert record.boundary == "nmb"


class TestUnpairedRecord:
    def _cell(self) -> UnpairedCell:
        return UnpairedCell(
            key=PairKey(corpus="zinc22", vocab_size=2048, boundary="nmb"),
            tier="conditional",
            cell_id="zinc22__smirk_gpe_v2048_nmb",
            arm="bpe",
            reason="conditional_negative_branch",
        )

    def test_propagates_pending_reason(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            fertility_io, "_arm_from_cell", lambda _cid, _arm: "no meta.yaml"
        )

        assert fertility_io._unpaired_record(self._cell()) == "no meta.yaml"

    def test_success_wraps_present_arm_with_complementary_missing_arm(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            fertility_io, "_arm_from_cell", lambda _cid, arm: _arm(arm=arm, cell_id=arm)
        )

        record = fertility_io._unpaired_record(self._cell())

        assert isinstance(record, UnpairedFertility)
        assert record.missing_arm == "unigram"  # complement of the present bpe arm
        assert record.unpaired_reason == "conditional_negative_branch"
