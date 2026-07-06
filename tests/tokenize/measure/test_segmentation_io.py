"""Tests for ``segmentation_io`` (deposition + aggregator).

Validate write / read round-trip, the by-construction freshness logic, and
aggregator behavior. The runtime encode path (``deposit_pair`` end-to-end) is
exercised by the sentinel / sweep runs; these tests stub it via records built
in-process so they exercise the IO contract without loading real artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from smiles_subword.tokenize.measure._cellmeta import CellMetaFields
from smiles_subword.tokenize.measure._cells import eval_split_sha
from smiles_subword.tokenize.measure._pairing import MatchedPair, PairKey, UnpairedCell
from smiles_subword.tokenize.measure.segmentation import (
    ArmSegmentation,
    MatchedPairSegmentation,
    UnpairedSegmentation,
    compute_bpe_arm_segmentation,
    compute_matched_pair_segmentation,
    compute_unpaired_segmentation,
)
from smiles_subword.tokenize.measure.segmentation import io as segmentation_io


def _unigram_arm(
    *,
    cell_id: str = "pubchem__smirk_unigram_v256_nmb",
    boundary: str = "nmb",
    sha: str = "sha-ul",
    eval_sha: str = "eval-A",
    per_mol: float = 2.5,
    per_glyph: float = 0.30,
) -> ArmSegmentation:
    return ArmSegmentation(
        cell_id=cell_id,
        arm="unigram",
        boundary=boundary,  # type: ignore[arg-type]
        n_molecules=100,
        total_glyphs=4000,
        total_entropy_nats=per_mol * 100,
        entropy_per_molecule_mean=per_mol,
        entropy_per_molecule_ci=(per_mol - 0.1, per_mol + 0.1),
        entropy_per_glyph=per_glyph,
        entropy_per_glyph_ci=(per_glyph - 0.01, per_glyph + 0.01),
        verified_by_construction=False,
        training_corpus_sha=sha,
        eval_split_sha=eval_sha,
        bootstrap_seed=42,
        n_resamples=1000,
    )


def _bpe_arm(
    *,
    cell_id: str = "pubchem__smirk_gpe_v256_nmb",
    boundary: str = "nmb",
    sha: str = "sha-bpe",
) -> ArmSegmentation:
    return compute_bpe_arm_segmentation(
        cell_id=cell_id,
        boundary=boundary,  # type: ignore[arg-type]
        training_corpus_sha=sha,
    )


def _matched_record(
    pair_key: str = "pubchem__v256_nmb",
    *,
    bpe_sha: str = "sha-bpe",
    ul_sha: str = "sha-ul",
    eval_sha: str = "eval-A",
    bpe_override: ArmSegmentation | None = None,
) -> MatchedPairSegmentation:
    bpe = bpe_override or _bpe_arm(sha=bpe_sha)
    ul = _unigram_arm(sha=ul_sha, eval_sha=eval_sha)
    return compute_matched_pair_segmentation(
        bpe,
        ul,
        pair_key=pair_key,
        tier="headline",
        corpus="pubchem",
        vocab_size=256,
        boundary="nmb",
    )


def _unpaired_bpe_record() -> UnpairedSegmentation:
    arm = _bpe_arm(cell_id="zinc22__smirk_gpe_v2048_nmb", sha="sha-cond")
    return compute_unpaired_segmentation(
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


def _unpaired_unigram_record() -> UnpairedSegmentation:
    arm = _unigram_arm(
        cell_id="pubchem__smirk_unigram_v1024_mb__seed_uncapped", boundary="mb"
    )
    return compute_unpaired_segmentation(
        arm,
        pair_key="pubchem__v1024_mb__seed_uncapped",
        tier="extras_seed_cap",
        corpus="pubchem",
        vocab_size=1024,
        boundary="mb",
        extras_kind="seed_cap",
        extras_label="uncapped",
        missing_arm="bpe",
        unpaired_reason="extras_single_arm_knob",
    )


def _write_meta(artifact_dir: Path, *, sha: str) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": artifact_dir.name,
        "vocab_size": 256,
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
            {"file": f"s-{i:05d}.parquet", "sha256": sha, "n_rows": 1, "n_bytes": 1}
            for i, sha in enumerate(shas)
        ],
    }
    (test_dir / "MANIFEST.yaml").write_text(yaml.safe_dump(payload, sort_keys=False))


@pytest.fixture(autouse=True)
def _redirect_data_dirs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root = tmp_path / "segmentation_root"
    monkeypatch.setattr(segmentation_io, "SEGMENTATION_DATA_DIR", root)
    monkeypatch.setattr(segmentation_io, "SEGMENTATION_CELL_DIR", root / "segmentation")
    monkeypatch.setattr(
        segmentation_io, "SEGMENTATION_TABLE_JSON", root / "segmentation_table.json"
    )
    monkeypatch.setattr(
        segmentation_io, "SEGMENTATION_TABLE_MD", root / "segmentation_table.md"
    )


class TestJsonPath:
    def test_path_is_under_the_segmentation_cell_dir(self, tmp_path: Path) -> None:
        assert segmentation_io.segmentation_json_path("pubchem__v256_nmb") == (
            tmp_path / "segmentation_root" / "segmentation" / "pubchem__v256_nmb.json"
        )


class TestWriteReadRoundTrip:
    def test_matched_record_round_trip(self) -> None:
        segmentation_io.write_segmentation_json(_matched_record())
        payload = segmentation_io.read_segmentation_json("pubchem__v256_nmb")

        assert payload is not None
        assert payload["schema_version"] == segmentation_io.SCHEMA_VERSION
        assert payload["pair_status"] == "matched"
        assert payload["delta_entropy_per_molecule"] == pytest.approx(2.5)
        bpe = payload["bpe"]
        assert isinstance(bpe, dict)
        assert bpe["verified_by_construction"] is True
        assert bpe["total_entropy_nats"] == 0.0

    def test_unpaired_bpe_round_trip(self) -> None:
        segmentation_io.write_segmentation_json(_unpaired_bpe_record())
        payload = segmentation_io.read_segmentation_json("zinc22__v2048_nmb")

        assert payload is not None
        assert payload["pair_status"] == "single_arm"
        assert payload["missing_arm"] == "unigram"
        assert payload["unigram"] is None
        assert payload["delta_entropy_per_molecule"] is None

    def test_unpaired_unigram_round_trip(self) -> None:
        segmentation_io.write_segmentation_json(_unpaired_unigram_record())
        payload = segmentation_io.read_segmentation_json(
            "pubchem__v1024_mb__seed_uncapped"
        )

        assert payload is not None
        assert payload["missing_arm"] == "bpe"
        assert payload["bpe"] is None
        unigram = payload["unigram"]
        assert isinstance(unigram, dict)
        assert unigram["verified_by_construction"] is False

    def test_read_returns_none_when_absent(self) -> None:
        assert segmentation_io.read_segmentation_json("nothing__here") is None


class TestIsSegmentationDone:
    def _setup(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        artifacts_root = tmp_path / "artifacts"
        data_root = tmp_path / "data"
        monkeypatch.setattr(
            "smiles_subword.tokenize.measure._cellmeta.tokenizer_artifact_dir",
            lambda corpus, name: artifacts_root / corpus / name,
        )
        monkeypatch.setattr("smiles_subword.paths.DATA_DIR", data_root)
        _write_meta(artifacts_root / "pubchem" / "smirk_gpe_v256_nmb", sha="sha-bpe")
        _write_meta(artifacts_root / "pubchem" / "smirk_unigram_v256_nmb", sha="sha-ul")
        _write_test_split_manifest(data_root / "processed" / "pubchem", ["aa", "bb"])

    def test_true_when_meta_and_eval_split_match(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._setup(monkeypatch, tmp_path)
        eval_sha = eval_split_sha("pubchem")
        segmentation_io.write_segmentation_json(_matched_record(eval_sha=eval_sha))

        assert segmentation_io.is_segmentation_done("pubchem__v256_nmb") is True

    def test_false_when_unigram_eval_split_drifted(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._setup(monkeypatch, tmp_path)
        segmentation_io.write_segmentation_json(
            _matched_record(eval_sha="eval-DIFFERENT")
        )

        assert segmentation_io.is_segmentation_done("pubchem__v256_nmb") is False

    def test_false_when_bpe_meta_sha_drifted(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._setup(monkeypatch, tmp_path)
        eval_sha = eval_split_sha("pubchem")
        segmentation_io.write_segmentation_json(
            _matched_record(eval_sha=eval_sha, bpe_sha="sha-bpe-STALE")
        )

        assert segmentation_io.is_segmentation_done("pubchem__v256_nmb") is False

    def test_false_when_meta_missing(self) -> None:
        segmentation_io.write_segmentation_json(_matched_record())

        assert segmentation_io.is_segmentation_done("pubchem__v256_nmb") is False


class TestBuildTable:
    def test_aggregates_matched_and_unpaired_records(
        self, monkeypatch: pytest.MonkeyPatch
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
            segmentation_io, "pair_all_cells", lambda: ([matched], [unpaired])
        )

        segmentation_io.write_segmentation_json(_matched_record())
        segmentation_io.write_segmentation_json(_unpaired_bpe_record())

        table_json_path, table_md_path = segmentation_io.build_segmentation_table()
        table = json.loads(table_json_path.read_text())

        assert table["n_pairs"] == 2
        assert table["n_matched_present"] == 1
        assert table["n_unpaired_present"] == 1
        assert table["pending"] == []
        assert table["bpe_zero_violations"] == []
        assert "Matched pairs" in table_md_path.read_text()

    def test_flags_bpe_zero_violation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        matched = MatchedPair(
            key=PairKey(corpus="pubchem", vocab_size=256, boundary="nmb"),
            tier="headline",
            bpe_cell_id="pubchem__smirk_gpe_v256_nmb",
            unigram_cell_id="pubchem__smirk_unigram_v256_nmb",
        )
        monkeypatch.setattr(segmentation_io, "pair_all_cells", lambda: ([matched], []))

        tampered_bpe = ArmSegmentation(
            cell_id="pubchem__smirk_gpe_v256_nmb",
            arm="bpe",
            boundary="nmb",
            n_molecules=100,
            total_glyphs=4000,
            total_entropy_nats=5.0,
            entropy_per_molecule_mean=0.05,
            entropy_per_molecule_ci=(0.0, 0.1),
            entropy_per_glyph=0.001,
            entropy_per_glyph_ci=(0.0, 0.002),
            verified_by_construction=False,
            training_corpus_sha="sha-bpe",
            eval_split_sha="eval-A",
            bootstrap_seed=1,
            n_resamples=1000,
        )
        segmentation_io.write_segmentation_json(
            _matched_record(bpe_override=tampered_bpe)
        )

        table_json_path, _ = segmentation_io.build_segmentation_table()
        table = json.loads(table_json_path.read_text())

        assert table["bpe_zero_violations"] == ["pubchem__v256_nmb"]

    def test_lists_pending_when_payload_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        matched = MatchedPair(
            key=PairKey(corpus="pubchem", vocab_size=512, boundary="nmb"),
            tier="headline",
            bpe_cell_id="pubchem__smirk_gpe_v512_nmb",
            unigram_cell_id="pubchem__smirk_unigram_v512_nmb",
        )
        monkeypatch.setattr(segmentation_io, "pair_all_cells", lambda: ([matched], []))

        table_json_path, _ = segmentation_io.build_segmentation_table()
        table = json.loads(table_json_path.read_text())

        assert table["pending"] == ["pubchem__v512_nmb"]


def _meta(
    corpus: str = "pubchem", name: str = "smirk_unigram_v256_nmb"
) -> CellMetaFields:
    return CellMetaFields(
        corpus=corpus,
        name=name,
        artifact_dir=Path("/art") / corpus / name,
        boundary="nmb",
        training_corpus_sha="T",
    )


class TestArmFromCell:
    """The arm bridge: BPE is derived from meta alone (no load/encode); Unigram
    loads the adapter and streams; unresolved cells pend rather than raise."""

    def test_bpe_arm_is_zero_without_loading_the_adapter(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The BPE arm is zero by construction: it must not load the adapter or
        # touch the held-out split — a load here would be both wrong and slow.
        monkeypatch.setattr(segmentation_io, "resolve_cell_meta", lambda _cid: _meta())

        def must_not_load(_c: str, _n: str) -> object:
            raise AssertionError("the BPE arm must not load the adapter")

        monkeypatch.setattr(segmentation_io, "load_cell_adapter", must_not_load)

        arm = segmentation_io._arm_from_cell("pubchem__smirk_gpe_v256_nmb", "bpe")

        assert isinstance(arm, ArmSegmentation)
        assert arm.verified_by_construction is True
        assert arm.total_entropy_nats == 0.0

    def test_unigram_arm_loads_and_runs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(segmentation_io, "resolve_cell_meta", lambda _cid: _meta())
        monkeypatch.setattr(
            segmentation_io, "load_cell_adapter", lambda _c, _n: object()
        )
        sentinel = _unigram_arm()
        captured: dict[str, object] = {}

        def fake_run(_adapter: object, **kw: object) -> ArmSegmentation:
            captured.update(kw)
            return sentinel

        monkeypatch.setattr(segmentation_io, "run_arm_segmentation", fake_run)

        arm = segmentation_io._arm_from_cell("pubchem__x", "unigram")

        assert arm is sentinel
        assert captured["arm"] == "unigram"
        assert captured["tokenizer_json"] == Path(
            "/art/pubchem/smirk_unigram_v256_nmb/tokenizer.json"
        )

    def test_meta_error_propagates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            segmentation_io, "resolve_cell_meta", lambda _cid: "no meta.yaml for x"
        )

        assert segmentation_io._arm_from_cell("x", "bpe") == "no meta.yaml for x"

    def test_unigram_missing_adapter_returns_reason(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(segmentation_io, "resolve_cell_meta", lambda _cid: _meta())

        def boom(_c: str, _n: str) -> object:
            raise FileNotFoundError("no tokenizer.json")

        monkeypatch.setattr(segmentation_io, "load_cell_adapter", boom)

        assert "no tokenizer.json" in segmentation_io._arm_from_cell("x", "unigram")


class TestRecordBuilderErrorPropagation:
    """Deposit-vs-pend: an unresolved arm pends; a boundary mismatch pends."""

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
            segmentation_io, "_arm_from_cell", lambda *_a, **_k: "no meta.yaml"
        )

        assert segmentation_io._matched_pair_record(self._pair()) == "no meta.yaml"

    def test_matched_propagates_unigram_reason(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake(cell_id: str, arm: str) -> ArmSegmentation | str:
            return _bpe_arm() if arm == "bpe" else "no meta.yaml for unigram"

        monkeypatch.setattr(segmentation_io, "_arm_from_cell", fake)

        result = segmentation_io._matched_pair_record(self._pair())
        assert result == "no meta.yaml for unigram"

    def test_matched_boundary_mismatch_pends(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake(cell_id: str, arm: str) -> ArmSegmentation:
            return _bpe_arm() if arm == "bpe" else _unigram_arm(boundary="mb")

        monkeypatch.setattr(segmentation_io, "_arm_from_cell", fake)

        result = segmentation_io._matched_pair_record(self._pair())
        assert isinstance(result, str)
        assert "boundary mismatch" in result

    def test_matched_success_builds_record(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake(cell_id: str, arm: str) -> ArmSegmentation:
            return _bpe_arm() if arm == "bpe" else _unigram_arm()

        monkeypatch.setattr(segmentation_io, "_arm_from_cell", fake)

        record = segmentation_io._matched_pair_record(self._pair())

        assert isinstance(record, MatchedPairSegmentation)
        assert record.pair_key == "pubchem__v256_nmb"

    def test_unpaired_propagates_reason(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            segmentation_io, "_arm_from_cell", lambda *_a, **_k: "no meta.yaml"
        )
        unpaired = UnpairedCell(
            key=PairKey(corpus="zinc22", vocab_size=2048, boundary="nmb"),
            tier="conditional",
            cell_id="zinc22__smirk_gpe_v2048_nmb",
            arm="bpe",
            reason="conditional_negative_branch",
        )

        assert segmentation_io._unpaired_record(unpaired) == "no meta.yaml"

    def test_unpaired_success_wraps_present_arm_with_complement(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            segmentation_io, "_arm_from_cell", lambda _cid, _arm: _bpe_arm()
        )
        unpaired = UnpairedCell(
            key=PairKey(corpus="zinc22", vocab_size=2048, boundary="nmb"),
            tier="conditional",
            cell_id="zinc22__smirk_gpe_v2048_nmb",
            arm="bpe",
            reason="conditional_negative_branch",
        )

        record = segmentation_io._unpaired_record(unpaired)

        assert isinstance(record, UnpairedSegmentation)
        assert record.missing_arm == "unigram"  # complement of the present bpe arm
        assert record.unpaired_reason == "conditional_negative_branch"
