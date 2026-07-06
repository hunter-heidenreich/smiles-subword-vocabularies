"""Tests for ``distribution_io`` (deposition + aggregator).

Validate write / read round-trip, freshness checks, and aggregator behavior.
The runtime encoding path (``deposit_pair`` end-to-end) is exercised separately
by the sentinel / sweep runs — these tests stub it via :func:`monkeypatch` so
they exercise the IO contract without training or loading real artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from smiles_subword.tokenize.measure._cellmeta import CellMetaFields
from smiles_subword.tokenize.measure._cells import eval_split_sha
from smiles_subword.tokenize.measure._pairing import MatchedPair, PairKey, UnpairedCell
from smiles_subword.tokenize.measure.distribution import (
    ArmDistribution,
    MatchedPairDistribution,
    UnpairedDistribution,
    compute_matched_pair_distribution,
    compute_unpaired_distribution,
)
from smiles_subword.tokenize.measure.distribution import io as distribution_io


def _arm(
    *,
    arm: str,
    cell_id: str,
    boundary: str = "nmb",
    d: float = 0.30,
    eta: float = 0.80,
    renyi: float = 0.70,
    v_effective: int = 256,
    live_token_count: int = 200,
    sha: str = "sha-A",
    eval_sha: str = "eval-A",
) -> ArmDistribution:
    return ArmDistribution(
        cell_id=cell_id,
        arm=arm,  # type: ignore[arg-type]
        boundary=boundary,  # type: ignore[arg-type]
        n_molecules=100,
        total_tokens=5000,
        vocab_size=v_effective + 3,
        v_effective=v_effective,
        live_token_count=live_token_count,
        d=d,
        d_ci=(d - 0.01, d + 0.01),
        eta=eta,
        eta_ci=(eta - 0.01, eta + 0.01),
        renyi=renyi,
        renyi_ci=(renyi - 0.01, renyi + 0.01),
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
    bpe_d: float = 0.40,
    ul_d: float = 0.25,
    bpe_v_effective: int = 256,
    ul_v_effective: int = 256,
) -> MatchedPairDistribution:
    bpe = _arm(
        arm="bpe",
        cell_id="pubchem__smirk_gpe_v256_nmb",
        d=bpe_d,
        sha=bpe_sha,
        eval_sha=eval_sha,
        v_effective=bpe_v_effective,
    )
    ul = _arm(
        arm="unigram",
        cell_id="pubchem__smirk_unigram_v256_nmb",
        d=ul_d,
        sha=ul_sha,
        eval_sha=eval_sha,
        v_effective=ul_v_effective,
    )
    return compute_matched_pair_distribution(
        bpe,
        ul,
        pair_key=pair_key,
        tier="headline",
        corpus="pubchem",
        vocab_size=256,
        boundary="nmb",
    )


def _unpaired_record() -> UnpairedDistribution:
    arm = _arm(arm="bpe", cell_id="zinc22__smirk_gpe_v2048_nmb", sha="sha-cond")
    return compute_unpaired_distribution(
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
        distribution_io, "DISTRIBUTION_DATA_DIR", tmp_path / "distribution_root"
    )
    monkeypatch.setattr(
        distribution_io,
        "DISTRIBUTION_CELL_DIR",
        tmp_path / "distribution_root" / "distribution",
    )
    monkeypatch.setattr(
        distribution_io,
        "DISTRIBUTION_TABLE_JSON",
        tmp_path / "distribution_root" / "distribution_table.json",
    )
    monkeypatch.setattr(
        distribution_io,
        "DISTRIBUTION_TABLE_MD",
        tmp_path / "distribution_root" / "distribution_table.md",
    )
    return tmp_path


class TestJsonPath:
    def test_path_is_under_the_distribution_cell_dir(self, tmp_path: Path) -> None:
        assert distribution_io.distribution_json_path("pubchem__v256_nmb") == (
            tmp_path / "distribution_root" / "distribution" / "pubchem__v256_nmb.json"
        )


class TestWriteReadRoundTrip:
    def test_matched_record_round_trip(self) -> None:
        distribution_io.write_distribution_json(_matched_record())
        payload = distribution_io.read_distribution_json("pubchem__v256_nmb")

        assert payload is not None
        assert payload["schema_version"] == distribution_io.SCHEMA_VERSION
        assert payload["pair_status"] == "matched"
        assert payload["delta_d"] == pytest.approx(0.15)
        assert payload["delta_d_exceeds_threshold"] is True
        assert payload["v_effective_consistent"] is True

    def test_unpaired_record_round_trip(self) -> None:
        distribution_io.write_distribution_json(_unpaired_record())
        payload = distribution_io.read_distribution_json("zinc22__v2048_nmb")

        assert payload is not None
        assert payload["pair_status"] == "single_arm"
        assert payload["missing_arm"] == "unigram"
        assert payload["unigram"] is None
        assert payload["bpe"] is not None
        assert payload["delta_d"] is None

    def test_write_leaves_no_tmp_file(self) -> None:
        path = distribution_io.write_distribution_json(_matched_record())

        assert list(path.parent.iterdir()) == [path]

    def test_read_returns_none_when_absent(self) -> None:
        assert distribution_io.read_distribution_json("nothing__here") is None

    def test_read_returns_none_for_corrupt_json(self) -> None:
        path = distribution_io.distribution_json_path("bad__pair")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("not json {")

        assert distribution_io.read_distribution_json("bad__pair") is None


class TestIsDistributionDone:
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
        distribution_io.write_distribution_json(_matched_record(eval_sha=eval_sha))

        assert distribution_io.is_distribution_done("pubchem__v256_nmb") is True

    def test_false_when_eval_split_sha_drifted(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._setup(monkeypatch, tmp_path)
        distribution_io.write_distribution_json(
            _matched_record(eval_sha="eval-DIFFERENT")
        )

        assert distribution_io.is_distribution_done("pubchem__v256_nmb") is False

    def test_false_when_meta_sha_drifted(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._setup(monkeypatch, tmp_path)
        eval_sha = eval_split_sha("pubchem")
        distribution_io.write_distribution_json(
            _matched_record(eval_sha=eval_sha, bpe_sha="sha-bpe-STALE")
        )

        assert distribution_io.is_distribution_done("pubchem__v256_nmb") is False

    def test_false_when_meta_missing(self) -> None:
        distribution_io.write_distribution_json(_matched_record())

        assert distribution_io.is_distribution_done("pubchem__v256_nmb") is False


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
            distribution_io, "pair_all_cells", lambda: ([matched], [unpaired])
        )

        distribution_io.write_distribution_json(_matched_record())
        distribution_io.write_distribution_json(_unpaired_record())

        table_json_path, table_md_path = distribution_io.build_distribution_table()
        table = json.loads(table_json_path.read_text())

        assert table["n_pairs"] == 2
        assert table["n_matched_present"] == 1
        assert table["n_unpaired_present"] == 1
        assert table["pending"] == []
        assert table["v_effective_violations"] == []
        assert "Matched pairs" in table_md_path.read_text()

    def test_flags_v_effective_violation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from smiles_subword.tokenize.measure._pairing import MatchedPair

        matched = MatchedPair(
            key=PairKey(corpus="pubchem", vocab_size=256, boundary="nmb"),
            tier="headline",
            bpe_cell_id="pubchem__smirk_gpe_v256_nmb",
            unigram_cell_id="pubchem__smirk_unigram_v256_nmb",
        )
        monkeypatch.setattr(distribution_io, "pair_all_cells", lambda: ([matched], []))

        distribution_io.write_distribution_json(
            _matched_record(bpe_v_effective=256, ul_v_effective=255)
        )

        table_json_path, _ = distribution_io.build_distribution_table()
        table = json.loads(table_json_path.read_text())

        assert table["v_effective_violations"] == ["pubchem__v256_nmb"]

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
        monkeypatch.setattr(distribution_io, "pair_all_cells", lambda: ([matched], []))

        table_json_path, _ = distribution_io.build_distribution_table()
        table = json.loads(table_json_path.read_text())

        assert table["pending"] == ["pubchem__v512_nmb"]


def _stub_arm_from_cell_seams(
    monkeypatch: pytest.MonkeyPatch,
    *,
    vocab_size: int = 260,
    special_ids: frozenset[int] = frozenset({0, 1, 2}),
) -> dict[str, object]:
    """Wire _arm_from_cell's seams and capture the v_effective it passes on."""
    monkeypatch.setattr(
        distribution_io,
        "resolve_cell_meta",
        lambda _cid: CellMetaFields(
            corpus="pubchem",
            name="n",
            artifact_dir=Path("/art"),
            boundary="nmb",
            training_corpus_sha="T",
        ),
    )
    monkeypatch.setattr(
        distribution_io,
        "load_cell_adapter",
        lambda _c, _n: SimpleNamespace(vocab_size=vocab_size),
    )
    monkeypatch.setattr(
        distribution_io, "collect_all_special_ids", lambda _a, _d: special_ids
    )
    captured: dict[str, object] = {}

    def fake_run(_adapter: object, **kw: object) -> ArmDistribution:
        captured.update(kw)
        return _arm(
            arm=kw["arm"],  # type: ignore[arg-type]
            cell_id=str(kw["cell_id"]),
            v_effective=int(kw["v_effective"]),  # type: ignore[call-overload]
        )

    monkeypatch.setattr(distribution_io, "run_arm_distribution", fake_run)
    return captured


class TestArmFromCellNormalizer:
    """The v_effective normalizer choice — the dead-glyph ΔD cancellation premise."""

    def test_default_normalizes_by_the_nominal_target_v(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Even though the realized adapter vocab is 260, a matched arm normalizes
        # by the nominal target V so both arms share |V| and dead glyphs cancel.
        captured = _stub_arm_from_cell_seams(monkeypatch, vocab_size=260)

        distribution_io._arm_from_cell("pubchem__x", "bpe", target_vocab_size=256)

        assert captured["v_effective"] == 256

    def test_realized_v_subtracts_in_vocab_specials(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The merge-exhaustion H-anchor normalizes by its realized non-special
        # count: vocab_size minus the specials that fall within the vocab range
        # (an out-of-range special id is not subtracted).
        captured = _stub_arm_from_cell_seams(
            monkeypatch, vocab_size=260, special_ids=frozenset({0, 1, 2, 300})
        )

        distribution_io._arm_from_cell(
            "x", "bpe", target_vocab_size=999, use_realized_v=True
        )

        assert captured["v_effective"] == 260 - 3  # 300 is out of range

    def test_meta_error_propagates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            distribution_io, "resolve_cell_meta", lambda _cid: "no meta.yaml for x"
        )

        result = distribution_io._arm_from_cell("x", "bpe", target_vocab_size=256)
        assert result == "no meta.yaml for x"

    def test_missing_adapter_returns_reason(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            distribution_io,
            "resolve_cell_meta",
            lambda _cid: CellMetaFields(
                corpus="pubchem",
                name="n",
                artifact_dir=Path("/art"),
                boundary="nmb",
                training_corpus_sha="T",
            ),
        )

        def boom(_c: str, _n: str) -> object:
            raise FileNotFoundError("no tokenizer.json")

        monkeypatch.setattr(distribution_io, "load_cell_adapter", boom)

        result = distribution_io._arm_from_cell("x", "bpe", target_vocab_size=256)
        assert isinstance(result, str)
        assert "no tokenizer.json" in result


class TestRecordBuilderErrorPropagation:
    """The deposit-vs-pend branches: an unresolved arm pends, not raises."""

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
            distribution_io, "_arm_from_cell", lambda *_a, **_k: "no meta.yaml"
        )

        assert distribution_io._matched_pair_record(self._pair()) == "no meta.yaml"

    def test_matched_propagates_unigram_reason(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake(cell_id: str, arm: str, **_k: object) -> ArmDistribution | str:
            return (
                _arm(arm="bpe", cell_id=cell_id)
                if arm == "bpe"
                else "no meta.yaml for unigram"
            )

        monkeypatch.setattr(distribution_io, "_arm_from_cell", fake)

        result = distribution_io._matched_pair_record(self._pair())
        assert result == "no meta.yaml for unigram"

    def test_matched_boundary_mismatch_pends(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake(cell_id: str, arm: str, **_k: object) -> ArmDistribution:
            return _arm(
                arm=arm, cell_id=cell_id, boundary="nmb" if arm == "bpe" else "mb"
            )

        monkeypatch.setattr(distribution_io, "_arm_from_cell", fake)

        result = distribution_io._matched_pair_record(self._pair())
        assert isinstance(result, str)
        assert "boundary mismatch" in result

    def test_unpaired_propagates_reason(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            distribution_io, "_arm_from_cell", lambda *_a, **_k: "no meta.yaml"
        )
        unpaired = UnpairedCell(
            key=PairKey(corpus="zinc22", vocab_size=2048, boundary="nmb"),
            tier="conditional",
            cell_id="zinc22__smirk_gpe_v2048_nmb",
            arm="bpe",
            reason="conditional_negative_branch",
        )

        assert distribution_io._unpaired_record(unpaired) == "no meta.yaml"


class TestRecordBuilderNormalizerWiring:
    """Which normalizer each record builder selects."""

    def _capture_arm_calls(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> list[dict[str, object]]:
        calls: list[dict[str, object]] = []

        def fake_arm(
            cell_id: str,
            arm: str,
            *,
            target_vocab_size: int,
            use_realized_v: bool = False,
        ) -> ArmDistribution:
            calls.append(
                {"arm": arm, "target": target_vocab_size, "realized": use_realized_v}
            )
            return _arm(arm=arm, cell_id=cell_id)

        monkeypatch.setattr(distribution_io, "_arm_from_cell", fake_arm)
        return calls

    def test_matched_uses_nominal_target_for_both_arms(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = self._capture_arm_calls(monkeypatch)
        pair = MatchedPair(
            key=PairKey(corpus="pubchem", vocab_size=256, boundary="nmb"),
            tier="headline",
            bpe_cell_id="pubchem__smirk_gpe_v256_nmb",
            unigram_cell_id="pubchem__smirk_unigram_v256_nmb",
        )

        distribution_io._matched_pair_record(pair)

        assert [c["arm"] for c in calls] == ["bpe", "unigram"]
        assert all(c["target"] == 256 and c["realized"] is False for c in calls)

    def test_merge_exhaustion_unpaired_uses_realized_v(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = self._capture_arm_calls(monkeypatch)
        unpaired = UnpairedCell(
            key=PairKey(
                corpus="real_space",
                vocab_size=50000,
                boundary="nmb",
                extras_kind="merge_exhaustion",
                extras_label="merge_exhaustion",
            ),
            tier="extras_merge_exhaustion",
            cell_id="real_space__smirk_gpe_v50000_nmb_merge_exhaustion",
            arm="bpe",
            reason="extras_single_arm_knob",
        )

        distribution_io._unpaired_record(unpaired)

        assert calls[0]["realized"] is True

    def test_non_merge_exhaustion_unpaired_uses_nominal_target(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = self._capture_arm_calls(monkeypatch)
        unpaired = UnpairedCell(
            key=PairKey(corpus="zinc22", vocab_size=2048, boundary="nmb"),
            tier="conditional",
            cell_id="zinc22__smirk_gpe_v2048_nmb",
            arm="bpe",
            reason="conditional_negative_branch",
        )

        distribution_io._unpaired_record(unpaired)

        assert calls[0]["realized"] is False
        assert calls[0]["target"] == 2048
