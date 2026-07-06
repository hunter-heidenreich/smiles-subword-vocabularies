"""Tests for ``smiles_subword.tokenize.measure.jaccard.io`` (deposit + aggregator)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from smiles_subword.tokenize.measure import _deposit
from smiles_subword.tokenize.measure._cellmeta import CellMetaFields
from smiles_subword.tokenize.measure._pairing import MatchedPair, PairKey, UnpairedCell
from smiles_subword.tokenize.measure.jaccard import (
    ArmJaccard,
    ArmJaccardInputs,
    JwMoleculeData,
    MatchedPairJaccard,
    UnpairedJaccard,
)
from smiles_subword.tokenize.measure.jaccard import io as jaccard_io


def _arm_inputs(arm: str, *, boundary: str = "nmb") -> ArmJaccardInputs:
    """A minimal ArmJaccardInputs for the record-builder tests."""
    jw = JwMoleculeData(
        n_molecules=2,
        mol_idx=np.asarray([0], dtype=np.int64),
        sub_local=np.asarray([0], dtype=np.int64),
        count=np.asarray([1.0], dtype=np.float64),
        local_tuples=(("C", "C"),),
    )
    return ArmJaccardInputs(
        cell_id=f"pubchem__{arm}",
        arm=arm,  # type: ignore[arg-type]
        boundary=boundary,  # type: ignore[arg-type]
        training_corpus_sha="T",
        eval_split_sha="E",
        multi_subwords=frozenset({("C", "C")}),
        structural_subwords=frozenset({("C", "C")}),
        bracket_internal_subwords=frozenset(),
        unseen_subwords=frozenset(),
        n_distinct_bracket_chunks=1,
        n_distinct_nonbracket_chunks=1,
        nonbracket_cap_bound=False,
        jw=jw,
        bootstrap_seed=7,
    )


def _arm(
    arm: str, *, cell_id: str, train_sha: str = "T", eval_sha: str = "E"
) -> ArmJaccard:
    return ArmJaccard(
        cell_id=cell_id,
        arm=arm,  # type: ignore[arg-type]
        boundary="nmb",
        n_multi_subwords=10,
        n_structural=7,
        n_bracket_internal=3,
        n_unseen=0,
        n_held_out_molecules=100,
        total_emitted_multi=500,
        n_distinct_bracket_chunks=42,
        n_distinct_nonbracket_chunks=999,
        nonbracket_cap_bound=False,
        training_corpus_sha=train_sha,
        eval_split_sha=eval_sha,
        bootstrap_seed=7,
        n_resamples=1000,
    )


def _matched(jaccard: float = 0.2, *, cap: bool = False) -> MatchedPairJaccard:
    bpe = _arm("bpe", cell_id="pubchem__smirk_gpe_v256_nmb")
    ul = _arm("unigram", cell_id="pubchem__smirk_unigram_v256_nmb")
    if cap:
        object.__setattr__(bpe, "nonbracket_cap_bound", True)
    return MatchedPairJaccard(
        pair_key="pubchem__v256_nmb",
        tier="headline",
        corpus="pubchem",
        vocab_size=256,
        boundary="nmb",
        extras_kind=None,
        extras_label=None,
        bpe=bpe,
        unigram=ul,
        jaccard=jaccard,
        jaccard_struct=jaccard + 0.01,
        weighted_jaccard=0.18,
        weighted_jaccard_ci=(0.17, 0.19),
        weighted_jaccard_struct=0.16,
        weighted_jaccard_struct_ci=(0.15, 0.17),
        jaccard_minus_struct=-0.01,
    )


@pytest.fixture(autouse=True)
def _isolate_data_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(jaccard_io, "JACCARD_CELL_DIR", tmp_path / "jaccard")
    monkeypatch.setattr(
        jaccard_io, "INVENTORY_DIR", tmp_path / "jaccard" / "_inventory"
    )
    monkeypatch.setattr(
        jaccard_io, "JACCARD_TABLE_JSON", tmp_path / "jaccard_table.json"
    )
    monkeypatch.setattr(jaccard_io, "JACCARD_TABLE_MD", tmp_path / "jaccard_table.md")


class TestJsonPath:
    def test_path_is_under_the_jaccard_cell_dir(self, tmp_path: Path) -> None:
        assert jaccard_io.jaccard_json_path("pubchem__v256_nmb") == (
            tmp_path / "jaccard" / "pubchem__v256_nmb.json"
        )


class TestJsonRoundTrip:
    def test_write_then_read_matched(self) -> None:
        path = jaccard_io.write_jaccard_json(_matched())

        payload = jaccard_io.read_jaccard_json("pubchem__v256_nmb")

        assert path.is_file()
        assert payload is not None
        assert payload["schema_version"] == jaccard_io.SCHEMA_VERSION
        assert payload["jaccard"] == pytest.approx(0.2)
        assert payload["bpe"]["cell_id"] == "pubchem__smirk_gpe_v256_nmb"

    def test_read_missing_returns_none(self) -> None:
        assert jaccard_io.read_jaccard_json("absent__v1_nmb") is None


class TestInventoryCachePath:
    def test_path_keyed_by_training_sha(self) -> None:
        path = jaccard_io.inventory_cache_path("abc123")
        assert path.name == "abc123.json"
        assert path.parent == jaccard_io.INVENTORY_DIR


class TestIsJaccardDone:
    def _deposit_and_monkeypatch(
        self, monkeypatch: pytest.MonkeyPatch, *, train_sha: str, eval_sha: str
    ) -> None:
        jaccard_io.write_jaccard_json(_matched())
        monkeypatch.setattr(_deposit, "eval_split_sha", lambda _c: eval_sha)
        monkeypatch.setattr(
            "smiles_subword.tokenize.measure._cellmeta.cell_meta",
            lambda _c, _n: {"training_corpus_sha": train_sha},
        )

    def test_true_when_shas_match(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._deposit_and_monkeypatch(monkeypatch, train_sha="T", eval_sha="E")
        assert jaccard_io.is_jaccard_done("pubchem__v256_nmb") is True

    def test_false_on_eval_sha_drift(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._deposit_and_monkeypatch(monkeypatch, train_sha="T", eval_sha="E2")
        assert jaccard_io.is_jaccard_done("pubchem__v256_nmb") is False

    def test_false_on_training_sha_drift(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._deposit_and_monkeypatch(monkeypatch, train_sha="T2", eval_sha="E")
        assert jaccard_io.is_jaccard_done("pubchem__v256_nmb") is False

    def test_false_when_record_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_deposit, "eval_split_sha", lambda _c: "E")
        assert jaccard_io.is_jaccard_done("pubchem__v256_nmb") is False


class TestDepositPair:
    def test_writes_record_when_inputs_resolve(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            jaccard_io, "_matched_pair_record", lambda _pair: _matched(jaccard=0.3)
        )
        pair = MatchedPair(
            key=PairKey(corpus="pubchem", vocab_size=256, boundary="nmb"),
            tier="headline",
            bpe_cell_id="pubchem__smirk_gpe_v256_nmb",
            unigram_cell_id="pubchem__smirk_unigram_v256_nmb",
        )

        path, reason = jaccard_io.deposit_pair(pair)

        assert reason is None
        assert path is not None
        assert path.is_file()

    def test_returns_reason_when_inputs_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            jaccard_io, "_matched_pair_record", lambda _pair: "no meta.yaml"
        )
        pair = MatchedPair(
            key=PairKey(corpus="pubchem", vocab_size=256, boundary="nmb"),
            tier="headline",
            bpe_cell_id="x",
            unigram_cell_id="y",
        )

        path, reason = jaccard_io.deposit_pair(pair)

        assert path is None
        assert reason == "no meta.yaml"


class TestBuildJaccardTable:
    def _unpaired(self) -> UnpairedJaccard:
        return UnpairedJaccard(
            pair_key="zinc22__v2048_nmb",
            tier="headline",
            corpus="zinc22",
            vocab_size=2048,
            boundary="nmb",
            extras_kind=None,
            extras_label=None,
            present_arm=_arm("bpe", cell_id="zinc22__smirk_gpe_v2048_nmb"),
            missing_arm="unigram",
            unpaired_reason="conditional_negative_branch",
        )

    def _patch_pairs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        matched = MatchedPair(
            key=PairKey(corpus="pubchem", vocab_size=256, boundary="nmb"),
            tier="headline",
            bpe_cell_id="pubchem__smirk_gpe_v256_nmb",
            unigram_cell_id="pubchem__smirk_unigram_v256_nmb",
        )
        unpaired = UnpairedCell(
            key=PairKey(corpus="zinc22", vocab_size=2048, boundary="nmb"),
            tier="headline",
            cell_id="zinc22__smirk_gpe_v2048_nmb",
            arm="bpe",
            reason="conditional_negative_branch",
        )
        monkeypatch.setattr(
            jaccard_io, "pair_all_cells", lambda **_kw: ([matched], [unpaired])
        )

    def test_aggregates_present_and_pending(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._patch_pairs(monkeypatch)
        jaccard_io.write_jaccard_json(_matched())

        table_json, table_md = jaccard_io.build_jaccard_table()
        payload = json.loads(table_json.read_text())

        assert payload["n_pairs"] == 2
        assert payload["n_present"] == 1
        assert payload["pending"] == ["zinc22__v2048_nmb"]
        assert table_md.is_file()

    def test_unpaired_row_and_cap_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._patch_pairs(monkeypatch)
        jaccard_io.write_jaccard_json(_matched(cap=True))
        jaccard_io.write_jaccard_json(self._unpaired())

        table_json, _ = jaccard_io.build_jaccard_table()
        payload = json.loads(table_json.read_text())

        assert payload["n_present"] == 2
        assert payload["n_unpaired_present"] == 1
        assert payload["nonbracket_cap_bound"] == ["pubchem__v256_nmb"]
        assert payload["pending"] == []


class TestTrainCounts:
    """``_train_counts`` remaps deposited F95 token-id counts to glyph tuples,
    excluding the single-glyph atomic base — the training rank-frequency input."""

    def test_remaps_ids_to_tuples_and_excludes_atomic(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        f95 = tmp_path / "f95.json"
        f95.write_text(
            json.dumps({"training_counts_by_id": {"5": 100, "6": 50, "3": 999}})
        )
        monkeypatch.setattr(jaccard_io, "f95_json_path", lambda _cid: f95)
        monkeypatch.setattr(
            jaccard_io, "tokenizer_artifact_dir", lambda _c, _n: tmp_path
        )
        # id 3 is a single-glyph atom; ids 5/6 are multi-glyph pieces.
        monkeypatch.setattr(
            jaccard_io,
            "glyph_tuple_map",
            lambda _d, _a: {3: ("C",), 5: ("C", "C"), 6: ("c", "c")},
        )

        counts = jaccard_io._train_counts(
            SimpleNamespace(cell_id="pubchem__x", arm="bpe")
        )

        assert counts == {("C", "C"): 100, ("c", "c"): 50}
        assert ("C",) not in counts  # atomic base excluded

    def test_none_when_f95_absent(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            jaccard_io, "f95_json_path", lambda _cid: tmp_path / "absent.json"
        )

        assert jaccard_io._train_counts(SimpleNamespace(cell_id="x", arm="bpe")) is None

    def test_none_when_counts_key_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        f95 = tmp_path / "f95.json"
        f95.write_text(json.dumps({"headline_clearance": 0.9}))  # no counts key
        monkeypatch.setattr(jaccard_io, "f95_json_path", lambda _cid: f95)

        assert jaccard_io._train_counts(SimpleNamespace(cell_id="x", arm="bpe")) is None


class TestArmInputsFromCell:
    """``_arm_inputs_from_cell`` bridges a cell_id to ArmJaccardInputs, returning
    a pending-reason string (not raising) when the cell is unresolved."""

    def test_returns_inputs_on_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            jaccard_io,
            "resolve_cell_meta",
            lambda _cid: CellMetaFields(
                corpus="pubchem",
                name="smirk_gpe_v256_nmb",
                artifact_dir=Path("/art"),
                boundary="nmb",
                training_corpus_sha="T",
            ),
        )
        monkeypatch.setattr(jaccard_io, "load_cell_adapter", lambda _c, _n: object())
        sentinel = _arm_inputs("bpe")
        monkeypatch.setattr(jaccard_io, "run_arm_jaccard", lambda *a, **k: sentinel)

        assert jaccard_io._arm_inputs_from_cell("pubchem__x", "bpe") is sentinel

    def test_propagates_meta_error_reason(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            jaccard_io, "resolve_cell_meta", lambda _cid: "no meta.yaml for x"
        )

        assert jaccard_io._arm_inputs_from_cell("x", "bpe") == "no meta.yaml for x"

    def test_returns_reason_when_adapter_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            jaccard_io,
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

        monkeypatch.setattr(jaccard_io, "load_cell_adapter", boom)

        assert "no tokenizer.json" in jaccard_io._arm_inputs_from_cell("x", "bpe")


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
            jaccard_io, "_arm_inputs_from_cell", lambda _cid, _arm: "no meta.yaml"
        )

        assert jaccard_io._matched_pair_record(self._pair()) == "no meta.yaml"

    def test_propagates_a_unigram_arm_pending_reason(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The bpe arm resolves but the unigram arm does not — its reason wins.
        def fake(_cid: str, arm: str) -> ArmJaccardInputs | str:
            return _arm_inputs("bpe") if arm == "bpe" else "no meta.yaml for unigram"

        monkeypatch.setattr(jaccard_io, "_arm_inputs_from_cell", fake)

        result = jaccard_io._matched_pair_record(self._pair())
        assert result == "no meta.yaml for unigram"

    def test_boundary_mismatch_pends_rather_than_computing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake(_cid: str, arm: str) -> ArmJaccardInputs:
            return _arm_inputs(arm, boundary="nmb" if arm == "bpe" else "mb")

        monkeypatch.setattr(jaccard_io, "_arm_inputs_from_cell", fake)

        result = jaccard_io._matched_pair_record(self._pair())
        assert isinstance(result, str)
        assert "boundary mismatch" in result

    def test_success_builds_record_and_deposits_characterization(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            jaccard_io, "_arm_inputs_from_cell", lambda _cid, arm: _arm_inputs(arm)
        )
        characterized: dict = {}
        monkeypatch.setattr(
            jaccard_io,
            "_deposit_pair_characterization",
            lambda pk, _b, _u: characterized.update(pair_key=pk),
        )

        record = jaccard_io._matched_pair_record(self._pair())

        assert isinstance(record, MatchedPairJaccard)
        assert record.pair_key == "pubchem__v256_nmb"
        assert record.boundary == "nmb"
        assert characterized["pair_key"] == "pubchem__v256_nmb"


class TestUnpairedRecord:
    def _cell(self) -> UnpairedCell:
        return UnpairedCell(
            key=PairKey(corpus="zinc22", vocab_size=2048, boundary="nmb"),
            tier="headline",
            cell_id="zinc22__smirk_gpe_v2048_nmb",
            arm="bpe",
            reason="conditional_negative_branch",
        )

    def test_propagates_pending_reason(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            jaccard_io, "_arm_inputs_from_cell", lambda _cid, _arm: "no meta.yaml"
        )

        assert jaccard_io._unpaired_record(self._cell()) == "no meta.yaml"

    def test_success_wraps_present_arm_with_complementary_missing_arm(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            jaccard_io, "_arm_inputs_from_cell", lambda _cid, arm: _arm_inputs(arm)
        )

        record = jaccard_io._unpaired_record(self._cell())

        assert isinstance(record, UnpairedJaccard)
        assert record.missing_arm == "unigram"  # complement of the present bpe arm
        assert record.unpaired_reason == "conditional_negative_branch"


class TestHoldoutCounts:
    def test_aggregates_emissions_and_keeps_zero_tail(self) -> None:
        """Counts sum across molecules; a never-emitted vocab piece stays at 0."""
        jw = JwMoleculeData(
            n_molecules=2,
            mol_idx=np.array([0, 0, 1]),
            sub_local=np.array([0, 1, 0]),
            count=np.array([3, 4, 2]),
            local_tuples=(("C", "C"), ("c", "c", "c")),
        )
        multi = frozenset({("C", "C"), ("c", "c", "c"), ("N", "S")})

        counts = jaccard_io._holdout_counts(multi, jw)

        assert counts == {("C", "C"): 5, ("c", "c", "c"): 4, ("N", "S"): 0}
