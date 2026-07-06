"""Tests for ``deadzone_io`` (deposition + aggregator)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from smiles_subword.tokenize.audit import f95_io
from smiles_subword.tokenize.measure._pairing import (
    MatchedPair,
    PairKey,
    UnpairedCell,
)
from smiles_subword.tokenize.measure.deadzone import (
    ArmF95Slice,
    MatchedPairDeadzone,
    UnpairedDeadzone,
    compute_matched_pair_deadzone,
    compute_unpaired_deadzone,
)
from smiles_subword.tokenize.measure.deadzone import io as deadzone_io


def _slice(
    *,
    arm: str,
    cell_id: str,
    f100: float = 0.90,
    sha: str = "sha-A",
    unsafe: bool = False,
) -> ArmF95Slice:
    return ArmF95Slice(
        cell_id=cell_id,
        arm=arm,  # type: ignore[arg-type]
        clearance_by_n={50: 1.0, 100: f100, 200: 0.5},
        headline_clearance=f100,
        embedding_tail_unsafe=unsafe,
        training_corpus_sha=sha,
        v_observed=256,
        n_non_atomic=80,
    )


def _f95_payload_on_disk(
    cell_id: str,
    *,
    arm: str,
    f100: float = 0.90,
    sha: str = "sha-A",
    unsafe: bool = False,
) -> None:
    path = f95_io.f95_json_path(cell_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "schema_version": 1,
        "cell_id": cell_id,
        "algo": "bpe" if arm == "bpe" else "unigram",
        "vocab_size": 256,
        "corpus": "pubchem",
        "boundary": "nmb",
        "tier": "headline",
        "training_corpus_sha": sha,
        "arm": arm,
        "v_observed": 256,
        "n_non_atomic": 80,
        "n_corpus_tokens": 1000,
        "n_corpus_molecules": 100,
        "fp_thresholds": [],
        "clearance_by_n": {"50": 1.0, "100": f100, "200": 0.5},
        "headline_clearance": f100,
        "embedding_tail_unsafe": unsafe,
    }
    path.write_text(json.dumps(payload, sort_keys=True))


def _matched_record(
    pair_key: str = "pubchem__v256_nmb",
    *,
    bpe_f100: float = 0.90,
    ul_f100: float = 0.60,
    bpe_sha: str = "sha-bpe",
    ul_sha: str = "sha-ul",
) -> MatchedPairDeadzone:
    bpe = _slice(
        arm="bpe", cell_id="pubchem__smirk_gpe_v256_nmb", f100=bpe_f100, sha=bpe_sha
    )
    ul = _slice(
        arm="unigram",
        cell_id="pubchem__smirk_unigram_v256_nmb",
        f100=ul_f100,
        sha=ul_sha,
    )
    return compute_matched_pair_deadzone(
        bpe,
        ul,
        pair_key=pair_key,
        tier="headline",
        corpus="pubchem",
        vocab_size=256,
        boundary="nmb",
    )


def _unpaired_record(pair_key: str = "zinc22__v2048_nmb") -> UnpairedDeadzone:
    arm = _slice(arm="bpe", cell_id="zinc22__smirk_gpe_v2048_nmb", unsafe=True)
    return compute_unpaired_deadzone(
        arm,
        pair_key=pair_key,
        tier="conditional",
        corpus="zinc22",
        vocab_size=2048,
        boundary="nmb",
        extras_kind=None,
        extras_label=None,
        missing_arm="unigram",
        unpaired_reason="conditional_negative_branch",
    )


@pytest.fixture(autouse=True)
def _redirect_data_dirs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(deadzone_io, "DEADZONE_DATA_DIR", tmp_path)
    monkeypatch.setattr(deadzone_io, "DEADZONE_CELL_DIR", tmp_path / "deadzone")
    monkeypatch.setattr(
        deadzone_io, "DEADZONE_TABLE_JSON", tmp_path / "deadzone_table.json"
    )
    monkeypatch.setattr(
        deadzone_io, "DEADZONE_TABLE_MD", tmp_path / "deadzone_table.md"
    )
    monkeypatch.setattr(f95_io, "F95_CELL_DIR", tmp_path / "f95_root" / "f95")
    monkeypatch.setattr(
        f95_io, "F95_TABLE_JSON", tmp_path / "f95_root" / "f95_table.json"
    )
    monkeypatch.setattr(f95_io, "F95_TABLE_MD", tmp_path / "f95_root" / "f95_table.md")


class TestJsonPath:
    def test_path_is_under_the_deadzone_cell_dir(self, tmp_path: Path) -> None:
        assert deadzone_io.deadzone_json_path("pubchem__v256_nmb") == (
            tmp_path / "deadzone" / "pubchem__v256_nmb.json"
        )


class TestWriteReadRoundTrip:
    def test_matched_record_round_trip(self) -> None:
        record = _matched_record(bpe_f100=0.90, ul_f100=0.60)

        deadzone_io.write_deadzone_json(record)
        payload = deadzone_io.read_deadzone_json("pubchem__v256_nmb")

        assert payload is not None
        assert payload["schema_version"] == deadzone_io.SCHEMA_VERSION
        assert payload["pair_status"] == "matched"
        assert payload["headline_delta_f"] == pytest.approx(0.30)

    def test_unpaired_record_round_trip(self) -> None:
        deadzone_io.write_deadzone_json(_unpaired_record())

        payload = deadzone_io.read_deadzone_json("zinc22__v2048_nmb")

        assert payload is not None
        assert payload["pair_status"] == "single_arm"
        assert payload["missing_arm"] == "unigram"
        assert payload["unigram"] is None
        assert payload["bpe"] is not None

    def test_write_leaves_no_tmp_file(self) -> None:
        path = deadzone_io.write_deadzone_json(_matched_record())

        siblings = list(path.parent.iterdir())
        assert siblings == [path]

    def test_read_returns_none_when_absent(self) -> None:
        assert deadzone_io.read_deadzone_json("nothing__here") is None


class TestIsDeadzoneDone:
    def test_true_when_deposited_and_both_f95_shas_match(self) -> None:
        deadzone_io.write_deadzone_json(
            _matched_record(bpe_sha="sha-bpe", ul_sha="sha-ul")
        )
        _f95_payload_on_disk("pubchem__smirk_gpe_v256_nmb", arm="bpe", sha="sha-bpe")
        _f95_payload_on_disk(
            "pubchem__smirk_unigram_v256_nmb", arm="unigram", sha="sha-ul"
        )

        assert deadzone_io.is_deadzone_done("pubchem__v256_nmb") is True

    def test_false_when_bpe_sha_drifted(self) -> None:
        deadzone_io.write_deadzone_json(
            _matched_record(bpe_sha="sha-bpe", ul_sha="sha-ul")
        )
        _f95_payload_on_disk(
            "pubchem__smirk_gpe_v256_nmb", arm="bpe", sha="sha-bpe-NEW"
        )
        _f95_payload_on_disk(
            "pubchem__smirk_unigram_v256_nmb", arm="unigram", sha="sha-ul"
        )

        assert deadzone_io.is_deadzone_done("pubchem__v256_nmb") is False

    def test_false_when_underlying_f95_missing(self) -> None:
        deadzone_io.write_deadzone_json(
            _matched_record(bpe_sha="sha-bpe", ul_sha="sha-ul")
        )

        assert deadzone_io.is_deadzone_done("pubchem__v256_nmb") is False

    def test_false_when_deadzone_json_absent(self) -> None:
        assert deadzone_io.is_deadzone_done("anything") is False

    def test_single_arm_record_only_checks_present_arm(self) -> None:
        deadzone_io.write_deadzone_json(_unpaired_record())
        _f95_payload_on_disk(
            "zinc22__smirk_gpe_v2048_nmb", arm="bpe", sha="sha-A", unsafe=True
        )

        assert deadzone_io.is_deadzone_done("zinc22__v2048_nmb") is True


class TestDepositPair:
    def test_emits_pending_when_an_input_f95_is_missing(self) -> None:
        pair = MatchedPair(
            key=PairKey(corpus="pubchem", vocab_size=256, boundary="nmb"),
            tier="headline",
            bpe_cell_id="pubchem__smirk_gpe_v256_nmb",
            unigram_cell_id="pubchem__smirk_unigram_v256_nmb",
        )
        _f95_payload_on_disk("pubchem__smirk_gpe_v256_nmb", arm="bpe")

        path, reason = deadzone_io.deposit_pair(pair)

        assert path is None
        assert reason is not None
        assert "smirk_unigram" in reason

    def test_deposits_when_both_inputs_present(self) -> None:
        pair = MatchedPair(
            key=PairKey(corpus="pubchem", vocab_size=256, boundary="nmb"),
            tier="headline",
            bpe_cell_id="pubchem__smirk_gpe_v256_nmb",
            unigram_cell_id="pubchem__smirk_unigram_v256_nmb",
        )
        _f95_payload_on_disk("pubchem__smirk_gpe_v256_nmb", arm="bpe", f100=0.90)
        _f95_payload_on_disk(
            "pubchem__smirk_unigram_v256_nmb", arm="unigram", f100=0.60
        )

        path, reason = deadzone_io.deposit_pair(pair)

        assert path is not None
        assert reason is None
        payload = deadzone_io.read_deadzone_json("pubchem__v256_nmb")
        assert payload is not None
        assert payload["headline_delta_f"] == pytest.approx(0.30)


class TestDepositUnpaired:
    def test_deposits_single_arm_when_f95_present(self) -> None:
        unpaired = UnpairedCell(
            key=PairKey(corpus="zinc22", vocab_size=2048, boundary="nmb"),
            tier="conditional",
            cell_id="zinc22__smirk_gpe_v2048_nmb",
            arm="bpe",
            reason="conditional_negative_branch",
        )
        _f95_payload_on_disk("zinc22__smirk_gpe_v2048_nmb", arm="bpe", unsafe=True)

        path, reason = deadzone_io.deposit_unpaired(unpaired)

        assert path is not None
        assert reason is None
        payload = deadzone_io.read_deadzone_json("zinc22__v2048_nmb")
        assert payload is not None
        assert payload["missing_arm"] == "unigram"
        assert payload["any_arm_unsafe"] is True


class TestBuildDeadzoneTable:
    def test_lists_pairs_without_a_json_as_pending(self) -> None:
        deadzone_io.write_deadzone_json(_matched_record())

        table_json, _md = deadzone_io.build_deadzone_table()
        table = json.loads(table_json.read_text())

        assert table["n_pairs"] == 40  # large-V anchor is jaccard/fertility-only
        assert table["n_present"] == 1
        assert "pubchem__v256_nmb" not in table["pending"]
        assert len(table["pending"]) == 39

    def test_records_an_unsafe_pair_in_flagged_list(self) -> None:
        bpe = _slice(
            arm="bpe", cell_id="pubchem__smirk_gpe_v256_nmb", f100=0.30, unsafe=True
        )
        ul = _slice(
            arm="unigram",
            cell_id="pubchem__smirk_unigram_v256_nmb",
            f100=0.20,
            unsafe=False,
        )
        unsafe_record = compute_matched_pair_deadzone(
            bpe,
            ul,
            pair_key="pubchem__v256_nmb",
            tier="headline",
            corpus="pubchem",
            vocab_size=256,
            boundary="nmb",
        )
        deadzone_io.write_deadzone_json(unsafe_record)

        table_json, _md = deadzone_io.build_deadzone_table()
        table = json.loads(table_json.read_text())

        assert "pubchem__v256_nmb" in table["flagged_pairs"]

    def test_writes_a_markdown_table(self) -> None:
        deadzone_io.write_deadzone_json(_matched_record())
        deadzone_io.write_deadzone_json(_unpaired_record())

        _json, table_md = deadzone_io.build_deadzone_table()
        text = table_md.read_text()

        assert "pubchem__v256_nmb" in text
        assert "zinc22__v2048_nmb" in text
        assert "Single-arm" in text


class TestDepositAll:
    def test_walks_committed_manifest_and_reports_pending(self) -> None:
        deposited, pending = deadzone_io.deposit_all()

        assert deposited == []
        assert len(pending) == 40  # large-V anchor is jaccard/fertility-only

    def test_only_pair_keys_filters_the_walk(self) -> None:
        only = frozenset({"pubchem__v256_nmb"})
        _f95_payload_on_disk("pubchem__smirk_gpe_v256_nmb", arm="bpe")
        _f95_payload_on_disk("pubchem__smirk_unigram_v256_nmb", arm="unigram")

        deposited, pending = deadzone_io.deposit_all(only_pair_keys=only)

        assert deposited == ["pubchem__v256_nmb"]
        assert pending == []

    def test_caches_fresh_pairs_on_re_run(self) -> None:
        _f95_payload_on_disk("pubchem__smirk_gpe_v256_nmb", arm="bpe")
        _f95_payload_on_disk("pubchem__smirk_unigram_v256_nmb", arm="unigram")
        only = frozenset({"pubchem__v256_nmb"})

        deadzone_io.deposit_all(only_pair_keys=only)
        deposited_again, pending_again = deadzone_io.deposit_all(only_pair_keys=only)

        assert deposited_again == ["pubchem__v256_nmb"]
        assert pending_again == []
