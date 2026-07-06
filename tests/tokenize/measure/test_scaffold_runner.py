"""Tests for ``scaffold_runner`` — scaffold log + dispatch."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from smiles_subword.tokenize.measure.scaffold import runner as scaffold_runner
from smiles_subword.tokenize.measure.scaffold.runner import (
    ScaffoldLogMissingError,
    read_scaffold_log,
    read_scaffold_log_sha,
    run_arm_scaffold,
    scaffold_log_path,
)


def _scaffold_jsonl(*records: dict[str, Any], vocab_size: int = 162) -> str:
    base_alphabet = [[tid, f"g{tid}"] for tid in range(159)]
    header = {
        "format": "smirk-scaffold-log/v1",
        "min_frequency": 0,
        "vocab_size": vocab_size,
        "merge_brackets": False,
        "limit_alphabet": None,
        "base_alphabet": base_alphabet,
    }
    lines = [json.dumps(header), *[json.dumps(r) for r in records]]
    return "\n".join(lines) + "\n"


def _write_cell(
    artifacts_root: Path,
    *,
    corpus: str,
    name: str,
    base_kind: str,
    vocab_size: int,
    n_merges: int | None,
    training_corpus_sha: str = "deadbeef",
    merge_brackets: bool = False,
    scaffold_jsonl: str | None = None,
) -> Path:
    cell_dir = artifacts_root / "tokenizer" / corpus / name
    cell_dir.mkdir(parents=True, exist_ok=True)
    meta: dict[str, Any] = {
        "name": name,
        "base_kind": base_kind,
        "vocab_size": vocab_size,
        "training_corpus_sha": training_corpus_sha,
        "merge_brackets": merge_brackets,
        "split_structure": True,
    }
    if n_merges is not None:
        meta["n_merges"] = n_merges
    (cell_dir / "meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
    if scaffold_jsonl is not None:
        (cell_dir / "scaffold.jsonl").write_text(scaffold_jsonl)
    return cell_dir


@pytest.fixture
def patched_artifacts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Redirect ``tokenizer_artifact_dir`` to a tmp tree for both call sites."""
    artifacts_root = tmp_path / "artifacts"
    artifacts_root.mkdir()

    def _redirect(corpus: str, name: str) -> Path:
        return artifacts_root / "tokenizer" / corpus / name

    monkeypatch.setattr(scaffold_runner, "tokenizer_artifact_dir", _redirect)
    return artifacts_root


class TestScaffoldLogPath:
    def test_points_at_artifact_dir(self, patched_artifacts: Path) -> None:
        path = scaffold_log_path("pubchem", "smirk_gpe_v256_nmb")
        assert path == (
            patched_artifacts
            / "tokenizer"
            / "pubchem"
            / "smirk_gpe_v256_nmb"
            / "scaffold.jsonl"
        )


class TestReadScaffoldLogSha:
    def test_returns_none_when_log_absent(self, patched_artifacts: Path) -> None:
        _write_cell(
            patched_artifacts,
            corpus="pubchem",
            name="smirk_gpe_v256_nmb",
            base_kind="smirk_gpe",
            vocab_size=256,
            n_merges=97,
        )

        assert read_scaffold_log_sha("pubchem", "smirk_gpe_v256_nmb") is None

    def test_returns_sha256_when_log_present(self, patched_artifacts: Path) -> None:
        body = _scaffold_jsonl()
        _write_cell(
            patched_artifacts,
            corpus="pubchem",
            name="smirk_gpe_v256_nmb",
            base_kind="smirk_gpe",
            vocab_size=256,
            n_merges=97,
            scaffold_jsonl=body,
        )

        sha = read_scaffold_log_sha("pubchem", "smirk_gpe_v256_nmb")

        assert sha == hashlib.sha256(body.encode("utf-8")).hexdigest()


class TestReadScaffoldLog:
    def test_parses_present_log(self, patched_artifacts: Path) -> None:
        body = _scaffold_jsonl(
            {
                "step": 0,
                "pair": [45, 45],
                "new_id": 159,
                "new_token": "CC",
                "candidate_freq": 10,
                "standalone": [[45, 4], [159, 9]],
            }
        )
        _write_cell(
            patched_artifacts,
            corpus="pubchem",
            name="smirk_gpe_v256_nmb",
            base_kind="smirk_gpe",
            vocab_size=256,
            n_merges=97,
            scaffold_jsonl=body,
        )

        header, records = read_scaffold_log("pubchem", "smirk_gpe_v256_nmb")

        assert header.format == "smirk-scaffold-log/v1"
        assert len(records) == 1
        assert records[0].new_id == 159

    def test_raises_scaffold_log_missing(self, patched_artifacts: Path) -> None:
        _write_cell(
            patched_artifacts,
            corpus="pubchem",
            name="smirk_gpe_v256_nmb",
            base_kind="smirk_gpe",
            vocab_size=256,
            n_merges=97,
        )

        with pytest.raises(ScaffoldLogMissingError, match=r"no scaffold\.jsonl"):
            read_scaffold_log("pubchem", "smirk_gpe_v256_nmb")


class TestRunArmScaffold:
    def test_unigram_short_circuits_to_zero_record(
        self, patched_artifacts: Path
    ) -> None:
        _write_cell(
            patched_artifacts,
            corpus="pubchem",
            name="smirk_unigram_v256_nmb",
            base_kind="smirk_unigram",
            vocab_size=256,
            n_merges=None,
        )

        arm = run_arm_scaffold(
            cell_id="pubchem__smirk_unigram_v256_nmb",
            corpus="pubchem",
            name="smirk_unigram_v256_nmb",
            arm="unigram",
            boundary="nmb",
        )

        assert arm.arm == "unigram"
        assert arm.scaffold_count == 0
        assert arm.verified_by_construction is True

    def test_bpe_arm_reads_log_and_applies_criterion(
        self, patched_artifacts: Path
    ) -> None:
        body = _scaffold_jsonl(
            {
                "step": 0,
                "pair": [45, 45],
                "new_id": 159,
                "new_token": "CC",
                "candidate_freq": 10,
                "standalone": [[159, 0]],
            },
            {
                "step": 1,
                "pair": [159, 102],
                "new_id": 160,
                "new_token": "CCO",
                "candidate_freq": 3,
                "standalone": [[160, 5]],
            },
        )
        _write_cell(
            patched_artifacts,
            corpus="pubchem",
            name="smirk_gpe_v161_nmb",
            base_kind="smirk_gpe",
            vocab_size=161,
            n_merges=2,
            scaffold_jsonl=body,
        )

        arm = run_arm_scaffold(
            cell_id="pubchem__smirk_gpe_v161_nmb",
            corpus="pubchem",
            name="smirk_gpe_v161_nmb",
            arm="bpe",
            boundary="nmb",
        )

        assert arm.arm == "bpe"
        assert arm.threshold == 3
        assert arm.scaffold_count == 1
        assert arm.scaffold_log_sha is not None

    def test_bpe_arm_raises_when_log_absent(self, patched_artifacts: Path) -> None:
        _write_cell(
            patched_artifacts,
            corpus="pubchem",
            name="smirk_gpe_v256_nmb",
            base_kind="smirk_gpe",
            vocab_size=256,
            n_merges=97,
        )

        with pytest.raises(ScaffoldLogMissingError):
            run_arm_scaffold(
                cell_id="pubchem__smirk_gpe_v256_nmb",
                corpus="pubchem",
                name="smirk_gpe_v256_nmb",
                arm="bpe",
                boundary="nmb",
            )

    def test_raises_on_missing_training_corpus_sha(
        self, patched_artifacts: Path
    ) -> None:
        # Sibling of the n_merges guard, checked before the arm branch — a meta
        # without training_corpus_sha cannot be fingerprinted for freshness.
        cell_dir = patched_artifacts / "tokenizer" / "pubchem" / "smirk_gpe_v256_nmb"
        cell_dir.mkdir(parents=True, exist_ok=True)
        (cell_dir / "meta.yaml").write_text(
            yaml.safe_dump({"name": "smirk_gpe_v256_nmb", "vocab_size": 256})
        )

        with pytest.raises(TypeError, match="missing training_corpus_sha"):
            run_arm_scaffold(
                cell_id="pubchem__smirk_gpe_v256_nmb",
                corpus="pubchem",
                name="smirk_gpe_v256_nmb",
                arm="bpe",
                boundary="nmb",
            )

    def test_raises_on_missing_n_merges_for_bpe(self, patched_artifacts: Path) -> None:
        body = _scaffold_jsonl()
        _write_cell(
            patched_artifacts,
            corpus="pubchem",
            name="smirk_gpe_v256_nmb",
            base_kind="smirk_gpe",
            vocab_size=256,
            n_merges=None,
            scaffold_jsonl=body,
        )

        with pytest.raises(TypeError, match="missing n_merges"):
            run_arm_scaffold(
                cell_id="pubchem__smirk_gpe_v256_nmb",
                corpus="pubchem",
                name="smirk_gpe_v256_nmb",
                arm="bpe",
                boundary="nmb",
            )
