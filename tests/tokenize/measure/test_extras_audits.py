"""Tests for ``smiles_subword.tokenize.measure.supplementary.extras_audits``.

The builders read trained tokenizer artifacts off disk. Here a synthetic
artifact tree is written to a tmp dir and ``tokenizer_artifact_dir`` /
``audit_path`` are redirected into it, so the full derivation runs end to end
against real files — the Unigram glyph-tuple loader, the ``meta.yaml`` reader,
the multi-glyph Jaccard, and the symmetric-difference count — without the large
committed artifacts present.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from smiles_subword.tokenize.measure.supplementary import extras_audits

# Per-cell synthetic multi-glyph vocabularies, keyed by artifact name. A
# single-glyph entry is a base piece (excluded from the multi-glyph set); a
# length->=2 entry is a learned piece the Jaccard compares.
_VOCABS: dict[str, list[list[str]]] = {
    # seed-cap: probe identical to its default-seed baseline (inert => J=1).
    "smirk_unigram_v1024_mb": [["C"], ["C", "C"], ["C", "O"]],
    "smirk_unigram_v1024_mb_seed_uncapped": [["C"], ["C", "C"], ["C", "O"]],
    # prune-schedule v256: probe drops one piece, adds one (sym-diff 2, J=1/3).
    "smirk_unigram_v256_mb": [["C", "C"], ["C", "N"]],
    "smirk_unigram_v256_mb_prune_shrink_0_9": [["C", "C"], ["N", "O"]],
    # prune-schedule v512: probe identical to baseline (J=1).
    "smirk_unigram_v512_mb": [["C", "C", "C"], ["O", "O"]],
    "smirk_unigram_v512_mb_prune_shrink_0_9": [["C", "C", "C"], ["O", "O"]],
}

_METAS: dict[str, dict[str, object]] = {
    "smirk_unigram_v1024_mb": {"seed_size": 1_000_000},
    "smirk_unigram_v256_mb": {"shrinking_factor": 0.75},
    "smirk_unigram_v512_mb": {"shrinking_factor": 0.75},
    "smirk_gpe_v50000_nmb_merge_exhaustion": {"vocab_size": 4331, "n_merges": 4172},
}


@pytest.fixture(autouse=True)
def _artifact_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Write a synthetic artifact tree and point the module's IO seams at it."""
    root = tmp_path / "tokenizer"

    def _dir(corpus: str, name: str) -> Path:
        d = root / corpus / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    for name, vocab in _VOCABS.items():
        d = _dir("pubchem", name)
        model = {"model": {"type": "Unigram", "vocab": [{"glyphs": g} for g in vocab]}}
        (d / "tokenizer.json").write_text(json.dumps(model))
    for name, meta in _METAS.items():
        corpus = "real_space" if name.startswith("smirk_gpe") else "pubchem"
        (_dir(corpus, name) / "meta.yaml").write_text(yaml.safe_dump(meta))

    monkeypatch.setattr(
        extras_audits,
        "tokenizer_artifact_dir",
        lambda corpus, name: root / corpus / name,
    )
    monkeypatch.setattr(
        extras_audits, "audit_path", lambda name: tmp_path / f"{name}.json"
    )


class TestSeedCapAudit:
    def test_identical_seed_pools_are_inert(self) -> None:
        out = extras_audits.build_seed_cap_audit()
        assert out["multi_glyph_jaccard"] == 1.0
        assert out["symmetric_difference_count"] == 0
        assert out["inert"] is True
        assert out["baseline_seed_size"] == 1_000_000
        assert (
            out["probe_seed_size"]
            == extras_audits.enumerate_seed_cap()[0].seed_size_override
        )


class TestPruneScheduleAudit:
    def test_per_v_jaccard_and_symmetric_difference(self) -> None:
        comps = {
            c["baseline_cell"]: c
            for c in extras_audits.build_prune_schedule_audit()["comparisons"]
        }
        v256 = comps["pubchem__smirk_unigram_v256_mb"]
        assert v256["multi_glyph_jaccard"] == pytest.approx(1 / 3)
        assert v256["multi_glyph_symmetric_difference_count"] == 2
        assert v256["baseline_shrinking_factor"] == 0.75
        v512 = comps["pubchem__smirk_unigram_v512_mb"]
        assert v512["multi_glyph_jaccard"] == 1.0


class TestMergeExhaustionAudit:
    def test_reads_realised_terminal_below_cap(self) -> None:
        out = extras_audits.build_merge_exhaustion_audit()
        assert out["vocab_size_realised"] == 4331
        assert out["vocab_size_cap"] == 50_000
        assert out["n_merges"] == 4172
        assert out["natural_termination"] is True


class TestWriteExtrasAudits:
    def test_writes_three_named_deposits(self) -> None:
        written = extras_audits.write_extras_audits()
        assert {p.name for p in written} == {
            "seed_cap.json",
            "prune_schedule.json",
            "merge_exhaustion.json",
        }
        for p in written:
            payload = json.loads(p.read_text())
            assert payload["schema_version"] == extras_audits.SCHEMA_VERSION
