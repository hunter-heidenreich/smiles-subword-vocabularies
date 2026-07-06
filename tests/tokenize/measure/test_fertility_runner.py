"""Tests for ``fertility_runner`` (encode pass + held-out aggregation).

The glyph-count builder the runner uses lives in ``_glyphmap`` and is covered
by ``test_glyphmap``.
"""

from __future__ import annotations

import pytest

from smiles_subword.tokenize.adapters.smirk import SmirkAdapter
from smiles_subword.tokenize.measure import _cells
from smiles_subword.tokenize.measure.fertility import runner as fertility_runner

_FIXTURE_SMILES: tuple[str, ...] = (
    "CCO",
    "CC(=O)O",
    "c1ccccc1",
    "[NH4+]",
    "CC(=O)Oc1ccccc1C(=O)O",
)


class TestEncodeForFertility:
    def test_atomic_baseline_one_glyph_per_token_with_default_map(self) -> None:
        adapter = SmirkAdapter.atomic()

        per_mol = fertility_runner.encode_for_fertility(adapter, _FIXTURE_SMILES, {})

        assert len(per_mol) == len(_FIXTURE_SMILES)
        for pm in per_mol:
            assert pm.n_tokens >= 1
            assert pm.n_glyphs == pm.n_tokens

    def test_glyph_map_scales_glyph_counts(self) -> None:
        adapter = SmirkAdapter.atomic()
        ids = adapter.encode(_FIXTURE_SMILES[0])
        glyph_map = dict.fromkeys(ids, 3)

        per_mol = fertility_runner.encode_for_fertility(
            adapter, [_FIXTURE_SMILES[0]], glyph_map
        )

        assert per_mol[0].n_glyphs == 3 * per_mol[0].n_tokens

    def test_excludes_special_tokens_from_the_token_count(self) -> None:
        # Fertility counts content tokens, not bos/eos: a regression to
        # add_special_tokens=True would inflate every molecule's count by the
        # specials and silently raise the reported fertility.
        adapter = SmirkAdapter.atomic()
        smi = "CC(=O)Oc1ccccc1C(=O)O"

        per_mol = fertility_runner.encode_for_fertility(adapter, [smi], {})

        assert per_mol[0].n_tokens == len(adapter.encode(smi, add_special_tokens=False))
        assert per_mol[0].n_tokens < len(adapter.encode(smi, add_special_tokens=True))

    def test_empty_smiles_iterator_yields_no_records(self) -> None:
        adapter = SmirkAdapter.atomic()

        per_mol = fertility_runner.encode_for_fertility(adapter, [], {})

        assert per_mol == []

    def test_batch_size_does_not_change_counts(self) -> None:
        adapter = SmirkAdapter.atomic()

        single = fertility_runner.encode_for_fertility(
            adapter, _FIXTURE_SMILES, {}, batch_size=1
        )
        bulk = fertility_runner.encode_for_fertility(
            adapter, _FIXTURE_SMILES, {}, batch_size=64
        )

        assert single == bulk


class TestRunArmFertility:
    def test_aggregates_held_out_stream(self, monkeypatch: pytest.MonkeyPatch) -> None:
        adapter = SmirkAdapter.atomic()
        monkeypatch.setattr(
            _cells,
            "iter_smiles_from_parquet",
            lambda _dir: list(_FIXTURE_SMILES),
        )
        monkeypatch.setattr(
            fertility_runner, "eval_split_sha", lambda _corpus: "eval-X"
        )
        monkeypatch.setattr(fertility_runner, "glyph_count_map", lambda _dir, _arm: {})

        arm = fertility_runner.run_arm_fertility(
            adapter,
            cell_id="pubchem__smirk_base",
            corpus="pubchem",
            name="smirk_base",
            arm="bpe",
            boundary="nmb",
            training_corpus_sha="sha-A",
        )

        assert arm.n_molecules == len(_FIXTURE_SMILES)
        assert arm.total_glyphs == arm.total_tokens
        assert arm.eval_split_sha == "eval-X"
        assert arm.fertility_mean == pytest.approx(
            arm.total_tokens / len(_FIXTURE_SMILES)
        )
