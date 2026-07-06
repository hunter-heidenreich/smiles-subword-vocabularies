"""Tests for ``absorption_runner`` (encode-and-classify)."""

from __future__ import annotations

import pytest

from smiles_subword.tokenize.adapters.smirk import SmirkAdapter
from smiles_subword.tokenize.measure import _cells
from smiles_subword.tokenize.measure.absorption import runner as absorption_runner

_FIXTURE_SMILES: tuple[str, ...] = (
    "CCO",
    "CC(=O)O",
    "c1ccccc1",
    "[NH4+]",
    "CC(=O)Oc1ccccc1C(=O)O",
)


@pytest.fixture
def atomic_adapter() -> SmirkAdapter:
    return SmirkAdapter.atomic()


class TestEncodeForAbsorption:
    def test_nmb_records_have_chunks_and_no_cross_chunk(
        self, atomic_adapter: SmirkAdapter
    ) -> None:
        per_mol = absorption_runner.encode_for_absorption(
            atomic_adapter, _FIXTURE_SMILES, boundary="nmb"
        )

        assert len(per_mol) == len(_FIXTURE_SMILES)
        for pm in per_mol:
            assert pm.n_chunks >= 1
            assert pm.n_cross_chunk is None

    def test_absorption_is_detected_per_char_aligned_offsets(
        self, atomic_adapter: SmirkAdapter
    ) -> None:
        # Exact golden pinning the offset<->chunk alignment: for c1ccccc1 the
        # Layer-B chunks are ['c','1','ccccc','1']; the atomic tokenizer emits one
        # token per character, so the three single-char chunks are absorbed and
        # the 5-char 'ccccc' run is not. A misaligned offset/chunk wiring would
        # miscount.
        (pm,) = absorption_runner.encode_for_absorption(
            atomic_adapter, ["c1ccccc1"], boundary="nmb"
        )

        assert pm.n_chunks == 4
        assert pm.n_absorbed == 3

    def test_mb_path_yields_integer_cross_chunk_per_molecule(
        self, atomic_adapter: SmirkAdapter
    ) -> None:
        per_mol = absorption_runner.encode_for_absorption(
            atomic_adapter, _FIXTURE_SMILES, boundary="mb"
        )

        for pm in per_mol:
            assert isinstance(pm.n_cross_chunk, int)

    def test_empty_smiles_iterator_yields_no_records(
        self, atomic_adapter: SmirkAdapter
    ) -> None:
        per_mol = absorption_runner.encode_for_absorption(
            atomic_adapter, [], boundary="nmb"
        )

        assert per_mol == []

    def test_batch_boundary_unchanged_by_batch_size(
        self, atomic_adapter: SmirkAdapter
    ) -> None:
        single = absorption_runner.encode_for_absorption(
            atomic_adapter, _FIXTURE_SMILES, boundary="nmb", batch_size=1
        )
        bulk = absorption_runner.encode_for_absorption(
            atomic_adapter, _FIXTURE_SMILES, boundary="nmb", batch_size=64
        )

        assert single == bulk


class TestRunArmAbsorption:
    def test_aggregates_held_out_stream(
        self, atomic_adapter: SmirkAdapter, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            _cells, "iter_smiles_from_parquet", lambda _dir: list(_FIXTURE_SMILES)
        )
        monkeypatch.setattr(
            absorption_runner, "eval_split_sha", lambda _corpus: "eval-X"
        )

        arm = absorption_runner.run_arm_absorption(
            atomic_adapter,
            cell_id="pubchem__smirk_base",
            corpus="pubchem",
            arm="bpe",
            boundary="nmb",
            training_corpus_sha="sha-A",
        )

        assert arm.n_molecules == len(_FIXTURE_SMILES)
        assert arm.eval_split_sha == "eval-X"
        assert 0.0 <= arm.absorbed_fraction <= 1.0
        assert arm.cross_chunk_fraction is None  # nmb
