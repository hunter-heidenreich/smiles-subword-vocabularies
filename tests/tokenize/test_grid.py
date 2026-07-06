"""Tests for the tokenizer grid (`smiles_subword.tokenize.grid`).

Pin the frozen grid to code: exactly 44 cells with the
36/4/4 tier split, the 2 conditional cells, the committed manifest equal
to the generating rule, and the cell -> TokenizerConfig mapping correct for
both algorithm arms.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import yaml
from pydantic import ValidationError

if TYPE_CHECKING:
    from pathlib import Path

from smiles_subword.config import TokenizerConfig
from smiles_subword.paths import tokenizer_artifact_dir
from smiles_subword.tokenize.grid import (
    GridCell,
    GridManifest,
    cells_for_tier,
    enumerate_all,
    enumerate_conditional,
    enumerate_grid,
    grid_cell_to_config,
    load_grid_manifest,
    write_grid_manifest,
)


def _cell(algo: str = "bpe", boundary: str = "nmb") -> GridCell:
    return next(
        c for c in enumerate_grid() if c.algo == algo and c.boundary == boundary
    )


class TestEnumerateGrid:
    def test_yields_exactly_44_cells(self) -> None:
        assert len(enumerate_grid()) == 44

    def test_tier_breakdown_is_36_4_4(self) -> None:
        tiers = [c.tier for c in enumerate_grid()]

        assert tiers.count("headline") == 36
        assert tiers.count("sensitivity") == 4
        assert tiers.count("anchor") == 4

    def test_headline_is_the_full_cartesian_product(self) -> None:
        headline = {
            (c.algo, c.vocab_size, c.corpus, c.boundary)
            for c in enumerate_grid()
            if c.tier == "headline"
        }

        expected = {
            (algo, v, corpus, boundary)
            for algo in ("bpe", "unigram")
            for v in (256, 512, 1024)
            for corpus in ("pubchem", "zinc22", "coconut")
            for boundary in ("nmb", "mb")
        }
        assert headline == expected

    def test_sensitivity_cells_are_v2048_pubchem(self) -> None:
        sensitivity = [c for c in enumerate_grid() if c.tier == "sensitivity"]

        assert all(c.vocab_size == 2048 for c in sensitivity)
        assert all(c.corpus == "pubchem" for c in sensitivity)

    def test_anchor_cells_are_v1024_real_space(self) -> None:
        anchor = [c for c in enumerate_grid() if c.tier == "anchor"]

        assert all(c.vocab_size == 1024 for c in anchor)
        assert all(c.corpus == "real_space" for c in anchor)

    def test_cell_ids_are_unique(self) -> None:
        cell_ids = [c.cell_id for c in enumerate_grid()]

        assert len(set(cell_ids)) == len(cell_ids)


class TestEnumerateConditional:
    def test_yields_exactly_2_cells(self) -> None:
        assert len(enumerate_conditional()) == 2

    def test_cells_are_zinc22_bpe_v2048_conditional(self) -> None:
        cells = enumerate_conditional()

        assert all(c.corpus == "zinc22" for c in cells)
        assert all(c.algo == "bpe" for c in cells)
        assert all(c.vocab_size == 2048 for c in cells)
        assert all(c.tier == "conditional" for c in cells)

    def test_cells_cover_both_boundary_modes(self) -> None:
        assert {c.boundary for c in enumerate_conditional()} == {"nmb", "mb"}


class TestEnumerateAll:
    def test_is_grid_plus_conditional(self) -> None:
        assert enumerate_all() == enumerate_grid() + enumerate_conditional()

    def test_yields_46_cells_with_unique_ids(self) -> None:
        cell_ids = [c.cell_id for c in enumerate_all()]

        assert len(cell_ids) == 46
        assert len(set(cell_ids)) == 46


class TestGridCellNaming:
    def test_bpe_name_uses_the_gpe_engine_tag(self) -> None:
        cell = GridCell(
            algo="bpe",
            vocab_size=256,
            corpus="pubchem",
            boundary="nmb",
            tier="headline",
        )

        assert cell.name == "smirk_gpe_v256_nmb"

    def test_unigram_name_uses_the_unigram_engine_tag(self) -> None:
        cell = GridCell(
            algo="unigram",
            vocab_size=1024,
            corpus="zinc22",
            boundary="mb",
            tier="headline",
        )

        assert cell.name == "smirk_unigram_v1024_mb"

    def test_cell_id_prefixes_the_name_with_the_corpus(self) -> None:
        cell = GridCell(
            algo="bpe",
            vocab_size=512,
            corpus="real_space",
            boundary="mb",
            tier="anchor",
        )

        assert cell.cell_id == "real_space__smirk_gpe_v512_mb"

    def test_bpe_maps_to_smirk_gpe_kind(self) -> None:
        assert _cell(algo="bpe").kind == "smirk_gpe"

    def test_unigram_maps_to_smirk_unigram_kind(self) -> None:
        assert _cell(algo="unigram").kind == "smirk_unigram"


class TestGridCellValidation:
    def test_rejects_an_unknown_field(self) -> None:
        with pytest.raises(ValidationError):
            GridCell(
                algo="bpe",
                vocab_size=256,
                corpus="pubchem",
                boundary="nmb",
                tier="headline",
                bogus=1,
            )

    def test_rejects_an_unknown_corpus(self) -> None:
        with pytest.raises(ValidationError):
            GridCell(
                algo="bpe",
                vocab_size=256,
                corpus="chembl",
                boundary="nmb",
                tier="headline",
            )

    def test_is_frozen(self) -> None:
        cell = enumerate_grid()[0]

        with pytest.raises(ValidationError):
            cell.vocab_size = 999


class TestGridManifest:
    def test_committed_manifest_matches_the_enumeration(self) -> None:
        assert load_grid_manifest() == list(enumerate_all())

    def test_committed_manifest_carries_46_cells(self) -> None:
        assert len(load_grid_manifest()) == 46

    def test_write_grid_manifest_round_trips(self, tmp_path: Path) -> None:
        path = write_grid_manifest(tmp_path / "grid.yaml")

        assert load_grid_manifest(path) == list(enumerate_all())

    def test_manifest_rejects_an_unknown_key(self) -> None:
        with pytest.raises(ValidationError):
            GridManifest.model_validate({"version": 1, "cells": [], "bogus": True})


class TestCellsForTier:
    def test_no_tier_returns_every_committed_cell(self) -> None:
        assert cells_for_tier() == load_grid_manifest()

    @pytest.mark.parametrize(
        ("tier", "expected"),
        [
            pytest.param("headline", 36, id="headline"),
            pytest.param("sensitivity", 4, id="sensitivity"),
            pytest.param("anchor", 4, id="anchor"),
            pytest.param("conditional", 2, id="conditional"),
        ],
    )
    def test_tier_selects_the_expected_count(self, tier: str, expected: int) -> None:
        cells = cells_for_tier(tier)

        assert len(cells) == expected
        assert all(c.tier == tier for c in cells)

    def test_the_four_tiers_partition_the_grid(self) -> None:
        ids = [
            c.cell_id
            for tier in ("headline", "sensitivity", "anchor", "conditional")
            for c in cells_for_tier(tier)
        ]

        assert sorted(ids) == sorted(c.cell_id for c in enumerate_all())


class TestGridCellToConfig:
    def test_bpe_cell_maps_to_smirk_gpe(self) -> None:
        assert grid_cell_to_config(_cell(algo="bpe")).kind == "smirk_gpe"

    def test_unigram_cell_maps_to_smirk_unigram(self) -> None:
        assert grid_cell_to_config(_cell(algo="unigram")).kind == "smirk_unigram"

    def test_nmb_cell_disables_merge_brackets(self) -> None:
        assert grid_cell_to_config(_cell(boundary="nmb")).merge_brackets is False

    def test_mb_cell_enables_merge_brackets(self) -> None:
        assert grid_cell_to_config(_cell(boundary="mb")).merge_brackets is True

    def test_split_structure_is_true_for_every_cell(self) -> None:
        assert all(grid_cell_to_config(c).split_structure for c in enumerate_all())

    def test_training_input_points_at_canon_dedup_v1_train(self) -> None:
        cell = enumerate_grid()[0]
        cfg = grid_cell_to_config(cell)

        assert cfg.training_input is not None
        assert cfg.training_input.parts[-3:] == (
            cell.corpus,
            "canon_dedup_v1",
            "train",
        )

    def test_output_dir_follows_the_artifact_contract(self) -> None:
        cell = enumerate_grid()[0]
        cfg = grid_cell_to_config(cell)

        assert cfg.output_dir == tokenizer_artifact_dir(cell.corpus, cell.name)

    def test_no_arm_carries_a_target_len(self) -> None:
        # The post-train trim was removed: both arms ship the natural artifact,
        # so the config has no target_len concept at all (guards re-adding it).
        for algo in ("bpe", "unigram"):
            cfg = grid_cell_to_config(_cell(algo=algo))
            assert not hasattr(cfg, "target_len")

    @pytest.mark.parametrize("cell", enumerate_all(), ids=lambda c: c.cell_id)
    def test_every_cell_maps_to_a_valid_config(self, cell: GridCell) -> None:
        cfg = grid_cell_to_config(cell)

        assert isinstance(cfg, TokenizerConfig)
        assert cfg.name == cell.name
        assert cfg.vocab_size == cell.vocab_size


def test_written_manifest_is_yaml_with_a_version_header(tmp_path: Path) -> None:
    path = write_grid_manifest(tmp_path / "grid.yaml")
    payload = yaml.safe_load(path.read_text())

    assert payload["version"] == 1
    assert len(payload["cells"]) == 46
