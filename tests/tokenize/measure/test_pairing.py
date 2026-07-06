"""Tests for ``smiles_subword.tokenize.measure._pairing`` (the cross-arm matchmaker)."""

from __future__ import annotations

import pytest

from smiles_subword.tokenize.extras import ExtrasCell, load_extras_manifest
from smiles_subword.tokenize.grid import GridCell, load_grid_manifest
from smiles_subword.tokenize.measure._pairing import (
    PairKey,
    pair_all_cells,
    pair_extras_cells,
    pair_grid_cells,
)


def _gpe(corpus: str, v: int, boundary: str, tier: str = "headline") -> GridCell:
    return GridCell(
        algo="bpe", vocab_size=v, corpus=corpus, boundary=boundary, tier=tier
    )


def _unigram(corpus: str, v: int, boundary: str, tier: str = "headline") -> GridCell:
    return GridCell(
        algo="unigram", vocab_size=v, corpus=corpus, boundary=boundary, tier=tier
    )


class TestPairKeySlug:
    def test_grid_slug_strips_algo(self) -> None:
        key = PairKey(corpus="pubchem", vocab_size=256, boundary="nmb")

        assert key.slug == "pubchem__v256_nmb"

    def test_extras_slug_includes_kind_and_label(self) -> None:
        key = PairKey(
            corpus="pubchem",
            vocab_size=512,
            boundary="nmb",
            extras_kind="subsample_redraw",
            extras_label="r1",
        )

        assert key.slug == "pubchem__v512_nmb__subsample_r1"

    def test_size_sweep_slug_uses_size_prefix(self) -> None:
        key = PairKey(
            corpus="pubchem",
            vocab_size=512,
            boundary="nmb",
            extras_kind="size_sweep",
            extras_label="5m",
        )

        assert key.slug == "pubchem__v512_nmb__size_5m"

    def test_extras_kind_without_a_label_raises(self) -> None:
        # PairKey is a public dataclass; an extras coordinate must carry a label
        # for the suffix, so slug guards the missing-label misuse.
        key = PairKey(
            corpus="pubchem", vocab_size=512, boundary="nmb", extras_kind="size_sweep"
        )

        with pytest.raises(ValueError, match="must carry a label"):
            _ = key.slug

    def test_merge_exhaustion_slug_uses_label_only(self) -> None:
        key = PairKey(
            corpus="real_space",
            vocab_size=50000,
            boundary="nmb",
            extras_kind="merge_exhaustion",
            extras_label="merge_exhaustion",
        )

        assert key.slug == "real_space__v50000_nmb__merge_exhaustion"

    def test_large_v_anchor_slug_uses_label_only(self) -> None:
        key = PairKey(
            corpus="pubchem",
            vocab_size=8192,
            boundary="nmb",
            extras_kind="large_v_anchor",
            extras_label="convergence_anchor",
        )

        assert key.slug == "pubchem__v8192_nmb__convergence_anchor"


class TestPairGridCells:
    def test_pairs_a_bpe_with_a_unigram_at_the_same_coordinate(self) -> None:
        matched, unpaired = pair_grid_cells(
            [_gpe("pubchem", 256, "nmb"), _unigram("pubchem", 256, "nmb")]
        )

        assert unpaired == []
        assert len(matched) == 1
        pair = matched[0]
        assert pair.bpe_cell_id == "pubchem__smirk_gpe_v256_nmb"
        assert pair.unigram_cell_id == "pubchem__smirk_unigram_v256_nmb"
        assert pair.key.slug == "pubchem__v256_nmb"
        assert pair.tier == "headline"

    def test_singleton_emits_unpaired_with_conditional_branch_reason(self) -> None:
        matched, unpaired = pair_grid_cells(
            [_gpe("zinc22", 2048, "nmb", tier="conditional")]
        )

        assert matched == []
        assert len(unpaired) == 1
        cell = unpaired[0]
        assert cell.cell_id == "zinc22__smirk_gpe_v2048_nmb"
        assert cell.arm == "bpe"
        assert cell.reason == "conditional_negative_branch"

    def test_committed_manifest_yields_22_matched_and_2_unpaired(self) -> None:
        matched, unpaired = pair_grid_cells(load_grid_manifest())

        assert len(matched) == 22
        assert len(unpaired) == 2
        assert {u.cell_id for u in unpaired} == {
            "zinc22__smirk_gpe_v2048_nmb",
            "zinc22__smirk_gpe_v2048_mb",
        }
        assert all(u.reason == "conditional_negative_branch" for u in unpaired)

    def test_raises_when_a_coordinate_has_mismatched_tiers(self) -> None:
        cells = [
            _gpe("pubchem", 256, "nmb", tier="headline"),
            _unigram("pubchem", 256, "nmb", tier="sensitivity"),
        ]

        with pytest.raises(ValueError, match="tier mismatch"):
            pair_grid_cells(cells)

    def test_raises_when_a_coordinate_has_more_than_two_cells(self) -> None:
        cells = [
            _gpe("pubchem", 256, "nmb"),
            _unigram("pubchem", 256, "nmb"),
            _unigram("pubchem", 256, "nmb"),
        ]

        with pytest.raises(ValueError, match="more than 2 grid cells"):
            pair_grid_cells(cells)

    def test_raises_when_a_pair_is_not_one_bpe_and_one_unigram(self) -> None:
        # Two cells at a coordinate, but both BPE — a manifest-integrity bug the
        # matched path must reject rather than silently pick one.
        cells = [_gpe("pubchem", 256, "nmb"), _gpe("pubchem", 256, "nmb")]

        with pytest.raises(ValueError, match=r"one bpe \+ one unigram"):
            pair_grid_cells(cells)

    def test_ordering_is_deterministic(self) -> None:
        first, _ = pair_grid_cells(load_grid_manifest())
        again, _ = pair_grid_cells(load_grid_manifest())

        assert [p.key.slug for p in first] == [p.key.slug for p in again]


class TestPairExtrasCells:
    def test_subsample_redraw_pairs_by_label(self) -> None:
        cells = [
            ExtrasCell(
                extras_kind="subsample_redraw",
                algo="bpe",
                vocab_size=512,
                corpus="pubchem",
                boundary="nmb",
                label="r1",
                training_subdir="redraw_r1",
            ),
            ExtrasCell(
                extras_kind="subsample_redraw",
                algo="unigram",
                vocab_size=512,
                corpus="pubchem",
                boundary="nmb",
                label="r1",
                training_subdir="redraw_r1",
            ),
            ExtrasCell(
                extras_kind="subsample_redraw",
                algo="bpe",
                vocab_size=512,
                corpus="pubchem",
                boundary="nmb",
                label="r2",
                training_subdir="redraw_r2",
            ),
            ExtrasCell(
                extras_kind="subsample_redraw",
                algo="unigram",
                vocab_size=512,
                corpus="pubchem",
                boundary="nmb",
                label="r2",
                training_subdir="redraw_r2",
            ),
        ]

        matched, unpaired = pair_extras_cells(cells)

        assert unpaired == []
        assert len(matched) == 2
        r1 = next(m for m in matched if m.key.extras_label == "r1")
        assert "subsample_r1" in r1.bpe_cell_id
        assert "subsample_r1" in r1.unigram_cell_id

    def test_seed_cap_is_single_arm_knob(self) -> None:
        cells = [
            ExtrasCell(
                extras_kind="seed_cap",
                algo="unigram",
                vocab_size=1024,
                corpus="pubchem",
                boundary="mb",
                label="uncapped",
                seed_size_override=8_000_000,
            )
        ]

        matched, unpaired = pair_extras_cells(cells)

        assert matched == []
        assert len(unpaired) == 1
        cell = unpaired[0]
        assert cell.reason == "extras_single_arm_knob"
        assert cell.tier == "extras_seed_cap"

    def test_committed_manifest_yields_12_matched_and_4_unpaired(self) -> None:
        # Default drops the large-V convergence anchor (reported only by the
        # convergence measurements; see test below).
        matched, unpaired = pair_extras_cells(load_extras_manifest())

        assert len(matched) == 12
        assert len(unpaired) == 4
        unpaired_kinds = {u.key.extras_kind for u in unpaired}
        assert unpaired_kinds == {"seed_cap", "prune_schedule", "merge_exhaustion"}
        assert "large_v_anchor" not in {m.key.extras_kind for m in matched}

    def test_large_v_anchor_included_only_when_opted_in(self) -> None:
        matched, _ = pair_extras_cells(
            load_extras_manifest(), include_large_v_anchor=True
        )

        assert len(matched) == 13  # +1 for the large-V convergence anchor
        anchor = [m for m in matched if m.key.extras_kind == "large_v_anchor"]
        assert len(anchor) == 1
        assert anchor[0].key.slug == "pubchem__v8192_nmb__convergence_anchor"

    def test_raises_when_a_pair_able_kind_has_only_one_cell(self) -> None:
        cells = [
            ExtrasCell(
                extras_kind="size_sweep",
                algo="bpe",
                vocab_size=512,
                corpus="pubchem",
                boundary="nmb",
                label="5m",
                training_subdir="size_5m",
            )
        ]

        with pytest.raises(ValueError, match="pair-able extras kind"):
            pair_extras_cells(cells)

    def test_raises_when_a_single_arm_kind_has_two_cells(self) -> None:
        cells = [
            ExtrasCell(
                extras_kind="seed_cap",
                algo="unigram",
                vocab_size=1024,
                corpus="pubchem",
                boundary="mb",
                label="uncapped",
                seed_size_override=8_000_000,
            ),
            ExtrasCell(
                extras_kind="seed_cap",
                algo="bpe",
                vocab_size=1024,
                corpus="pubchem",
                boundary="mb",
                label="uncapped",
            ),
        ]

        with pytest.raises(ValueError, match="single-arm extras kind"):
            pair_extras_cells(cells)

    def test_raises_when_a_coordinate_has_more_than_two_cells(self) -> None:
        cells = [
            ExtrasCell(
                extras_kind="subsample_redraw",
                algo=algo,  # type: ignore[arg-type]
                vocab_size=512,
                corpus="pubchem",
                boundary="nmb",
                label="r1",
                training_subdir="redraw_r1",
            )
            for algo in ("bpe", "unigram", "unigram")
        ]

        with pytest.raises(ValueError, match="more than 2 extras cells"):
            pair_extras_cells(cells)


class TestPairAllCells:
    def test_combines_grid_and_extras_totals(self) -> None:
        # Default excludes the large-V convergence anchor.
        matched, unpaired = pair_all_cells()

        assert len(matched) == 34
        assert len(unpaired) == 6
        assert "large_v_anchor" not in {m.key.extras_kind for m in matched}

    def test_includes_large_v_anchor_when_opted_in(self) -> None:
        matched, unpaired = pair_all_cells(include_large_v_anchor=True)

        assert len(matched) == 35  # +1 for the large-V convergence anchor
        assert len(unpaired) == 6
