"""Tests for the robustness extras and sensitivity battery
(`smiles_subword.tokenize.extras`).

Pin the spec to code: the 28 structural-probe cells (12/4/8/1/2/1) plus the 2
large-V convergence-anchor cells plus the 62 sensitivity cells (2 shared anchors
+ 5 off-default OFAT ladders + the off-anchor cells of 3 interaction grids + 3
off-anchor BPE references), no overlap with the headline grid, the committed
manifest equal to the generating rule, and per-kind ``TokenizerConfig`` overrides
wired correctly.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import pytest
import yaml
from pydantic import ValidationError

if TYPE_CHECKING:
    from pathlib import Path

from smiles_subword.tokenize.extras import (
    EXTRAS_MANIFEST_PATH,
    MINFREQ_DEFAULT,
    MINFREQ_VALUES,
    MPL_DEFAULT,
    MPL_VALUES,
    PRUNE_SCHEDULE_SHRINKING_FACTOR,
    SEED_CAP_OVERRIDE,
    SEED_DEFAULT,
    SEED_VALUES,
    SENSITIVITY_BOUNDARY,
    SENSITIVITY_CORPUS,
    SENSITIVITY_VOCAB_SIZE,
    SHRINK_DEFAULT,
    SHRINK_VALUES,
    SUBITER_DEFAULT,
    SUBITER_VALUES,
    ExtrasCell,
    ExtrasManifest,
    cells_for_extras_kind,
    enumerate_all,
    enumerate_interaction_bpe_refs,
    enumerate_interaction_mpl_typology,
    enumerate_interaction_mpl_v,
    enumerate_interaction_subiter_shrink,
    enumerate_large_v_anchor,
    enumerate_merge_exhaustion,
    enumerate_minfreq_ladder,
    enumerate_mpl_ladder,
    enumerate_prune_schedule,
    enumerate_seed_cap,
    enumerate_seed_ladder,
    enumerate_sensitivity_anchor,
    enumerate_shrink_ladder,
    enumerate_size_matched,
    enumerate_size_sweep,
    enumerate_subiter_ladder,
    enumerate_subsample_redraws,
    extras_cell_to_config,
    extras_training_dir,
    load_extras_manifest,
    write_extras_manifest,
)
from smiles_subword.tokenize.grid import enumerate_all as enumerate_grid_all


def test_yields_expected_cell_total() -> None:
    """28 structural-probe cells (incl. 8 size-matched) + 2 large-V anchor cells
    + 62 sensitivity cells (59 swept + 3 off-anchor BPE interaction
    references)."""
    assert len(enumerate_all()) == 92


def test_group_breakdown() -> None:
    counts = Counter(c.extras_kind for c in enumerate_all())
    assert counts == {
        "subsample_redraw": 12,
        "size_sweep": 4,
        "size_matched": 8,
        "seed_cap": 1,
        "prune_schedule": 2,
        "merge_exhaustion": 1,
        "large_v_anchor": 2,
        "sensitivity_anchor": 2,
        "mpl_ladder": 5,
        "seed_ladder": 5,
        "subiter_ladder": 3,
        "shrink_ladder": 4,
        "minfreq_ladder": 4,
        "interaction_subiter_shrink": 12,
        "interaction_mpl_v": 12,
        "interaction_mpl_typology": 12,
        "interaction_bpe_ref": 3,
    }


def test_subsample_redraws_full_cartesian_product() -> None:
    cells = enumerate_subsample_redraws()
    assert len(cells) == 12
    assert {(c.corpus, c.algo, c.label) for c in cells} == {
        (corpus, algo, label)
        for corpus in ("zinc22", "pubchem")
        for algo in ("bpe", "unigram")
        for label in ("r1", "r2", "r3")
    }
    assert {c.vocab_size for c in cells} == {512}
    assert {c.boundary for c in cells} == {"nmb"}


def test_size_sweep_full_cartesian_product() -> None:
    cells = enumerate_size_sweep()
    assert len(cells) == 4
    assert {(c.algo, c.label) for c in cells} == {
        (algo, label) for algo in ("bpe", "unigram") for label in ("5m", "15m")
    }
    assert {c.corpus for c in cells} == {"pubchem"}
    assert {c.vocab_size for c in cells} == {512}
    assert {c.boundary for c in cells} == {"nmb"}


def test_size_matched_is_pubchem_zinc22_v1024_size700k() -> None:
    """8 cells: PubChem + ZINC-22 at V=1024, both arms, both boundaries, all
    trained on the size_700k (COCONUT-scale) subsample."""
    cells = enumerate_size_matched()
    assert len(cells) == 8
    assert {(c.corpus, c.algo, c.boundary) for c in cells} == {
        (corpus, algo, boundary)
        for corpus in ("pubchem", "zinc22")
        for algo in ("bpe", "unigram")
        for boundary in ("nmb", "mb")
    }
    assert {c.vocab_size for c in cells} == {1024}
    for c in cells:
        assert c.training_subdir == "size_700k"
        training_dir = str(extras_training_dir(c))
        assert "canon_dedup_v1_extras" in training_dir
        assert "size_700k" in training_dir


def test_seed_cap_targets_worst_case_unigram_cell() -> None:
    cells = enumerate_seed_cap()
    assert len(cells) == 1
    (cell,) = cells
    assert cell.algo == "unigram"
    assert cell.corpus == "pubchem"
    assert cell.boundary == "mb"
    assert cell.vocab_size == 1024
    assert cell.seed_size_override == SEED_CAP_OVERRIDE
    assert cell.shrinking_factor_override is None


def test_prune_schedule_targets_worst_case_unigram_cell() -> None:
    """V=256 is the headline probe; V=512 is the contingency cell.
    Both are PubChem MB Unigram with the coarsened shrinking_factor."""
    cells = enumerate_prune_schedule()
    assert len(cells) == 2
    assert {c.vocab_size for c in cells} == {256, 512}
    for cell in cells:
        assert cell.algo == "unigram"
        assert cell.corpus == "pubchem"
        assert cell.boundary == "mb"
        assert cell.shrinking_factor_override == PRUNE_SCHEDULE_SHRINKING_FACTOR
        assert cell.seed_size_override is None


def test_merge_exhaustion_is_real_space_bpe_nmb() -> None:
    cells = enumerate_merge_exhaustion()
    assert len(cells) == 1
    (cell,) = cells
    assert cell.algo == "bpe"
    assert cell.corpus == "real_space"
    assert cell.boundary == "nmb"
    assert cell.vocab_size == 50_000


def test_large_v_anchor_is_pubchem_both_arms_v8192_nmb() -> None:
    cells = enumerate_large_v_anchor()
    assert len(cells) == 2
    assert {c.algo for c in cells} == {"bpe", "unigram"}
    assert {c.corpus for c in cells} == {"pubchem"}
    assert {c.boundary for c in cells} == {"nmb"}
    assert {c.vocab_size for c in cells} == {8192}
    # Both arms land on the same coordinate, so the pair slug is well-formed.
    names = {c.name for c in cells}
    assert "smirk_gpe_v8192_nmb_convergence_anchor" in names
    assert "smirk_unigram_v8192_nmb_convergence_anchor" in names


def test_cell_ids_are_unique() -> None:
    ids = [c.cell_id for c in enumerate_all()]
    assert len(ids) == len(set(ids))


def test_no_overlap_with_grid_cell_ids() -> None:
    grid_ids = {c.cell_id for c in enumerate_grid_all()}
    extras_ids = {c.cell_id for c in enumerate_all()}
    assert grid_ids.isdisjoint(extras_ids)


def test_cell_tier_starts_with_extras_prefix() -> None:
    for cell in enumerate_all():
        assert cell.tier == f"extras_{cell.extras_kind}"
        assert cell.tier.startswith("extras_")


def test_committed_manifest_equals_enumeration() -> None:
    enumerated = list(enumerate_all())
    manifest = load_extras_manifest()
    assert manifest == enumerated


def test_write_extras_manifest_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "extras.yaml"
    write_extras_manifest(path)
    parsed = ExtrasManifest.from_yaml(path)
    assert parsed.cells == list(enumerate_all())


def test_extras_kind_filter() -> None:
    seed_cap_only = cells_for_extras_kind("seed_cap")
    assert len(seed_cap_only) == 1
    assert seed_cap_only[0].extras_kind == "seed_cap"
    all_cells = cells_for_extras_kind(None)
    assert len(all_cells) == 92


def test_cell_to_config_subsample_points_at_extras_training_dir() -> None:
    cell = next(c for c in enumerate_all() if c.extras_kind == "subsample_redraw")
    cfg = extras_cell_to_config(cell)
    assert cfg.training_input is not None
    assert "canon_dedup_v1_extras" in str(cfg.training_input)
    assert cell.training_subdir is not None
    assert cell.training_subdir in str(cfg.training_input)


def test_cell_to_config_size_sweep_points_at_extras_training_dir() -> None:
    cell = next(c for c in enumerate_all() if c.extras_kind == "size_sweep")
    cfg = extras_cell_to_config(cell)
    assert cfg.training_input is not None
    assert "canon_dedup_v1_extras" in str(cfg.training_input)


def test_cell_to_config_seed_cap_propagates_seed_size_override() -> None:
    (cell,) = enumerate_seed_cap()
    cfg = extras_cell_to_config(cell)
    assert cfg.kind == "smirk_unigram"
    assert cfg.seed_size == SEED_CAP_OVERRIDE
    assert cfg.shrinking_factor is None


def test_cell_to_config_prune_schedule_propagates_shrinking_factor() -> None:
    for cell in enumerate_prune_schedule():
        cfg = extras_cell_to_config(cell)
        assert cfg.kind == "smirk_unigram"
        assert cfg.shrinking_factor == PRUNE_SCHEDULE_SHRINKING_FACTOR
        assert cfg.seed_size is None


def test_cell_to_config_merge_exhaustion_shape() -> None:
    (cell,) = enumerate_merge_exhaustion()
    cfg = extras_cell_to_config(cell)
    assert cfg.kind == "smirk_gpe"
    assert cfg.vocab_size == 50_000


def test_no_extras_cell_carries_a_target_len() -> None:
    # The post-train trim was removed: every extras cell ships the natural
    # artifact, so no config carries a target_len concept (guards re-adding it).
    for cell in enumerate_all():
        assert not hasattr(extras_cell_to_config(cell), "target_len")


def test_extras_training_dir_redraw_paths_distinct_across_redraws() -> None:
    """Each redraw label drives a distinct training corpus directory."""
    pubchem_redraws = [
        c
        for c in enumerate_subsample_redraws()
        if c.corpus == "pubchem" and c.algo == "bpe"
    ]
    dirs = {extras_training_dir(c) for c in pubchem_redraws}
    assert len(dirs) == 3


def test_extras_training_dir_size_sweep_uses_one_dir_per_size() -> None:
    """5m and 15m must drive different on-disk corpora even though they share
    a `hash_domain` (nesting in the subsampling sense, not directory-sharing)."""
    pubchem_size_sweep = [
        c for c in enumerate_size_sweep() if c.corpus == "pubchem" and c.algo == "bpe"
    ]
    dirs = {extras_training_dir(c) for c in pubchem_size_sweep}
    assert len(dirs) == 2


def test_extras_training_dir_seed_cap_falls_back_to_headline_train() -> None:
    """Seed-cap, prune-schedule, and merge-exhaustion cells train on the
    headline `canon_dedup_v1/train` — their probe is the hyperparameter
    override, not the training corpus."""
    for cell in (
        *enumerate_seed_cap(),
        *enumerate_prune_schedule(),
        *enumerate_merge_exhaustion(),
    ):
        d = extras_training_dir(cell)
        assert "canon_dedup_v1/train" in str(d)
        assert "canon_dedup_v1_extras" not in str(d)


def test_committed_manifest_uses_relative_paths_only() -> None:
    """The on-disk manifest must not bake in any local filesystem paths."""
    raw = yaml.safe_load(EXTRAS_MANIFEST_PATH.read_text())
    payload = yaml.safe_dump(raw)
    assert "/Users/" not in payload
    assert "/home/" not in payload


def test_subsample_label_must_be_non_empty() -> None:
    with pytest.raises(ValidationError):
        ExtrasCell(
            extras_kind="subsample_redraw",
            algo="bpe",
            vocab_size=512,
            corpus="pubchem",
            boundary="nmb",
            label="",
            training_subdir="redraw_r1",
        )


def test_seed_size_override_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        ExtrasCell(
            extras_kind="seed_cap",
            algo="unigram",
            vocab_size=1024,
            corpus="pubchem",
            boundary="mb",
            label="uncapped",
            seed_size_override=0,
        )


def test_shrinking_factor_override_in_open_unit_interval() -> None:
    with pytest.raises(ValidationError):
        ExtrasCell(
            extras_kind="prune_schedule",
            algo="unigram",
            vocab_size=256,
            corpus="pubchem",
            boundary="mb",
            label="shrink_1_0",
            shrinking_factor_override=1.0,
        )


def test_sensitivity_anchor_is_two_all_default_cells() -> None:
    """One Unigram + one BPE anchor, every knob at its reference default."""
    cells = enumerate_sensitivity_anchor()
    assert {c.algo for c in cells} == {"unigram", "bpe"}
    for c in cells:
        assert (c.corpus, c.vocab_size, c.boundary) == (
            SENSITIVITY_CORPUS,
            SENSITIVITY_VOCAB_SIZE,
            SENSITIVITY_BOUNDARY,
        )
        assert c.max_piece_length_override is None
        assert c.seed_size_override is None
        assert c.n_sub_iterations_override is None
        assert c.shrinking_factor_override is None
        assert c.min_frequency_override is None


def test_mpl_ladder_sweeps_off_default_rungs() -> None:
    cells = enumerate_mpl_ladder()
    assert {c.max_piece_length_override for c in cells} == set(MPL_VALUES) - {
        MPL_DEFAULT
    }
    assert all(c.algo == "unigram" for c in cells)


def test_seed_ladder_sweeps_off_default_rungs() -> None:
    cells = enumerate_seed_ladder()
    assert all(c.algo == "unigram" for c in cells)
    assert {c.seed_size_override for c in cells} == set(SEED_VALUES) - {SEED_DEFAULT}


def test_subiter_ladder_sweeps_off_default_rungs() -> None:
    cells = enumerate_subiter_ladder()
    assert all(c.algo == "unigram" for c in cells)
    assert {c.n_sub_iterations_override for c in cells} == set(SUBITER_VALUES) - {
        SUBITER_DEFAULT
    }


def test_shrink_ladder_sweeps_off_default_rungs_with_float_labels() -> None:
    """The only ladder on a float knob: off-default shrinking_factor rungs,
    labelled via _flabel (0.9 -> "0_9")."""
    cells = enumerate_shrink_ladder()
    assert all(c.algo == "unigram" for c in cells)
    assert {c.shrinking_factor_override for c in cells} == set(SHRINK_VALUES) - {
        SHRINK_DEFAULT
    }
    labels = {c.label for c in cells}
    assert "0_9" in labels  # _flabel(0.9)
    assert "0_75" not in labels  # default excluded


def test_minfreq_ladder_includes_zero_floor() -> None:
    """The BPE merge-frequency ladder reaches 0 (no floor) — GpeTrainer's own
    default — which the relaxed config bound (`ge=0`) must admit."""
    cells = enumerate_minfreq_ladder()
    swept = {c.min_frequency_override for c in cells}
    assert swept == set(MINFREQ_VALUES) - {MINFREQ_DEFAULT}
    assert 0 in swept
    assert all(c.algo == "bpe" for c in cells)


def test_interaction_subiter_shrink_is_off_default_interior() -> None:
    """A's 12 interior cells: both knobs off-default (the default row/column
    live in the subiter and shrink ladders, not here)."""
    cells = enumerate_interaction_subiter_shrink()
    assert len(cells) == 12
    for c in cells:
        assert c.n_sub_iterations_override not in (None, 2)
        assert c.shrinking_factor_override not in (None, 0.75)


def test_interaction_mpl_v_covers_off_anchor_vocab_rows() -> None:
    cells = enumerate_interaction_mpl_v()
    assert {c.vocab_size for c in cells} == {256, 1024}
    assert {c.max_piece_length_override for c in cells} == set(MPL_VALUES)


def test_interaction_mpl_typology_matches_coconut_on_full_train() -> None:
    """ZINC-22 uses its size_700k subsample; COCONUT (already ~702K) trains on
    its full headline corpus, so its cells carry no extras subdir."""
    cells = enumerate_interaction_mpl_typology()
    assert {c.corpus for c in cells} == {"zinc22", "coconut"}
    for c in (c for c in cells if c.corpus == "coconut"):
        assert c.training_subdir is None
        assert "canon_dedup_v1/train" in str(extras_training_dir(c))


def test_interaction_bpe_refs_are_size_matched_bpe_anchors() -> None:
    """Three default-knob BPE references on the size_700k subsample: PubChem at
    V in {256, 1024} (interaction B rows) and ZINC-22 at V=512 (interaction C
    column), each giving a swept Unigram cell a same-size cross-arm contrast."""
    cells = enumerate_interaction_bpe_refs()
    assert len(cells) == 3
    assert all(c.algo == "bpe" for c in cells)
    assert all(c.boundary == SENSITIVITY_BOUNDARY for c in cells)
    assert all(c.training_subdir == "size_700k" for c in cells)
    assert {(c.corpus, c.vocab_size) for c in cells} == {
        ("pubchem", 256),
        ("pubchem", 1024),
        ("zinc22", 512),
    }


def test_cell_to_config_threads_max_piece_length() -> None:
    cell = next(iter(enumerate_mpl_ladder()))
    cfg = extras_cell_to_config(cell)
    assert cfg.max_piece_length == cell.max_piece_length_override


def test_cell_to_config_threads_zero_min_frequency() -> None:
    cell = next(c for c in enumerate_minfreq_ladder() if c.min_frequency_override == 0)
    cfg = extras_cell_to_config(cell)
    assert cfg.min_frequency == 0


def test_cell_to_config_threads_both_interaction_knobs() -> None:
    cell = next(iter(enumerate_interaction_subiter_shrink()))
    cfg = extras_cell_to_config(cell)
    assert cfg.n_sub_iterations == cell.n_sub_iterations_override
    assert cfg.shrinking_factor == cell.shrinking_factor_override


def test_min_frequency_override_allows_zero_floor() -> None:
    cell = ExtrasCell(
        extras_kind="minfreq_ladder",
        algo="bpe",
        vocab_size=512,
        corpus="pubchem",
        boundary="nmb",
        label="0",
        min_frequency_override=0,
    )
    assert cell.min_frequency_override == 0
