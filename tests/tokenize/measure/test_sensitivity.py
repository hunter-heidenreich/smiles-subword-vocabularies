"""Sensitivity curve-aggregation structure.

These exercise the pure assembly in
:mod:`smiles_subword.tokenize.measure.supplementary.sensitivity` against
synthetic per-cell measurements, so the ladder/interaction shapes are
verified without training or measuring any tokenizer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from smiles_subword.tokenize.measure.supplementary.sensitivity import (
    CellMeasured,
    build_interaction_minfreq_mpl,
    build_interaction_mpl_typology,
    build_interaction_mpl_v,
    build_interaction_subiter_shrink,
    build_report,
)
from smiles_subword.tokenize.measure.supplementary.sensitivity import io as sens_io
from smiles_subword.tokenize.measure.supplementary.sensitivity.runner import (
    cells_to_measure,
)

if TYPE_CHECKING:
    from pathlib import Path

    from smiles_subword.tokenize.measure.jaccard import Arm


def _synthetic(cell_id: str, arm: Arm) -> CellMeasured:
    """A measured cell whose vocab set keys off the arm, so J is deterministic.

    Both arms share one common piece; each arm adds a distinct private piece, so
    the cross-arm Jaccard is a fixed ``1/3`` regardless of cell. Fertility
    differs by arm so the relative gap is non-zero and finite.
    """
    corpus = cell_id.split("__", 1)[0]
    shared = ("C", "C")
    private = (arm, "x")
    pieces = frozenset({shared, private})
    return CellMeasured(
        cell_id=cell_id,
        arm=arm,
        corpus=corpus,
        vocab_size=512,
        boundary="nmb",
        multi=pieces,
        fertility=10.0 if arm == "bpe" else 12.0,
    )


@pytest.fixture
def measured() -> dict[str, CellMeasured]:
    return {cid: _synthetic(cid, arm) for cid, arm in cells_to_measure()}


def test_cells_to_measure_covers_every_battery_cell_plus_coconut_ref() -> None:
    targets = cells_to_measure()
    ids = [cid for cid, _ in targets]
    assert len(ids) == len(set(ids))
    assert "coconut__smirk_gpe_v512_nmb" in ids
    assert any(cid.endswith("_sens_anchor") for cid in ids)


def test_report_has_five_ladders_and_four_interactions(
    measured: dict[str, CellMeasured],
) -> None:
    report = build_report(measured)
    assert {c.knob for c in report.ladders} == {
        "mpl",
        "seed",
        "subiter",
        "shrink",
        "minfreq",
    }
    assert {g.name for g in report.interactions} == {
        "subiter_shrink",
        "mpl_v",
        "mpl_typology",
        "minfreq_mpl",
    }


@pytest.mark.parametrize(
    ("knob", "n_points"),
    [("mpl", 6), ("seed", 6), ("subiter", 4), ("shrink", 5), ("minfreq", 5)],
)
def test_ladder_point_counts(
    measured: dict[str, CellMeasured], knob: str, n_points: int
) -> None:
    report = build_report(measured)
    curve = next(c for c in report.ladders if c.knob == knob)
    assert len(curve.points) == n_points
    assert sum(1 for p in curve.points if p.is_default) == 1


def test_ladder_x_axis_includes_the_reference_default(
    measured: dict[str, CellMeasured],
) -> None:
    report = build_report(measured)
    mpl = next(c for c in report.ladders if c.knob == "mpl")
    assert [p.x for p in mpl.points] == [4.0, 8.0, 16.0, 32.0, 64.0, 128.0]
    assert next(p for p in mpl.points if p.x == 16.0).is_default


def test_full_interaction_grids_are_rectangular(
    measured: dict[str, CellMeasured],
) -> None:
    assert len(build_interaction_subiter_shrink(measured).points) == 20
    assert len(build_interaction_mpl_v(measured).points) == 18
    assert len(build_interaction_mpl_typology(measured).points) == 18
    assert len(build_interaction_minfreq_mpl(measured).points) == 30


def test_contrast_values_are_finite_and_as_constructed(
    measured: dict[str, CellMeasured],
) -> None:
    report = build_report(measured)
    for curve in report.ladders:
        for p in curve.points:
            assert p.jaccard == pytest.approx(1.0 / 3.0)
            assert p.delta_fertility_relative == pytest.approx(2.0 / 11.0)


def test_minfreq_mpl_crossing_pairs_distinct_cells(
    measured: dict[str, CellMeasured],
) -> None:
    grid = build_interaction_minfreq_mpl(measured)
    keys = {(p.x, p.y) for p in grid.points}
    assert (2.0, 16.0) in keys
    assert next(p for p in grid.points if p.x == 2.0 and p.y == 16.0).is_default
    bpe_cells = {p.bpe_cell_id for p in grid.points}
    unigram_cells = {p.unigram_cell_id for p in grid.points}
    assert len(bpe_cells) == 5
    assert len(unigram_cells) == 6


def test_report_io_round_trip(
    measured: dict[str, CellMeasured],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sens_io, "SENSITIVITY_DIR", tmp_path)
    report = build_report(measured)

    path = sens_io.write_report(report)
    payload = sens_io.read_report()

    assert path == tmp_path / "sensitivity_report.json"
    assert payload["schema_version"] == 1
    assert set(payload["ladders"]) == {"mpl", "seed", "subiter", "shrink", "minfreq"}
    assert payload["anchor"]["corpus"] == "pubchem"
