"""Hyperparameter-sensitivity curve aggregation.

The battery is read as *response curves*, not the headline matched-pair join
(excluded from the pairing walk; see
:data:`smiles_subword.tokenize.measure._pairing.SENSITIVITY_KINDS`). Each swept
hyperparameter is varied one-factor-at-a-time across a ladder; at every rung we
recompute the two cross-arm contrasts — unweighted vocabulary overlap ``J`` and
the relative fertility gap ``|Δf|``. (Structural overlap ``J_struct`` is reported
at the headline grid, not swept: it needs a per-cell training-corpus inventory
classification, infeasible across the battery.)

The swept arm is paired against the opposite arm at its reference default for the
same ``(corpus, V, boundary, subsample)``: the OFAT ladders and interaction A use
the shared ``sensitivity_anchor``; interactions B (mpl × V) and C (mpl ×
typology) use the off-anchor ``interaction_bpe_ref`` cells (plus COCONUT's
full-corpus headline BPE); interaction D crosses the BPE minfreq and Unigram mpl
ladders directly. This module is the pure aggregation; the measurement bridge is
in :mod:`.runner`, the JSON deposit in :mod:`.io`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from smiles_subword.tokenize.extras import ExtrasKind, cells_for_extras_kind
from smiles_subword.tokenize.measure.fertility import relative_fertility_gap
from smiles_subword.tokenize.measure.jaccard import (
    Arm,
    Boundary,
    GlyphTuple,
    jaccard,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from smiles_subword.tokenize.extras import ExtrasCell

SCHEMA_VERSION = 1

_ANCHOR_DEFAULTS: dict[str, float] = {
    "mpl": 16.0,
    "seed": 1_000_000.0,
    "subiter": 2.0,
    "shrink": 0.75,
    "minfreq": 2.0,
}

COCONUT_BPE_REF = "coconut__smirk_gpe_v512_nmb"
"""Interaction C's COCONUT column reference: the full-corpus (~702K) headline
BPE V=512 cell, size-matched to COCONUT's Unigram interaction rungs."""


@dataclass(frozen=True)
class CellMeasured:
    """One trained cell's curve inputs: its multi-glyph set and held-out fertility."""

    cell_id: str
    arm: Arm
    corpus: str
    vocab_size: int
    boundary: Boundary
    multi: frozenset[GlyphTuple]
    fertility: float


@dataclass(frozen=True)
class ContrastPoint:
    """A cross-arm contrast at one ladder rung or interaction grid cell."""

    x: float
    is_default: bool
    jaccard: float
    delta_fertility_relative: float
    bpe_cell_id: str
    unigram_cell_id: str
    y: float | None = None

    def as_dict(self) -> dict[str, object]:
        d: dict[str, object] = {
            "x": self.x,
            "is_default": self.is_default,
            "jaccard": self.jaccard,
            "delta_fertility_relative": self.delta_fertility_relative,
            "bpe_cell_id": self.bpe_cell_id,
            "unigram_cell_id": self.unigram_cell_id,
        }
        if self.y is not None:
            d["y"] = self.y
        return d


@dataclass(frozen=True)
class LadderCurve:
    """One OFAT response curve (knob → contrast points, default flagged)."""

    knob: str
    label: str
    swept_arm: Arm
    default_x: float
    points: list[ContrastPoint]

    def as_dict(self) -> dict[str, object]:
        return {
            "knob": self.knob,
            "label": self.label,
            "swept_arm": self.swept_arm,
            "default_x": self.default_x,
            "points": [p.as_dict() for p in self.points],
        }


@dataclass(frozen=True)
class InteractionGrid:
    """One pairwise interaction surface (two crossed knobs → contrast points)."""

    name: str
    x_knob: str
    y_knob: str
    points: list[ContrastPoint]

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "x_knob": self.x_knob,
            "y_knob": self.y_knob,
            "points": [p.as_dict() for p in self.points],
        }


@dataclass(frozen=True)
class LadderSpec:
    """Declarative OFAT ladder: which kind sweeps which arm's override."""

    knob: str
    label: str
    swept_kind: ExtrasKind
    swept_arm: Arm
    override_attr: str


_LADDERS: tuple[LadderSpec, ...] = (
    LadderSpec(
        "mpl",
        "Unigram max piece length",
        "mpl_ladder",
        "unigram",
        "max_piece_length_override",
    ),
    LadderSpec(
        "seed", "Unigram seed pool", "seed_ladder", "unigram", "seed_size_override"
    ),
    LadderSpec(
        "subiter",
        "Unigram EM sub-iterations",
        "subiter_ladder",
        "unigram",
        "n_sub_iterations_override",
    ),
    LadderSpec(
        "shrink",
        "Unigram shrinking factor",
        "shrink_ladder",
        "unigram",
        "shrinking_factor_override",
    ),
    LadderSpec(
        "minfreq",
        "BPE merge frequency",
        "minfreq_ladder",
        "bpe",
        "min_frequency_override",
    ),
)


def _opposite(arm: Arm) -> Arm:
    return "unigram" if arm == "bpe" else "bpe"


def _anchor_cell_id(arm: Arm) -> str:
    """The ``sensitivity_anchor`` cell id for one arm (PubChem V=512 NMB)."""
    for cell in cells_for_extras_kind("sensitivity_anchor"):
        if cell.algo == arm:
            return cell.cell_id
    raise KeyError(f"no sensitivity_anchor cell for arm {arm!r}")


def _bpe_ref_cell_id(corpus: str, vocab_size: int) -> str:
    """The BPE reference cell id for an interaction coordinate.

    PubChem V=512 is the anchor; COCONUT V=512 is the full-corpus headline cell;
    every other coordinate is an ``interaction_bpe_ref`` cell.
    """
    if corpus == "pubchem" and vocab_size == 512:
        return _anchor_cell_id("bpe")
    if corpus == "coconut" and vocab_size == 512:
        return COCONUT_BPE_REF
    for cell in cells_for_extras_kind("interaction_bpe_ref"):
        if cell.corpus == corpus and cell.vocab_size == vocab_size:
            return cell.cell_id
    raise KeyError(f"no BPE reference for corpus={corpus!r} V={vocab_size}")


def _override_x(cell: ExtrasCell, attr: str) -> float:
    value = getattr(cell, attr)
    if value is None:
        raise ValueError(f"cell {cell.cell_id} missing override {attr}")
    return float(value)


def contrast_point(
    *,
    bpe: CellMeasured,
    unigram: CellMeasured,
    x: float,
    is_default: bool,
    y: float | None = None,
) -> ContrastPoint:
    """Compute the (J, |Δf|) cross-arm contrast for a BPE/Unigram cell pair."""
    j = jaccard(bpe.multi, unigram.multi)
    dfr = relative_fertility_gap(bpe.fertility, unigram.fertility)
    return ContrastPoint(
        x=x,
        is_default=is_default,
        jaccard=j,
        delta_fertility_relative=dfr,
        bpe_cell_id=bpe.cell_id,
        unigram_cell_id=unigram.cell_id,
        y=y,
    )


def _pair_for_arm(
    swept_arm: Arm, swept: CellMeasured, ref: CellMeasured
) -> tuple[CellMeasured, CellMeasured]:
    """Order a (swept, reference) pair into (bpe, unigram)."""
    return (ref, swept) if swept_arm == "unigram" else (swept, ref)


def build_ladder(spec: LadderSpec, measured: Mapping[str, CellMeasured]) -> LadderCurve:
    """Assemble one OFAT curve: the anchor default rung plus the swept rungs."""
    ref = measured[_anchor_cell_id(_opposite(spec.swept_arm))]
    anchor_swept = measured[_anchor_cell_id(spec.swept_arm)]
    default_x = _ANCHOR_DEFAULTS[spec.knob]
    bpe, uni = _pair_for_arm(spec.swept_arm, anchor_swept, ref)
    points = [contrast_point(bpe=bpe, unigram=uni, x=default_x, is_default=True)]
    for cell in cells_for_extras_kind(spec.swept_kind):
        swept = measured[cell.cell_id]
        bpe, uni = _pair_for_arm(spec.swept_arm, swept, ref)
        points.append(
            contrast_point(
                bpe=bpe,
                unigram=uni,
                x=_override_x(cell, spec.override_attr),
                is_default=False,
            )
        )
    points.sort(key=lambda p: p.x)
    return LadderCurve(spec.knob, spec.label, spec.swept_arm, default_x, points)


def _unigram_grid_point(
    uni: CellMeasured, ref_bpe: CellMeasured, *, x: float, y: float, is_default: bool
) -> ContrastPoint:
    return contrast_point(bpe=ref_bpe, unigram=uni, x=x, is_default=is_default, y=y)


def build_interaction_subiter_shrink(
    measured: Mapping[str, CellMeasured],
) -> InteractionGrid:
    """Interaction A: Unigram sub-iterations × shrinking factor (vs anchor BPE)."""
    ref = measured[_anchor_cell_id("bpe")]
    pts: list[ContrastPoint] = []
    anchor = measured[_anchor_cell_id("unigram")]
    pts.append(_unigram_grid_point(anchor, ref, x=2.0, y=0.75, is_default=True))
    for cell in cells_for_extras_kind("subiter_ladder"):
        m = measured[cell.cell_id]
        pts.append(
            _unigram_grid_point(
                m,
                ref,
                x=_override_x(cell, "n_sub_iterations_override"),
                y=0.75,
                is_default=False,
            )
        )
    for cell in cells_for_extras_kind("shrink_ladder"):
        m = measured[cell.cell_id]
        pts.append(
            _unigram_grid_point(
                m,
                ref,
                x=2.0,
                y=_override_x(cell, "shrinking_factor_override"),
                is_default=False,
            )
        )
    for cell in cells_for_extras_kind("interaction_subiter_shrink"):
        m = measured[cell.cell_id]
        pts.append(
            _unigram_grid_point(
                m,
                ref,
                x=_override_x(cell, "n_sub_iterations_override"),
                y=_override_x(cell, "shrinking_factor_override"),
                is_default=False,
            )
        )
    return InteractionGrid("subiter_shrink", "subiter", "shrink", pts)


def _mpl_axis_grid(
    measured: Mapping[str, CellMeasured],
    *,
    name: str,
    second_knob: str,
    off_kind: ExtrasKind,
    anchor_y: float,
    ref_for: dict[str, tuple[str, int]],
) -> InteractionGrid:
    """Shared builder for interaction B (mpl × V) and C (mpl × typology).

    ``ref_for`` maps a cell_id to its BPE reference ``(corpus, V)``; the anchor
    column (PubChem V=512, mpl ladder + anchor) sits at ``y == anchor_y``.
    """
    pts: list[ContrastPoint] = []
    anchor = measured[_anchor_cell_id("unigram")]
    ref_anchor = measured[_anchor_cell_id("bpe")]
    pts.append(
        _unigram_grid_point(anchor, ref_anchor, x=16.0, y=anchor_y, is_default=True)
    )
    for cell in cells_for_extras_kind("mpl_ladder"):
        m = measured[cell.cell_id]
        pts.append(
            _unigram_grid_point(
                m,
                ref_anchor,
                x=_override_x(cell, "max_piece_length_override"),
                y=anchor_y,
                is_default=False,
            )
        )
    for cell in cells_for_extras_kind(off_kind):
        m = measured[cell.cell_id]
        ref_corpus, ref_v = ref_for[cell.cell_id]
        ref = measured[_bpe_ref_cell_id(ref_corpus, ref_v)]
        y = float(cell.vocab_size) if second_knob == "V" else _TYPOLOGY_Y[cell.corpus]
        pts.append(
            _unigram_grid_point(
                m,
                ref,
                x=_override_x(cell, "max_piece_length_override"),
                y=y,
                is_default=False,
            )
        )
    return InteractionGrid(name, "mpl", second_knob, pts)


_TYPOLOGY_Y: dict[str, float] = {"pubchem": 0.0, "zinc22": 1.0, "coconut": 2.0}
"""Categorical y-codes for interaction C's corpus axis (figure tick labels)."""


def build_interaction_mpl_v(measured: Mapping[str, CellMeasured]) -> InteractionGrid:
    """Interaction B: Unigram max piece length × V (each vs same-V BPE)."""
    ref_for = {
        c.cell_id: (c.corpus, c.vocab_size)
        for c in cells_for_extras_kind("interaction_mpl_v")
    }
    return _mpl_axis_grid(
        measured,
        name="mpl_v",
        second_knob="V",
        off_kind="interaction_mpl_v",
        anchor_y=512.0,
        ref_for=ref_for,
    )


def build_interaction_mpl_typology(
    measured: Mapping[str, CellMeasured],
) -> InteractionGrid:
    """Interaction C: Unigram max piece length × typology (vs same-corpus BPE)."""
    ref_for = {
        c.cell_id: (c.corpus, 512)
        for c in cells_for_extras_kind("interaction_mpl_typology")
    }
    return _mpl_axis_grid(
        measured,
        name="mpl_typology",
        second_knob="typology",
        off_kind="interaction_mpl_typology",
        anchor_y=_TYPOLOGY_Y["pubchem"],
        ref_for=ref_for,
    )


def build_interaction_minfreq_mpl(
    measured: Mapping[str, CellMeasured],
) -> InteractionGrid:
    """Interaction D: BPE minfreq × Unigram mpl --- a crossing of the two ladders.

    Both arms run permissive at once: every (minfreq, mpl) point pairs the BPE
    cell at that merge frequency with the Unigram cell at that max piece length.
    """
    bpe_by_x = {2.0: measured[_anchor_cell_id("bpe")]}
    for cell in cells_for_extras_kind("minfreq_ladder"):
        bpe_by_x[_override_x(cell, "min_frequency_override")] = measured[cell.cell_id]
    uni_by_y = {16.0: measured[_anchor_cell_id("unigram")]}
    for cell in cells_for_extras_kind("mpl_ladder"):
        uni_by_y[_override_x(cell, "max_piece_length_override")] = measured[
            cell.cell_id
        ]
    pts: list[ContrastPoint] = []
    for mf, bpe in bpe_by_x.items():
        for mpl, uni in uni_by_y.items():
            pts.append(
                contrast_point(
                    bpe=bpe,
                    unigram=uni,
                    x=mf,
                    y=mpl,
                    is_default=(mf == 2.0 and mpl == 16.0),
                )
            )
    return InteractionGrid("minfreq_mpl", "minfreq", "mpl", pts)


@dataclass
class SensitivityReport:
    """The full deposited figure payload: anchor, OFAT ladders, interactions."""

    ladders: list[LadderCurve]
    interactions: list[InteractionGrid]
    anchor: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "anchor": self.anchor,
            "ladders": {c.knob: c.as_dict() for c in self.ladders},
            "interactions": {g.name: g.as_dict() for g in self.interactions},
        }


def build_report(measured: Mapping[str, CellMeasured]) -> SensitivityReport:
    """Assemble every ladder and interaction from the measured cells."""
    ladders = [build_ladder(spec, measured) for spec in _LADDERS]
    interactions = [
        build_interaction_subiter_shrink(measured),
        build_interaction_mpl_v(measured),
        build_interaction_mpl_typology(measured),
        build_interaction_minfreq_mpl(measured),
    ]
    anchor_pt = ladders[0].points[
        next(i for i, p in enumerate(ladders[0].points) if p.is_default)
    ]
    anchor = {
        "corpus": "pubchem",
        "vocab_size": 512,
        "boundary": "nmb",
        "subsample": "size_700k",
        "jaccard": anchor_pt.jaccard,
        "delta_fertility_relative": anchor_pt.delta_fertility_relative,
    }
    return SensitivityReport(ladders, interactions, anchor)


__all__ = [
    "CellMeasured",
    "ContrastPoint",
    "InteractionGrid",
    "LadderCurve",
    "LadderSpec",
    "SensitivityReport",
    "build_interaction_minfreq_mpl",
    "build_interaction_mpl_typology",
    "build_interaction_mpl_v",
    "build_interaction_subiter_shrink",
    "build_ladder",
    "build_report",
    "contrast_point",
]
