"""Coordinate-keyed pairing of grid + extras cells — the cross-arm matchmaker.

Groups a BPE cell with its Unigram counterpart at the same
``(V, corpus, boundary[, extras_kind, label])`` coordinate so any cross-arm
measurement can join the two arms without filename arithmetic. Deadzone's
``ΔF_{p,n} = F^BPE − F^UL`` F95 join is the original consumer, but Jaccard,
Fertility, and Distribution all walk these pairs too (via
:func:`pair_all_cells`).

The grid pair_key strips the algo from the cell name; the extras pair_key
also strips it but adds the ``extras_kind`` + ``label`` axes the grid lacks
(``r1`` BPE pairs with ``r1`` Unigram and not with ``r2`` Unigram).
Singletons emit as :class:`UnpairedCell` with a reason — the 2 ZINC-22
BPE ``V=2048`` conditional cells (Unigram arm intentionally untrained)
and the four single-arm knob extras (seed-cap, two prune-schedule,
merge-exhaustion).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from smiles_subword.tokenize.extras import ExtrasCell, ExtrasKind, load_extras_manifest
from smiles_subword.tokenize.grid import GridCell, load_grid_manifest

if TYPE_CHECKING:
    from collections.abc import Sequence

UnpairedReason = Literal["conditional_negative_branch", "extras_single_arm_knob"]

_SINGLE_ARM_EXTRAS_KINDS: frozenset[ExtrasKind] = frozenset(
    {"seed_cap", "prune_schedule", "merge_exhaustion"}
)

SENSITIVITY_KINDS: frozenset[ExtrasKind] = frozenset(
    {
        "sensitivity_anchor",
        "mpl_ladder",
        "seed_ladder",
        "subiter_ladder",
        "shrink_ladder",
        "minfreq_ladder",
        "interaction_subiter_shrink",
        "interaction_mpl_v",
        "interaction_mpl_typology",
        "interaction_bpe_ref",
    }
)
"""The hyperparameter sensitivity battery. These cells
are read as per-knob response curves, not as the matched-pair ΔF join, so they
are excluded from the matched-pair walk below and measured by the sensitivity
pipeline instead."""

# The large-V convergence anchor is reported only by the
# convergence measurements — the cross-arm Jaccard, fertility, and the F95
# learnability audit (and vocab characterization). Every other matched-pair
# measurement leaves it out, so the shared pairing walk drops it by default and
# the two drivers that report it opt back in via ``include_large_v_anchor``.
_LARGE_V_ANCHOR_KINDS: frozenset[ExtrasKind] = frozenset({"large_v_anchor"})


@dataclass(frozen=True)
class PairKey:
    """Coordinate of one cross-arm pair_key.

    Grid pairs leave ``extras_kind`` / ``extras_label`` ``None``; extras pairs
    set both. The :attr:`slug` is the on-disk pair_key string and the JSON
    filename stem.
    """

    corpus: str
    vocab_size: int
    boundary: str
    extras_kind: ExtrasKind | None = None
    extras_label: str | None = None

    @property
    def slug(self) -> str:
        """The pair_key string — ``{corpus}__v{V}_{boundary}[__{kind}_{label}]``."""
        base = f"{self.corpus}__v{self.vocab_size}_{self.boundary}"
        if self.extras_kind is None:
            return base
        suffix = _extras_suffix(self.extras_kind, self.extras_label)
        return f"{base}__{suffix}"


@dataclass(frozen=True)
class MatchedPair:
    """One coordinate with both arms present — a cross-arm pair candidate."""

    key: PairKey
    tier: str
    bpe_cell_id: str
    unigram_cell_id: str


@dataclass(frozen=True)
class UnpairedCell:
    """One coordinate with only one arm — structurally single-arm.

    ``reason`` distinguishes the conditional negative branch (ZINC-22
    BPE ``V=2048``, Unigram intentionally untrained) from the
    single-arm-knob extras (seed-cap / prune-schedule are Unigram-only
    knobs; merge-exhaustion is a BPE-only ``GpeTrainer`` probe).
    """

    key: PairKey
    tier: str
    cell_id: str
    arm: Literal["bpe", "unigram"]
    reason: UnpairedReason


def _extras_suffix(kind: ExtrasKind, label: str | None) -> str:
    if label is None:
        raise ValueError(f"extras kind {kind!r} must carry a label")
    if kind == "subsample_redraw":
        return f"subsample_{label}"
    if kind == "size_sweep":
        return f"size_{label}"
    if kind == "size_matched":
        return f"size_matched_{label}"
    if kind == "seed_cap":
        return f"seed_{label}"
    if kind == "prune_schedule":
        return f"prune_{label}"
    if kind == "merge_exhaustion":
        return label
    if kind == "large_v_anchor":
        return label
    raise ValueError(f"unknown extras kind: {kind!r}")


def _group_by_coordinate(
    cells: Sequence[GridCell] | Sequence[ExtrasCell],
    *,
    extras: bool,
) -> dict[PairKey, list[GridCell] | list[ExtrasCell]]:
    groups: dict[PairKey, list[GridCell] | list[ExtrasCell]] = defaultdict(list)
    for cell in cells:
        if extras:
            assert isinstance(cell, ExtrasCell)
            key = PairKey(
                corpus=cell.corpus,
                vocab_size=cell.vocab_size,
                boundary=cell.boundary,
                extras_kind=cell.extras_kind,
                extras_label=cell.label,
            )
        else:
            assert isinstance(cell, GridCell)
            key = PairKey(
                corpus=cell.corpus,
                vocab_size=cell.vocab_size,
                boundary=cell.boundary,
            )
        groups[key].append(cell)  # type: ignore[arg-type]
    return groups


def pair_grid_cells(
    cells: Sequence[GridCell],
) -> tuple[list[MatchedPair], list[UnpairedCell]]:
    """Group grid cells by ``(corpus, V, boundary)`` and split into pairs.

    Coordinates with 2 cells (one BPE + one Unigram) yield :class:`MatchedPair`;
    coordinates with 1 cell yield :class:`UnpairedCell` with
    ``reason="conditional_negative_branch"`` (the only structurally single-arm
    grid case is the ZINC-22 BPE ``V=2048``).

    Raises:
        ValueError: a coordinate carries more than 2 cells or its two cells
            disagree on tier — both indicate a manifest bug.
    """
    groups = _group_by_coordinate(cells, extras=False)
    matched: list[MatchedPair] = []
    unpaired: list[UnpairedCell] = []
    for key, members in sorted(groups.items(), key=_pair_sort_key):
        tiers = {c.tier for c in members}
        if len(tiers) != 1:
            raise ValueError(f"tier mismatch at {key.slug}: {tiers}")
        tier = tiers.pop()
        if len(members) == 2:
            matched.append(_matched_from_members(key, tier, members))
        elif len(members) == 1:
            (cell,) = members
            unpaired.append(
                UnpairedCell(
                    key=key,
                    tier=tier,
                    cell_id=cell.cell_id,
                    arm=cell.algo,
                    reason="conditional_negative_branch",
                )
            )
        else:
            raise ValueError(
                f"more than 2 grid cells at {key.slug}: {[c.cell_id for c in members]}"
            )
    return matched, unpaired


def pair_extras_cells(
    cells: Sequence[ExtrasCell],
    *,
    include_large_v_anchor: bool = False,
) -> tuple[list[MatchedPair], list[UnpairedCell]]:
    """Group extras cells by ``(extras_kind, corpus, V, boundary, label)``.

    Pair-able kinds (subsample-redraw, size-sweep) have 2 cells per
    coordinate; single-arm knob kinds (seed-cap, prune-schedule,
    merge-exhaustion) have 1, emitted with
    ``reason="extras_single_arm_knob"``. The sensitivity-battery kinds
    (:data:`SENSITIVITY_KINDS`) are not part of this matched-pair join and
    are dropped before grouping. The large-V convergence anchor
    (:data:`_LARGE_V_ANCHOR_KINDS`) is likewise dropped unless
    ``include_large_v_anchor`` is set — only the convergence measurements
    (Jaccard, fertility) report it.

    Raises:
        ValueError: any coordinate carries more than 2 cells, or a
            pair-able-kind coordinate carries only 1 cell.
    """
    cells = [c for c in cells if c.extras_kind not in SENSITIVITY_KINDS]
    if not include_large_v_anchor:
        cells = [c for c in cells if c.extras_kind not in _LARGE_V_ANCHOR_KINDS]
    groups = _group_by_coordinate(cells, extras=True)
    matched: list[MatchedPair] = []
    unpaired: list[UnpairedCell] = []
    for key, members in sorted(groups.items(), key=_pair_sort_key):
        assert all(isinstance(c, ExtrasCell) for c in members)
        kind = key.extras_kind
        assert kind is not None
        tier = f"extras_{kind}"
        if len(members) == 2:
            if kind in _SINGLE_ARM_EXTRAS_KINDS:
                raise ValueError(
                    f"single-arm extras kind {kind!r} unexpectedly has 2 cells at "
                    f"{key.slug}: {[c.cell_id for c in members]}"
                )
            matched.append(_matched_from_members(key, tier, members))
        elif len(members) == 1:
            (cell,) = members
            if kind not in _SINGLE_ARM_EXTRAS_KINDS:
                raise ValueError(
                    f"pair-able extras kind {kind!r} has only 1 cell at "
                    f"{key.slug}: {cell.cell_id}"
                )
            unpaired.append(
                UnpairedCell(
                    key=key,
                    tier=tier,
                    cell_id=cell.cell_id,
                    arm=cell.algo,
                    reason="extras_single_arm_knob",
                )
            )
        else:
            raise ValueError(
                f"more than 2 extras cells at {key.slug}: "
                f"{[c.cell_id for c in members]}"
            )
    return matched, unpaired


def pair_all_cells(
    *, include_large_v_anchor: bool = False
) -> tuple[list[MatchedPair], list[UnpairedCell]]:
    """Pair every committed grid + extras cell. Deterministically ordered.

    The large-V convergence anchor is excluded unless
    ``include_large_v_anchor`` is set; only the convergence measurements
    (Jaccard, fertility) report it (see :func:`pair_extras_cells`).
    """
    grid_matched, grid_unpaired = pair_grid_cells(load_grid_manifest())
    extras_matched, extras_unpaired = pair_extras_cells(
        load_extras_manifest(), include_large_v_anchor=include_large_v_anchor
    )
    return grid_matched + extras_matched, grid_unpaired + extras_unpaired


def _matched_from_members(
    key: PairKey,
    tier: str,
    members: list[GridCell] | list[ExtrasCell],
) -> MatchedPair:
    algos = {c.algo for c in members}
    if algos != {"bpe", "unigram"}:
        raise ValueError(
            f"matched coordinate {key.slug} does not have one bpe + one unigram: "
            f"{[c.cell_id for c in members]}"
        )
    bpe = next(c for c in members if c.algo == "bpe")
    unigram = next(c for c in members if c.algo == "unigram")
    return MatchedPair(
        key=key,
        tier=tier,
        bpe_cell_id=bpe.cell_id,
        unigram_cell_id=unigram.cell_id,
    )


def _pair_sort_key(item: tuple[PairKey, object]) -> tuple[str, int, str, str, str]:
    """Deterministic ordering over (corpus, V, boundary, extras_kind, label)."""
    key = item[0]
    return (
        key.corpus,
        key.vocab_size,
        key.boundary,
        key.extras_kind or "",
        key.extras_label or "",
    )


__all__ = [
    "SENSITIVITY_KINDS",
    "MatchedPair",
    "PairKey",
    "UnpairedCell",
    "pair_all_cells",
    "pair_extras_cells",
    "pair_grid_cells",
]
