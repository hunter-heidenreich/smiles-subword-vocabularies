"""Tidy-row extraction from the deposited measurement aggregators.

This turns the on-disk measurement ``*_table.json`` aggregators under
``results/data/`` into the rows the results tables and figures consume. It
retokenizes nothing — it is a pure read over already-deposited JSON, the
canonical per-step joins.

The results tables and cross-V figures draw from:

- **ΔF** ← ``deadzone_table`` (per-arm F95 clearance + cross-arm ΔF).
- **three Jaccards** ← ``jaccard_table`` (``J``, ``J_struct``, ``J_w`` + CI).
- **seven measurements** ← the ``deadzone_table`` matched spine (ΔF) joined by
  ``pair_key`` to Jaccard (``J``), Fertility (relΔf), Distribution (``|ΔD|``),
  ``absorption`` (Δabsorbed), ``scaffold`` (Δscaffold), ``segmentation``
  (ΔH/glyph).
- **cross-axis cells** ← Jaccard / Fertility / Distribution joined per matched
  cell (:func:`cross_axis_cells`), the substrate for the trend + interaction
  figures.

By default only the **grid** coordinates (``extras_kind is None``: the 44-cell
frozen grid) are returned — the robustness extras are a separate
results subsection, surfaced via ``include_extras=True``. Rows sort by corpus typology
order (PubChem → ZINC-22 → COCONUT → REAL-Space), then ``V`` ascending, then
boundary (NMB before MB), so the tables read in the paper's narrative order.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from _corpora import CORPUS_RANK

from smiles_subword._io import read_json_or_none
from smiles_subword.paths import RESULTS_DATA_DIR, audit_path

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

DEADZONE_TABLE = "deadzone_table"
ABSORPTION_TABLE = "absorption_table"
SCAFFOLD_TABLE = "scaffold_table"
JACCARD_TABLE = "jaccard_table"
SEGMENTATION_TABLE = "segmentation_table"
FERTILITY_TABLE = "fertility_table"
DISTRIBUTION_TABLE = "distribution_table"
NESTEDNESS_TABLE = "nestedness_table"
CLOSURE_TABLE = "closure_table"
FG_ALIGNMENT_TABLE = "fg_alignment_table"
NONCANON_TABLE = "noncanon_table"

REQUIRED_TABLES = (
    DEADZONE_TABLE,
    ABSORPTION_TABLE,
    SCAFFOLD_TABLE,
    JACCARD_TABLE,
    SEGMENTATION_TABLE,
    FERTILITY_TABLE,
    DISTRIBUTION_TABLE,
    NESTEDNESS_TABLE,
    CLOSURE_TABLE,
    FG_ALIGNMENT_TABLE,
    NONCANON_TABLE,
)

AUDIT_ITEMS = ("seed_cap", "prune_schedule", "merge_exhaustion")

_BOUNDARY_ORDER = {"nmb": 0, "mb": 1}


# --------------------------------------------------------------------------- #
# Reading                                                                     #
# --------------------------------------------------------------------------- #


def table_path(name: str) -> Path:
    """Return the on-disk path of the aggregator ``{name}.json``."""
    return RESULTS_DATA_DIR / f"{name}.json"


def read_table(name: str) -> dict[str, Any] | None:
    """Return the deposited aggregator payload for ``name``, or ``None``."""
    return read_json_or_none(table_path(name))


def missing_tables() -> list[str]:
    """Names of the required aggregators that are not on disk (sorted)."""
    return sorted(name for name in REQUIRED_TABLES if read_table(name) is None)


def read_audit(name: str) -> dict[str, Any] | None:
    """Return the deposited robustness-extras audit payload, or ``None``."""
    return read_json_or_none(audit_path(name))


def read_deadzone_cell(pair_key: str) -> dict[str, Any] | None:
    """Return the per-condition Deadzone deposit for ``pair_key``, or ``None``.

    The aggregate carries only the headline $F_{.95,100}$; the per-condition
    deposit additionally holds ``clearance_by_n`` (the n-sweep), read here for the
    learnability detail table without re-aggregating ``deadzone_table.json``.
    """
    path = RESULTS_DATA_DIR / DEADZONE_TABLE.removesuffix("_table") / f"{pair_key}.json"
    return read_json_or_none(path)


def missing_audits() -> list[str]:
    """Robustness-extras audit items (extras-table inputs) not on disk (sorted)."""
    return sorted(i for i in AUDIT_ITEMS if read_audit(i) is None)


def _payload_sha(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.blake2b(canonical.encode("utf-8"), digest_size=16).hexdigest()


def upstream_sha_map() -> dict[str, str]:
    """Map each required aggregator to a content SHA, or ``"absent"``.

    The results-table manifest stores this so a regenerated table can detect
    that an upstream measurement was re-run (or removed) since it was built.
    """
    out: dict[str, str] = {}
    for name in REQUIRED_TABLES:
        payload = read_table(name)
        out[name] = "absent" if payload is None else _payload_sha(payload)
    return out


def audit_upstream_sha_map() -> dict[str, str]:
    """Map each robustness-extras audit input to a content SHA, or ``"absent"``.

    Stored in the results-table manifest so the extras table can detect that an
    audit (seed-cap, prune-schedule, merge-exhaustion) was re-run since build.
    """
    out: dict[str, str] = {}
    for item in AUDIT_ITEMS:
        payload = read_audit(item)
        out[item] = "absent" if payload is None else _payload_sha(payload)
    return out


def _section(table: dict[str, Any] | None, section: str) -> list[dict[str, Any]]:
    if table is None:
        return []
    rows = table.get(section)
    return rows if isinstance(rows, list) else []


def _index(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(r["pair_key"]): r for r in rows}


def _is_grid(row: dict[str, Any]) -> bool:
    return row.get("extras_kind") is None


def _coord_sort_key(corpus: str, vocab_size: int, boundary: str) -> tuple[Any, ...]:
    return (
        CORPUS_RANK.get(corpus, 99),
        vocab_size,
        _BOUNDARY_ORDER.get(boundary, 99),
        corpus,
        boundary,
    )


def _opt_bool(value: object) -> bool | None:
    return None if value is None else bool(value)


def _opt_float(value: object) -> float | None:
    return None if value is None else float(value)  # type: ignore[arg-type]


def _opt_int(value: object) -> int | None:
    return None if value is None else int(value)  # type: ignore[arg-type]


def _keep(row: dict[str, Any], *, include_extras: bool) -> bool:
    return include_extras or _is_grid(row)


def _row_ci(row: dict[str, Any], prefix: str) -> tuple[float, float] | None:
    """Return the ``(lo, hi)`` bootstrap CI for ``prefix`` in a row, or None."""
    lo = _opt_float(row.get(f"{prefix}_ci_lo"))
    hi = _opt_float(row.get(f"{prefix}_ci_hi"))
    return (lo, hi) if lo is not None and hi is not None else None


# --------------------------------------------------------------------------- #
# Row records                                                                 #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CrossAxisCell:
    """One matched grid cell's cross-arm scalars, joined across measurements.

    The substrate for the cross-V trend and algorithm×boundary interaction
    figures: the trend uses the magnitude scalars (``jaccard`` / ``rel_fertility``
    / ``abs_delta_d``); the interaction uses the signed cross-arm deltas
    (``jaccard`` itself for J, ``delta_fertility_signed``, ``delta_d_signed``).
    """

    pair_key: str
    corpus: str
    vocab_size: int
    boundary: str
    jaccard: float
    rel_fertility: float
    abs_delta_d: float
    delta_fertility_signed: float
    delta_d_signed: float


@dataclass(frozen=True)
class JaccardRow:
    """The three vocabulary-Jaccard quantities for one matched pair (Jaccard)."""

    pair_key: str
    corpus: str
    vocab_size: int
    boundary: str
    tier: str
    jaccard: float
    jaccard_struct: float | None
    weighted_jaccard: float | None
    weighted_jaccard_ci: tuple[float, float] | None
    weighted_jaccard_struct: float | None
    weighted_jaccard_struct_ci: tuple[float, float] | None
    bpe_n_multi: int | None
    unigram_n_multi: int | None


@dataclass(frozen=True)
class FertilityRow:
    """Absolute per-arm fertility and compression ratio for one matched pair.

    Surfaces the per-arm tokens-per-molecule (``*_fertility``) and
    glyphs-per-token compression ratio (``*_glyphs_per_token``), each with its
    bootstrap CI, alongside the cross-arm gap the body reports. The means and
    ratios live in the ``fertility_table`` aggregate; the CIs were folded into it
    from the per-condition deposits (mirrors the Jaccard-CI path).
    """

    pair_key: str
    corpus: str
    vocab_size: int
    boundary: str
    tier: str
    bpe_fertility: float
    bpe_fertility_ci: tuple[float, float] | None
    unigram_fertility: float
    unigram_fertility_ci: tuple[float, float] | None
    bpe_glyphs_per_token: float
    bpe_glyphs_per_token_ci: tuple[float, float] | None
    unigram_glyphs_per_token: float
    unigram_glyphs_per_token_ci: tuple[float, float] | None
    delta_fertility: float
    rel_fertility: float


@dataclass(frozen=True)
class NestednessRow:
    """Cross-arm boundary-nestedness reading for one matched pair (Nestedness).

    The positional companion to fertility: both arms cut the same glyph stream,
    so their boundaries are subsets of the same inter-glyph positions. Surfaces
    the boundary Jaccard (over *cut* positions, excluding agree-merge), the
    ``nest`` rate (Unigram cuts where BPE merges, the fertility gap read
    positionally) and ``conflict`` rate (the genuine crossing disagreement, over
    *all* positions), the nested-molecule fraction (zero-conflict molecules), and
    conflict localization by substructure class (heteroatom / unsaturated /
    saturated carbon). All point values live in the ``nestedness_table``
    aggregate; ``cut_rate_sat_c`` is ``None`` where the corpus emits no such
    piece (REAL-Space).
    """

    pair_key: str
    corpus: str
    vocab_size: int
    boundary: str
    tier: str
    boundary_jaccard: float
    conflict_rate: float
    nest_rate: float
    nested_molecule_fraction: float
    cut_rate_heteroatom: float | None
    cut_rate_unsat_c: float | None
    cut_rate_sat_c: float | None


@dataclass(frozen=True)
class ClosureRow:
    """Within-arm compositional-closure reading for one matched pair (Closure).

    A construction-independent read of each vocabulary's internal self-structure.
    ``bpe_c_bin`` is the binary-split closure anchor (``1.000`` by
    BPE's merge-closure invariant) and ``bpe_c_orph`` is ``0`` by the same
    invariant, so the contrast lives in the Unigram-LM columns: ``ul_c_bin``
    (how many of its pieces decompose into in-vocab parts) and ``ul_c_orph``
    (how many float free). ``c_full`` is the stronger full-substring closure,
    non-trivial for both arms. All point values live in the ``closure_table``
    aggregate; closure is an exact-set quantity and carries no CI.
    """

    pair_key: str
    corpus: str
    vocab_size: int
    boundary: str
    tier: str
    bpe_c_bin: float
    ul_c_bin: float
    ul_c_orph: float
    bpe_c_full: float
    ul_c_full: float


@dataclass(frozen=True)
class FgAlignmentRow:
    """Within-arm functional-bond-locality reading for one matched pair.

    Locality is the fraction of multiply-bonded heteroatoms (the ``=O`` / ``#N``
    cores of the functional groups, read off the graph) whose bond the arm kept
    inside a single token, on the held-out split. ``bpe_locality`` /
    ``ul_locality`` are the overall per-arm fractions and ``delta_locality`` the
    cross-arm gap; ``*_carbonyl`` surfaces the headline carbonyl (``C{=}O``) bond
    class. Point values live in the ``fg_alignment_table`` aggregate; the
    bootstrap CIs were folded into the per-condition deposits.
    """

    pair_key: str
    corpus: str
    vocab_size: int
    boundary: str
    tier: str
    bpe_locality: float
    ul_locality: float
    delta_locality: float
    bpe_carbonyl: float
    ul_carbonyl: float


@dataclass(frozen=True)
class NoncanonRow:
    """Within-arm robustness to the SMILES rewrite orbit, for one matched pair.

    Each value is the per-arm piece-bag instability (the fraction of the token
    multiset that changes vs the canonical string) on the held-out subsample:
    ``random`` is RDKit's restricted randomization, ``kekule`` the
    aromatic-to-Kekule rewrite, ``explicitH`` the all-explicit-hydrogen rewrite,
    and ``obcanon`` the cross-toolkit swap to OpenBabel's canonical SMILES.
    ``gap_canon`` / ``gap_rand`` are the symmetric relative fertility gap
    ``rel|df| = |df|/mean`` (the paper's standard granularity metric, converted
    from the deposited Unigram/BPE fertility ratio) on canonical strings and on
    the randomized orbit; their closeness is the test of whether the granularity
    gap is a canonical-notation artifact. Point values live in the
    ``noncanon_table`` aggregate; bootstrap CIs are in the deposits.
    """

    pair_key: str
    corpus: str
    vocab_size: int
    boundary: str
    tier: str
    bpe_random: float
    ul_random: float
    bpe_kekule: float
    ul_kekule: float
    bpe_explicit_h: float
    ul_explicit_h: float
    bpe_obcanon: float
    ul_obcanon: float
    gap_canon: float
    gap_rand: float


@dataclass(frozen=True)
class DistributionRow:
    """Per-arm token-distribution intrinsics for one matched pair.

    Surfaces the three within-family distribution quantities: imbalance ``d``
    (divergence from uniform), normalized entropy ``eta``, and Rényi efficiency
    ``renyi``, per arm with bootstrap CIs, plus the live-token counts and the
    cross-arm ``abs_delta_d`` token-imbalance gap. Point values live in the
    ``distribution_table`` aggregate; the CIs were folded into it from the
    per-condition deposits (mirrors the fertility path).
    """

    pair_key: str
    corpus: str
    vocab_size: int
    boundary: str
    tier: str
    bpe_d: float
    bpe_d_ci: tuple[float, float] | None
    unigram_d: float
    unigram_d_ci: tuple[float, float] | None
    bpe_eta: float
    bpe_eta_ci: tuple[float, float] | None
    unigram_eta: float
    unigram_eta_ci: tuple[float, float] | None
    bpe_renyi: float
    bpe_renyi_ci: tuple[float, float] | None
    unigram_renyi: float
    unigram_renyi_ci: tuple[float, float] | None
    bpe_live: int | None
    unigram_live: int | None
    abs_delta_d: float


@dataclass(frozen=True)
class AbsorptionRow:
    """Per-arm whole-pretoken absorption for one matched pair (Absorption).

    The fraction of pretokens emitted as a single token, per arm with its
    bootstrap CI, plus the cross-arm gap the body reports. Point values live in
    the ``absorption_table`` aggregate; the CIs were folded in from the
    per-condition deposits (mirrors the fertility path).
    """

    pair_key: str
    corpus: str
    vocab_size: int
    boundary: str
    tier: str
    bpe_absorbed: float
    bpe_absorbed_ci: tuple[float, float] | None
    unigram_absorbed: float
    unigram_absorbed_ci: tuple[float, float] | None
    delta_absorbed: float


@dataclass(frozen=True)
class DeadzoneNSweepRow:
    """Per-arm rare-token clearance across the firing-count sweep (Deadzone).

    ``c_n`` is the fraction of vocabulary pieces firing at least ``n`` times in
    the training corpus; the learnability bar $F_{p,n}$ is ``c_n >= p`` with
    $F_{95\\%,100}$ the headline, so any $(p,n)$ bar reads off these. Sourced
    from the per-condition Deadzone deposits (``clearance_by_n``), not the
    aggregate.
    """

    pair_key: str
    corpus: str
    vocab_size: int
    boundary: str
    tier: str
    bpe_c: dict[int, float]
    unigram_c: dict[int, float]
    any_arm_unsafe: bool


@dataclass(frozen=True)
class DeltaFRow:
    """Per-arm F95 clearance and the cross-arm ΔF for one matched pair (Deadzone)."""

    pair_key: str
    corpus: str
    vocab_size: int
    boundary: str
    tier: str
    bpe_clearance: float
    unigram_clearance: float
    headline_delta_f: float
    any_arm_unsafe: bool


@dataclass(frozen=True)
class MeasurementRow:
    """The seven cross-arm measurement scalars for one matched pair."""

    pair_key: str
    corpus: str
    vocab_size: int
    boundary: str
    tier: str
    delta_f: float
    delta_absorbed: float | None
    delta_scaffold: float | None
    rel_fertility: float
    jaccard: float
    abs_delta_d: float
    delta_entropy_per_glyph: float | None


@dataclass(frozen=True)
class RedrawSpread:
    """Unigram $F_{.95,100}$ clearance across a subsample-redraw triplet."""

    corpus: str
    clearances: tuple[float, ...]

    @property
    def spread(self) -> float:
        """Max-minus-min clearance across the redraws (0.0 if fewer than two)."""
        return max(self.clearances) - min(self.clearances) if self.clearances else 0.0


@dataclass(frozen=True)
class SizeSweepPoint:
    """Unigram $F_{.95,100}$ clearance at one training-corpus size."""

    n_label: str
    unigram_clearance: float


@dataclass(frozen=True)
class PruneComparison:
    """Multi-glyph Jaccard of a prune-schedule probe vs its baseline."""

    vocab_size: int
    jaccard: float


@dataclass(frozen=True)
class RobustnessExtras:
    """The five robustness-extras readings (results subsection)."""

    redraws: tuple[RedrawSpread, ...]
    size_sweep: tuple[SizeSweepPoint, ...]
    seed_cap_jaccard: float | None
    seed_cap_symmetric_difference: int | None
    prune: tuple[PruneComparison, ...]
    merge_exhaustion_realised_v: int | None
    merge_exhaustion_cap: int | None
    merge_exhaustion_natural: bool | None


# --------------------------------------------------------------------------- #
# Builders                                                                     #
# --------------------------------------------------------------------------- #


def _sorted(rows: list[Any]) -> list[Any]:
    rows.sort(key=lambda r: _coord_sort_key(r.corpus, r.vocab_size, r.boundary))
    return rows


def cross_axis_cells(*, include_extras: bool = False) -> list[CrossAxisCell]:
    """Per-cell cross-arm scalars joined across J / fertility / distribution.

    The matched grid cells present in all three measurement tables; the
    substrate the cross-V trend and interaction figures are assembled from.
    """
    jaccard = _index(_section(read_table(JACCARD_TABLE), "matched"))
    fertility = _index(_section(read_table(FERTILITY_TABLE), "matched"))
    distribution = _index(_section(read_table(DISTRIBUTION_TABLE), "matched"))
    out: list[CrossAxisCell] = []
    for key, jr in jaccard.items():
        if not _keep(jr, include_extras=include_extras):
            continue
        fr = fertility.get(key)
        dr = distribution.get(key)
        if fr is None or dr is None:
            continue
        out.append(
            CrossAxisCell(
                pair_key=str(jr["pair_key"]),
                corpus=str(jr["corpus"]),
                vocab_size=int(jr["vocab_size"]),
                boundary=str(jr["boundary"]),
                jaccard=float(jr["jaccard"]),
                rel_fertility=float(fr["delta_fertility_relative"]),
                abs_delta_d=float(dr["abs_delta_d"]),
                delta_fertility_signed=float(fr["delta_fertility"]),
                delta_d_signed=float(dr["delta_d"]),
            )
        )
    return _sorted(out)


def jaccard_rows(*, include_extras: bool = False) -> list[JaccardRow]:
    """The three Jaccards per matched pair from ``jaccard_table``."""
    out: list[JaccardRow] = []
    for r in _section(read_table(JACCARD_TABLE), "matched"):
        if not _keep(r, include_extras=include_extras):
            continue
        ci_lo = _opt_float(r.get("weighted_jaccard_ci_lo"))
        ci_hi = _opt_float(r.get("weighted_jaccard_ci_hi"))
        ci = (ci_lo, ci_hi) if ci_lo is not None and ci_hi is not None else None
        sci_lo = _opt_float(r.get("weighted_jaccard_struct_ci_lo"))
        sci_hi = _opt_float(r.get("weighted_jaccard_struct_ci_hi"))
        sci = (sci_lo, sci_hi) if sci_lo is not None and sci_hi is not None else None
        out.append(
            JaccardRow(
                pair_key=str(r["pair_key"]),
                corpus=str(r["corpus"]),
                vocab_size=int(r["vocab_size"]),
                boundary=str(r["boundary"]),
                tier=str(r["tier"]),
                jaccard=float(r["jaccard"]),
                jaccard_struct=_opt_float(r.get("jaccard_struct")),
                weighted_jaccard=_opt_float(r.get("weighted_jaccard")),
                weighted_jaccard_ci=ci,
                weighted_jaccard_struct=_opt_float(r.get("weighted_jaccard_struct")),
                weighted_jaccard_struct_ci=sci,
                bpe_n_multi=_opt_int(r.get("bpe_n_multi")),
                unigram_n_multi=_opt_int(r.get("unigram_n_multi")),
            )
        )
    return _sorted(out)


def fertility_rows(*, include_extras: bool = False) -> list[FertilityRow]:
    """Absolute per-arm fertility + compression ratio per matched pair."""

    out: list[FertilityRow] = []
    for r in _section(read_table(FERTILITY_TABLE), "matched"):
        if not _keep(r, include_extras=include_extras):
            continue
        out.append(
            FertilityRow(
                pair_key=str(r["pair_key"]),
                corpus=str(r["corpus"]),
                vocab_size=int(r["vocab_size"]),
                boundary=str(r["boundary"]),
                tier=str(r["tier"]),
                bpe_fertility=float(r["bpe_fertility"]),
                bpe_fertility_ci=_row_ci(r, "bpe_fertility"),
                unigram_fertility=float(r["unigram_fertility"]),
                unigram_fertility_ci=_row_ci(r, "unigram_fertility"),
                bpe_glyphs_per_token=float(r["bpe_glyphs_per_token"]),
                bpe_glyphs_per_token_ci=_row_ci(r, "bpe_glyphs_per_token"),
                unigram_glyphs_per_token=float(r["unigram_glyphs_per_token"]),
                unigram_glyphs_per_token_ci=_row_ci(r, "unigram_glyphs_per_token"),
                delta_fertility=float(r["delta_fertility"]),
                rel_fertility=float(r["delta_fertility_relative"]),
            )
        )
    return _sorted(out)


def nestedness_rows(*, include_extras: bool = False) -> list[NestednessRow]:
    """Cross-arm boundary-nestedness readings per matched pair."""
    out: list[NestednessRow] = []
    for r in _section(read_table(NESTEDNESS_TABLE), "matched"):
        if not _keep(r, include_extras=include_extras):
            continue
        out.append(
            NestednessRow(
                pair_key=str(r["pair_key"]),
                corpus=str(r["corpus"]),
                vocab_size=int(r["vocab_size"]),
                boundary=str(r["boundary"]),
                tier=str(r["tier"]),
                boundary_jaccard=float(r["boundary_jaccard"]),
                conflict_rate=float(r["conflict_rate"]),
                nest_rate=float(r["nest_rate"]),
                nested_molecule_fraction=float(r["nested_molecule_fraction"]),
                cut_rate_heteroatom=_opt_float(r.get("cut_rate_heteroatom")),
                cut_rate_unsat_c=_opt_float(r.get("cut_rate_unsat_c")),
                cut_rate_sat_c=_opt_float(r.get("cut_rate_sat_c")),
            )
        )
    return _sorted(out)


def closure_rows(*, include_extras: bool = False) -> list[ClosureRow]:
    """Within-arm compositional-closure readings per matched pair."""
    out: list[ClosureRow] = []
    for r in _section(read_table(CLOSURE_TABLE), "matched"):
        if not _keep(r, include_extras=include_extras):
            continue
        out.append(
            ClosureRow(
                pair_key=str(r["pair_key"]),
                corpus=str(r["corpus"]),
                vocab_size=int(r["vocab_size"]),
                boundary=str(r["boundary"]),
                tier=str(r["tier"]),
                bpe_c_bin=float(r["bpe_c_bin"]),
                ul_c_bin=float(r["ul_c_bin"]),
                ul_c_orph=float(r["ul_c_orph"]),
                bpe_c_full=float(r["bpe_c_full"]),
                ul_c_full=float(r["ul_c_full"]),
            )
        )
    return _sorted(out)


def fg_alignment_rows(*, include_extras: bool = False) -> list[FgAlignmentRow]:
    """Within-arm functional-bond-locality readings per matched pair."""
    out: list[FgAlignmentRow] = []
    for r in _section(read_table(FG_ALIGNMENT_TABLE), "matched"):
        if not _keep(r, include_extras=include_extras):
            continue
        out.append(
            FgAlignmentRow(
                pair_key=str(r["pair_key"]),
                corpus=str(r["corpus"]),
                vocab_size=int(r["vocab_size"]),
                boundary=str(r["boundary"]),
                tier=str(r["tier"]),
                bpe_locality=float(r["bpe_locality"]),
                ul_locality=float(r["ul_locality"]),
                delta_locality=float(r["delta_locality"]),
                bpe_carbonyl=float(r["bpe_carbonyl"]),
                ul_carbonyl=float(r["ul_carbonyl"]),
            )
        )
    return _sorted(out)


def _ratio_to_rel_gap(ratio: float) -> float:
    """Symmetric relative fertility gap rel|df| = |df|/mean = 2(r-1)/(r+1),
    the paper's standard granularity metric, from a Unigram/BPE ratio r."""
    return 2.0 * (ratio - 1.0) / (ratio + 1.0)


def noncanon_rows(*, include_extras: bool = False) -> list[NoncanonRow]:
    """Within-arm SMILES-rewrite-robustness readings per matched pair."""
    out: list[NoncanonRow] = []
    for r in _section(read_table(NONCANON_TABLE), "matched"):
        if not _keep(r, include_extras=include_extras):
            continue
        out.append(
            NoncanonRow(
                pair_key=str(r["pair_key"]),
                corpus=str(r["corpus"]),
                vocab_size=int(r["vocab_size"]),
                boundary=str(r["boundary"]),
                tier=str(r["tier"]),
                bpe_random=float(r["bpe_bag_random"]),
                ul_random=float(r["ul_bag_random"]),
                bpe_kekule=float(r["bpe_bag_kekule"]),
                ul_kekule=float(r["ul_bag_kekule"]),
                bpe_explicit_h=float(r["bpe_bag_explicitH"]),
                ul_explicit_h=float(r["ul_bag_explicitH"]),
                bpe_obcanon=float(r["bpe_bag_obcanon"]),
                ul_obcanon=float(r["ul_bag_obcanon"]),
                gap_canon=_ratio_to_rel_gap(float(r["gap_canon"])),
                gap_rand=_ratio_to_rel_gap(float(r["gap_rand"])),
            )
        )
    return _sorted(out)


def distribution_rows(*, include_extras: bool = False) -> list[DistributionRow]:
    """Per-arm token-distribution intrinsics (D, eta, Renyi) per matched pair."""

    out: list[DistributionRow] = []
    for r in _section(read_table(DISTRIBUTION_TABLE), "matched"):
        if not _keep(r, include_extras=include_extras):
            continue
        out.append(
            DistributionRow(
                pair_key=str(r["pair_key"]),
                corpus=str(r["corpus"]),
                vocab_size=int(r["vocab_size"]),
                boundary=str(r["boundary"]),
                tier=str(r["tier"]),
                bpe_d=float(r["bpe_d"]),
                bpe_d_ci=_row_ci(r, "bpe_d"),
                unigram_d=float(r["unigram_d"]),
                unigram_d_ci=_row_ci(r, "unigram_d"),
                bpe_eta=float(r["bpe_eta"]),
                bpe_eta_ci=_row_ci(r, "bpe_eta"),
                unigram_eta=float(r["unigram_eta"]),
                unigram_eta_ci=_row_ci(r, "unigram_eta"),
                bpe_renyi=float(r["bpe_renyi"]),
                bpe_renyi_ci=_row_ci(r, "bpe_renyi"),
                unigram_renyi=float(r["unigram_renyi"]),
                unigram_renyi_ci=_row_ci(r, "unigram_renyi"),
                bpe_live=_opt_int(r.get("bpe_live")),
                unigram_live=_opt_int(r.get("unigram_live")),
                abs_delta_d=float(r["abs_delta_d"]),
            )
        )
    return _sorted(out)


def absorption_rows(*, include_extras: bool = False) -> list[AbsorptionRow]:
    """Per-arm whole-pretoken absorption per matched pair."""

    out: list[AbsorptionRow] = []
    for r in _section(read_table(ABSORPTION_TABLE), "matched"):
        if not _keep(r, include_extras=include_extras):
            continue
        out.append(
            AbsorptionRow(
                pair_key=str(r["pair_key"]),
                corpus=str(r["corpus"]),
                vocab_size=int(r["vocab_size"]),
                boundary=str(r["boundary"]),
                tier=str(r["tier"]),
                bpe_absorbed=float(r["bpe_absorbed_fraction"]),
                bpe_absorbed_ci=_row_ci(r, "bpe_absorbed"),
                unigram_absorbed=float(r["unigram_absorbed_fraction"]),
                unigram_absorbed_ci=_row_ci(r, "unigram_absorbed"),
                delta_absorbed=float(r["delta_absorbed"]),
            )
        )
    return _sorted(out)


def deadzone_nsweep_rows(*, include_extras: bool = False) -> list[DeadzoneNSweepRow]:
    """Per-arm rare-token clearance across n, read from per-condition deposits."""

    def _by_n(arm_block: object) -> dict[int, float]:
        raw = arm_block.get("clearance_by_n", {}) if isinstance(arm_block, dict) else {}
        return {int(n): float(v) for n, v in raw.items()}

    out: list[DeadzoneNSweepRow] = []
    for r in _section(read_table(DEADZONE_TABLE), "matched"):
        if not _keep(r, include_extras=include_extras):
            continue
        cell = read_deadzone_cell(str(r["pair_key"]))
        if cell is None:
            continue
        out.append(
            DeadzoneNSweepRow(
                pair_key=str(r["pair_key"]),
                corpus=str(r["corpus"]),
                vocab_size=int(r["vocab_size"]),
                boundary=str(r["boundary"]),
                tier=str(r["tier"]),
                bpe_c=_by_n(cell.get("bpe")),
                unigram_c=_by_n(cell.get("unigram")),
                any_arm_unsafe=bool(r.get("any_arm_unsafe")),
            )
        )
    return _sorted(out)


def delta_f_rows(*, include_extras: bool = False) -> list[DeltaFRow]:
    """Per-arm F95 clearance and the cross-arm ΔF per matched pair (Deadzone)."""
    out: list[DeltaFRow] = []
    for r in _section(read_table(DEADZONE_TABLE), "matched"):
        if not _keep(r, include_extras=include_extras):
            continue
        out.append(
            DeltaFRow(
                pair_key=str(r["pair_key"]),
                corpus=str(r["corpus"]),
                vocab_size=int(r["vocab_size"]),
                boundary=str(r["boundary"]),
                tier=str(r["tier"]),
                bpe_clearance=float(r["bpe_headline_clearance"]),
                unigram_clearance=float(r["unigram_headline_clearance"]),
                headline_delta_f=float(r["headline_delta_f"]),
                any_arm_unsafe=bool(r.get("any_arm_unsafe")),
            )
        )
    return _sorted(out)


def measurement_rows(*, include_extras: bool = False) -> list[MeasurementRow]:
    """The seven cross-arm scalars per matched pair.

    Deadzone (the matched-grid spine + ΔF) ⋈ Jaccard / Fertility / Distribution
    / Absorption / Scaffold / Segmentation, joined by ``pair_key``.
    """
    jaccard = _index(_section(read_table(JACCARD_TABLE), "matched"))
    fertility = _index(_section(read_table(FERTILITY_TABLE), "matched"))
    distribution = _index(_section(read_table(DISTRIBUTION_TABLE), "matched"))
    absorption = _index(_section(read_table(ABSORPTION_TABLE), "matched"))
    scaffold = _index(_section(read_table(SCAFFOLD_TABLE), "matched"))
    segmentation = _index(_section(read_table(SEGMENTATION_TABLE), "matched"))
    out: list[MeasurementRow] = []
    for r in _section(read_table(DEADZONE_TABLE), "matched"):
        if not _keep(r, include_extras=include_extras):
            continue
        key = str(r["pair_key"])
        out.append(
            MeasurementRow(
                pair_key=key,
                corpus=str(r["corpus"]),
                vocab_size=int(r["vocab_size"]),
                boundary=str(r["boundary"]),
                tier=str(r["tier"]),
                delta_f=float(r["headline_delta_f"]),
                delta_absorbed=_opt_float(
                    absorption.get(key, {}).get("delta_absorbed")
                ),
                delta_scaffold=_opt_float(
                    scaffold.get(key, {}).get("delta_scaffold_fraction")
                ),
                rel_fertility=float(fertility[key]["delta_fertility_relative"]),
                jaccard=float(jaccard[key]["jaccard"]),
                abs_delta_d=float(distribution[key]["abs_delta_d"]),
                delta_entropy_per_glyph=_opt_float(
                    segmentation.get(key, {}).get("delta_entropy_per_glyph")
                ),
            )
        )
    return _sorted(out)


_SIZE_SWEEP_POINTS = (
    ("pubchem__v512_nmb__size_5m", "5M"),
    ("pubchem__v512_nmb__size_15m", "15M"),
    ("pubchem__v512_nmb", "50M"),
)


def _redraw_index(pair_key: str) -> int:
    return int(pair_key.rsplit("_r", 1)[-1])


def _vocab_from_cell(cell: str) -> int:
    for part in cell.split("_"):
        if part.startswith("v") and part[1:].isdigit():
            return int(part[1:])
    return 0


def robustness_extras() -> RobustnessExtras:
    """Assemble the five robustness-extras readings from Deadzone rows + audit JSONs.

    Subsample-redraw spread and the PubChem size sweep come from the Deadzone
    per-arm clearance (the embedding-tail bar the extras probe); the seed-cap,
    prune-schedule, and REAL-Space merge-exhaustion readings come from the
    bespoke ``data/audits/*.json`` deposits.
    """
    deadzone_matched = _section(read_table(DEADZONE_TABLE), "matched")
    deadzone_by_key = {str(r["pair_key"]): r for r in deadzone_matched}

    by_corpus: dict[str, list[tuple[int, float]]] = {}
    for r in deadzone_matched:
        if r.get("extras_kind") != "subsample_redraw":
            continue
        clr = _opt_float(r.get("unigram_headline_clearance"))
        if clr is not None:
            by_corpus.setdefault(str(r["corpus"]), []).append(
                (_redraw_index(str(r["pair_key"])), clr)
            )
    redraws = tuple(
        RedrawSpread(corpus=c, clearances=tuple(v for _, v in sorted(pairs)))
        for c, pairs in sorted(
            by_corpus.items(), key=lambda kv: CORPUS_RANK.get(kv[0], 99)
        )
    )

    size_sweep: list[SizeSweepPoint] = []
    for key, label in _SIZE_SWEEP_POINTS:
        clr = _opt_float(
            (deadzone_by_key.get(key) or {}).get("unigram_headline_clearance")
        )
        if clr is not None:
            size_sweep.append(SizeSweepPoint(n_label=label, unigram_clearance=clr))

    seed = read_audit("seed_cap") or {}
    prune = read_audit("prune_schedule") or {}
    merge = read_audit("merge_exhaustion") or {}

    prune_rows = tuple(
        PruneComparison(
            vocab_size=_vocab_from_cell(str(c.get("baseline_cell", ""))),
            jaccard=float(c["multi_glyph_jaccard"]),
        )
        for c in prune.get("comparisons", [])
        if c.get("multi_glyph_jaccard") is not None
    )

    return RobustnessExtras(
        redraws=redraws,
        size_sweep=tuple(size_sweep),
        seed_cap_jaccard=_opt_float(seed.get("multi_glyph_jaccard")),
        seed_cap_symmetric_difference=(
            None
            if seed.get("symmetric_difference_count") is None
            else int(seed["symmetric_difference_count"])
        ),
        prune=prune_rows,
        merge_exhaustion_realised_v=(
            None
            if merge.get("vocab_size_realised") is None
            else int(merge["vocab_size_realised"])
        ),
        merge_exhaustion_cap=(
            None
            if merge.get("vocab_size_cap") is None
            else int(merge["vocab_size_cap"])
        ),
        merge_exhaustion_natural=_opt_bool(merge.get("natural_termination")),
    )


__all__ = [
    "ABSORPTION_TABLE",
    "AUDIT_ITEMS",
    "CLOSURE_TABLE",
    "DEADZONE_TABLE",
    "DISTRIBUTION_TABLE",
    "FERTILITY_TABLE",
    "FG_ALIGNMENT_TABLE",
    "JACCARD_TABLE",
    "NESTEDNESS_TABLE",
    "NONCANON_TABLE",
    "REQUIRED_TABLES",
    "SCAFFOLD_TABLE",
    "SEGMENTATION_TABLE",
    "AbsorptionRow",
    "ClosureRow",
    "CrossAxisCell",
    "DeadzoneNSweepRow",
    "DeltaFRow",
    "DistributionRow",
    "FertilityRow",
    "FgAlignmentRow",
    "JaccardRow",
    "MeasurementRow",
    "NestednessRow",
    "NoncanonRow",
    "PruneComparison",
    "RedrawSpread",
    "RobustnessExtras",
    "SizeSweepPoint",
    "absorption_rows",
    "audit_upstream_sha_map",
    "closure_rows",
    "cross_axis_cells",
    "deadzone_nsweep_rows",
    "delta_f_rows",
    "distribution_rows",
    "fertility_rows",
    "fg_alignment_rows",
    "jaccard_rows",
    "measurement_rows",
    "missing_audits",
    "missing_tables",
    "nestedness_rows",
    "noncanon_rows",
    "read_audit",
    "read_deadzone_cell",
    "read_table",
    "robustness_extras",
    "table_path",
    "upstream_sha_map",
]
