"""The robustness extras: axes the 44-cell grid lacks.

The grid + conditional cells (``grid.py``) cover the four study axes (algo, V,
corpus, boundary); the extras add training-subsample redraw, training-corpus
size, Unigram seed-cap, Unigram prune-schedule, and BPE merge-exhaustion, so
they get their own spec module. Single code-side source of truth:
:func:`enumerate_all` generates the 92 cells, ``configs/tokenizer/extras.yaml``
is the committed manifest, and a test pins manifest to enumeration so neither
drifts.

The core structural probes:

1. **Training-subsample spot-check** (12 cells): (V=512, ZINC-22, NMB) +
   (V=512, PubChem, NMB) × both algos × 3 independent redraws each. Bounds
   subsample noise on the headline (J, |Δfertility|, |ΔD|) triple.
2. **Within-PubChem size sweep** (4 cells): (V=512, PubChem, NMB) × both
   algos × {~5M, ~15M} training-corpus subsamples nested under the ~50M draw.
3. **Seed-cap spot-check** (1 cell): worst-case Unigram cell
   (V=1024, PubChem, MB), ``seed_size`` raised until non-binding. Piece-set
   identity vs the 10⁶-capped headline ⇒ cap inert.
4. **Prune-schedule spot-check** (1 + 1 contingency): worst-case Unigram cell
   (V=256, PubChem, MB), ``shrinking_factor=0.9`` (default 0.75). The V=256
   multi-glyph Jaccard landed at 0.83, firing the contingency: a second cell
   at V=512.
5. **REAL-Space merge-exhaustion continuity cell** (1 cell): NMB
   ``GpeTrainer`` on REAL-Space ``canon_dedup_v1`` targeting Wadell's
   ``V=50_000``; terminates at ``V≈2.3K`` when bigram merges exhaust. Matched-V
   continuity check vs Wadell.

Beyond these, the spec enumerates a **size-matched** pair (PubChem / ZINC-22 at
``V=1024``), a **large-V convergence anchor**, and the **hyperparameter
sensitivity battery** (shared anchors, off-default OFAT ladders, three
interaction grids + same-size BPE references). Per-group breakdown in
:func:`enumerate_all`; 92 cells total.

Each cell is trained twice for the per-arm determinism assertion: BPE
byte-identical, Unigram piece-set identical.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from smiles_subword.config import (
    TokenizerConfig,
    TokenizerKind,
    YamlLoadable,
    algo_to_kind,
    cell_artifact_name,
)
from smiles_subword.paths import (
    CONFIGS_DIR,
    processed_corpus_dir,
    tokenizer_artifact_dir,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

EXTRAS_MANIFEST_VERSION = 1
EXTRAS_MANIFEST_PATH = CONFIGS_DIR / "tokenizer" / "extras.yaml"

ExtrasKind = Literal[
    "subsample_redraw",
    "size_sweep",
    "size_matched",
    "seed_cap",
    "prune_schedule",
    "merge_exhaustion",
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
    "large_v_anchor",
]
ExtrasCorpus = Literal["pubchem", "zinc22", "real_space", "coconut"]
ExtrasAlgo = Literal["bpe", "unigram"]
ExtrasBoundary = Literal["nmb", "mb"]

_SUBSAMPLE_REDRAW_LABELS: tuple[str, ...] = ("r1", "r2", "r3")
_SIZE_SWEEP_LABELS: tuple[str, ...] = ("5m", "15m")
_SUBSAMPLE_VOCAB_SIZE = 512
_SIZE_SWEEP_VOCAB_SIZE = 512
_SIZE_MATCHED_VOCAB_SIZE = 1024
_SIZE_MATCHED_CORPORA: tuple[ExtrasCorpus, ...] = ("pubchem", "zinc22")
_SIZE_MATCHED_LABEL = "700k"
_SEED_CAP_VOCAB_SIZE = 1024
_PRUNE_SCHEDULE_VOCAB_SIZES: tuple[int, ...] = (256, 512)
"""V=256 is the headline probe; V=512 is the contingency, added because the
V=256 multi-glyph Jaccard vs the default-schedule headline landed at 0.83."""
_MERGE_EXHAUSTION_VOCAB_SIZE = 50_000
"""Wadell's BPE target (`smirk` v0.2.0); `GpeTrainer` terminates when bigram
merges exhaust well below this. Realised terminal `V` is recorded in
``meta.yaml`` and asserted < this cap by the post-train audit."""
# Large-V convergence anchor: both arms at one power-of-2 V above the headline
# range, on PubChem (widest alphabet, so the Unigram arm saturates least).
# Shows the leading edge of convergence (J rising, fertility gap shrinking).
_LARGE_V_ANCHOR_VOCAB_SIZE = 8192
_LARGE_V_ANCHOR_CORPUS: ExtrasCorpus = "pubchem"
_LARGE_V_ANCHOR_BOUNDARY: ExtrasBoundary = "nmb"

SEED_CAP_OVERRIDE: int = 8_000_000
"""Seed-cap probe. 8× the default `seed_size=1_000_000`, above any plausible
Layer-B-chunk substring count for the worst-case PubChem MB cell. Piece-set
identity vs the 10⁶-capped headline ⇒ cap inert."""

PRUNE_SCHEDULE_SHRINKING_FACTOR: float = 0.9
"""Prune-schedule probe. Coarsened shrinking factor (default 0.75 → 0.9);
``n_sub_iterations`` stays at its default of 2."""

# --- Hyperparameter sensitivity battery ---------------------
# Ladders and interactions share one anchor coordinate; the all-defaults anchor
# is enumerated once (``sensitivity_anchor``) and reassembled into each ladder /
# interaction grid at measurement time, so no tokenizer trains more than once.
SENSITIVITY_CORPUS: ExtrasCorpus = "pubchem"
SENSITIVITY_BOUNDARY: ExtrasBoundary = "nmb"
SENSITIVITY_VOCAB_SIZE = 512
SENSITIVITY_SUBSAMPLE = "size_700k"
"""The subsample the sensitivity battery trains on: PubChem at COCONUT scale
(~702K). Only this sweep is subsampled; the headline grid stays full-corpus."""

MPL_VALUES: tuple[int, ...] = (4, 8, 16, 32, 64, 128)
SEED_VALUES: tuple[int, ...] = (
    250_000,
    500_000,
    1_000_000,
    2_000_000,
    4_000_000,
    8_000_000,
)
SUBITER_VALUES: tuple[int, ...] = (1, 2, 3, 4)
SHRINK_VALUES: tuple[float, ...] = (0.5, 0.6, 0.75, 0.9, 0.95)
MINFREQ_VALUES: tuple[int, ...] = (0, 1, 2, 4, 8)

MPL_DEFAULT = 16
SEED_DEFAULT = 1_000_000
SUBITER_DEFAULT = 2
SHRINK_DEFAULT = 0.75
MINFREQ_DEFAULT = 2

INTERACTION_V_RUNGS: tuple[int, ...] = (256, 1024)
"""Off-anchor V rows for interaction B; V=512 is the anchor column."""
INTERACTION_TYPOLOGY_CORPORA: tuple[ExtrasCorpus, ...] = ("zinc22", "coconut")
"""Off-anchor corpora for interaction C; PubChem is the anchor column."""

# The interaction-surface cross-arm contrast (J, J_struct, |Δf|) pairs each
# swept Unigram cell against the matching BPE arm at the *same* (corpus, V,
# boundary, subsample). BPE has no piece-length cap, so one default-knob BPE
# reference per off-anchor coordinate suffices. The anchor column (PubChem
# V=512) reuses ``sensitivity_anchor`` and COCONUT V=512 is the headline cell;
# only these three references are enumerated here.
INTERACTION_BPE_REF_PUBCHEM_V: tuple[int, ...] = (256, 1024)
"""Interaction B off-anchor V rows need a same-size BPE reference (PubChem
``size_700k``); V=512 is covered by the BPE anchor."""


def _flabel(x: float) -> str:
    """Filesystem-safe label for a numeric rung (``0.5`` → ``"0_5"``)."""
    return str(x).replace(".", "_")


class ExtrasCell(BaseModel):
    """One robustness-extra cell.

    ``label`` is the per-kind discriminator (e.g. ``"r1"`` for a subsample
    redraw, ``"uncapped"`` for the seed-cap probe). Frozen and hashable so
    manifest / enumeration equality can be checked as sets.

    Extras-specific overrides on the training pipeline:

    * ``training_subdir`` — trains from
      ``data/processed/<corpus>/canon_dedup_v1_extras/<training_subdir>/``
      rather than the default ``canon_dedup_v1/train/``.
    * ``seed_size_override`` — overrides ``seed_size=1_000_000`` (Unigram).
    * ``shrinking_factor_override`` — overrides ``shrinking_factor=0.75`` (Unigram).
    * ``max_piece_length_override`` / ``min_frequency_override`` /
      ``n_sub_iterations_override`` — sensitivity-ladder overrides on the
      Unigram (``max_piece_length``, ``n_sub_iterations``) or BPE
      (``min_frequency``) trainer.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    extras_kind: ExtrasKind
    algo: ExtrasAlgo
    vocab_size: int = Field(ge=1)
    corpus: ExtrasCorpus
    boundary: ExtrasBoundary
    label: str = Field(min_length=1)
    training_subdir: str | None = None
    seed_size_override: int | None = Field(default=None, ge=1)
    shrinking_factor_override: float | None = Field(default=None, gt=0.0, lt=1.0)
    max_piece_length_override: int | None = Field(default=None, ge=1)
    min_frequency_override: int | None = Field(default=None, ge=0)
    n_sub_iterations_override: int | None = Field(default=None, ge=1)

    @property
    def kind(self) -> TokenizerKind:
        """The :data:`TokenizerKind` the algo axis maps to (:func:`algo_to_kind`).

        Shares the :func:`algo_to_kind` rule with
        :attr:`smiles_subword.tokenize.grid.GridCell.kind`, so the audit
        functions accept either cell type through a structural bound.
        """
        return algo_to_kind(self.algo)

    @property
    def name(self) -> str:
        """Artifact directory name — ``smirk_{gpe|unigram}_v<V>_<boundary>_<label>``.

        Built via :func:`cell_artifact_name` with the per-kind discriminator
        suffix from :data:`_SUFFIX_BY_KIND`.
        """
        suffix = _SUFFIX_BY_KIND[self.extras_kind].format(label=self.label)
        return cell_artifact_name(
            self.algo, self.vocab_size, self.boundary, suffix=suffix
        )

    @property
    def cell_id(self) -> str:
        """Globally-unique cell identifier — ``{corpus}__{name}``."""
        return f"{self.corpus}__{self.name}"

    @property
    def tier(self) -> str:
        """Audit JSON category — ``"extras_" + extras_kind``, keeping extras
        rows distinct from grid tiers in joins over deposited F95 / determinism
        payloads.
        """
        return f"extras_{self.extras_kind}"


_SUFFIX_BY_KIND: dict[ExtrasKind, str] = {
    "subsample_redraw": "subsample_{label}",
    "size_sweep": "size_{label}",
    "size_matched": "sizematched_{label}",
    "seed_cap": "seed_{label}",
    "prune_schedule": "prune_{label}",
    "merge_exhaustion": "{label}",
    "sensitivity_anchor": "sens_anchor",
    "mpl_ladder": "mpl_{label}",
    "seed_ladder": "seedsweep_{label}",
    "subiter_ladder": "subiter_{label}",
    "shrink_ladder": "shrink_{label}",
    "minfreq_ladder": "minfreq_{label}",
    "interaction_subiter_shrink": "ix_subiter_shrink_{label}",
    "interaction_mpl_v": "ix_mpl_v_{label}",
    "interaction_mpl_typology": "ix_mpl_typ_{label}",
    "interaction_bpe_ref": "ix_bperef",
    "large_v_anchor": "{label}",
}


class ExtrasManifest(YamlLoadable):
    """The committed extras manifest — ``configs/tokenizer/extras.yaml``."""

    model_config = ConfigDict(extra="forbid")

    version: int = Field(ge=1)
    cells: list[ExtrasCell]


def enumerate_subsample_redraws() -> tuple[ExtrasCell, ...]:
    """The 12-cell training-subsample spot-check.

    ``(V=512, ZINC-22, NMB)`` + ``(V=512, PubChem, NMB)`` × both algos
    × 3 independent training-corpus redraws each.
    """
    corpora: tuple[ExtrasCorpus, ...] = ("zinc22", "pubchem")
    algos: tuple[ExtrasAlgo, ...] = ("bpe", "unigram")
    return tuple(
        ExtrasCell(
            extras_kind="subsample_redraw",
            algo=algo,
            vocab_size=_SUBSAMPLE_VOCAB_SIZE,
            corpus=corpus,
            boundary="nmb",
            label=label,
            training_subdir=f"redraw_{label}",
        )
        for corpus in corpora
        for algo in algos
        for label in _SUBSAMPLE_REDRAW_LABELS
    )


def enumerate_size_sweep() -> tuple[ExtrasCell, ...]:
    """The 4-cell within-PubChem size sweep.

    ``(V=512, PubChem, NMB)`` × both algos × {~5M, ~15M} nested subsamples.
    """
    algos: tuple[ExtrasAlgo, ...] = ("bpe", "unigram")
    return tuple(
        ExtrasCell(
            extras_kind="size_sweep",
            algo=algo,
            vocab_size=_SIZE_SWEEP_VOCAB_SIZE,
            corpus="pubchem",
            boundary="nmb",
            label=label,
            training_subdir=f"size_{label}",
        )
        for algo in algos
        for label in _SIZE_SWEEP_LABELS
    )


def enumerate_size_matched() -> tuple[ExtrasCell, ...]:
    """The 8-cell size-matched typology probe.

    Subsample PubChem and ZINC-22 to COCONUT's ~702K size and retrain the
    ``V=1024`` matched pairs under both boundary policies, isolating the
    size×typology confound behind COCONUT's ``V=1024`` imbalance shortfall.
    A *paired* kind (both arms at one coordinate), so Jaccard/Distribution
    read it as a cross-arm contrast; trains from
    ``canon_dedup_v1_extras/size_700k``.
    """
    algos: tuple[ExtrasAlgo, ...] = ("bpe", "unigram")
    boundaries: tuple[ExtrasBoundary, ...] = ("nmb", "mb")
    return tuple(
        ExtrasCell(
            extras_kind="size_matched",
            algo=algo,
            vocab_size=_SIZE_MATCHED_VOCAB_SIZE,
            corpus=corpus,
            boundary=boundary,
            label=_SIZE_MATCHED_LABEL,
            training_subdir=SENSITIVITY_SUBSAMPLE,
        )
        for corpus in _SIZE_MATCHED_CORPORA
        for algo in algos
        for boundary in boundaries
    )


def enumerate_seed_cap() -> tuple[ExtrasCell, ...]:
    """The 1-cell seed-cap spot-check.

    Worst-case Unigram cell: widest-alphabet PubChem, MB, largest headline
    ``V=1024``. ``seed_size`` raised to :data:`SEED_CAP_OVERRIDE`.
    """
    return (
        ExtrasCell(
            extras_kind="seed_cap",
            algo="unigram",
            vocab_size=_SEED_CAP_VOCAB_SIZE,
            corpus="pubchem",
            boundary="mb",
            label="uncapped",
            seed_size_override=SEED_CAP_OVERRIDE,
        ),
    )


def enumerate_prune_schedule() -> tuple[ExtrasCell, ...]:
    """The prune-schedule spot-check cells.

    Worst-case Unigram cell: widest-alphabet PubChem, MB; ``shrinking_factor``
    coarsened from 0.75 to :data:`PRUNE_SCHEDULE_SHRINKING_FACTOR`. V=256 is the
    headline probe; V=512 the contingency, added after the V=256 Jaccard landed
    at 0.83.
    """
    return tuple(
        ExtrasCell(
            extras_kind="prune_schedule",
            algo="unigram",
            vocab_size=v,
            corpus="pubchem",
            boundary="mb",
            label="shrink_0_9",
            shrinking_factor_override=PRUNE_SCHEDULE_SHRINKING_FACTOR,
        )
        for v in _PRUNE_SCHEDULE_VOCAB_SIZES
    )


def enumerate_merge_exhaustion() -> tuple[ExtrasCell, ...]:
    """The 1-cell REAL-Space merge-exhaustion continuity cell.

    NMB stock ``GpeTrainer`` on REAL-Space ``canon_dedup_v1`` targeting Wadell's
    ``V=50_000``; terminates at ``V≈2.3K`` when bigram merges exhaust. Realised
    terminal ``V`` recorded in ``meta.yaml`` and asserted < the cap by the
    post-train audit.
    """
    return (
        ExtrasCell(
            extras_kind="merge_exhaustion",
            algo="bpe",
            vocab_size=_MERGE_EXHAUSTION_VOCAB_SIZE,
            corpus="real_space",
            boundary="nmb",
            label="merge_exhaustion",
        ),
    )


def enumerate_large_v_anchor() -> tuple[ExtrasCell, ...]:
    """The 2-cell PubChem large-V convergence anchor.

    Both arms at ``V=8192``, one power-of-2 rung above the headline grid, on
    full-corpus PubChem NMB. Paired by coordinate like a grid pair, so the
    cross-arm Jaccard and fertility drivers pick it up automatically.
    """
    return tuple(
        ExtrasCell(
            extras_kind="large_v_anchor",
            algo=algo,
            vocab_size=_LARGE_V_ANCHOR_VOCAB_SIZE,
            corpus=_LARGE_V_ANCHOR_CORPUS,
            boundary=_LARGE_V_ANCHOR_BOUNDARY,
            label="convergence_anchor",
        )
        for algo in ("bpe", "unigram")
    )


def enumerate_sensitivity_anchor() -> tuple[ExtrasCell, ...]:
    """The two shared anchor cells (Unigram + BPE) for the sensitivity battery.

    All knobs at reference defaults, PubChem V512 NMB, ``size_700k`` subsample.
    Every ladder and interaction grid includes this point, so it is enumerated
    once rather than as a rung of each sweep.
    """
    return tuple(
        ExtrasCell(
            extras_kind="sensitivity_anchor",
            algo=algo,
            vocab_size=SENSITIVITY_VOCAB_SIZE,
            corpus=SENSITIVITY_CORPUS,
            boundary=SENSITIVITY_BOUNDARY,
            label="anchor",
            training_subdir=SENSITIVITY_SUBSAMPLE,
        )
        for algo in ("unigram", "bpe")
    )


def _ladder(
    extras_kind: ExtrasKind,
    algo: ExtrasAlgo,
    *,
    override_field: str,
    values: tuple[float, ...],
    default: float,
    label: Callable[[float], str] = str,
) -> tuple[ExtrasCell, ...]:
    """One off-default OFAT ladder at the sensitivity-anchor coordinate.

    Emits one cell per off-default rung in ``values``, setting the single
    ``*_override`` field named by ``override_field``. Shared body of the five
    ``enumerate_*_ladder`` factories. ``override`` is ``dict[str, Any]`` because
    ``override_field`` selects one of several heterogeneously-typed fields (int
    for most, float for ``shrinking_factor``); validated by ``ExtrasCell``.
    """
    cells: list[ExtrasCell] = []
    for v in values:
        if v == default:
            continue
        override: dict[str, Any] = {override_field: v}
        cells.append(
            ExtrasCell(
                extras_kind=extras_kind,
                algo=algo,
                vocab_size=SENSITIVITY_VOCAB_SIZE,
                corpus=SENSITIVITY_CORPUS,
                boundary=SENSITIVITY_BOUNDARY,
                label=label(v),
                training_subdir=SENSITIVITY_SUBSAMPLE,
                **override,
            )
        )
    return tuple(cells)


def enumerate_mpl_ladder() -> tuple[ExtrasCell, ...]:
    """Unigram ``max_piece_length`` ladder (off-default rungs) at the anchor."""
    return _ladder(
        "mpl_ladder",
        "unigram",
        override_field="max_piece_length_override",
        values=MPL_VALUES,
        default=MPL_DEFAULT,
    )


def enumerate_seed_ladder() -> tuple[ExtrasCell, ...]:
    """Unigram ``seed_size`` ladder (off-default rungs) at the anchor."""
    return _ladder(
        "seed_ladder",
        "unigram",
        override_field="seed_size_override",
        values=SEED_VALUES,
        default=SEED_DEFAULT,
    )


def enumerate_subiter_ladder() -> tuple[ExtrasCell, ...]:
    """Unigram ``n_sub_iterations`` ladder (off-default rungs) at the anchor."""
    return _ladder(
        "subiter_ladder",
        "unigram",
        override_field="n_sub_iterations_override",
        values=SUBITER_VALUES,
        default=SUBITER_DEFAULT,
    )


def enumerate_shrink_ladder() -> tuple[ExtrasCell, ...]:
    """Unigram ``shrinking_factor`` ladder (off-default rungs) at the anchor."""
    return _ladder(
        "shrink_ladder",
        "unigram",
        override_field="shrinking_factor_override",
        values=SHRINK_VALUES,
        default=SHRINK_DEFAULT,
        label=_flabel,
    )


def enumerate_minfreq_ladder() -> tuple[ExtrasCell, ...]:
    """BPE ``min_frequency`` ladder (off-default rungs) at the anchor."""
    return _ladder(
        "minfreq_ladder",
        "bpe",
        override_field="min_frequency_override",
        values=MINFREQ_VALUES,
        default=MINFREQ_DEFAULT,
    )


def enumerate_interaction_subiter_shrink() -> tuple[ExtrasCell, ...]:
    """Interaction A: Unigram ``n_sub_iterations`` × ``shrinking_factor`` interior.

    The 4×5 grid minus its default row/column (the subiter and shrink ladders)
    and the anchor — the off-default×off-default interior, reassembled full at
    measurement time.
    """
    return tuple(
        ExtrasCell(
            extras_kind="interaction_subiter_shrink",
            algo="unigram",
            vocab_size=SENSITIVITY_VOCAB_SIZE,
            corpus=SENSITIVITY_CORPUS,
            boundary=SENSITIVITY_BOUNDARY,
            label=f"si{si}_sf{_flabel(sf)}",
            training_subdir=SENSITIVITY_SUBSAMPLE,
            n_sub_iterations_override=si,
            shrinking_factor_override=sf,
        )
        for si in SUBITER_VALUES
        if si != SUBITER_DEFAULT
        for sf in SHRINK_VALUES
        if sf != SHRINK_DEFAULT
    )


def enumerate_interaction_mpl_v() -> tuple[ExtrasCell, ...]:
    """Interaction B: Unigram ``max_piece_length`` × V, off-anchor V rows.

    Full 6×3 grid; V=512 is the anchor column (mpl ladder + anchor), so only
    V ∈ {256, 1024} × the full ``max_piece_length`` set is enumerated here.
    """
    return tuple(
        ExtrasCell(
            extras_kind="interaction_mpl_v",
            algo="unigram",
            vocab_size=v,
            corpus=SENSITIVITY_CORPUS,
            boundary=SENSITIVITY_BOUNDARY,
            label=str(mpl),
            training_subdir=SENSITIVITY_SUBSAMPLE,
            max_piece_length_override=mpl,
        )
        for v in INTERACTION_V_RUNGS
        for mpl in MPL_VALUES
    )


def enumerate_interaction_mpl_typology() -> tuple[ExtrasCell, ...]:
    """Interaction C: Unigram ``max_piece_length`` × typology, off-anchor corpora.

    Full 6×3 grid; PubChem is the anchor column, so only ZINC-22 and COCONUT ×
    the full ``max_piece_length`` set is enumerated here, at COCONUT-matched
    size: ZINC-22 on its ``size_700k`` subsample, COCONUT on its full (~702K)
    train.
    """
    return tuple(
        ExtrasCell(
            extras_kind="interaction_mpl_typology",
            algo="unigram",
            vocab_size=SENSITIVITY_VOCAB_SIZE,
            corpus=corpus,
            boundary=SENSITIVITY_BOUNDARY,
            label=str(mpl),
            training_subdir=(SENSITIVITY_SUBSAMPLE if corpus == "zinc22" else None),
            max_piece_length_override=mpl,
        )
        for corpus in INTERACTION_TYPOLOGY_CORPORA
        for mpl in MPL_VALUES
    )


def enumerate_interaction_bpe_refs() -> tuple[ExtrasCell, ...]:
    """Off-anchor BPE references for the interaction surfaces (B and C).

    Three default-knob BPE cells on the ``size_700k`` subsample: PubChem at
    V ∈ {256, 1024} (interaction B off-anchor rows) and ZINC-22 at V=512
    (interaction C off-anchor column). The anchor column (PubChem V=512) and
    COCONUT V=512 (full-corpus headline) are already trained, so the swept
    Unigram cells of both grids each have a same-size BPE arm to contrast.
    """
    refs = [
        ExtrasCell(
            extras_kind="interaction_bpe_ref",
            algo="bpe",
            vocab_size=v,
            corpus="pubchem",
            boundary=SENSITIVITY_BOUNDARY,
            label="ref",
            training_subdir=SENSITIVITY_SUBSAMPLE,
        )
        for v in INTERACTION_BPE_REF_PUBCHEM_V
    ]
    refs.append(
        ExtrasCell(
            extras_kind="interaction_bpe_ref",
            algo="bpe",
            vocab_size=SENSITIVITY_VOCAB_SIZE,
            corpus="zinc22",
            boundary=SENSITIVITY_BOUNDARY,
            label="ref",
            training_subdir=SENSITIVITY_SUBSAMPLE,
        )
    )
    return tuple(refs)


def enumerate_all() -> tuple[ExtrasCell, ...]:
    """All robustness extras and sensitivity cells.

    28 structural probes (12 subsample redraws, 4 size-sweep, 8 size-matched, 1
    seed-cap, 2 prune-schedule, 1 merge-exhaustion) plus the 2 large-V
    convergence-anchor cells, plus the 62-cell sensitivity battery: 2 shared
    anchors, five off-default OFAT ladders (mpl 5, seed 5, subiter 3, shrink 4,
    minfreq 4), and the off-anchor cells of three interaction grids (A 12, B
    12, C 12), with 3 off-anchor BPE references giving the B/C interaction
    surfaces a same-size cross-arm contrast. Interaction D (BPE ``min_frequency``
    × Unigram ``max_piece_length``) adds no cells — it is a measurement-time
    crossing of the minfreq and mpl ladders. Total: 92.
    """
    return (
        enumerate_subsample_redraws()
        + enumerate_size_sweep()
        + enumerate_size_matched()
        + enumerate_seed_cap()
        + enumerate_prune_schedule()
        + enumerate_merge_exhaustion()
        + enumerate_large_v_anchor()
        + enumerate_sensitivity_anchor()
        + enumerate_mpl_ladder()
        + enumerate_seed_ladder()
        + enumerate_subiter_ladder()
        + enumerate_shrink_ladder()
        + enumerate_minfreq_ladder()
        + enumerate_interaction_subiter_shrink()
        + enumerate_interaction_mpl_v()
        + enumerate_interaction_mpl_typology()
        + enumerate_interaction_bpe_refs()
    )


def load_extras_manifest(path: Path = EXTRAS_MANIFEST_PATH) -> list[ExtrasCell]:
    """Read and validate the committed extras manifest into a cell list."""
    return ExtrasManifest.from_yaml(path).cells


def cells_for_extras_kind(extras_kind: ExtrasKind | None = None) -> list[ExtrasCell]:
    """Return the committed extras cells, optionally narrowed to one ``extras_kind``."""
    cells = load_extras_manifest()
    if extras_kind is None:
        return cells
    return [c for c in cells if c.extras_kind == extras_kind]


def write_extras_manifest(path: Path = EXTRAS_MANIFEST_PATH) -> Path:
    """Regenerate the committed manifest from :func:`enumerate_all`.

    A developer action, not a runtime one: run once, commit the result.
    """
    manifest = ExtrasManifest(
        version=EXTRAS_MANIFEST_VERSION, cells=list(enumerate_all())
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(manifest.model_dump(), sort_keys=False))
    return path


def extras_training_dir(cell: ExtrasCell) -> Path:
    """Return the corpus training directory for an :class:`ExtrasCell`.

    Cells with ``training_subdir`` set point at the robustness-extras
    subsample (``data/processed/<corpus>/canon_dedup_v1_extras/<subdir>/``);
    others train on the headline ``canon_dedup_v1/train`` of their corpus.
    """
    if cell.training_subdir is not None:
        return (
            processed_corpus_dir(cell.corpus)
            / "canon_dedup_v1_extras"
            / cell.training_subdir
        )
    return processed_corpus_dir(cell.corpus) / "canon_dedup_v1" / "train"


def extras_cell_to_config(cell: ExtrasCell) -> TokenizerConfig:
    """Map an extras cell to its runnable :class:`TokenizerConfig`.

    Both arms ship the natural artifact their trainer produces (no post-train
    trim); ``vocab_size`` is the matched *target* each arm realizes on its own
    terms.
    """
    return TokenizerConfig(
        name=cell.name,
        kind=cell.kind,
        vocab_size=cell.vocab_size,
        corpus=cell.corpus,
        training_input=extras_training_dir(cell),
        output_dir=tokenizer_artifact_dir(cell.corpus, cell.name),
        merge_brackets=cell.boundary == "mb",
        split_structure=True,
        seed_size=cell.seed_size_override,
        shrinking_factor=cell.shrinking_factor_override,
        max_piece_length=cell.max_piece_length_override,
        n_sub_iterations=cell.n_sub_iterations_override,
        min_frequency=(
            cell.min_frequency_override
            if cell.min_frequency_override is not None
            else MINFREQ_DEFAULT
        ),
    )
