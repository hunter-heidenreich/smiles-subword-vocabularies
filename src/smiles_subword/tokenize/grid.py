"""The tokenizer grid: the frozen grid and sensitivity-analysis enumeration.

The study trains a frozen 44-cell grid over four axes — algorithm, vocab size
``V``, corpus, boundary mode. Single code-side source of truth:
:func:`enumerate_grid` is the generating rule for the 44 cells,
:func:`enumerate_conditional` adds the ``V=2048``-on-ZINC-22 cells,
``configs/tokenizer/grid.yaml`` is the committed manifest of both, and a test
pins manifest to enumeration so neither drifts from the frozen design.

This module only enumerates and maps cells; training happens via the dispatch
driver. The robustness extras (92 cells) introduce axes the grid lacks and get
their own spec (``extras.py``); the conditional cells share all four axes and
belong here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

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
    from pathlib import Path

GRID_MANIFEST_VERSION = 1
GRID_MANIFEST_PATH = CONFIGS_DIR / "tokenizer" / "grid.yaml"

_ALGOS: tuple[Literal["bpe", "unigram"], ...] = ("bpe", "unigram")
_BOUNDARIES: tuple[Literal["nmb", "mb"], ...] = ("nmb", "mb")
_HEADLINE_VOCAB_SIZES: tuple[int, ...] = (256, 512, 1024)
_HEADLINE_CORPORA: tuple[Literal["pubchem", "zinc22", "coconut"], ...] = (
    "pubchem",
    "zinc22",
    "coconut",
)
_SENSITIVITY_VOCAB_SIZE = 2048
_SENSITIVITY_CORPUS: Literal["pubchem"] = "pubchem"
_ANCHOR_VOCAB_SIZE = 1024
_ANCHOR_CORPUS: Literal["real_space"] = "real_space"
_CONDITIONAL_ALGO: Literal["bpe"] = "bpe"
_CONDITIONAL_VOCAB_SIZE = 2048
_CONDITIONAL_CORPUS: Literal["zinc22"] = "zinc22"


class GridCell(BaseModel):
    """One grid cell — a point in (algo, V, corpus, boundary).

    ``tier`` records the family: ``headline`` / ``sensitivity`` / ``anchor``, or
    ``conditional`` for the ``V=2048``-on-ZINC-22 cells outside the frozen 44.
    Stored at enumeration time so the enumerator stays the single source of
    truth. Frozen and hashable so manifest / enumeration equality checks as sets.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    algo: Literal["bpe", "unigram"]
    vocab_size: int = Field(ge=1)
    corpus: Literal["pubchem", "zinc22", "coconut", "real_space"]
    boundary: Literal["nmb", "mb"]
    tier: Literal["headline", "sensitivity", "anchor", "conditional"]

    @property
    def kind(self) -> TokenizerKind:
        """The :data:`TokenizerKind` the algo axis maps to (:func:`algo_to_kind`)."""
        return algo_to_kind(self.algo)

    @property
    def name(self) -> str:
        """Artifact directory name — ``smirk_{gpe|unigram}_v{V}_{boundary}``.

        Built via :func:`cell_artifact_name`. Corpus is the parent directory
        under ``artifacts/tokenizer/``, so it is not in the name; the name
        alone is therefore not unique across the grid — use :attr:`cell_id`
        as the dispatch key.
        """
        return cell_artifact_name(self.algo, self.vocab_size, self.boundary)

    @property
    def cell_id(self) -> str:
        """Globally-unique cell identifier — ``{corpus}__{name}``."""
        return f"{self.corpus}__{self.name}"


class GridManifest(YamlLoadable):
    """The committed grid manifest — ``configs/tokenizer/grid.yaml``."""

    model_config = ConfigDict(extra="forbid")

    version: int = Field(ge=1)
    cells: list[GridCell]


def enumerate_grid() -> tuple[GridCell, ...]:
    """Return the 44 cells: 36 headline + 4 sensitivity + 4 anchor.

    This is the generating rule. The committed manifest is regenerated from it
    by :func:`write_grid_manifest` and pinned to it by a test.
    """
    headline = [
        GridCell(algo=a, vocab_size=v, corpus=c, boundary=b, tier="headline")
        for a in _ALGOS
        for v in _HEADLINE_VOCAB_SIZES
        for c in _HEADLINE_CORPORA
        for b in _BOUNDARIES
    ]
    sensitivity = [
        GridCell(
            algo=a,
            vocab_size=_SENSITIVITY_VOCAB_SIZE,
            corpus=_SENSITIVITY_CORPUS,
            boundary=b,
            tier="sensitivity",
        )
        for a in _ALGOS
        for b in _BOUNDARIES
    ]
    anchor = [
        GridCell(
            algo=a,
            vocab_size=_ANCHOR_VOCAB_SIZE,
            corpus=_ANCHOR_CORPUS,
            boundary=b,
            tier="anchor",
        )
        for a in _ALGOS
        for b in _BOUNDARIES
    ]
    return tuple(headline + sensitivity + anchor)


def enumerate_conditional() -> tuple[GridCell, ...]:
    """Return the 2 conditional cells: ZINC-22 BPE ``V=2048``, nmb + mb.

    Included because the F95 confirmation clears only the ZINC-22 BPE arm at
    ``V=2048``: COCONUT BPE is embedding-tail-unsafe already at ``V=1024`` and
    both corpora's Unigram arm is unsafe at ``V≥512``. These two cells are the
    positive branch; COCONUT and the ZINC-22 Unigram arm are left untrained.
    Outside the frozen 44 but sharing its four axes, so they live here.
    """
    return tuple(
        GridCell(
            algo=_CONDITIONAL_ALGO,
            vocab_size=_CONDITIONAL_VOCAB_SIZE,
            corpus=_CONDITIONAL_CORPUS,
            boundary=b,
            tier="conditional",
        )
        for b in _BOUNDARIES
    )


def enumerate_all() -> tuple[GridCell, ...]:
    """Return every committed cell: the 44-cell grid plus the 2 conditional cells."""
    return enumerate_grid() + enumerate_conditional()


def load_grid_manifest(path: Path = GRID_MANIFEST_PATH) -> list[GridCell]:
    """Read and validate the committed grid manifest into a cell list."""
    return GridManifest.from_yaml(path).cells


def cells_for_tier(tier: str | None = None) -> list[GridCell]:
    """Return the committed cells, optionally narrowed to one ``tier``.

    Training proceeds one tier at a time (``headline`` / ``sensitivity`` /
    ``anchor`` / ``conditional``); ``tier=None`` returns every committed
    cell — the 44-cell grid plus the 2 conditional cells.
    """
    cells = load_grid_manifest()
    if tier is None:
        return cells
    return [c for c in cells if c.tier == tier]


def write_grid_manifest(path: Path = GRID_MANIFEST_PATH) -> Path:
    """Regenerate the committed manifest from :func:`enumerate_all`.

    A developer action, not a runtime one: run once, commit the result.
    """
    manifest = GridManifest(version=GRID_MANIFEST_VERSION, cells=list(enumerate_all()))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(manifest.model_dump(), sort_keys=False))
    return path


def corpus_training_dir(corpus: str) -> Path:
    """Return the ``canon_dedup_v1`` training split directory for ``corpus``."""
    return processed_corpus_dir(corpus) / "canon_dedup_v1" / "train"


def grid_cell_to_config(cell: GridCell) -> TokenizerConfig:
    """Map a grid cell to its runnable :class:`TokenizerConfig`.

    Both arms ship the natural artifact their trainer produces (no post-train
    trim): ``vocab_size`` is the matched *target*, which each arm realizes on
    its own terms.
    """
    return TokenizerConfig(
        name=cell.name,
        kind=cell.kind,
        vocab_size=cell.vocab_size,
        corpus=cell.corpus,
        training_input=corpus_training_dir(cell.corpus),
        output_dir=tokenizer_artifact_dir(cell.corpus, cell.name),
        merge_brackets=cell.boundary == "mb",
        split_structure=True,
    )
