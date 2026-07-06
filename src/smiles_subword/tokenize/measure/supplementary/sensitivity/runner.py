"""Per-cell measurement bridge for the sensitivity battery.

Builds each cell's curve inputs — the multi-glyph vocab set (off
``tokenizer.json``, for ``J``) and the held-out relative fertility (via
:func:`run_arm_fertility`). The pure aggregation over these lives in
:mod:`.math`; :mod:`scripts.compute_sensitivity` drives the measurement and
deposits the result via :mod:`.io`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import yaml

from smiles_subword.paths import tokenizer_artifact_dir
from smiles_subword.tokenize.extras import cells_for_extras_kind
from smiles_subword.tokenize.measure._cells import load_cell_adapter
from smiles_subword.tokenize.measure._glyphmap import glyph_tuple_map
from smiles_subword.tokenize.measure._pairing import SENSITIVITY_KINDS
from smiles_subword.tokenize.measure.fertility.runner import run_arm_fertility
from smiles_subword.tokenize.measure.supplementary.sensitivity.math import (
    COCONUT_BPE_REF,
    CellMeasured,
)

if TYPE_CHECKING:
    from smiles_subword.tokenize.measure.jaccard import Arm, Boundary

__all__ = [
    "DEFAULT_FERTILITY_SAMPLE",
    "cells_to_measure",
    "measure_cell",
]

DEFAULT_FERTILITY_SAMPLE = 20_000
"""Held-out molecules encoded per cell for the relative fertility gap. ``|Δf|``
is a ratio of per-arm means, stable well below the full split; the same fixed
prefix is taken for every cell so the comparison is order-consistent."""


def cells_to_measure() -> list[tuple[str, Arm]]:
    """Every (cell_id, arm) the report needs, de-duplicated and ordered.

    The battery cells (anchor, ladders, interactions, BPE references) plus
    COCONUT's full-corpus headline BPE V=512 (interaction C's COCONUT reference,
    which lives in the headline grid rather than the extras manifest).
    """
    out: list[tuple[str, Arm]] = []
    seen: set[str] = set()
    for kind in sorted(SENSITIVITY_KINDS):
        for cell in cells_for_extras_kind(kind):
            if cell.cell_id not in seen:
                seen.add(cell.cell_id)
                out.append((cell.cell_id, cell.algo))
    if COCONUT_BPE_REF not in seen:
        out.append((COCONUT_BPE_REF, "bpe"))
    return out


def measure_cell(
    cell_id: str,
    arm: Arm,
    *,
    fertility_sample: int = DEFAULT_FERTILITY_SAMPLE,
) -> CellMeasured:
    """Build one cell's curve inputs: its multi-glyph vocab set and fertility.

    The multi-glyph set (``J``) is read straight off ``tokenizer.json``; Fertility
    supplies fertility over the first ``fertility_sample`` held-out molecules. No
    training-corpus inventory pass is taken (``J_struct`` is not swept).
    """
    corpus, _, name = cell_id.partition("__")
    meta = yaml.safe_load(
        (tokenizer_artifact_dir(corpus, name) / "meta.yaml").read_text()
    )
    boundary: Boundary = "mb" if bool(meta["merge_brackets"]) else "nmb"
    training_corpus_sha = meta["training_corpus_sha"]
    adapter = load_cell_adapter(corpus, name)

    glyph_tuples = glyph_tuple_map(tokenizer_artifact_dir(corpus, name), arm)
    multi = frozenset(t for t in glyph_tuples.values() if len(t) >= 2)
    fertility = run_arm_fertility(
        adapter,
        cell_id=cell_id,
        corpus=corpus,
        name=name,
        arm=arm,
        boundary=boundary,
        training_corpus_sha=training_corpus_sha,
        limit_molecules=fertility_sample,
    )
    return CellMeasured(
        cell_id=cell_id,
        arm=arm,
        corpus=corpus,
        vocab_size=int(meta["vocab_size"]),
        boundary=boundary,
        multi=multi,
        fertility=fertility.fertility_mean,
    )
