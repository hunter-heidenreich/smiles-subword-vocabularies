"""Per-cell Jaccard build: glyph-tuple vocab, structural split, held-out J_w data.

Given a trained tokenizer artifact, its training input, and the corpus's
held-out split, produces the :class:`ArmJaccardInputs` the pure math in
:mod:`jaccard` joins into per-arm and matched-pair records.

Three pieces:

* **glyph-tuple vocab** — the multi-glyph subword set, keyed by the exact glyph
  sequence so the two arms are comparable. Unigram stores each piece's glyphs;
  BPE is walked through its merge tree (id ``base_size + k`` = the ``k``-th
  merge's operand tuples concatenated).
* **structural split** — the bracket-internal / structural partition, from
  the training-corpus chunk inventory (:mod:`inventory`).
* **held-out J_w data** — per-molecule emitted multi-glyph counts on the shared
  held-out split, in sparse coordinate form for the molecule-resampled
  bootstrap.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import numpy as np

from smiles_subword.paths import tokenizer_artifact_dir
from smiles_subword.tokenize._batched import ENCODE_BATCH_SIZE, iter_encoded_batches
from smiles_subword.tokenize.extras import extras_training_dir, load_extras_manifest
from smiles_subword.tokenize.grid import corpus_training_dir
from smiles_subword.tokenize.measure._cells import (
    eval_split_sha,
    iter_test_split,
)
from smiles_subword.tokenize.measure._glyphmap import glyph_tuple_map
from smiles_subword.tokenize.measure.jaccard.inventory import (
    classify_subwords,
    get_or_build_inventory,
)
from smiles_subword.tokenize.measure.jaccard.math import (
    Arm,
    ArmJaccardInputs,
    Boundary,
    GlyphTuple,
    JwMoleculeData,
    bootstrap_seed,
)

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from smiles_subword.tokenize.adapters.smirk import SmirkAdapter
    from smiles_subword.tokenize.adapters.smirk_unigram import UnigramSmirkAdapter

__all__ = [
    "build_jw_data",
    "resolve_training_dir",
    "run_arm_jaccard",
]


def resolve_training_dir(corpus: str, name: str) -> Path:
    """Return the training-input directory a cell was trained on.

    Extras cells with a ``training_subdir`` point at their robustness-extras
    subsample; every other cell trains on the headline ``canon_dedup_v1/train``.
    """
    for cell in load_extras_manifest():
        if cell.corpus == corpus and cell.name == name:
            return extras_training_dir(cell)
    return corpus_training_dir(corpus)


def build_jw_data(
    adapter: SmirkAdapter | UnigramSmirkAdapter,
    smiles: Iterable[str],
    glyph_tuple_by_id: dict[int, GlyphTuple],
    *,
    batch_size: int = ENCODE_BATCH_SIZE,
) -> JwMoleculeData:
    """Stream the held-out split and build the sparse per-molecule J_w data.

    One coordinate entry per (molecule, multi-glyph subword) with the
    per-molecule emission count; molecules emitting no multi-glyph token still
    advance the molecule index so the resample unit count stays exact.
    """
    local_ids: dict[GlyphTuple, int] = {}
    local_tuples: list[GlyphTuple] = []
    mol_idx: list[int] = []
    sub_local: list[int] = []
    count: list[int] = []
    n_molecules = 0
    for ids in iter_encoded_batches(
        adapter, smiles, add_special_tokens=False, batch_size=batch_size
    ):
        per_mol: Counter[GlyphTuple] = Counter()
        for tid in ids:
            tup = glyph_tuple_by_id.get(tid)
            if tup is not None and len(tup) >= 2:
                per_mol[tup] += 1
        for tup, c in per_mol.items():
            lid = local_ids.get(tup)
            if lid is None:
                lid = len(local_tuples)
                local_ids[tup] = lid
                local_tuples.append(tup)
            mol_idx.append(n_molecules)
            sub_local.append(lid)
            count.append(c)
        n_molecules += 1
    return JwMoleculeData(
        n_molecules=n_molecules,
        mol_idx=np.asarray(mol_idx, dtype=np.int64),
        sub_local=np.asarray(sub_local, dtype=np.int64),
        count=np.asarray(count, dtype=np.float64),
        local_tuples=tuple(local_tuples),
    )


def run_arm_jaccard(
    adapter: SmirkAdapter | UnigramSmirkAdapter,
    *,
    cell_id: str,
    corpus: str,
    name: str,
    arm: Arm,
    boundary: Boundary,
    training_corpus_sha: str,
    inventory_cache_path: Path,
    eval_split_sha_value: str | None = None,
    limit_molecules: int | None = None,
    inventory_limit: int | None = None,
    batch_size: int = ENCODE_BATCH_SIZE,
) -> ArmJaccardInputs:
    """Build one arm's Jaccard inputs: vocab set, structural split, held-out J_w data.

    ``limit_molecules`` / ``inventory_limit`` truncate the held-out and training
    streams respectively — debug/test hooks only; production passes ``None``.
    """
    artifact_dir = tokenizer_artifact_dir(corpus, name)
    glyph_tuple_by_id = glyph_tuple_map(artifact_dir, arm)
    multi = frozenset(t for t in glyph_tuple_by_id.values() if len(t) >= 2)

    inventory = get_or_build_inventory(
        adapter,
        resolve_training_dir(corpus, name),
        inventory_cache_path,
        training_corpus_sha=training_corpus_sha,
        limit_molecules=inventory_limit,
    )
    split = classify_subwords(adapter, glyph_tuple_by_id, multi, inventory)

    eval_sha = eval_split_sha_value or eval_split_sha(corpus)
    smiles = iter_test_split(corpus, limit_molecules=limit_molecules)
    jw = build_jw_data(adapter, smiles, glyph_tuple_by_id, batch_size=batch_size)

    return ArmJaccardInputs(
        cell_id=cell_id,
        arm=arm,
        boundary=boundary,
        training_corpus_sha=training_corpus_sha,
        eval_split_sha=eval_sha,
        multi_subwords=multi,
        structural_subwords=split.structural,
        bracket_internal_subwords=split.bracket_internal,
        unseen_subwords=split.unseen,
        n_distinct_bracket_chunks=len(inventory.bracket_chunks),
        n_distinct_nonbracket_chunks=len(inventory.nonbracket_chunks),
        nonbracket_cap_bound=inventory.nonbracket_cap_bound,
        jw=jw,
        bootstrap_seed=bootstrap_seed(cell_id),
    )
