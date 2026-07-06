"""Per-cell Distribution held-out encode pass.

Streams the corpus's held-out split once through a trained tokenizer and builds
the sparse per-molecule token-count structure (:class:`DistributionMoleculeData`)
that :mod:`distribution` aggregates. The simplest runner: token-id frequencies
only — no glyph map, merge-tree walk, or training-corpus inventory — just an
encode pass with special-token ids dropped.

``v_effective`` is the nominal target ``|V|`` (165 base glyphs plus ``V − 165``
learned subwords, identical across arms at a matched cell). The caller supplies
it so both arms normalize by the same ``|V|`` — the premise under which dead
glyphs cancel in ``ΔD``. The single-arm merge-exhaustion H-anchor, whose
realized vocabulary falls short of its nominal target, is normalized by its
realized non-special count instead.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from smiles_subword.tokenize._batched import ENCODE_BATCH_SIZE, iter_encoded_batches
from smiles_subword.tokenize.base import collect_special_ids
from smiles_subword.tokenize.measure._cells import (
    eval_split_sha,
    iter_test_split,
)
from smiles_subword.tokenize.measure.distribution.math import (
    Arm,
    ArmDistribution,
    Boundary,
    DistributionMoleculeData,
    compute_arm_distribution,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from smiles_subword.tokenize.adapters.smirk import SmirkAdapter
    from smiles_subword.tokenize.adapters.smirk_unigram import UnigramSmirkAdapter

__all__ = [
    "build_distribution_data",
    "collect_all_special_ids",
    "run_arm_distribution",
]


def collect_all_special_ids(
    adapter: SmirkAdapter | UnigramSmirkAdapter, artifact_dir: Path
) -> frozenset[int]:
    """Every special-token id to drop from the token-frequency distribution.

    The :class:`~smiles_subword.tokenize.base.Tokenizer` protocol surfaces only
    ``bos``/``eos``/``pad``/``unk``; the artifacts also register ``[SEP]``,
    ``[CLS]``, ``[MASK]`` as ``added_tokens``. We union the protocol four with
    every ``added_tokens`` entry flagged ``special`` in ``tokenizer.json`` so
    none can leak into the distribution.
    """
    ids = set(collect_special_ids(adapter))
    tj_path = Path(artifact_dir) / "tokenizer.json"
    if tj_path.is_file():
        added = json.loads(tj_path.read_text()).get("added_tokens", [])
        ids.update(
            int(tok["id"])
            for tok in added
            if tok.get("special") and isinstance(tok.get("id"), int)
        )
    return frozenset(ids)


def build_distribution_data(
    adapter: SmirkAdapter | UnigramSmirkAdapter,
    smiles: Iterable[str],
    *,
    v_effective: int,
    special_ids: frozenset[int],
    batch_size: int = ENCODE_BATCH_SIZE,
) -> DistributionMoleculeData:
    """Stream ``smiles`` once and build the sparse per-molecule token-count data.

    One coordinate entry per (molecule, non-special token id) with the
    per-molecule emission count; molecules emitting only specials still advance
    the molecule index so the resample-unit count stays exact.
    """
    local_ids: dict[int, int] = {}
    local_token_ids: list[int] = []
    mol_idx: list[int] = []
    sub_local: list[int] = []
    count: list[int] = []
    n_molecules = 0
    for ids in iter_encoded_batches(
        adapter, smiles, add_special_tokens=False, batch_size=batch_size
    ):
        per_mol: Counter[int] = Counter()
        for tid in ids:
            if tid not in special_ids:
                per_mol[tid] += 1
        for tid, c in per_mol.items():
            lid = local_ids.get(tid)
            if lid is None:
                lid = len(local_token_ids)
                local_ids[tid] = lid
                local_token_ids.append(tid)
            mol_idx.append(n_molecules)
            sub_local.append(lid)
            count.append(c)
        n_molecules += 1
    return DistributionMoleculeData(
        n_molecules=n_molecules,
        mol_idx=np.asarray(mol_idx, dtype=np.int64),
        sub_local=np.asarray(sub_local, dtype=np.int64),
        count=np.asarray(count, dtype=np.float64),
        local_token_ids=tuple(local_token_ids),
        v_effective=v_effective,
    )


def run_arm_distribution(
    adapter: SmirkAdapter | UnigramSmirkAdapter,
    *,
    cell_id: str,
    corpus: str,
    arm: Arm,
    boundary: Boundary,
    v_effective: int,
    special_ids: frozenset[int],
    training_corpus_sha: str,
    eval_split_sha_value: str | None = None,
    limit_molecules: int | None = None,
    batch_size: int = ENCODE_BATCH_SIZE,
) -> ArmDistribution:
    """Encode the corpus's held-out split through ``adapter`` and aggregate.

    ``v_effective`` is the caller-chosen normalizer ``|V|`` (see the module
    docstring); ``special_ids`` are dropped. ``eval_split_sha_value`` defaults to
    the corpus test-split MANIFEST when ``None``. ``limit_molecules`` truncates
    the stream — debug/test hook only; production passes ``None``.
    """
    sha = eval_split_sha_value or eval_split_sha(corpus)
    smiles_iter = iter_test_split(corpus, limit_molecules=limit_molecules)
    data = build_distribution_data(
        adapter,
        smiles_iter,
        v_effective=v_effective,
        special_ids=special_ids,
        batch_size=batch_size,
    )
    return compute_arm_distribution(
        data,
        cell_id=cell_id,
        arm=arm,
        boundary=boundary,
        vocab_size=adapter.vocab_size,
        training_corpus_sha=training_corpus_sha,
        eval_split_sha=sha,
    )
