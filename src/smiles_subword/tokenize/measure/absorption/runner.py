"""Per-cell Absorption encode-and-classify pass.

Streams the corpus's held-out split once through a trained tokenizer and
produces the per-molecule :class:`PerMoleculeAbsorption` records that
:mod:`absorption` aggregates. Kept distinct from the pure math so the I/O is
testable in isolation and the math stays mockable from synthetic offsets.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from smiles_subword.tokenize.measure._cells import (
    eval_split_sha,
    iter_test_split,
)
from smiles_subword.tokenize.measure.absorption.math import (
    Arm,
    ArmAbsorption,
    Boundary,
    PerMoleculeAbsorption,
    classify_chunks,
    compute_arm_absorption,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from smirk import SmirkTokenizerFast

    from smiles_subword.tokenize.adapters.smirk import SmirkAdapter
    from smiles_subword.tokenize.adapters.smirk_unigram import UnigramSmirkAdapter

ENCODE_BATCH_SIZE = 1024
"""Batch size for the held-out encoding pass."""


def _hf_tokenizer(
    adapter: SmirkAdapter | UnigramSmirkAdapter,
) -> SmirkTokenizerFast:
    return adapter.hf_tokenizer


def encode_for_absorption(
    adapter: SmirkAdapter | UnigramSmirkAdapter,
    smiles: Iterable[str],
    *,
    boundary: Boundary,
    batch_size: int = ENCODE_BATCH_SIZE,
) -> list[PerMoleculeAbsorption]:
    """Stream ``smiles`` once and produce per-molecule Absorption counts.

    Batches through the HF tokenizer with ``return_offsets_mapping=True`` for
    character-aligned token spans, matches them against
    :meth:`pretokenize_layer_b` chunks, and dispatches to :func:`classify_chunks`.
    """
    tok = _hf_tokenizer(adapter)
    out: list[PerMoleculeAbsorption] = []
    batch: list[str] = []

    def _flush() -> None:
        if not batch:
            return
        enc = tok(batch, add_special_tokens=False, return_offsets_mapping=True)
        offsets_batch = cast("list[list[tuple[int, int]]]", enc["offset_mapping"])
        for smi, offsets in zip(batch, offsets_batch, strict=True):
            chunks = adapter.pretokenize_layer_b(smi)
            out.append(classify_chunks(chunks, offsets, boundary=boundary))
        batch.clear()

    for smi in smiles:
        batch.append(smi)
        if len(batch) >= batch_size:
            _flush()
    _flush()
    return out


def run_arm_absorption(
    adapter: SmirkAdapter | UnigramSmirkAdapter,
    *,
    cell_id: str,
    corpus: str,
    arm: Arm,
    boundary: Boundary,
    training_corpus_sha: str,
    eval_split_sha_value: str | None = None,
    limit_molecules: int | None = None,
    batch_size: int = ENCODE_BATCH_SIZE,
) -> ArmAbsorption:
    """Encode the corpus's held-out split through ``adapter`` and aggregate.

    ``eval_split_sha_value`` is computed once per corpus from the test
    split's MANIFEST when ``None``. ``limit_molecules`` truncates the
    stream — debug/test hook only; production deposits pass ``None``.
    """
    sha = eval_split_sha_value or eval_split_sha(corpus)
    smiles_iter = iter_test_split(corpus, limit_molecules=limit_molecules)
    per_molecule = encode_for_absorption(
        adapter, smiles_iter, boundary=boundary, batch_size=batch_size
    )
    return compute_arm_absorption(
        per_molecule,
        cell_id=cell_id,
        arm=arm,
        boundary=boundary,
        training_corpus_sha=training_corpus_sha,
        eval_split_sha=sha,
    )


__all__ = [
    "ENCODE_BATCH_SIZE",
    "encode_for_absorption",
    "run_arm_absorption",
]
