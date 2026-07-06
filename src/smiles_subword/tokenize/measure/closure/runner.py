"""Per-arm Closure build — a vocabulary-only read of one trained cell.

Closure is a property of the realized vocabulary alone, so this runner does no
encoding pass: it reads ``tokenizer.json`` and reconstructs the glyph-tuple
vocabulary via :func:`glyph_tuple_map` (Unigram stores each piece's glyphs; BPE
is walked through its merge tree), then hands the tuples to :mod:`closure`.
"""

from __future__ import annotations

from smiles_subword.paths import tokenizer_artifact_dir
from smiles_subword.tokenize.measure._glyphmap import glyph_tuple_map
from smiles_subword.tokenize.measure.closure.math import (
    Arm,
    ArmClosure,
    Boundary,
    compute_arm_closure,
)

__all__ = ["run_arm_closure"]


def run_arm_closure(
    *,
    cell_id: str,
    corpus: str,
    name: str,
    arm: Arm,
    boundary: Boundary,
    vocab_size: int,
    training_corpus_sha: str,
) -> ArmClosure:
    """Compute one arm's closure from its on-disk ``tokenizer.json``.

    Reads only the tokenizer artifact — no ``meta.yaml`` (the caller passes the
    resolved cell facts) and no corpus. Propagates :func:`glyph_tuple_map`
    errors, which the deposit step turns into a pending reason.
    """
    artifact_dir = tokenizer_artifact_dir(corpus, name)
    vocab_tuples = list(glyph_tuple_map(artifact_dir, arm).values())
    return compute_arm_closure(
        vocab_tuples,
        cell_id=cell_id,
        arm=arm,
        boundary=boundary,
        vocab_size=vocab_size,
        training_corpus_sha=training_corpus_sha,
    )
