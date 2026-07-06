"""Per-cell Fertility glyph-map build and held-out encode pass.

Builds the per-token glyph-count map once, streams every held-out molecule
through the tokenizer once, and produces the per-molecule ``(n_tokens,
n_glyphs)`` records :mod:`fertility` aggregates. Kept distinct from the pure
math so the heavy I/O is testable in isolation.

Glyph counting is exact and model-faithful (takes the 165-glyph Smirk
alphabet as the base unit, so ``len(surface_string)`` is the *wrong* count —
``Cl`` is one glyph, a bracketed atom is several):

* **Unigram** stores each piece's glyph list explicitly in ``tokenizer.json``
  (``model.vocab[i]["glyphs"]``); the count is ``len(glyphs)``.
* **BPE** stores only surface strings + ordered merges. We walk the merge
  tree: every base glyph is one glyph, and each merge ``(a, b) → ab`` has
  ``count[ab] = count[a] + count[b]``. Base glyphs never appear as a merge
  result, so the map self-adapts to the actual base size with no hardcoding.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from smiles_subword.paths import tokenizer_artifact_dir
from smiles_subword.tokenize._batched import ENCODE_BATCH_SIZE, iter_encoded_batches
from smiles_subword.tokenize.measure._cells import (
    eval_split_sha,
    iter_test_split,
)
from smiles_subword.tokenize.measure._glyphmap import glyph_count_map
from smiles_subword.tokenize.measure.fertility.math import (
    Arm,
    ArmFertility,
    Boundary,
    PerMoleculeFertility,
    compute_arm_fertility,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from smiles_subword.tokenize.adapters.smirk import SmirkAdapter
    from smiles_subword.tokenize.adapters.smirk_unigram import UnigramSmirkAdapter

__all__ = [
    "encode_for_fertility",
    "run_arm_fertility",
]


def encode_for_fertility(
    adapter: SmirkAdapter | UnigramSmirkAdapter,
    smiles: Iterable[str],
    glyph_map: dict[int, int],
    *,
    batch_size: int = ENCODE_BATCH_SIZE,
) -> list[PerMoleculeFertility]:
    """Stream ``smiles`` once and produce per-molecule Fertility counts.

    Each molecule's token count is the encoded length; its glyph count sums
    the per-token glyph counts. A token absent from ``glyph_map`` (e.g. an
    ``[UNK]``) counts as one glyph — it is a single base unit.
    """
    out: list[PerMoleculeFertility] = []
    for ids in iter_encoded_batches(
        adapter, smiles, add_special_tokens=False, batch_size=batch_size
    ):
        n_glyphs = sum(glyph_map.get(tid, 1) for tid in ids)
        out.append(PerMoleculeFertility(n_tokens=len(ids), n_glyphs=n_glyphs))
    return out


def run_arm_fertility(
    adapter: SmirkAdapter | UnigramSmirkAdapter,
    *,
    cell_id: str,
    corpus: str,
    name: str,
    arm: Arm,
    boundary: Boundary,
    training_corpus_sha: str,
    eval_split_sha_value: str | None = None,
    limit_molecules: int | None = None,
    batch_size: int = ENCODE_BATCH_SIZE,
) -> ArmFertility:
    """Encode the corpus's held-out split through ``adapter`` and aggregate.

    ``eval_split_sha_value`` is computed once per corpus from the test
    split's MANIFEST when ``None``. ``limit_molecules`` truncates the
    stream — debug/test hook only; production deposits pass ``None``.
    """
    sha = eval_split_sha_value or eval_split_sha(corpus)
    glyph_map = glyph_count_map(tokenizer_artifact_dir(corpus, name), arm)
    smiles_iter = iter_test_split(corpus, limit_molecules=limit_molecules)
    per_molecule = encode_for_fertility(
        adapter, smiles_iter, glyph_map, batch_size=batch_size
    )
    return compute_arm_fertility(
        per_molecule,
        cell_id=cell_id,
        arm=arm,
        boundary=boundary,
        training_corpus_sha=training_corpus_sha,
        eval_split_sha=sha,
    )
