"""Per-pair Nestedness dual-encode pass over the held-out split.

Streams every held-out molecule once, encodes it through *both* arms, and
compares their token boundaries into the per-molecule
:class:`PerMoleculeNestedness` records :mod:`nestedness` aggregates. Kept
distinct from the pure math so the heavy I/O is testable in isolation.

Both arms must see the identical molecule in lockstep, so each batch is encoded
through both adapters before comparison — a single parquet pass. The BPE arm
needs only per-token glyph *counts* (for its boundaries); the Unigram-LM arm
needs the per-token glyph *tuples* (boundary plus the substructure class used to
localize conflict).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from smiles_subword.paths import tokenizer_artifact_dir
from smiles_subword.tokenize._batched import ENCODE_BATCH_SIZE
from smiles_subword.tokenize.measure._cells import (
    eval_split_sha,
    iter_test_split,
)
from smiles_subword.tokenize.measure._glyphmap import (
    glyph_count_map,
    glyph_tuple_map,
)
from smiles_subword.tokenize.measure.nestedness.math import (
    Boundary,
    GlyphTuple,
    MatchedPairNestedness,
    PerMoleculeNestedness,
    compare_molecule,
    compute_pair_nestedness,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from smiles_subword.tokenize.adapters.smirk import SmirkAdapter
    from smiles_subword.tokenize.adapters.smirk_unigram import UnigramSmirkAdapter

__all__ = [
    "compare_held_out",
    "run_pair_nestedness",
]


def compare_held_out(
    bpe: SmirkAdapter | UnigramSmirkAdapter,
    unigram: SmirkAdapter | UnigramSmirkAdapter,
    smiles: Iterable[str],
    bpe_glyph_counts: dict[int, int],
    ul_glyph_tuples: dict[int, GlyphTuple],
    *,
    batch_size: int = ENCODE_BATCH_SIZE,
) -> tuple[list[PerMoleculeNestedness], int]:
    """Dual-encode ``smiles`` and compare boundaries per molecule.

    Returns ``(per_molecule, n_length_mismatch)``. A molecule whose two arms
    disagree on total glyph length (an ``[UNK]`` edge case; absent on the
    conformance-filtered corpora) is skipped and counted rather than aborting
    the pass.
    """
    out: list[PerMoleculeNestedness] = []
    mismatch = 0
    batch: list[str] = []

    def _flush() -> int:
        if not batch:
            return 0
        local_mismatch = 0
        bpe_ids = bpe.encode_batch(batch, add_special_tokens=False)
        ul_ids = unigram.encode_batch(batch, add_special_tokens=False)
        for bids, uids in zip(bpe_ids, ul_ids, strict=True):
            counts = [bpe_glyph_counts.get(tid, 1) for tid in bids]
            tuples = [ul_glyph_tuples.get(tid, ("?",)) for tid in uids]
            try:
                out.append(compare_molecule(counts, tuples))
            except ValueError:
                local_mismatch += 1
        batch.clear()
        return local_mismatch

    for smi in smiles:
        batch.append(smi)
        if len(batch) >= batch_size:
            mismatch += _flush()
    mismatch += _flush()
    return out, mismatch


def run_pair_nestedness(
    bpe: SmirkAdapter | UnigramSmirkAdapter,
    unigram: SmirkAdapter | UnigramSmirkAdapter,
    *,
    pair_key: str,
    tier: str,
    corpus: str,
    vocab_size: int,
    boundary: Boundary,
    bpe_cell_id: str,
    unigram_cell_id: str,
    bpe_name: str,
    unigram_name: str,
    bpe_training_corpus_sha: str,
    unigram_training_corpus_sha: str,
    extras_kind: str | None = None,
    extras_label: str | None = None,
    eval_split_sha_value: str | None = None,
    limit_molecules: int | None = None,
    batch_size: int = ENCODE_BATCH_SIZE,
) -> MatchedPairNestedness:
    """Encode the held-out split through both arms and aggregate nestedness.

    ``eval_split_sha_value`` is computed once per corpus from the test split's
    MANIFEST when ``None``. ``limit_molecules`` truncates the stream — a
    debug/test hook only; production deposits pass ``None``.
    """
    sha = eval_split_sha_value or eval_split_sha(corpus)
    bpe_counts = glyph_count_map(tokenizer_artifact_dir(corpus, bpe_name), "bpe")
    ul_tuples = glyph_tuple_map(tokenizer_artifact_dir(corpus, unigram_name), "unigram")
    smiles_iter = iter_test_split(corpus, limit_molecules=limit_molecules)
    per_molecule, n_mismatch = compare_held_out(
        bpe, unigram, smiles_iter, bpe_counts, ul_tuples, batch_size=batch_size
    )
    return compute_pair_nestedness(
        per_molecule,
        pair_key=pair_key,
        tier=tier,
        corpus=corpus,
        vocab_size=vocab_size,
        boundary=boundary,
        bpe_cell_id=bpe_cell_id,
        unigram_cell_id=unigram_cell_id,
        bpe_training_corpus_sha=bpe_training_corpus_sha,
        unigram_training_corpus_sha=unigram_training_corpus_sha,
        eval_split_sha=sha,
        extras_kind=extras_kind,
        extras_label=extras_label,
        n_length_mismatch=n_mismatch,
    )
