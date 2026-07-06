"""Per-cell Segmentation held-out segmentation-entropy pass.

Given a trained tokenizer artifact and the corpus's held-out test split, streams
every molecule once and computes its exact Unigram segmentation entropy (sum over
Layer-B chunks) and glyph count, the per-molecule structure :mod:`segmentation`
aggregates into per-arm readings.

Two pieces feed the lattice DP, both read from ``tokenizer.json``:

* **piece scores** — ``{glyph_tuple: log_prob}`` over the Unigram ``model.vocab``
  (each entry's ``glyphs`` + ``score``), the lattice edge weights.
* **glyph-tuple map** — ``{token_id: glyph_tuple}`` (reused from Jaccard), used to
  reconstruct a chunk's Layer-A glyph sequence: encode the chunk and concatenate
  its pieces' glyphs. Tokenization is chunk-local, so this reproduces the chunk's
  glyph sequence exactly without any new fork API.

Distinct Layer-B chunks recur heavily, so each chunk's ``(entropy, glyph_count)``
is memoized. BPE arms skip the encode entirely — their segmentation is unique, so
:func:`run_arm_segmentation` returns the zero record without touching the split.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from smiles_subword.tokenize.measure._cells import (
    eval_split_sha,
    iter_test_split,
)
from smiles_subword.tokenize.measure._glyphmap import build_unigram_glyph_tuples
from smiles_subword.tokenize.measure.segmentation.math import (
    Arm,
    ArmSegmentation,
    Boundary,
    GlyphTuple,
    PerMoleculeSegmentation,
    chunk_segmentation_entropy,
    compute_arm_segmentation,
    compute_bpe_arm_segmentation,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from pathlib import Path

    from smiles_subword.tokenize.adapters.smirk import SmirkAdapter
    from smiles_subword.tokenize.adapters.smirk_unigram import UnigramSmirkAdapter

__all__ = [
    "build_segmentation_data",
    "build_unigram_piece_scores",
    "run_arm_segmentation",
]

_UNK_GLYPH = "\x00unk"
"""Sentinel for an out-of-vocabulary held-out glyph.

HF Unigram emits one ``[UNK]`` per Layer-A glyph it cannot form from any
vocabulary piece; reconstructing the glyph sequence from encoded ids loses that
glyph's identity, so we substitute this sentinel (distinct from every real SMILES
glyph). Installed in the per-chunk score map as a length-1 piece, each ``[UNK]``
becomes a forced lattice boundary: 0 entropy (unique edge ⇒ posterior 1), no
piece can span it, and it counts as one glyph (matching Fertility's convention).
"""


def build_unigram_piece_scores(tokenizer_json: Path) -> dict[GlyphTuple, float]:
    """Return ``{glyph_tuple: log_prob}`` for a Unigram ``tokenizer.json``.

    Each ``model.vocab`` entry carries its piece's ``glyphs`` list and its
    Unigram ``score`` (log-probability); these are the lattice edge weights.
    """
    model = json.loads(tokenizer_json.read_text())["model"]
    if model.get("type") != "Unigram":
        raise ValueError(f"{tokenizer_json}: model type is not Unigram")
    return {tuple(entry["glyphs"]): float(entry["score"]) for entry in model["vocab"]}


def build_segmentation_data(
    adapter: SmirkAdapter | UnigramSmirkAdapter,
    smiles: Iterable[str],
    *,
    piece_scores: Mapping[GlyphTuple, float],
    glyph_tuple_by_id: Mapping[int, GlyphTuple],
    max_piece_len: int,
) -> list[PerMoleculeSegmentation]:
    """Stream ``smiles`` once and produce per-molecule entropy + glyph counts.

    Each molecule's entropy is the sum of its Layer-B chunks' segmentation
    entropies (the distribution factorizes over chunks); its glyph count is the
    sum of chunk glyph counts. Distinct chunks are memoized so each is encoded
    and run through the DP at most once. Out-of-vocabulary held-out glyphs
    (encoded as ``[UNK]``, an id outside ``glyph_tuple_by_id``) become the
    :data:`_UNK_GLYPH` sentinel — a forced lattice boundary worth one glyph and
    zero entropy.
    """
    scores: dict[GlyphTuple, float] = {**piece_scores, (_UNK_GLYPH,): 0.0}
    chunk_cache: dict[str, tuple[float, int]] = {}
    out: list[PerMoleculeSegmentation] = []
    for smi in smiles:
        total_entropy = 0.0
        total_glyphs = 0
        for chunk, _span in adapter.pretokenize_layer_b(smi):
            cached = chunk_cache.get(chunk)
            if cached is None:
                ids = adapter.encode(chunk, add_special_tokens=False)
                glyphs: GlyphTuple = tuple(
                    g for tid in ids for g in glyph_tuple_by_id.get(tid, (_UNK_GLYPH,))
                )
                entropy = chunk_segmentation_entropy(
                    glyphs, scores, max_piece_len=max_piece_len
                )
                cached = (entropy, len(glyphs))
                chunk_cache[chunk] = cached
            total_entropy += cached[0]
            total_glyphs += cached[1]
        out.append(
            PerMoleculeSegmentation(entropy_nats=total_entropy, n_glyphs=total_glyphs)
        )
    return out


def run_arm_segmentation(
    adapter: SmirkAdapter | UnigramSmirkAdapter,
    *,
    cell_id: str,
    corpus: str,
    arm: Arm,
    boundary: Boundary,
    training_corpus_sha: str,
    tokenizer_json: Path | None = None,
    eval_split_sha_value: str | None = None,
    limit_molecules: int | None = None,
) -> ArmSegmentation:
    """Compute Segmentation for one arm.

    BPE arms return the zero-by-construction record without encoding.
    Unigram arms read the piece scores + glyph-tuple map from ``tokenizer_json``,
    stream the corpus's held-out split, and aggregate per-molecule entropies.
    ``eval_split_sha_value`` is computed once per corpus from the test split's
    MANIFEST when ``None``; ``limit_molecules`` truncates the stream
    (debug/test hook — production passes ``None``).
    """
    if arm == "bpe":
        return compute_bpe_arm_segmentation(
            cell_id=cell_id,
            boundary=boundary,
            training_corpus_sha=training_corpus_sha,
        )
    if tokenizer_json is None:
        raise ValueError("the Unigram arm requires a tokenizer_json path")

    piece_scores = build_unigram_piece_scores(tokenizer_json)
    glyph_tuple_by_id = build_unigram_glyph_tuples(tokenizer_json)
    max_piece_len = max((len(k) for k in piece_scores), default=1)

    sha = eval_split_sha_value or eval_split_sha(corpus)
    smiles = iter_test_split(corpus, limit_molecules=limit_molecules)

    per_molecule = build_segmentation_data(
        adapter,
        smiles,
        piece_scores=piece_scores,
        glyph_tuple_by_id=glyph_tuple_by_id,
        max_piece_len=max_piece_len,
    )
    return compute_arm_segmentation(
        per_molecule,
        cell_id=cell_id,
        arm=arm,
        boundary=boundary,
        training_corpus_sha=training_corpus_sha,
        eval_split_sha=sha,
    )
