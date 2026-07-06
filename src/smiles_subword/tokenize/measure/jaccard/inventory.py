"""Training-corpus chunk inventory + occurrence-based subword classification.

``J_struct`` splits each arm's multi-glyph subwords into *bracket-internal*
(every training-corpus occurrence falls inside a bracketed-atom Layer-B chunk)
and *structural* (otherwise). A read-only check established that a glyph-only
rule is **insufficient** — NMB charge/isotope pieces like ``O-`` and ``13C``
carry no bracket-exclusive glyph yet only ever occur inside brackets — so the
split must be sourced from actual training-corpus occurrence.

Two facts make this tractable:

* ``pretokenize_layer_b`` is model- and ``merge_brackets``-independent, so the
  distinct-chunk inventory is a property of the *training input* alone — built
  once per ``training_corpus_sha`` and shared across every ``V`` and both arms.
* Tokenization is **chunk-local** (``split_structure=True`` ⇒ tokens never span
  Layer-B chunks; verified by Absorption's zero cross-chunk fraction), so encoding a
  *distinct* chunk once reproduces every occurrence's emission.

So: one streaming ``pretokenize_layer_b`` pass collects the distinct chunk
surfaces (a chunk is *bracketed* iff its surface starts with ``[``); then each
cell encodes that small inventory and attributes every emitted multi-glyph
piece to ``seen_bracket`` / ``seen_nonbracket``.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
from typing import TYPE_CHECKING

from smiles_subword._io import atomic_write_json, read_json_or_none
from smiles_subword.tokenize._corpus import iter_smiles_from_parquet
from smiles_subword.tokenize.measure.jaccard.math import GlyphTuple

if TYPE_CHECKING:
    from pathlib import Path

    from smiles_subword.tokenize.adapters.smirk import SmirkAdapter
    from smiles_subword.tokenize.adapters.smirk_unigram import UnigramSmirkAdapter

INVENTORY_SCHEMA_VERSION = 1

DEFAULT_NONBRACKET_CAP = 3_000_000
"""Cap on retained distinct non-bracket chunks (memory bound for huge corpora).

Bracket chunks are always retained in full (they saturate at a few hundred).
Non-bracket distinct chunks grow with long organic runs; the cap bounds memory
and per-cell encode cost. Because emitted multi-glyph pieces saturate well below
the cap, a bound cap leaves frequent-piece classification unchanged — but the
``nonbracket_cap_bound`` flag records when it bound so it is never silent.
"""

ENCODE_CHUNK_BATCH = 4096


@dataclass(frozen=True)
class ChunkInventory:
    """Distinct Layer-B chunk surfaces of one training input, by bracket flag."""

    training_corpus_sha: str
    bracket_chunks: tuple[str, ...]
    nonbracket_chunks: tuple[str, ...]
    n_molecules_scanned: int
    nonbracket_cap_bound: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": INVENTORY_SCHEMA_VERSION,
            "training_corpus_sha": self.training_corpus_sha,
            "n_molecules_scanned": self.n_molecules_scanned,
            "nonbracket_cap_bound": self.nonbracket_cap_bound,
            "bracket_chunks": list(self.bracket_chunks),
            "nonbracket_chunks": list(self.nonbracket_chunks),
        }


def build_chunk_inventory(
    adapter: SmirkAdapter | UnigramSmirkAdapter,
    training_dir: Path,
    *,
    training_corpus_sha: str,
    nonbracket_cap: int | None = DEFAULT_NONBRACKET_CAP,
    limit_molecules: int | None = None,
) -> ChunkInventory:
    """Stream the training split once and collect distinct Layer-B chunks.

    ``adapter`` is used only for its model-independent ``pretokenize_layer_b``,
    so any cell sharing the training input yields the same inventory.
    """
    bracket: dict[str, None] = {}
    nonbracket: dict[str, None] = {}
    cap_bound = False
    n_scanned = 0
    smiles = iter_smiles_from_parquet(training_dir)
    if limit_molecules is not None:
        smiles = islice(smiles, limit_molecules)
    for smi in smiles:
        n_scanned += 1
        for chunk, _span in adapter.pretokenize_layer_b(smi):
            if chunk.startswith("["):
                bracket.setdefault(chunk, None)
            elif chunk not in nonbracket:
                if nonbracket_cap is not None and len(nonbracket) >= nonbracket_cap:
                    cap_bound = True
                    continue
                nonbracket[chunk] = None
    return ChunkInventory(
        training_corpus_sha=training_corpus_sha,
        bracket_chunks=tuple(bracket),
        nonbracket_chunks=tuple(nonbracket),
        n_molecules_scanned=n_scanned,
        nonbracket_cap_bound=cap_bound,
    )


def write_inventory(path: Path, inventory: ChunkInventory) -> Path:
    """Atomically cache ``inventory`` as JSON; return its path."""
    atomic_write_json(path, inventory.as_dict())
    return path


def read_inventory(path: Path, *, training_corpus_sha: str) -> ChunkInventory | None:
    """Return the cached inventory iff present and its SHA matches; else None."""
    payload = read_json_or_none(path)
    if payload is None:
        return None
    if payload.get("training_corpus_sha") != training_corpus_sha:
        return None
    return ChunkInventory(
        training_corpus_sha=training_corpus_sha,
        bracket_chunks=tuple(payload.get("bracket_chunks", [])),
        nonbracket_chunks=tuple(payload.get("nonbracket_chunks", [])),
        n_molecules_scanned=int(payload.get("n_molecules_scanned", 0)),
        nonbracket_cap_bound=bool(payload.get("nonbracket_cap_bound", False)),
    )


def get_or_build_inventory(
    adapter: SmirkAdapter | UnigramSmirkAdapter,
    training_dir: Path,
    cache_path: Path,
    *,
    training_corpus_sha: str,
    nonbracket_cap: int | None = DEFAULT_NONBRACKET_CAP,
    limit_molecules: int | None = None,
) -> ChunkInventory:
    """Read the cached inventory for ``training_corpus_sha`` or build + cache it."""
    cached = read_inventory(cache_path, training_corpus_sha=training_corpus_sha)
    if cached is not None:
        return cached
    inventory = build_chunk_inventory(
        adapter,
        training_dir,
        training_corpus_sha=training_corpus_sha,
        nonbracket_cap=nonbracket_cap,
        limit_molecules=limit_molecules,
    )
    write_inventory(cache_path, inventory)
    return inventory


@dataclass(frozen=True)
class StructuralSplit:
    """Occurrence-based partition of one arm's multi-glyph subword set."""

    structural: frozenset[GlyphTuple]
    bracket_internal: frozenset[GlyphTuple]
    unseen: frozenset[GlyphTuple]


def classify_subwords(
    adapter: SmirkAdapter | UnigramSmirkAdapter,
    glyph_tuple_by_id: dict[int, GlyphTuple],
    multi_subwords: frozenset[GlyphTuple],
    inventory: ChunkInventory,
    *,
    batch_size: int = ENCODE_CHUNK_BATCH,
) -> StructuralSplit:
    """Attribute each multi-glyph subword to structural / bracket-internal.

    Encodes the inventory's bracket and non-bracket chunks with this arm's
    tokenizer; a multi-glyph piece emitted in any non-bracket chunk is
    *structural*, one emitted only in bracket chunks is *bracket-internal*, one
    emitted by no chunk is *unseen* (a vocab piece the corpus never realizes).
    """
    seen_bracket = _emitted_multi(
        adapter, inventory.bracket_chunks, glyph_tuple_by_id, batch_size=batch_size
    )
    seen_nonbracket = _emitted_multi(
        adapter, inventory.nonbracket_chunks, glyph_tuple_by_id, batch_size=batch_size
    )
    structural = multi_subwords & seen_nonbracket
    bracket_internal = (multi_subwords & seen_bracket) - seen_nonbracket
    unseen = multi_subwords - seen_bracket - seen_nonbracket
    return StructuralSplit(
        structural=frozenset(structural),
        bracket_internal=frozenset(bracket_internal),
        unseen=frozenset(unseen),
    )


def _emitted_multi(
    adapter: SmirkAdapter | UnigramSmirkAdapter,
    chunks: tuple[str, ...],
    glyph_tuple_by_id: dict[int, GlyphTuple],
    *,
    batch_size: int,
) -> set[GlyphTuple]:
    emitted: set[GlyphTuple] = set()
    for start in range(0, len(chunks), batch_size):
        batch = list(chunks[start : start + batch_size])
        for ids in adapter.encode_batch(batch, add_special_tokens=False):
            for tid in ids:
                tup = glyph_tuple_by_id.get(tid)
                if tup is not None and len(tup) >= 2:
                    emitted.add(tup)
    return emitted


__all__ = [
    "DEFAULT_NONBRACKET_CAP",
    "ChunkInventory",
    "StructuralSplit",
    "build_chunk_inventory",
    "classify_subwords",
    "get_or_build_inventory",
    "read_inventory",
    "write_inventory",
]
