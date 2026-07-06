"""Transfer-matrix I/O: load a cell, read it on another corpus, deposit.

Loads a tokenizer trained on ``train_corpus``, streams ``eval_corpus``'s held-out
test split through it once, counts tokens / glyphs / ``[UNK]``s per molecule, and
aggregates into a :class:`TransferRecord` (:mod:`.math`). Per-cell JSONs are
deposited under ``results/data/transfer/``.

The cell grid (:func:`enumerate_transfer_cells`) is the off-diagonal of the 4x4
corpus matrix at ``V=1024``, ``nmb``, both arms (24 cells). The diagonal is the
penalty normalizer, reused from each corpus's fertility. Eval splits are read via
a fixed-seed random sample (``run_transfer(sample=...)``) since the fertility mean
converges well before the full 1M-molecule split.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import yaml

from smiles_subword._io import atomic_write_json
from smiles_subword.config import cell_artifact_name
from smiles_subword.paths import tokenizer_artifact_dir
from smiles_subword.tokenize._batched import iter_encoded_batches
from smiles_subword.tokenize._corpus import iter_smiles_from_parquet
from smiles_subword.tokenize.measure._bootstrap import bootstrap_seed
from smiles_subword.tokenize.measure._cells import (
    corpus_test_split_dir,
    eval_split_sha,
    load_cell_adapter,
)
from smiles_subword.tokenize.measure._glyphmap import glyph_count_map
from smiles_subword.tokenize.measure.supplementary.transfer.math import (
    TRANSFER_DIR,
    Arm,
    Boundary,
    PerMoleculeTransfer,
    TransferRecord,
    compute_transfer_record,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from pathlib import Path

    from smiles_subword.tokenize.base import Tokenizer

CORPORA: tuple[str, ...] = ("pubchem", "zinc22", "coconut", "real_space")
"""Display / iteration order for the transfer matrix."""

HEADLINE_V = 1024
HEADLINE_BOUNDARY: Boundary = "nmb"
ARMS: tuple[Arm, ...] = ("bpe", "unigram")


def cell_name(arm: Arm, vocab_size: int, boundary: Boundary) -> str:
    """Artifact directory name for a grid cell, e.g. ``smirk_gpe_v1024_nmb``."""
    return cell_artifact_name(arm, vocab_size, boundary)


def count_per_molecule(
    adapter: Tokenizer,
    glyph_map: Mapping[int, int],
    unk_id: int | None,
    smiles: Iterable[str],
) -> list[PerMoleculeTransfer]:
    """Encode ``smiles`` once and tally per-molecule tokens / glyphs / ``[UNK]``s.

    Shared by the in-grid transfer matrix (:func:`run_transfer`) and the OOD eval
    (:func:`.ood.run_ood_eval`); ``glyph_map`` maps each token id to its glyph
    span (absent ids count as 1) and ``unk_id`` is the atom-level OOV id (``None``
    for arms without one).
    """
    per_molecule: list[PerMoleculeTransfer] = []
    for ids in iter_encoded_batches(adapter, smiles):
        n_unk = 0 if unk_id is None else sum(1 for i in ids if i == unk_id)
        per_molecule.append(
            PerMoleculeTransfer(
                n_tokens=len(ids),
                n_glyphs=sum(glyph_map.get(i, 1) for i in ids),
                n_unk=n_unk,
            )
        )
    return per_molecule


def _training_corpus_sha(corpus: str, name: str) -> str:
    """Read ``training_corpus_sha`` from a trained cell's ``meta.yaml``."""
    meta_path = tokenizer_artifact_dir(corpus, name) / "meta.yaml"
    meta = yaml.safe_load(meta_path.read_text())
    return str(meta.get("training_corpus_sha", ""))


def run_transfer(
    *,
    train_corpus: str,
    eval_corpus: str,
    arm: Arm,
    vocab_size: int = HEADLINE_V,
    boundary: Boundary = HEADLINE_BOUNDARY,
    sample: int | None = None,
    limit: int | None = None,
) -> TransferRecord:
    """Read the ``(train_corpus, arm, V, boundary)`` cell on ``eval_corpus``.

    Loads the trained tokenizer, builds its per-token glyph-count map, and
    reads ``eval_corpus``'s held-out test split. ``sample`` draws a fixed-seed
    random subset of that many molecules (the unbiased lean-run path — first-N
    would skew toward early-ingested, simpler molecules); ``limit`` takes the
    first N instead (tests / debug only).
    """
    name = cell_name(arm, vocab_size, boundary)
    adapter = load_cell_adapter(train_corpus, name)
    glyph_map = glyph_count_map(tokenizer_artifact_dir(train_corpus, name), arm)
    unk_id = getattr(adapter, "unk_id", None)

    smiles: Iterable[str] = iter_smiles_from_parquet(corpus_test_split_dir(eval_corpus))
    if sample is not None:
        pool = list(smiles)
        if len(pool) > sample:
            rng = random.Random(bootstrap_seed(f"sample__{eval_corpus}__{name}"))
            pool = rng.sample(pool, sample)
        smiles = pool
    elif limit is not None:
        smiles = (s for i, s in enumerate(smiles) if i < limit)

    per_molecule = count_per_molecule(adapter, glyph_map, unk_id, smiles)

    return compute_transfer_record(
        train_corpus=train_corpus,
        eval_corpus=eval_corpus,
        arm=arm,
        vocab_size=vocab_size,
        boundary=boundary,
        per_molecule=per_molecule,
        train_corpus_sha=_training_corpus_sha(train_corpus, name),
        eval_split_sha=eval_split_sha(eval_corpus),
    )


def enumerate_transfer_cells() -> list[tuple[str, str, Arm, int, Boundary]]:
    """Return the off-diagonal transfer cells ``(train, eval, arm, V, boundary)``.

    The lean scope: every ``train != eval`` corpus pair at ``V=1024``, ``nmb``,
    both arms (12 ordered pairs x 2 arms = 24). The on-domain diagonal is *not*
    recomputed --- it equals each corpus's fertility and is reused from there
    as the penalty normalizer.
    """
    return [
        (train, ev, arm, HEADLINE_V, HEADLINE_BOUNDARY)
        for arm in ARMS
        for train in CORPORA
        for ev in CORPORA
        if train != ev
    ]


def transfer_record_path(record: TransferRecord) -> Path:
    """Per-cell deposit path for a :class:`TransferRecord`."""
    return TRANSFER_DIR / f"{record.cell_key}.json"


def write_transfer_record(record: TransferRecord) -> Path:
    """Deposit a :class:`TransferRecord` as per-cell JSON; return its path."""
    path = transfer_record_path(record)
    atomic_write_json(path, record.as_dict())
    return path


__all__ = [
    "ARMS",
    "CORPORA",
    "HEADLINE_BOUNDARY",
    "HEADLINE_V",
    "cell_name",
    "enumerate_transfer_cells",
    "run_transfer",
    "transfer_record_path",
    "write_transfer_record",
]
