"""Loading trained grid cells and locating their held-out split.

Shared infrastructure for the measurement runners/io: load a trained cell's
adapter by ``(corpus, name)``, locate its deterministic held-out test split,
fingerprint that split for freshness, and stream it. These consolidate a
``load_cell_adapter`` and held-out-split reader that were previously copy-pasted
across the per-topic runners; they are measurement-agnostic, so they belong in a
neutral leaf.

A leaf module (``paths`` + ``adapters`` + ``_corpus`` + stdlib/yaml), so any
runner can import it without a cycle.
"""

from __future__ import annotations

from itertools import islice
from typing import TYPE_CHECKING

import yaml

from smiles_subword.paths import processed_corpus_dir, tokenizer_artifact_dir
from smiles_subword.tokenize._corpus import (
    iter_smiles_from_parquet,
    manifest_shard_fingerprint,
)
from smiles_subword.tokenize.adapters.smirk import SmirkAdapter
from smiles_subword.tokenize.adapters.smirk_unigram import UnigramSmirkAdapter

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


def corpus_test_split_dir(corpus: str) -> Path:
    """Return ``data/processed/<corpus>/canon_dedup_v1/test/`` for ``corpus``."""
    return processed_corpus_dir(corpus) / "canon_dedup_v1" / "test"


def iter_test_split(
    corpus: str, *, limit_molecules: int | None = None
) -> Iterator[str]:
    """Stream a corpus's held-out test split, optionally truncated.

    The held-out reader every measurement runner shares: SMILES from
    ``canon_dedup_v1/test/`` in deterministic shard order. ``limit_molecules``
    yields only the first ``n`` molecules — a debug/test hook; production
    deposits pass ``None`` to stream the whole split.
    """
    smiles = iter_smiles_from_parquet(corpus_test_split_dir(corpus))
    if limit_molecules is not None:
        return islice(smiles, limit_molecules)
    return smiles


def eval_split_sha(corpus: str) -> str:
    """Stable 32-hex fingerprint of a corpus's held-out test split.

    Fingerprints the test-split ``MANIFEST.yaml`` (written by the preprocessing
    pipeline) via :func:`smiles_subword.tokenize._corpus.manifest_shard_fingerprint`
    — the same recipe as the training-corpus fingerprint. Recorded in the per-cell
    measurement JSON so a re-run of the held-out split invalidates every
    downstream record automatically.
    """
    return manifest_shard_fingerprint(corpus_test_split_dir(corpus) / "MANIFEST.yaml")


def load_cell_adapter(corpus: str, name: str) -> SmirkAdapter | UnigramSmirkAdapter:
    """Load a trained cell artifact by ``(corpus, name)``.

    Reads the cell's ``meta.yaml`` to dispatch on ``base_kind``: ``smirk_gpe``
    cells return a :class:`SmirkAdapter`, ``smirk_unigram`` cells return a
    :class:`UnigramSmirkAdapter`. Other kinds raise — the grid only trains the
    two Smirk arms.
    """
    artifact_dir = tokenizer_artifact_dir(corpus, name)
    meta_path = artifact_dir / "meta.yaml"
    if not meta_path.is_file():
        raise FileNotFoundError(
            f"no meta.yaml under {artifact_dir!r}; cell not trained?"
        )
    meta = yaml.safe_load(meta_path.read_text())
    base_kind = str(meta.get("base_kind", "")).strip()
    if base_kind == "smirk_gpe":
        return SmirkAdapter.load(artifact_dir)
    if base_kind == "smirk_unigram":
        return UnigramSmirkAdapter.load(artifact_dir)
    raise ValueError(
        f"cell adapter not implemented for base_kind={base_kind!r} at {artifact_dir}"
    )


__all__ = [
    "corpus_test_split_dir",
    "eval_split_sha",
    "iter_test_split",
    "load_cell_adapter",
]
