"""Shared corpus / hashing helpers for the tokenize submodule.

Primitives shared by every tokenizer build path (smirk_base, smirk_gpe,
smirk_unigram): stream the ``smiles`` column out of a sorted Parquet shard set,
materialize it to a one-SMILES-per-line text file, and fingerprint the corpus by
hashing its shard SHAs. Private; public helpers re-exported from ``__init__``.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import pyarrow.parquet as pq
import yaml

from smiles_subword._io import atomic_write_text

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

_SMILES_BATCH_ROWS = 131072


def _iter_smiles_batches(parquet_dir: Path) -> Iterator[list[str]]:
    """Yield ``smiles`` columns batch-by-batch from sorted Parquet shards."""
    for shard in sorted(parquet_dir.glob("*.parquet")):
        pf = pq.ParquetFile(shard)
        for batch in pf.iter_batches(batch_size=_SMILES_BATCH_ROWS, columns=["smiles"]):
            yield batch.column("smiles").to_pylist()


def iter_smiles_from_parquet(parquet_dir: Path) -> Iterator[str]:
    """Stream the ``smiles`` column from every Parquet shard under ``parquet_dir``.

    Sorted by shard filename for determinism. One source of truth shared by the
    Stage-5 tokenizer-builders and the intrinsics driver.
    """
    for batch in _iter_smiles_batches(parquet_dir):
        yield from batch


def materialize_smiles_txt(parquet_dir: Path, out_path: Path) -> Path:
    """Write every ``smiles`` row of a Parquet shard set to a single text file.

    Atomic (tmp+rename) so a crash mid-write can't leave a partial ``.smi`` next
    to a stale ``.sha`` sidecar. Returns ``out_path``.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp_path.open("w") as fh:
        for batch in _iter_smiles_batches(parquet_dir):
            if batch:
                fh.write("\n".join(batch))
                fh.write("\n")
    tmp_path.replace(out_path)
    return out_path


def manifest_shard_fingerprint(manifest_path: Path) -> str:
    """Stable 32-hex-char BLAKE2b-128 fingerprint of a stage's shard set.

    Sorts the per-shard SHA256s from ``manifest_path`` (a ``MANIFEST.yaml``) and
    hashes the newline-joined list. Single recipe behind both the training-corpus
    fingerprint and the held-out-split fingerprint
    (``measure._cells.eval_split_sha``), so the two cannot drift.
    """
    manifest = yaml.safe_load(manifest_path.read_text())
    shas = sorted(s["sha256"] for s in manifest.get("shards", []))
    if not shas:
        raise ValueError(f"no shards listed in {manifest_path}")
    h = hashlib.blake2b(digest_size=16)
    for sha in shas:
        h.update(sha.encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()


def training_corpus_sha(parquet_dir: Path) -> str:
    """Stable 32-hex-char fingerprint of a Parquet stage's input shards.

    Fingerprints ``<parquet_dir>/MANIFEST.yaml`` (written by the upstream Stage 4
    subsample driver) via :func:`manifest_shard_fingerprint`. Stored as
    ``training_corpus_sha`` in the tokenizer's ``meta.yaml`` so a measurement can
    pin the exact corpus a cell was trained on.
    """
    return manifest_shard_fingerprint(parquet_dir / "MANIFEST.yaml")


def ensure_smi_cache(parquet_dir: Path, *, sha: str) -> Path:
    """Get-or-rebuild the cached one-SMILES-per-line text dump.

    Lives at ``<parent>/<name>.smi`` beside ``parquet_dir``, with a ``.sha``
    sidecar holding ``sha`` (caller's corpus fingerprint). Rewrites only when the
    sidecar mismatches; both writes are atomic tmp+rename.
    """
    txt_path = parquet_dir.parent / f"{parquet_dir.name}.smi"
    sha_path = parquet_dir.parent / f"{parquet_dir.name}.smi.sha"
    if txt_path.exists() and sha_path.exists() and sha_path.read_text().strip() == sha:
        return txt_path
    materialize_smiles_txt(parquet_dir, txt_path)
    atomic_write_text(sha_path, sha + "\n")
    return txt_path


def write_meta_yaml(meta_path: Path, payload: dict[str, object]) -> None:
    """Atomically write ``payload`` to ``meta_path`` as YAML.

    Every tokenizer save path uses this; ``meta.yaml`` is the only canonical
    record of a tokenizer's identity / intrinsics, so a truncated write is fatal.
    """
    atomic_write_text(meta_path, yaml.safe_dump(payload, sort_keys=False))
