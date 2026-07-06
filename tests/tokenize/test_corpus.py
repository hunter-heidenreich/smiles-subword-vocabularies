"""Unit tests for the shared tokenize corpus / fingerprint helpers.

`_corpus` is exercised transitively by every build / audit / measure path, but
its provenance-load-bearing and branching contracts deserve direct pins:
`manifest_shard_fingerprint` (the single recipe behind both the training-corpus
and held-out-split fingerprints, so the two cannot drift), `ensure_smi_cache`'s
staleness branch, and the streaming / atomic write helpers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from smiles_subword.tokenize._corpus import (
    ensure_smi_cache,
    iter_smiles_from_parquet,
    manifest_shard_fingerprint,
    materialize_smiles_txt,
    training_corpus_sha,
)

if TYPE_CHECKING:
    from pathlib import Path


def _write_manifest(path: Path, shas: list[str]) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "shards": [
                    {"file": f"s{i}.parquet", "sha256": s} for i, s in enumerate(shas)
                ]
            }
        )
    )
    return path


def _write_parquet_dir(parent: Path, shards: dict[str, list[str]]) -> Path:
    corpus = parent / "corpus"
    corpus.mkdir()
    for name, smiles in shards.items():
        pq.write_table(pa.table({"smiles": smiles}), corpus / name)
    return corpus


# --- manifest_shard_fingerprint ----------------------------------------------


def test_fingerprint_is_sort_invariant(tmp_path: Path) -> None:
    # The recipe sorts the SHAs, so manifest shard order must not matter.
    a = _write_manifest(tmp_path / "a.yaml", ["c" * 64, "a" * 64, "b" * 64])
    b = _write_manifest(tmp_path / "b.yaml", ["a" * 64, "b" * 64, "c" * 64])
    assert manifest_shard_fingerprint(a) == manifest_shard_fingerprint(b)


def test_fingerprint_changes_when_a_sha_changes(tmp_path: Path) -> None:
    a = _write_manifest(tmp_path / "a.yaml", ["a" * 64, "b" * 64])
    b = _write_manifest(tmp_path / "b.yaml", ["a" * 64, "c" * 64])
    assert manifest_shard_fingerprint(a) != manifest_shard_fingerprint(b)


def test_fingerprint_raises_on_no_shards(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text(yaml.safe_dump({"shards": []}))
    with pytest.raises(ValueError, match="no shards listed"):
        manifest_shard_fingerprint(path)


def test_fingerprint_golden_value(tmp_path: Path) -> None:
    # Frozen against the pinned BLAKE2b-128 recipe; a deliberate change to the
    # algorithm or framing must update this and re-fingerprint every cell.
    path = _write_manifest(tmp_path / "m.yaml", ["0" * 64, "1" * 64])
    assert manifest_shard_fingerprint(path) == "66d6592c4fd2ef9711b916b230ccb594"


def test_training_corpus_sha_fingerprints_dir_manifest(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    manifest = _write_manifest(corpus / "MANIFEST.yaml", ["a" * 64, "b" * 64])
    assert training_corpus_sha(corpus) == manifest_shard_fingerprint(manifest)


# --- ensure_smi_cache --------------------------------------------------------


def test_ensure_smi_cache_builds_when_absent(tmp_path: Path) -> None:
    corpus = _write_parquet_dir(tmp_path, {"s0.parquet": ["CCO", "c1ccccc1"]})

    txt = ensure_smi_cache(corpus, sha="abc")

    assert txt == tmp_path / "corpus.smi"
    assert txt.read_text().splitlines() == ["CCO", "c1ccccc1"]
    assert (tmp_path / "corpus.smi.sha").read_text().strip() == "abc"


def test_ensure_smi_cache_hit_skips_rebuild(tmp_path: Path) -> None:
    corpus = _write_parquet_dir(tmp_path, {"s0.parquet": ["CCO"]})
    txt = ensure_smi_cache(corpus, sha="abc")
    txt.write_text("SENTINEL\n")  # tamper; a rebuild would overwrite this

    again = ensure_smi_cache(corpus, sha="abc")  # same sha -> cache hit

    assert again.read_text() == "SENTINEL\n"


def test_ensure_smi_cache_rebuilds_on_sha_mismatch(tmp_path: Path) -> None:
    corpus = _write_parquet_dir(tmp_path, {"s0.parquet": ["CCO"]})
    txt = ensure_smi_cache(corpus, sha="abc")
    txt.write_text("SENTINEL\n")

    again = ensure_smi_cache(corpus, sha="def")  # stale sidecar -> rebuild

    assert again.read_text().splitlines() == ["CCO"]
    assert (tmp_path / "corpus.smi.sha").read_text().strip() == "def"


# --- materialize_smiles_txt / iter_smiles_from_parquet -----------------------


def test_materialize_writes_all_smiles_newline_joined(tmp_path: Path) -> None:
    corpus = _write_parquet_dir(
        tmp_path, {"s0.parquet": ["CCO", "c1ccccc1"], "s1.parquet": ["CCN"]}
    )
    out = tmp_path / "sub" / "out.smi"

    materialize_smiles_txt(corpus, out)

    assert out.read_text().splitlines() == ["CCO", "c1ccccc1", "CCN"]
    assert not out.with_name("out.smi.tmp").exists()  # atomic: no leftover tmp


def test_iter_streams_in_sorted_shard_order(tmp_path: Path) -> None:
    # Written b-first; iteration must follow sorted filename order (a before b).
    corpus = _write_parquet_dir(
        tmp_path, {"b-shard.parquet": ["N"], "a-shard.parquet": ["C"]}
    )
    assert list(iter_smiles_from_parquet(corpus)) == ["C", "N"]
