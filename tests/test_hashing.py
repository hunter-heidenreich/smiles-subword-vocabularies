"""Unit tests for the shared SHA256 provenance primitives.

`sha256_file`/`sha256_bytes` are exercised transitively by every
ingest/preprocess manifest test, but they are provenance-load-bearing (Zenodo
deposit SHAs, manifest verification) and the multi-chunk read path is never hit
by the tiny test corpora. These tests pin the lowercase-hex contract against a
known vector, exercise the chunked read loop directly with a >1 MiB input, and
assert the in-memory `sha256_bytes` agrees with the streamed `sha256_file`.
"""

from __future__ import annotations

import hashlib
import re
from typing import TYPE_CHECKING

from smiles_subword._hashing import _SHA_READ_CHUNK, sha256_bytes, sha256_file

if TYPE_CHECKING:
    from pathlib import Path

# SHA256("abc"), the canonical FIPS 180-4 test vector.
_ABC_SHA256 = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_matches_known_vector(tmp_path: Path) -> None:
    path = tmp_path / "abc.txt"
    path.write_bytes(b"abc")
    assert sha256_file(path) == _ABC_SHA256


def test_empty_file_hashes_to_sha256_of_empty(tmp_path: Path) -> None:
    path = tmp_path / "empty"
    path.write_bytes(b"")
    assert sha256_file(path) == hashlib.sha256(b"").hexdigest()


def test_output_is_lowercase_hex(tmp_path: Path) -> None:
    path = tmp_path / "x"
    path.write_bytes(b"some provenance bytes")
    digest = sha256_file(path)
    assert re.fullmatch(r"[0-9a-f]{64}", digest)


def test_multi_chunk_read_matches_one_shot(tmp_path: Path) -> None:
    # Span several read chunks so the `iter(... read ...)` loop runs more than
    # once, and assert it equals a single-shot hash of the same bytes.
    data = bytes(range(256)) * ((_SHA_READ_CHUNK * 3) // 256 + 1)
    assert len(data) > _SHA_READ_CHUNK * 3
    path = tmp_path / "big.bin"
    path.write_bytes(data)
    assert sha256_file(path) == hashlib.sha256(data).hexdigest()


def test_bytes_matches_known_vector() -> None:
    assert sha256_bytes(b"abc") == _ABC_SHA256


def test_bytes_matches_file_on_same_input(tmp_path: Path) -> None:
    # The in-memory and streamed primitives must agree on identical bytes.
    data = b"some provenance bytes"
    path = tmp_path / "x"
    path.write_bytes(data)
    assert sha256_bytes(data) == sha256_file(path)
