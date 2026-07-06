"""Unit tests for the shared on-disk I/O primitives.

`atomic_write_text`/`atomic_write_json`/`read_json_or_none`/`atomic_output_dir`
back every deposit and manifest write in the package. They are exercised
transitively by the integration tests, but the branching error paths
(`atomic_output_dir`'s `keep_on_error` split, stale-staging cleanup, the
`JSONDecodeError`-swallowing read) are never hit on the happy path. These tests
pin the observable contracts: serialization format, parent-dir creation, no
leftover `.tmp`, and the replace/rollback semantics.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from smiles_subword._io import (
    atomic_output_dir,
    atomic_write_json,
    atomic_write_text,
    read_json_or_none,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_write_text_roundtrips(tmp_path: Path) -> None:
    path = tmp_path / "out.txt"
    atomic_write_text(path, "hello\nworld\n")
    assert path.read_text() == "hello\nworld\n"


def test_write_text_creates_parent_dirs(tmp_path: Path) -> None:
    path = tmp_path / "a" / "b" / "c.txt"
    atomic_write_text(path, "x")
    assert path.read_text() == "x"


def test_write_text_leaves_no_tmp_sibling(tmp_path: Path) -> None:
    path = tmp_path / "out.txt"
    atomic_write_text(path, "x")
    assert list(tmp_path.iterdir()) == [path]


def test_write_text_overwrites_existing(tmp_path: Path) -> None:
    path = tmp_path / "out.txt"
    path.write_text("old")
    atomic_write_text(path, "new")
    assert path.read_text() == "new"


def test_write_json_serialization_format(tmp_path: Path) -> None:
    # Sorted keys, two-space indent, trailing newline: the deposit format other
    # tooling reads back. Pin the exact bytes.
    path = tmp_path / "out.json"
    atomic_write_json(path, {"b": 1, "a": 2})
    assert path.read_text() == '{\n  "a": 2,\n  "b": 1\n}\n'


def test_write_json_roundtrips_through_reader(tmp_path: Path) -> None:
    path = tmp_path / "out.json"
    payload = {"nested": {"x": [1, 2, 3]}, "flag": True}
    atomic_write_json(path, payload)
    assert read_json_or_none(path) == payload


def test_read_json_returns_dict_for_valid(tmp_path: Path) -> None:
    path = tmp_path / "in.json"
    path.write_text(json.dumps({"k": "v"}))
    assert read_json_or_none(path) == {"k": "v"}


def test_read_json_returns_none_for_missing(tmp_path: Path) -> None:
    assert read_json_or_none(tmp_path / "nope.json") is None


def test_read_json_returns_none_for_corrupt(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.json"
    path.write_text("{not valid json")
    assert read_json_or_none(path) is None


def test_output_dir_renames_staging_into_place(tmp_path: Path) -> None:
    out = tmp_path / "result"
    with atomic_output_dir(out, keep_on_error=False) as staging:
        assert staging != out
        (staging / "data.txt").write_text("payload")
    assert (out / "data.txt").read_text() == "payload"
    assert not (tmp_path / "result.tmp").exists()


def test_output_dir_replaces_existing(tmp_path: Path) -> None:
    out = tmp_path / "result"
    out.mkdir()
    (out / "stale.txt").write_text("old")
    with atomic_output_dir(out, keep_on_error=False) as staging:
        (staging / "fresh.txt").write_text("new")
    assert (out / "fresh.txt").read_text() == "new"
    assert not (out / "stale.txt").exists()


def test_output_dir_error_discards_staging_and_preserves_prior(tmp_path: Path) -> None:
    out = tmp_path / "result"
    out.mkdir()
    (out / "kept.txt").write_text("prior")

    def run_and_fail() -> None:
        with atomic_output_dir(out, keep_on_error=False) as staging:
            (staging / "half.txt").write_text("partial")
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        run_and_fail()
    assert (out / "kept.txt").read_text() == "prior"
    assert not (tmp_path / "result.tmp").exists()


def test_output_dir_error_keeps_staging_when_requested(tmp_path: Path) -> None:
    out = tmp_path / "result"

    def run_and_fail() -> None:
        with atomic_output_dir(out, keep_on_error=True) as staging:
            (staging / "half.txt").write_text("partial")
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        run_and_fail()
    staging_dir = tmp_path / "result.tmp"
    assert (staging_dir / "half.txt").read_text() == "partial"
    assert not out.exists()


def test_output_dir_clears_stale_staging_first(tmp_path: Path) -> None:
    out = tmp_path / "result"
    stale_staging = tmp_path / "result.tmp"
    stale_staging.mkdir()
    (stale_staging / "leftover.txt").write_text("from a crashed run")
    with atomic_output_dir(out, keep_on_error=False) as staging:
        assert list(staging.iterdir()) == []
        (staging / "fresh.txt").write_text("new")
    assert (out / "fresh.txt").read_text() == "new"
    assert not (out / "leftover.txt").exists()
