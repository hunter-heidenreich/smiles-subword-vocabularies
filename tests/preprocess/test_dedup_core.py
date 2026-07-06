"""Unit tests for the pure pieces of the DuckDB window-dedup core.

`_dedup_core` is covered end-to-end through test_canon_dedup.py, but three
contract-bearing pieces are only ever exercised transitively: `bucket_key`'s
padding + the prefix-ordering invariant that underwrites bucket==single_pass
equivalence, `apply_duckdb_pragmas`'s documented ValueError guard, and (since
the tie-break was made total) deterministic survival when `(smiles, source_id)`
collides. Pin them directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from smiles_subword.preprocess._dedup_core import (
    apply_duckdb_pragmas,
    bucket_key,
    dedup_to_writer,
)
from smiles_subword.preprocess._io import ShardWriter

if TYPE_CHECKING:
    from pathlib import Path


# --- bucket_key --------------------------------------------------------------


def test_bucket_key_pads_short_smiles() -> None:
    assert bucket_key("") == "0000"  # no bytes -> two padding bytes
    assert bucket_key("C") == "4300"  # 'C' == 0x43, padded with 0x00


def test_bucket_key_is_always_four_hex_chars() -> None:
    for smiles in ["", "C", "CCO", "c1ccccc1", "[Na+]", "Brc1ccccc1"]:
        key = bucket_key(smiles)
        assert len(key) == 4
        assert all(ch in "0123456789abcdef" for ch in key)


def test_bucket_key_order_matches_smiles_prefix_order() -> None:
    # The load-bearing invariant: sorting bucket keys reproduces the byte order
    # of the (zero-padded) 2-byte SMILES prefixes. This is what makes buckets,
    # concatenated in sorted-key order, byte-equivalent to single-pass dedup.
    smiles = ["CCO", "CCN", "Oc1ccccc1", "c1ccccc1", "[Na+]", "Brc1ccccc1", "C", ""]
    by_key = sorted(smiles, key=bucket_key)
    by_prefix = sorted(smiles, key=lambda s: s.encode("utf-8")[:2].ljust(2, b"\x00"))
    assert by_key == by_prefix


# --- apply_duckdb_pragmas ----------------------------------------------------


def test_apply_pragmas_rejects_malformed_memory_limit() -> None:
    con = duckdb.connect(":memory:")
    try:
        with pytest.raises(ValueError, match="malformed duckdb_memory_limit"):
            apply_duckdb_pragmas(con, threads=None, memory_limit="8GB; DROP TABLE t")
    finally:
        con.close()


def test_apply_pragmas_accepts_valid_settings() -> None:
    con = duckdb.connect(":memory:")
    try:
        apply_duckdb_pragmas(con, threads=2, memory_limit="512MB")  # must not raise
    finally:
        con.close()


# --- dedup_to_writer total tie-break -----------------------------------------

_TIE_SCHEMA = pa.schema(
    [("source_id", pa.string()), ("smiles", pa.string()), ("source", pa.string())]
)


def test_total_tie_break_picks_deterministic_survivor(tmp_path: Path) -> None:
    # Two rows share (smiles, source_id) but differ in `source`. The total
    # tie-break (source_id, then the remaining emitted columns) must keep the
    # lexicographically smallest full row deterministically — here source "x",
    # despite it appearing second in the input.
    table = pa.table(
        {
            "source_id": ["a", "a"],
            "smiles": ["CCO", "CCO"],
            "source": ["y", "x"],
        },
        schema=_TIE_SCHEMA,
    )
    src = tmp_path / "in.parquet"
    pq.write_table(table, src)

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    writer = ShardWriter(
        out_dir,
        schema=_TIE_SCHEMA,
        shard_prefix="dedup",
        target_bytes=2**20,
    )
    con = duckdb.connect(":memory:")
    try:
        dedup_to_writer(
            con,
            parquet_source=str(src),
            output_columns=("source_id", "smiles", "source"),
            writer=writer,
            out_schema=_TIE_SCHEMA,
            rows_per_batch=1024,
        )
        writer.close_current()  # dedup_to_writer leaves closing to the caller
    finally:
        con.close()

    result = pa.concat_tables(
        pq.read_table(s) for s in sorted(out_dir.glob("dedup-*.parquet"))
    )
    assert result.num_rows == 1
    assert result.column("source_id").to_pylist() == ["a"]
    assert result.column("source").to_pylist() == ["x"]
