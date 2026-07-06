"""Unit tests for the shared SQL-identifier guard.

`quote_ident` is the single defense behind every f-string-interpolated column
name in DuckDB query construction (the ingest CSV reader, the preprocess dedup
window). It is exercised transitively wherever those queries run, but the
reject path is the load-bearing one — pin it directly against a known set of
non-identifiers plus the valid-name quoting contract.
"""

from __future__ import annotations

import pytest

from smiles_subword._sql import quote_ident


@pytest.mark.parametrize(
    "bad_name",
    [
        "1abc",  # leading digit
        "a-b",  # hyphen
        "a b",  # space
        'a"b',  # embedded quote
        "smiles; DROP TABLE t",  # injection attempt
        "",  # empty
    ],
)
def test_rejects_non_identifier(bad_name: str) -> None:
    with pytest.raises(ValueError, match="invalid SQL identifier"):
        quote_ident(bad_name)


def test_double_quotes_valid_name() -> None:
    assert quote_ident("source_id") == '"source_id"'
