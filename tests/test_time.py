"""Unit test for the shared wall-clock stamp.

`utc_now_naive_seconds` is the single source of the `ingest_ts` data-column
stamp and the preprocess stage-start stamp. Its naive + second-truncated shape
is the reproducibility contract behind the `timestamp[us]` cast and
byte-identical same-second reruns — pin it directly.
"""

from __future__ import annotations

from smiles_subword._time import utc_now_naive_seconds


def test_utc_now_naive_seconds_is_naive_and_truncated() -> None:
    ts = utc_now_naive_seconds()
    assert ts.tzinfo is None  # naive, so it casts to a parquet timestamp[us]
    assert ts.microsecond == 0  # truncated, so same-second reruns hash identically
