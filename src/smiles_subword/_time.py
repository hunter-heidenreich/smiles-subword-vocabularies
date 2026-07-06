"""Wall-clock timestamp helper shared across pipeline stages.

A leaf module (stdlib only), like :mod:`smiles_subword._hashing` and
:mod:`smiles_subword._sql`, so any subpackage can import it without a cycle.
"""

from __future__ import annotations

from datetime import UTC, datetime


def utc_now_naive_seconds() -> datetime:
    """Return the current UTC time, made naive (no tzinfo) and truncated to seconds.

    Naive so it casts cleanly to a parquet ``timestamp[us]``; truncated to whole
    seconds so stamps within the same second read back identically across reruns.
    Shared by the ``ingest_ts`` data-column stamp and the preprocess stage-start
    stamp.
    """
    return datetime.now(tz=UTC).replace(microsecond=0).replace(tzinfo=None)


__all__ = ["utc_now_naive_seconds"]
