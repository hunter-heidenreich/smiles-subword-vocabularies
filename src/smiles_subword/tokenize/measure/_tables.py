"""Shared rendering helpers for the measurement ``*_table.md`` aggregates.

A leaf module (stdlib only), so any ``*_io`` aggregator can import it without a
cycle.
"""

from __future__ import annotations

from typing import cast


def fmt_md(value: object, *, spec: str = ".4f") -> str:
    """Format a numeric cell for a measurement ``*_table.md``.

    ``None`` renders as an em dash and ``NaN`` as ``"nan"``; otherwise the value
    is formatted with ``spec`` — 4 decimals by default, ``"+.4f"`` for the signed
    delta columns. One precision (4 decimals) across every measurement's table.
    """
    if value is None:
        return "—"
    f = float(cast("float", value))
    if f != f:
        return "nan"
    return format(f, spec)


__all__ = ["fmt_md"]
