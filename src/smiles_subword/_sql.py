"""Shared SQL-identifier quoting for DuckDB query construction.

Column names are interpolated into DuckDB SQL rather than bound as ``?`` (SQL
cannot parameterize identifiers); validating + double-quoting them makes a
malformed or injected name fail loudly instead of reaching the query.

A leaf module (stdlib only) so any subpackage can import it without a cycle.
"""

from __future__ import annotations

import re

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def quote_ident(name: str) -> str:
    """Validate `name` as a plain SQL identifier and return it double-quoted.

    Raises:
        ValueError: if `name` is not `^[A-Za-z_][A-Za-z0-9_]*$` — guards the
            f-string interpolation of column names into DuckDB SQL.
    """
    if not _IDENT_RE.match(name):
        raise ValueError(f"invalid SQL identifier: {name!r}")
    return f'"{name}"'


__all__ = ["quote_ident"]
