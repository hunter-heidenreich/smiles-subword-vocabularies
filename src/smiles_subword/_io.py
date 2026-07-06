"""Generic on-disk I/O helpers shared across the package.

A leaf module (stdlib only), like :mod:`smiles_subword._shards` and
:mod:`smiles_subword._hashing`, so any subpackage can import it without a cycle.
"""

from __future__ import annotations

import json
import shutil
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path
    from typing import Any


def atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically.

    Stages to a sibling ``.tmp`` and ``replace``-s it into place, so a crashed
    write never leaves a partial file; creates the parent dir if needed.
    (Manifest writes keep their own ``fsync``-ing variant for stronger
    durability.)
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text)
    tmp_path.replace(path)


def atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    """Write ``payload`` as pretty-printed JSON to ``path`` atomically.

    ``indent=2, sort_keys=True`` with a trailing newline, via
    :func:`atomic_write_text`.
    """
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def read_json_or_none(path: Path) -> dict[str, Any] | None:
    """Return the JSON object at ``path``, or ``None`` if missing or corrupt.

    Callers layer their own freshness / SHA checks on the returned payload.
    """
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


@contextmanager
def atomic_output_dir(output_dir: Path, *, keep_on_error: bool) -> Iterator[Path]:
    """Yield a staging dir that atomically replaces ``output_dir`` on clean exit.

    Work happens in a sibling ``<output_dir>.tmp`` (a stale one is cleared
    first); on clean exit any existing ``output_dir`` is removed and the staging
    dir renamed into place. ``keep_on_error`` picks the failure path: ``False``
    removes the half-written staging dir and re-raises (prior ``output_dir``
    intact); ``True`` leaves it for inspection.
    """
    staging_dir = output_dir.parent / (output_dir.name + ".tmp")
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)
    try:
        yield staging_dir
    except BaseException:
        if not keep_on_error:
            shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    else:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        staging_dir.rename(output_dir)


__all__ = [
    "atomic_output_dir",
    "atomic_write_json",
    "atomic_write_text",
    "read_json_or_none",
]
