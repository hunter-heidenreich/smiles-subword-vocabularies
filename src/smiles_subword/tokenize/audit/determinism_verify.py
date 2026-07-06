"""Per-cell determinism verification orchestration (grid + robustness extras).

:func:`verify_cell` (grid) and :func:`verify_extras_cell` (extras): retrain a
trained cell once via the exact dispatch path, compare rerun vs canonical (BPE
byte-identity; Unigram piece-set identity), deposit per-cell JSON. Both are thin
bindings of :func:`_runtime.run_verify`, differing only in cell→config resolver,
expected-jitter predicate, scratch prefix, and tag.

Failure handling:

- **BPE non-byte-identity** — impossible by construction: deposit JSON as
  evidence, then :class:`RuntimeError` halts the sweep.
- **Unigram jitter, expected** (grid ``V=1024 NMB``) — flag, record spread, no halt.
- **Unigram jitter, unexpected** — flag loudly; any extras jitter is unexpected.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from smiles_subword.tokenize.audit import _runtime
from smiles_subword.tokenize.extras import extras_cell_to_config
from smiles_subword.tokenize.grid import grid_cell_to_config

if TYPE_CHECKING:
    from smiles_subword.tokenize.audit.determinism import DeterminismResult
    from smiles_subword.tokenize.extras import ExtrasCell
    from smiles_subword.tokenize.grid import GridCell

_log = _runtime.make_logger("determinism")
_extras_log = _runtime.make_logger("determinism-extras")


def is_expected_unigram_jitter(cell: GridCell) -> bool:
    """True for the known-jitter set — Unigram, ``V=1024``, NMB.

    One frequency-tied piece is non-deterministic here (flag, not halt);
    piece-set identical at ``V≤512`` and ``V=1024 MB``.
    """
    return cell.algo == "unigram" and cell.vocab_size == 1024 and cell.boundary == "nmb"


def _extras_expected_jitter(cell: ExtrasCell) -> bool:
    """Always False: extras Unigram cells (V=512 NMB, V=1024 MB, V=256 MB) fall
    outside the ``V=1024 NMB`` jitter set, so any extras jitter is UNEXPECTED.
    """
    del cell
    return False


def verify_cell(
    cell: GridCell, *, force: bool = False, dry_run: bool = False
) -> DeterminismResult | None:
    """Verify per-arm determinism for one grid cell; return the result or None.

    Grid binding of :func:`_runtime.run_verify`: ``grid_cell_to_config``, the
    ``V=1024 NMB`` jitter predicate, ``det-`` scratch prefix.

    Raises:
        RuntimeError: a BPE cell's artifacts are not byte-identical across
            retraining.
    """
    return _runtime.run_verify(
        cell,
        to_config=grid_cell_to_config,
        is_expected_jitter=is_expected_unigram_jitter,
        log=_log,
        prefix=f"det-{cell.cell_id}-",
        force=force,
        dry_run=dry_run,
    )


def verify_extras_cell(
    cell: ExtrasCell, *, force: bool = False, dry_run: bool = False
) -> DeterminismResult | None:
    """Verify per-arm determinism for one extras cell; return the result or None.

    Extras binding of :func:`_runtime.run_verify`: ``extras_cell_to_config``, the
    always-False jitter predicate, ``det-extras-`` prefix.

    Raises:
        RuntimeError: a BPE cell's artifacts are not byte-identical across
            retraining.
    """
    return _runtime.run_verify(
        cell,
        to_config=extras_cell_to_config,
        is_expected_jitter=_extras_expected_jitter,
        log=_extras_log,
        prefix=f"det-extras-{cell.cell_id}-",
        force=force,
        dry_run=dry_run,
    )


__all__ = ["is_expected_unigram_jitter", "verify_cell", "verify_extras_cell"]
