"""Per-cell F_{p,n} confirmation orchestration (grid + robustness extras).

The glyph alphabet defining the non-atomic vocabulary (see :mod:`.f95`) resolves
from a Smirk-GPE artifact: for a BPE cell, the cell itself; for a Unigram cell, a
matched same-corpus, same-boundary BPE cell (the alphabet is identical across
arms by construction). The grid resolver searches the headline grid; the extras
resolver searches the grid first (always trained), then the extras manifest.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from smiles_subword.tokenize import SmirkAdapter, UnigramSmirkAdapter
from smiles_subword.tokenize.audit import _runtime
from smiles_subword.tokenize.extras import extras_cell_to_config, load_extras_manifest
from smiles_subword.tokenize.grid import grid_cell_to_config, load_grid_manifest

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from smiles_subword.tokenize.audit.f95 import F95Result
    from smiles_subword.tokenize.extras import ExtrasCell
    from smiles_subword.tokenize.grid import GridCell

_log = _runtime.make_logger("f95")
_extras_log = _runtime.make_logger("f95-extras")


def _first_loadable_bpe_vocab(output_dirs: Iterable[Path]) -> frozenset[str] | None:
    """Glyph alphabet of the first Smirk-GPE artifact in ``output_dirs`` to load.

    The alphabet is identical across same-(corpus, boundary) BPE cells, so any
    one that loads serves; ``None`` means none is trained yet.
    """
    for output_dir in output_dirs:
        try:
            bpe_tok = SmirkAdapter.load(output_dir)
        except Exception:  # noqa: BLE001 - try the next candidate
            continue
        return frozenset(bpe_tok.hf_tokenizer.get_vocab())
    return None


# --------------------------------------------------------------------------- #
# Grid                                                                        #
# --------------------------------------------------------------------------- #


def _matched_bpe_cells(cell: GridCell) -> list[GridCell]:
    """Same-corpus, same-boundary grid BPE cells, smallest vocab first."""
    candidates = [
        c
        for c in load_grid_manifest()
        if c.algo == "bpe" and c.corpus == cell.corpus and c.boundary == cell.boundary
    ]
    return sorted(candidates, key=lambda c: c.vocab_size)


def _grid_resolve_atomic_tokens(
    cell: GridCell, tok: SmirkAdapter | UnigramSmirkAdapter
) -> frozenset[str] | None:
    """Corpus glyph alphabet for a grid cell, or None if unresolvable.

    BPE cell: its own ``get_vocab()`` (Smirk-GPE surfaces only the atomic base).
    Unigram cell: a matched BPE artifact, or None if none is trained yet.
    """
    if cell.algo == "bpe":
        assert isinstance(tok, SmirkAdapter)
        return frozenset(tok.hf_tokenizer.get_vocab())
    return _first_loadable_bpe_vocab(
        grid_cell_to_config(c).output_dir for c in _matched_bpe_cells(cell)
    )


def confirm_cell(
    cell: GridCell, *, force: bool = False, dry_run: bool = False
) -> F95Result | None:
    """Confirm ``F_{p,n}`` for one grid cell; grid binding of
    :func:`_runtime.run_confirm`."""
    return _runtime.run_confirm(
        cell,
        to_config=grid_cell_to_config,
        resolve_atomic_tokens=_grid_resolve_atomic_tokens,
        log=_log,
        force=force,
        dry_run=dry_run,
    )


# --------------------------------------------------------------------------- #
# Robustness extras                                                           #
# --------------------------------------------------------------------------- #


def _grid_bpe_alphabet(corpus: str, boundary: str) -> frozenset[str] | None:
    """Glyph alphabet from a same-(corpus, boundary) grid BPE cell.

    Smallest vocab first — the alphabet is identical, but the smaller artifact
    loads faster.
    """
    candidates = [
        c
        for c in load_grid_manifest()
        if c.algo == "bpe" and c.corpus == corpus and c.boundary == boundary
    ]
    return _first_loadable_bpe_vocab(
        grid_cell_to_config(c).output_dir
        for c in sorted(candidates, key=lambda c: c.vocab_size)
    )


def _extras_bpe_alphabet(corpus: str, boundary: str) -> frozenset[str] | None:
    """Fallback to an extras BPE cell on the same corpus + boundary."""
    return _first_loadable_bpe_vocab(
        extras_cell_to_config(c).output_dir
        for c in load_extras_manifest()
        if c.algo == "bpe" and c.corpus == corpus and c.boundary == boundary
    )


def _extras_resolve_atomic_tokens(
    cell: ExtrasCell, tok: SmirkAdapter | UnigramSmirkAdapter
) -> frozenset[str] | None:
    """Corpus glyph alphabet for an extras cell, or None.

    BPE cell: its own ``get_vocab()``. Unigram cell: a matched BPE artifact,
    grid first (always trained) then the extras manifest.
    """
    if cell.algo == "bpe":
        assert isinstance(tok, SmirkAdapter)
        return frozenset(tok.hf_tokenizer.get_vocab())
    grid_alphabet = _grid_bpe_alphabet(cell.corpus, cell.boundary)
    if grid_alphabet is not None:
        return grid_alphabet
    return _extras_bpe_alphabet(cell.corpus, cell.boundary)


def confirm_extras_cell(
    cell: ExtrasCell, *, force: bool = False, dry_run: bool = False
) -> F95Result | None:
    """Confirm ``F_{p,n}`` for one extras cell; extras binding of
    :func:`_runtime.run_confirm`."""
    return _runtime.run_confirm(
        cell,
        to_config=extras_cell_to_config,
        resolve_atomic_tokens=_extras_resolve_atomic_tokens,
        log=_extras_log,
        force=force,
        dry_run=dry_run,
    )


__all__ = ["confirm_cell", "confirm_extras_cell"]
