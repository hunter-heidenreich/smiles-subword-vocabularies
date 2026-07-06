"""Shared per-cell training dispatch for the grid and robustness-extras sweeps.

Resolving a cell against its manifest, materializing its ``TokenizerConfig`` to
a gitignored cache YAML, shelling out to ``scripts/tokenize/train_tokenizer.py``,
skip-if-already-trained resume, and the three post-train audit hooks (F95
confirmation, train-twice determinism, scaffold-log retrain) are identical for
both sweeps — only the cell type, cell→config seam, audit-hook bindings, and
cache directory differ. That orchestration lives here once (mirroring
:mod:`smiles_subword.tokenize.audit._runtime`); each ``scripts/tokenize/dispatch_*``
driver is a thin binding supplying a :class:`DispatchSeams` bundle, the
argparser, and a ``log`` callable.

IO calls (:func:`subprocess.run`, the artifact loaders, ``record_results``) are
module-level so the test suite patches them on ``dispatch`` directly. Progress
goes through the injected ``log`` (drivers pass ``print``); this module emits
nothing itself.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Generic, Protocol, TypeVar

import yaml

from smiles_subword.config import TokenizerConfig
from smiles_subword.paths import TRAIN_TOKENIZER_SCRIPT
from smiles_subword.tokenize import (
    SmirkAdapter,
    TokenizerMeta,
    UnigramSmirkAdapter,
    training_corpus_sha,
)
from smiles_subword.tokenize.audit.scaffold_retrain import retrain_with_scaffold_log
from smiles_subword.tokenize.audit.scaffold_retrain_io import record_results

if TYPE_CHECKING:
    from collections.abc import Sequence

    from smiles_subword.tokenize.audit.scaffold_retrain import ScaffoldRetrainResult


class DispatchCell(Protocol):
    """Structural cell type for dispatch: anything carrying a stable ``cell_id``."""

    @property
    def cell_id(self) -> str: ...


_Cell = TypeVar("_Cell", bound=DispatchCell)


@dataclass(frozen=True)
class DispatchSeams(Generic[_Cell]):
    """The grid-vs-extras seams a driver binds.

    ``to_config`` maps a cell to its :class:`TokenizerConfig`; ``confirm_hook`` /
    ``verify_hook`` are the sweep's audit entrypoints (called as
    ``hook(cell, force=..., dry_run=...)``); ``cache_dir`` is the gitignored
    directory the materialized per-cell config YAMLs are written to.
    """

    to_config: Callable[[_Cell], TokenizerConfig]
    confirm_hook: Callable[..., object]
    verify_hook: Callable[..., object]
    cache_dir: Path


def resolve_cell(cell_id: str, cells: Sequence[_Cell]) -> _Cell:
    """Return ``cells``' member with ``cell_id``, or raise ``FileNotFoundError``."""
    by_id = {cell.cell_id: cell for cell in cells}
    cell = by_id.get(cell_id)
    if cell is None:
        valid = ", ".join(sorted(by_id)) or "(empty manifest)"
        raise FileNotFoundError(f"unknown cell {cell_id!r}; valid cells: {valid}")
    return cell


def artifact_reloads(artifact_dir: Path, *, kind: str) -> bool:
    """True if the saved tokenizer reloads cleanly.

    A bare existence check misses a truncated ``tokenizer.json`` from a run that
    crashed mid-``save``: any reload failure means the cell must be re-trained.
    """
    loader = SmirkAdapter.load if kind == "smirk_gpe" else UnigramSmirkAdapter.load
    try:
        loader(artifact_dir)
    except Exception:  # noqa: BLE001 - any failure mode means "re-train this cell"
        return False
    return True


def is_cell_done(cell: _Cell, *, to_config: Callable[[_Cell], TokenizerConfig]) -> bool:
    """True if ``cell`` is already trained against the current corpus.

    Done means all hold: the artifact-contract files exist; the tokenizer
    reloads (see :func:`artifact_reloads`); the training corpus is present on
    disk; and the recorded ``training_corpus_sha`` matches it — a corpus
    reprocessed since training invalidates the stale tokenizer.
    """
    cfg = to_config(cell)
    out = cfg.output_dir
    required = [out / "tokenizer.json", out / "meta.yaml"]
    if cfg.kind == "smirk_gpe":
        required.append(out / "merges.txt")
    if not all(p.is_file() for p in required):
        return False
    if not artifact_reloads(out, kind=cfg.kind):
        return False
    meta = TokenizerMeta.model_validate(yaml.safe_load((out / "meta.yaml").read_text()))
    assert cfg.training_input is not None
    if not cfg.training_input.is_dir():
        return False
    return meta.training_corpus_sha == training_corpus_sha(cfg.training_input)


def materialize_config(
    cell: _Cell, *, to_config: Callable[[_Cell], TokenizerConfig], cache_dir: Path
) -> Path:
    """Write ``cell``'s ``TokenizerConfig`` to ``cache_dir`` (always regenerated).

    ``mode="json"`` stringifies the resolved ``RepoPath`` fields so
    ``yaml.safe_dump`` accepts them.
    """
    cfg = to_config(cell)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / f"{cell.cell_id}.yaml"
    out_path.write_text(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False))
    return out_path


def build_command(config_path: Path) -> list[str]:
    """Return the argv that trains one cell via ``train_tokenizer.py``."""
    return [
        "uv",
        "run",
        "python",
        str(TRAIN_TOKENIZER_SCRIPT),
        "--config",
        str(config_path),
    ]


def dispatch_one(
    cell: _Cell,
    seams: DispatchSeams[_Cell],
    *,
    log: Callable[[str], None],
    dry_run: bool,
    force: bool,
    confirm_f95: bool,
    verify_determinism: bool,
    retrain_scaffold: bool,
    scaffold_results: list[ScaffoldRetrainResult] | None = None,
) -> None:
    """Train one cell, skipping it when already done (unless ``force``).

    The post-train hooks run after training — including for a cell skipped as
    already-trained, so a resumed sweep backfills trained-but-unhooked cells.
    With ``confirm_f95``, the ``F_{p,n}`` confirmation runs; with
    ``verify_determinism``, the train-twice determinism assertion runs; with
    ``retrain_scaffold``, the scaffold-instrumentation supplementary retrain
    runs (BPE arms only; byte-identity asserted against the canonical artifact).
    """
    if not force and is_cell_done(cell, to_config=seams.to_config):
        log(f"skip {cell.cell_id} (already trained)")
    else:
        config_path = materialize_config(
            cell, to_config=seams.to_config, cache_dir=seams.cache_dir
        )
        cmd = build_command(config_path)
        log(f"dispatch {cell.cell_id}")
        if dry_run:
            log(f"  config: {config_path}")
            log(f"  command: {' '.join(cmd)}")
        else:
            subprocess.run(cmd, check=True)

    if confirm_f95:
        seams.confirm_hook(cell, force=force, dry_run=dry_run)
    if verify_determinism:
        seams.verify_hook(cell, force=force, dry_run=dry_run)
    if retrain_scaffold:
        cfg = seams.to_config(cell)
        result = retrain_with_scaffold_log(
            cell_id=cell.cell_id,
            canonical_dir=cfg.output_dir,
            base_config=cfg,
            force=force,
            dry_run=dry_run,
        )
        if scaffold_results is not None:
            scaffold_results.append(result)


def run_dispatch(
    cells: Sequence[_Cell],
    seams: DispatchSeams[_Cell],
    *,
    log: Callable[[str], None],
    dry_run: bool,
    force: bool,
    confirm_f95: bool,
    verify_determinism: bool,
    retrain_scaffold: bool,
) -> None:
    """Dispatch every cell in ``cells``, then deposit the scaffold-retrain audit.

    The scaffold audit is written once over all cells when ``retrain_scaffold``
    is set and at least one result was produced.
    """
    scaffold_results: list[ScaffoldRetrainResult] | None = (
        [] if retrain_scaffold else None
    )
    for cell in cells:
        dispatch_one(
            cell,
            seams,
            log=log,
            dry_run=dry_run,
            force=force,
            confirm_f95=confirm_f95,
            verify_determinism=verify_determinism,
            retrain_scaffold=retrain_scaffold,
            scaffold_results=scaffold_results,
        )
    if scaffold_results:
        json_path, md_path = record_results(scaffold_results)
        log(f"[scaffold-retrain] audit → {json_path}")
        log(f"[scaffold-retrain] audit → {md_path}")


__all__ = [
    "DispatchCell",
    "DispatchSeams",
    "artifact_reloads",
    "build_command",
    "dispatch_one",
    "is_cell_done",
    "materialize_config",
    "resolve_cell",
    "run_dispatch",
]
