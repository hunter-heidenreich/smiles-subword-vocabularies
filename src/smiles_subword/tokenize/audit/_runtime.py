"""Shared orchestration helpers for the per-cell audit pilots.

The grid and robustness-extras confirm/verify drivers share the same cell
plumbing: a tagged stderr progress logger, loading a trained artifact (or None
when the cell is not trained yet), and — for determinism — retraining into a
scratch directory through the exact dispatch path and comparing against the
canonical. The confirm/verify orchestration (:func:`run_confirm` /
:func:`run_verify`) lives here too; the IO seams it calls (:func:`load_trained`,
:func:`train_into`, ``is_*_done``, ``write_*_json``) are module-level functions
so the test suite patches them on ``_runtime`` directly. Each grid/extras driver
is a thin binding supplying the cell→config resolver and the log tag.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

import yaml

from smiles_subword.paths import TRAIN_TOKENIZER_SCRIPT
from smiles_subword.tokenize import (
    SmirkAdapter,
    UnigramSmirkAdapter,
    iter_smiles_from_parquet,
    training_corpus_sha,
)
from smiles_subword.tokenize.audit.determinism import (
    DeterminismResult,
    compare_artifacts,
    digest_artifact,
    unigram_glyph_set,
)
from smiles_subword.tokenize.audit.determinism_io import (
    is_determinism_done,
    write_determinism_json,
)
from smiles_subword.tokenize.audit.f95 import compute_f95
from smiles_subword.tokenize.audit.f95_io import is_f95_done, write_f95_json

if TYPE_CHECKING:
    from collections.abc import Callable

    from smiles_subword.config import TokenizerConfig
    from smiles_subword.tokenize.audit._celldeposit import AuditableCell
    from smiles_subword.tokenize.audit.f95 import F95Result

# Binds a confirm/verify call's cell to its seam callables — a GridCell driver
# passes GridCell-typed seams, an ExtrasCell driver ExtrasCell-typed ones.
_Cell = TypeVar("_Cell", bound="AuditableCell")


def make_logger(tag: str) -> Callable[[str], None]:
    """Return a stderr logger that prefixes each line with ``[tag]``."""

    def _log(message: str) -> None:
        sys.stderr.write(f"[{tag}] {message}\n")
        sys.stderr.flush()

    return _log


def load_trained(
    cell: AuditableCell, output_dir: Path
) -> SmirkAdapter | UnigramSmirkAdapter | None:
    """Load ``cell``'s trained tokenizer, or None if it is not trained yet.

    Any load failure — missing directory, truncated ``tokenizer.json`` — means
    the cell is not ready, not an error to raise.
    """
    loader = SmirkAdapter.load if cell.kind == "smirk_gpe" else UnigramSmirkAdapter.load
    try:
        return loader(output_dir)
    except Exception:  # noqa: BLE001 - any failure means "not trained / unusable"
        return None


def train_into(
    cfg: TokenizerConfig,
    scratch: Path,
    *,
    extra_update: dict[str, object] | None = None,
) -> None:
    """Retrain ``cfg``'s cell into ``scratch`` via the exact dispatch path.

    Shells out to ``scripts/tokenize/train_tokenizer.py`` so the rerun goes
    through the identical build path that produced the canonical artifact — the
    comparison is apples-to-apples. ``extra_update`` overlays extra config fields
    onto the rerun (the scaffold audit sets ``scaffold_log=True``).
    """
    update: dict[str, object] = {"output_dir": scratch, **(extra_update or {})}
    rerun_cfg = cfg.model_copy(update=update)
    config_path = scratch / "rerun_config.yaml"
    config_path.write_text(
        yaml.safe_dump(rerun_cfg.model_dump(mode="json"), sort_keys=False)
    )
    subprocess.run(
        [
            "uv",
            "run",
            "python",
            str(TRAIN_TOKENIZER_SCRIPT),
            "--config",
            str(config_path),
        ],
        check=True,
    )


def retrain_and_compare(
    cell: AuditableCell, cfg: TokenizerConfig, *, prefix: str
) -> DeterminismResult:
    """Retrain ``cell`` into a scratch dir and compare it against the canonical.

    ``prefix`` names the scratch ``mkdtemp`` directory. The scratch tree is
    always removed; only the rerun's digest is kept.
    """
    scratch = Path(tempfile.mkdtemp(prefix=prefix))
    try:
        train_into(cfg, scratch)
        canonical = digest_artifact(cfg.output_dir, algo=cell.algo)
        rerun = digest_artifact(scratch, algo=cell.algo)
        pieces = (
            (unigram_glyph_set(cfg.output_dir), unigram_glyph_set(scratch))
            if cell.algo == "unigram"
            else None
        )
        return compare_artifacts(canonical, rerun, pieces=pieces)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def run_confirm(
    cell: _Cell,
    *,
    to_config: Callable[[_Cell], TokenizerConfig],
    resolve_atomic_tokens: Callable[
        [_Cell, SmirkAdapter | UnigramSmirkAdapter], frozenset[str] | None
    ],
    log: Callable[[str], None],
    force: bool = False,
    dry_run: bool = False,
) -> F95Result | None:
    """Confirm ``F_{p,n}`` for one cell — the grid/extras-agnostic orchestration.

    Load the trained cell, compute ``F_{p,n}`` over its full training corpus,
    deposit the per-cell JSON. Returns None when the work is skipped: a dry run,
    a cell not trained yet, a cell already confirmed against the current corpus
    (unless ``force``), or a Unigram cell whose glyph alphabet is not yet
    resolvable. The grid and extras drivers differ only in ``to_config``,
    ``resolve_atomic_tokens`` (the glyph-alphabet lookup), and the log tag.
    """
    cfg = to_config(cell)
    assert cfg.training_input is not None

    if dry_run:
        log(f"confirm {cell.cell_id} (dry-run): F95 over {cfg.training_input}")
        return None

    tok = load_trained(cell, cfg.output_dir)
    if tok is None:
        log(f"skip {cell.cell_id} (not trained yet)")
        return None

    sha = training_corpus_sha(cfg.training_input)
    if not force and is_f95_done(cell, training_corpus_sha=sha):
        log(f"skip {cell.cell_id} (f95 already confirmed)")
        return None

    atomic_tokens = resolve_atomic_tokens(cell, tok)
    if atomic_tokens is None:
        log(f"skip {cell.cell_id} (matched BPE cell not trained — glyph alphabet)")
        return None

    result = compute_f95(
        tok,
        iter_smiles_from_parquet(cfg.training_input),
        arm=cell.algo,
        atomic_tokens=atomic_tokens,
    )
    write_f95_json(cell, result, training_corpus_sha=sha)
    flag = " EMBEDDING-TAIL-UNSAFE" if result.embedding_tail_unsafe else ""
    log(f"confirmed {cell.cell_id}: F95,100={result.headline_clearance:.4f}{flag}")
    return result


def run_verify(
    cell: _Cell,
    *,
    to_config: Callable[[_Cell], TokenizerConfig],
    is_expected_jitter: Callable[[_Cell], bool],
    log: Callable[[str], None],
    prefix: str,
    force: bool = False,
    dry_run: bool = False,
) -> DeterminismResult | None:
    """Verify per-arm determinism for one cell — grid/extras-agnostic.

    Retrain the cell into scratch, compare against the canonical, deposit the
    per-cell JSON, and log the outcome. Returns None when skipped (dry run, not
    trained, or already verified unless ``force``). The grid and extras drivers
    differ only in ``to_config``, ``is_expected_jitter`` (the expected-jitter
    set), the scratch-dir ``prefix``, and the log tag.

    Raises:
        RuntimeError: a BPE cell's artifacts are not byte-identical across
            retraining (a halt-and-investigate bug; the JSON is deposited first
            as evidence).
    """
    cfg = to_config(cell)
    assert cfg.training_input is not None

    if dry_run:
        log(f"verify {cell.cell_id} (dry-run): retrain + compare {cfg.output_dir}")
        return None

    if load_trained(cell, cfg.output_dir) is None:
        log(f"skip {cell.cell_id} (not trained yet)")
        return None

    sha = training_corpus_sha(cfg.training_input)
    if not force and is_determinism_done(cell, training_corpus_sha=sha):
        log(f"skip {cell.cell_id} (determinism already verified)")
        return None

    result = retrain_and_compare(cell, cfg, prefix=prefix)
    expected = is_expected_jitter(cell)
    write_determinism_json(
        cell, result, training_corpus_sha=sha, expected_failure=expected
    )
    _report_determinism(cell, result, expected=expected, log=log)
    return result


def _report_determinism(
    cell: AuditableCell,
    result: DeterminismResult,
    *,
    expected: bool,
    log: Callable[[str], None],
) -> None:
    """Log a determinism outcome; raise on a BPE mismatch (a halt bug)."""
    if result.deterministic:
        log(f"verified {cell.cell_id}: {result.arm} determinism holds")
        return
    if result.mismatch_kind == "bpe_byte":
        raise RuntimeError(
            f"{cell.cell_id}: BPE artifacts are NOT byte-identical across "
            "retraining. BPE is deterministic by construction — this is a bug, "
            "not a science deviation. Halt and investigate."
        )
    spread = result.rerun_spread
    if expected:
        log(
            f"FLAGGED {cell.cell_id}: Unigram piece set jitters (rerun spread "
            f"{spread}) — a known, expected Unigram jitter case."
        )
    else:
        log(
            f"UNEXPECTED {cell.cell_id}: Unigram piece set jitters (rerun spread "
            f"{spread}) — halt and investigate."
        )


__all__ = [
    "load_trained",
    "make_logger",
    "retrain_and_compare",
    "run_confirm",
    "run_verify",
    "train_into",
]
