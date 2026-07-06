"""Resolve a trained cell's ``meta.yaml`` into deposit fields + freshness.

Every measurement's ``*_io`` module reads back each cell's ``meta.yaml`` to
build its per-cell/per-pair record and to check freshness (has the recorded
``training_corpus_sha`` drifted?). That meta-read + resolution + freshness core
lives here once rather than copy-pasted per measurement.

This is a leaf module (depends only on ``paths`` + stdlib/yaml), so any ``*_io``
module can import it without a cycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import yaml

from smiles_subword.paths import tokenizer_artifact_dir


def cell_meta(corpus: str, name: str) -> dict[str, object] | None:
    """Return a trained cell's ``meta.yaml`` payload, or None if absent.

    Used by the freshness checks to re-read each cell's recorded
    ``training_corpus_sha`` and compare it against the deposit.
    """
    meta_path = tokenizer_artifact_dir(corpus, name) / "meta.yaml"
    if not meta_path.is_file():
        return None
    return cast("dict[str, object]", yaml.safe_load(meta_path.read_text()))


@dataclass(frozen=True)
class CellMetaFields:
    """A trained cell's deposit-relevant meta, resolved from its ``cell_id``."""

    corpus: str
    name: str
    artifact_dir: Path
    boundary: Literal["nmb", "mb"]
    training_corpus_sha: str


def resolve_cell_meta(cell_id: str) -> CellMetaFields | str:
    """Resolve ``cell_id`` to its deposit-relevant meta fields.

    Returns the resolved :class:`CellMetaFields`, or a human-readable error
    reason (for the deposit-pend path) when ``cell_id`` is malformed, the
    ``meta.yaml`` is absent, or it lacks ``merge_brackets`` /
    ``training_corpus_sha``. Used by the held-out-evaluated measurements, whose
    per-arm record build needs the boundary and artifact dir.
    """
    corpus, _, name = cell_id.partition("__")
    if not name:
        return f"malformed cell_id {cell_id!r}"
    artifact_dir = tokenizer_artifact_dir(corpus, name)
    meta = cell_meta(corpus, name)
    if meta is None:
        return f"no meta.yaml for {cell_id}"
    boundary_raw = meta.get("merge_brackets")
    if boundary_raw is None:
        return f"meta.yaml for {cell_id} missing merge_brackets"
    boundary: Literal["nmb", "mb"] = "mb" if bool(boundary_raw) else "nmb"
    training_corpus_sha = meta.get("training_corpus_sha")
    if not isinstance(training_corpus_sha, str):
        return f"meta.yaml for {cell_id} missing training_corpus_sha"
    return CellMetaFields(corpus, name, artifact_dir, boundary, training_corpus_sha)


@dataclass(frozen=True)
class ArmInfo:
    """The cell facts the vocabulary-only / boundary-independent deposit needs."""

    name: str
    training_corpus_sha: str


def arm_info(cell_id: str) -> ArmInfo | str:
    """Resolve ``cell_id`` to its ``(name, training_corpus_sha)``, or an error reason.

    The lighter sibling of :func:`resolve_cell_meta` for the deposit builders
    that need only the cell name and training SHA — no boundary, no artifact dir.
    """
    corpus, _, name = cell_id.partition("__")
    if not name:
        return f"malformed cell_id {cell_id!r}"
    meta = cell_meta(corpus, name)
    if meta is None:
        return f"no meta.yaml for {cell_id}"
    sha = meta.get("training_corpus_sha")
    if not isinstance(sha, str):
        return f"meta.yaml for {cell_id} missing training_corpus_sha"
    return ArmInfo(name=name, training_corpus_sha=sha)


def cell_training_sha_fresh(
    cell_id: object, deposited_corpus_sha: object
) -> tuple[str, str] | None:
    """Resolve ``cell_id`` to ``(corpus, name)`` iff its training SHA is fresh.

    The shared core of every measurement's per-arm freshness check: ``cell_id``
    must name a real cell whose ``meta.yaml`` ``training_corpus_sha`` still
    equals ``deposited_corpus_sha``. Returns the cell's ``(corpus, name)`` so
    callers can layer their own measurement-specific freshness inputs (eval-split
    sha, scaffold-log sha, ...), or None when the SHA has drifted or either
    argument is malformed.
    """
    if not isinstance(cell_id, str) or not isinstance(deposited_corpus_sha, str):
        return None
    corpus, _, name = cell_id.partition("__")
    if not name:
        return None
    meta = cell_meta(corpus, name)
    if meta is None or meta.get("training_corpus_sha") != deposited_corpus_sha:
        return None
    return corpus, name


__all__ = [
    "ArmInfo",
    "CellMetaFields",
    "arm_info",
    "cell_meta",
    "cell_training_sha_fresh",
    "resolve_cell_meta",
]
