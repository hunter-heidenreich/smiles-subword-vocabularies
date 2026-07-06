"""Per-cell scaffold-log read + arm dispatch.

Given a trained cell artifact directory, loads the sidecar
``scaffold.jsonl`` for BPE arms and the cell's ``meta.yaml`` for the
identifying fields, then dispatches to :mod:`scaffold` for the pure
Lian-2024 computation. Unigram arms have no log by construction and
short-circuit to a zero record.

Kept distinct from :mod:`scaffold` so the heavy I/O is testable in
isolation from the pure math, mirroring the Deadzone/Absorption split.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import yaml

from smiles_subword._hashing import sha256_file
from smiles_subword.paths import tokenizer_artifact_dir
from smiles_subword.tokenize.measure.scaffold.math import (
    Arm,
    ArmScaffold,
    Boundary,
    ScaffoldLogHeader,
    ScaffoldRecord,
    compute_bpe_arm_scaffold,
    compute_unigram_arm_scaffold,
    parse_scaffold_log,
)

if TYPE_CHECKING:
    from pathlib import Path


class ScaffoldLogMissingError(FileNotFoundError):
    """A BPE cell has no sidecar ``scaffold.jsonl``.

    Indicates the cell has not been retrained-with-scaffold-logging
    (GPE per-merge-step instrumentation), so Scaffold cannot be computed. The
    dispatcher's ``--retrain-scaffold`` hook is the remedy.
    """


def scaffold_log_path(corpus: str, name: str) -> Path:
    """Sidecar log path for a trained cell ``(corpus, name)``."""
    return tokenizer_artifact_dir(corpus, name) / "scaffold.jsonl"


def read_scaffold_log_sha(corpus: str, name: str) -> str | None:
    """Return SHA256 of the cell's sidecar log, or None if absent."""
    path = scaffold_log_path(corpus, name)
    if not path.is_file():
        return None
    return sha256_file(path)


def read_scaffold_log(
    corpus: str, name: str
) -> tuple[ScaffoldLogHeader, list[ScaffoldRecord]]:
    """Parse the cell's sidecar log into header + records.

    Raises:
        ScaffoldLogMissingError: the cell's artifact dir has no
            ``scaffold.jsonl`` sidecar.
    """
    path = scaffold_log_path(corpus, name)
    if not path.is_file():
        raise ScaffoldLogMissingError(
            f"no scaffold.jsonl at {path}; retrain with --retrain-scaffold "
            "to populate the per-merge-step instrumentation log"
        )
    with path.open("r", encoding="utf-8") as fh:
        return parse_scaffold_log(fh)


def _cell_meta(corpus: str, name: str) -> dict[str, object]:
    meta_path = tokenizer_artifact_dir(corpus, name) / "meta.yaml"
    if not meta_path.is_file():
        raise FileNotFoundError(f"no meta.yaml at {meta_path}")
    return cast("dict[str, object]", yaml.safe_load(meta_path.read_text()))


def run_arm_scaffold(
    *,
    cell_id: str,
    corpus: str,
    name: str,
    arm: Arm,
    boundary: Boundary,
) -> ArmScaffold:
    """Compute Scaffold for one arm of one cell from on-disk artifacts.

    BPE arms parse the sidecar ``scaffold.jsonl`` and apply
    :func:`compute_bpe_arm_scaffold`. Unigram arms short-circuit to
    :func:`compute_unigram_arm_scaffold` — no log to read.

    Reads ``training_corpus_sha`` + ``n_merges`` + ``vocab_size`` from
    the cell's ``meta.yaml``. ``scaffold_log_sha`` is computed from the
    log bytes (BPE) or ``None`` (Unigram).
    """
    meta = _cell_meta(corpus, name)
    training_corpus_sha = meta.get("training_corpus_sha")
    if not isinstance(training_corpus_sha, str):
        raise TypeError(f"{cell_id}: meta.yaml missing training_corpus_sha")
    vocab_size = int(cast("int", meta["vocab_size"]))
    if arm == "unigram":
        return compute_unigram_arm_scaffold(
            cell_id=cell_id,
            boundary=boundary,
            vocab_size=vocab_size,
            training_corpus_sha=training_corpus_sha,
        )
    n_merges_raw = meta.get("n_merges")
    if not isinstance(n_merges_raw, int):
        raise TypeError(f"{cell_id}: meta.yaml missing n_merges for BPE arm")
    log_sha = read_scaffold_log_sha(corpus, name)
    if log_sha is None:
        raise ScaffoldLogMissingError(
            f"{cell_id}: no scaffold.jsonl; retrain with --retrain-scaffold"
        )
    header, records = read_scaffold_log(corpus, name)
    return compute_bpe_arm_scaffold(
        records,
        cell_id=cell_id,
        boundary=boundary,
        vocab_size=vocab_size,
        n_merges=int(n_merges_raw),
        atomic_vocab_size=len(header.base_alphabet),
        training_corpus_sha=training_corpus_sha,
        scaffold_log_sha=log_sha,
    )


__all__ = [
    "ScaffoldLogMissingError",
    "read_scaffold_log",
    "read_scaffold_log_sha",
    "run_arm_scaffold",
    "scaffold_log_path",
]
