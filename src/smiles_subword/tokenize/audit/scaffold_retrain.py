"""Scaffold-log supplementary retrain (scaffold byte-identity audit).

For each canonical BPE cell on disk, retrain its tokenizer in a scratch
directory with ``scaffold_log=True``, then assert the scratch
``tokenizer.json`` + ``merges.txt`` are byte-identical to the canonical
artifact. On success the sidecar ``scaffold.jsonl`` is copied into the
canonical cell directory and its ``meta.yaml`` is patched in place with the
resulting ``scaffold_log_sha``.

The byte-identity assertion operationally verifies the scaffold
instrumentation's "logging-only does not alter merge selection" claim; any
mismatch is a contract violation and halts the sweep with :class:`RuntimeError`.

Symmetric across grid + extras: both dispatchers call this with the cell's
``TokenizerConfig`` so the retrain goes through the same
``scripts/tokenize/train_tokenizer.py`` driver that produced the canonical
artifact (matching :mod:`determinism_verify`).
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import yaml

from smiles_subword._hashing import sha256_file
from smiles_subword.tokenize._corpus import write_meta_yaml
from smiles_subword.tokenize.audit import _runtime

if TYPE_CHECKING:
    from smiles_subword.config import TokenizerConfig

ScaffoldRetrainStatus = Literal["ok", "already_done", "skipped", "failed"]

_log = _runtime.make_logger("scaffold-retrain")


@dataclass(frozen=True)
class ScaffoldRetrainResult:
    """Outcome of one supplementary scaffold-log retrain.

    - ``ok`` — retrain succeeded and the canonical cell now has a
      ``scaffold.jsonl`` whose SHA256 matches ``scaffold_log_sha``.
    - ``already_done`` — the canonical cell already carries a
      ``scaffold.jsonl`` whose SHA matches the meta-recorded value;
      nothing was done.
    - ``skipped`` — the cell's kind is not ``smirk_gpe`` (e.g. Unigram
      arm); the scaffold measurement has no scaffold log for these by construction.
    - ``failed`` — see ``reason``. A ``byte_mismatch`` reason indicates
      a scaffold-instrumentation contract violation and the caller raises.
    """

    cell_id: str
    status: ScaffoldRetrainStatus
    reason: str | None
    scaffold_log_sha: str | None


def _existing_log_sha_if_fresh(canonical_dir: Path) -> str | None:
    """Return canonical/scaffold.jsonl's SHA iff present and meta records a match.

    Returns the recorded SHA (reused by the caller to avoid re-hashing a
    possibly multi-MB log) when the log is fresh, else ``None``.
    """
    log_path = canonical_dir / "scaffold.jsonl"
    meta_path = canonical_dir / "meta.yaml"
    if not log_path.is_file() or not meta_path.is_file():
        return None
    meta = yaml.safe_load(meta_path.read_text())
    recorded = meta.get("scaffold_log_sha")
    if not isinstance(recorded, str):
        return None
    return recorded if recorded == sha256_file(log_path) else None


def _byte_compare(scratch: Path, canonical: Path) -> str | None:
    """Compare BPE artifact bytes; return mismatch reason or None on match."""
    for name in ("tokenizer.json", "merges.txt"):
        scratch_bytes = (scratch / name).read_bytes()
        canonical_bytes = (canonical / name).read_bytes()
        if scratch_bytes != canonical_bytes:
            return f"{name}_mismatch"
    return None


def _patch_meta_with_scaffold_sha(meta_path: Path, *, sha: str) -> None:
    """Edit ``meta.yaml`` in place to set ``scaffold_log_sha``.

    Patches only the meta file (via the atomic :func:`write_meta_yaml`, the same
    writer every tokenizer save path uses), leaving the rest of the canonical
    artifact untouched — calling ``SmirkAdapter.save`` would re-serialize
    ``tokenizer.json`` too and break the byte-identity the audit just asserted.
    """
    payload = yaml.safe_load(meta_path.read_text())
    payload["scaffold_log_sha"] = sha
    write_meta_yaml(meta_path, payload)


def _train_into(cfg: TokenizerConfig, scratch: Path) -> Path:
    """Run ``train_tokenizer.py`` with ``scaffold_log=True`` into a scratch dir.

    Reuses :func:`_runtime.train_into` (the exact dispatch path the determinism
    audit retrains through) with ``scaffold_log`` overlaid. Returns the scratch
    ``scaffold.jsonl`` path; raises if the fork did not emit the log.
    """
    _runtime.train_into(cfg, scratch, extra_update={"scaffold_log": True})
    log = scratch / "scaffold.jsonl"
    if not log.is_file():
        raise RuntimeError(
            f"scaffold.jsonl absent under {scratch} after training with "
            "scaffold_log=True — the smirk fork did not emit the log"
        )
    return log


def retrain_with_scaffold_log(
    *,
    cell_id: str,
    canonical_dir: Path,
    base_config: TokenizerConfig,
    force: bool = False,
    dry_run: bool = False,
) -> ScaffoldRetrainResult:
    """Supplementary retrain into scratch + byte-identity assert + log copy.

    Returns a :class:`ScaffoldRetrainResult` summarizing the outcome.
    Raises :class:`RuntimeError` on a ``byte_mismatch`` failure — a
    scaffold-instrumentation contract violation that halts the sweep.
    """
    if base_config.kind != "smirk_gpe":
        return ScaffoldRetrainResult(
            cell_id=cell_id, status="skipped", reason="not-bpe", scaffold_log_sha=None
        )

    canonical_tok = canonical_dir / "tokenizer.json"
    if not canonical_tok.is_file():
        return ScaffoldRetrainResult(
            cell_id=cell_id,
            status="failed",
            reason="canonical_artifact_missing",
            scaffold_log_sha=None,
        )

    fresh_sha = None if force else _existing_log_sha_if_fresh(canonical_dir)
    if fresh_sha is not None:
        return ScaffoldRetrainResult(
            cell_id=cell_id,
            status="already_done",
            reason=None,
            scaffold_log_sha=fresh_sha,
        )

    if dry_run:
        _log(f"{cell_id}: dry-run; would retrain into scratch + byte-compare")
        return ScaffoldRetrainResult(
            cell_id=cell_id, status="skipped", reason="dry_run", scaffold_log_sha=None
        )

    with tempfile.TemporaryDirectory(prefix=f"scaffold-{cell_id}-") as scratch_str:
        scratch = Path(scratch_str)
        scratch_log = _train_into(base_config, scratch)
        mismatch = _byte_compare(scratch, canonical_dir)
        if mismatch is not None:
            _log(
                f"{cell_id}: BYTE-IDENTITY VIOLATION ({mismatch}). The "
                "scaffold-instrumentation contract is fixed — halt "
                "and investigate."
            )
            raise RuntimeError(
                f"{cell_id}: scaffold-instrumented retrain produced a "
                f"{mismatch} against the canonical artifact at {canonical_dir}. "
                "The scaffold instrumentation's 'logging-only does not alter "
                "merge selection' claim must hold; this is a contract "
                "violation."
            )
        target = canonical_dir / "scaffold.jsonl"
        target.write_bytes(scratch_log.read_bytes())
    log_sha = sha256_file(canonical_dir / "scaffold.jsonl")
    _patch_meta_with_scaffold_sha(canonical_dir / "meta.yaml", sha=log_sha)
    _log(f"{cell_id}: ok (scaffold.jsonl sha {log_sha[:12]}…)")
    return ScaffoldRetrainResult(
        cell_id=cell_id, status="ok", reason=None, scaffold_log_sha=log_sha
    )


__all__ = [
    "ScaffoldRetrainResult",
    "ScaffoldRetrainStatus",
    "retrain_with_scaffold_log",
]
