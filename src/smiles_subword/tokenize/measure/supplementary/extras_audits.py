"""Robustness-extras audit summaries.

Re-derives the three bespoke audit deposits the results table reads
(``results/data/audits/{seed_cap,prune_schedule,merge_exhaustion}.json``) from the
trained extras tokenizers:

* **seed-cap** and **prune-schedule** compare a one-armed Unigram-LM
  hyperparameter probe's multi-glyph vocabulary against its reference-default
  baseline arm (multi-glyph Jaccard ``=1`` ⇒ the knob is inert / the piece set
  is identical).
* **merge-exhaustion** reads the BPE arm's natural terminal vocabulary off the
  trained tokenizer's ``meta.yaml`` (realised ``|V|`` below the training cap).

Each is a pure vocabulary-set computation over committed tokenizer artifacts —
no corpus pass — so it runs anywhere the artifacts live. The multi-glyph set and
Jaccard use the same definitions as the headline membership measurement
(:mod:`smiles_subword.tokenize.measure.jaccard`): the subword identity is the
glyph-tuple, and a piece is multi-glyph iff its length is ``>= 2``.

This module regenerates the three deposits deterministically.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import yaml

from smiles_subword._io import atomic_write_json
from smiles_subword.paths import audit_path, tokenizer_artifact_dir
from smiles_subword.tokenize.extras import (
    _MERGE_EXHAUSTION_VOCAB_SIZE,
    enumerate_merge_exhaustion,
    enumerate_prune_schedule,
    enumerate_seed_cap,
)
from smiles_subword.tokenize.measure._glyphmap import glyph_tuple_map
from smiles_subword.tokenize.measure.jaccard import GlyphTuple, jaccard

if TYPE_CHECKING:
    from pathlib import Path

SCHEMA_VERSION = 1

__all__ = [
    "build_merge_exhaustion_audit",
    "build_prune_schedule_audit",
    "build_seed_cap_audit",
    "write_extras_audits",
]


def _unigram_baseline_name(vocab_size: int, boundary: str) -> str:
    """Reference-default grid arm a Unigram-LM probe is compared against."""
    return f"smirk_unigram_v{vocab_size}_{boundary}"


def _multi_glyph_set(corpus: str, name: str) -> frozenset[GlyphTuple]:
    """The arm's multi-glyph subword set (glyph-tuples of length ``>= 2``)."""
    tuples = glyph_tuple_map(tokenizer_artifact_dir(corpus, name), "unigram")
    return frozenset(t for t in tuples.values() if len(t) >= 2)


def _read_meta(corpus: str, name: str) -> dict[str, Any]:
    meta = tokenizer_artifact_dir(corpus, name) / "meta.yaml"
    return yaml.safe_load(meta.read_text())


def build_seed_cap_audit() -> dict[str, Any]:
    """Multi-glyph Jaccard of the uncapped-seed probe vs its default-seed arm."""
    (cell,) = enumerate_seed_cap()
    baseline_name = _unigram_baseline_name(cell.vocab_size, cell.boundary)
    probe = _multi_glyph_set(cell.corpus, cell.name)
    base = _multi_glyph_set(cell.corpus, baseline_name)
    return {
        "audit": "seed_cap",
        "schema_version": SCHEMA_VERSION,
        "probe_cell": cell.cell_id,
        "baseline_cell": f"{cell.corpus}__{baseline_name}",
        "probe_multi_glyph_count": len(probe),
        "baseline_multi_glyph_count": len(base),
        "probe_seed_size": cell.seed_size_override,
        "baseline_seed_size": _read_meta(cell.corpus, baseline_name).get("seed_size"),
        "multi_glyph_jaccard": jaccard(probe, base),
        "symmetric_difference_count": len(probe ^ base),
        "inert": probe == base,
    }


def build_prune_schedule_audit() -> dict[str, Any]:
    """Multi-glyph Jaccard of each shrink-schedule probe vs its default arm."""
    comparisons: list[dict[str, Any]] = []
    for cell in enumerate_prune_schedule():
        baseline_name = _unigram_baseline_name(cell.vocab_size, cell.boundary)
        probe = _multi_glyph_set(cell.corpus, cell.name)
        base = _multi_glyph_set(cell.corpus, baseline_name)
        comparisons.append(
            {
                "baseline_cell": f"{cell.corpus}__{baseline_name}",
                "probe_cell": cell.cell_id,
                "baseline_multi_glyph_count": len(base),
                "probe_multi_glyph_count": len(probe),
                "baseline_shrinking_factor": _read_meta(cell.corpus, baseline_name).get(
                    "shrinking_factor"
                ),
                "probe_shrinking_factor": cell.shrinking_factor_override,
                "multi_glyph_jaccard": jaccard(probe, base),
                "multi_glyph_symmetric_difference_count": len(probe ^ base),
                "status": "done",
            }
        )
    return {
        "audit": "prune_schedule",
        "schema_version": SCHEMA_VERSION,
        "comparisons": comparisons,
    }


def build_merge_exhaustion_audit() -> dict[str, Any]:
    """Natural merge-exhaustion terminal vocabulary read off the BPE arm's meta."""
    (cell,) = enumerate_merge_exhaustion()
    meta = _read_meta(cell.corpus, cell.name)
    realised = int(meta["vocab_size"])
    cap = _MERGE_EXHAUSTION_VOCAB_SIZE
    return {
        "audit": "merge_exhaustion",
        "schema_version": SCHEMA_VERSION,
        "cell": cell.cell_id,
        "vocab_size_realised": realised,
        "vocab_size_cap": cap,
        "n_merges": int(meta["n_merges"]),
        "natural_termination": realised < cap,
    }


_BUILDERS = {
    "seed_cap": build_seed_cap_audit,
    "prune_schedule": build_prune_schedule_audit,
    "merge_exhaustion": build_merge_exhaustion_audit,
}


def write_extras_audits() -> list[Path]:
    """Build and deposit all three audit summaries; return the written paths."""
    written: list[Path] = []
    for name, builder in _BUILDERS.items():
        path = audit_path(name)
        atomic_write_json(path, builder())
        written.append(path)
    return written
