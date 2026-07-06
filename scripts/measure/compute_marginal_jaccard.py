"""Compute the marginal cross-arm Jaccard across vocabulary-size steps.

For each ``(corpus, boundary)`` and consecutive vocabulary-size step
``v_lower -> v_upper`` where both arms are trained, this loads the two arms'
multi-glyph piece sets (glyph-tuple convention of :mod:`jaccard`, single-glyph
base excluded), forms each arm's freshly-added pieces, and computes the Jaccard
of those fresh sets. It quantifies the claim that the cross-$V$ rise in
overlap on PubChem/ZINC-22 is carried by BPE catching up to pieces Unigram-LM
already had — so what each arm *adds* between consecutive $V$ is near-disjoint.

Deposits one aggregate record per step to
``results/data/marginal_jaccard_table.json`` and a companion ``.md``. Lenient:
steps whose tokenizer artifact is missing are skipped and reported.

Examples::

    uv run python scripts/measure/compute_marginal_jaccard.py
"""

from __future__ import annotations

import argparse
import math
import sys
from itertools import pairwise
from typing import TYPE_CHECKING

from smiles_subword._io import atomic_write_json
from smiles_subword.config import TokenizerAlgo, cell_artifact_name
from smiles_subword.paths import tokenizer_artifact_dir
from smiles_subword.tokenize.grid import enumerate_all
from smiles_subword.tokenize.measure._glyphmap import (
    build_bpe_glyph_tuples,
    build_unigram_glyph_tuples,
)
from smiles_subword.tokenize.measure.supplementary.marginal_jaccard import (
    MARGINAL_JACCARD_SCHEMA_VERSION,
    MARGINAL_JACCARD_TABLE,
    MarginalStep,
    build_step,
)

if TYPE_CHECKING:
    from smiles_subword.tokenize.measure.jaccard import GlyphTuple


def _multi_set(
    corpus: str, arm: TokenizerAlgo, v: int, boundary: str
) -> frozenset[GlyphTuple] | None:
    """Multi-glyph (glyph_count >= 2) piece set for one cell, or None if absent."""
    tokenizer_json = (
        tokenizer_artifact_dir(corpus, cell_artifact_name(arm, v, boundary))
        / "tokenizer.json"
    )
    if not tokenizer_json.is_file():
        return None
    builder = build_bpe_glyph_tuples if arm == "bpe" else build_unigram_glyph_tuples
    return frozenset(t for t in builder(tokenizer_json).values() if len(t) >= 2)


def _arm_multi_sets(
    corpus: str, boundary: str
) -> dict[tuple[TokenizerAlgo, int], frozenset[GlyphTuple]]:
    """Load every present cell's multi-glyph set for one (corpus, boundary), once.

    Keyed by ``(arm, V)``; cells with no trained artifact are omitted. Computing
    each set a single time here lets the both-arms-present scan and every per-step
    difference reuse it instead of re-parsing the same ``tokenizer.json`` files.
    """
    sets: dict[tuple[TokenizerAlgo, int], frozenset[GlyphTuple]] = {}
    for cell in enumerate_all():
        if cell.corpus != corpus or cell.boundary != boundary:
            continue
        ms = _multi_set(corpus, cell.algo, cell.vocab_size, boundary)
        if ms is not None:
            sets[(cell.algo, cell.vocab_size)] = ms
    return sets


def compute_steps() -> list[MarginalStep]:
    """Every consecutive-V marginal step over the committed grid."""
    pairs = sorted({(c.corpus, c.boundary) for c in enumerate_all()})
    steps: list[MarginalStep] = []
    for corpus, boundary in pairs:
        sets = _arm_multi_sets(corpus, boundary)
        vs = sorted(
            {v for (_arm, v) in sets if ("bpe", v) in sets and ("unigram", v) in sets}
        )
        for v_lower, v_upper in pairwise(vs):
            steps.append(
                build_step(
                    corpus=corpus,
                    boundary=boundary,
                    v_lower=v_lower,
                    v_upper=v_upper,
                    bpe_lower=sets[("bpe", v_lower)],
                    bpe_upper=sets[("bpe", v_upper)],
                    unigram_lower=sets[("unigram", v_lower)],
                    unigram_upper=sets[("unigram", v_upper)],
                )
            )
    return steps


def deposit(steps: list[MarginalStep]) -> tuple[str, str]:
    """Write the aggregate JSON + Markdown table; return both paths."""
    finite = [s.marginal_jaccard for s in steps if not math.isnan(s.marginal_jaccard)]
    payload = {
        "schema_version": MARGINAL_JACCARD_SCHEMA_VERSION,
        "n_steps": len(steps),
        "max_marginal_jaccard": max(finite) if finite else None,
        "steps": [s.as_dict() for s in steps],
    }
    atomic_write_json(MARGINAL_JACCARD_TABLE, payload)

    md_path = MARGINAL_JACCARD_TABLE.with_suffix(".md")
    lines = [
        "# Marginal cross-arm Jaccard per V-step\n",
        "`fresh_arm = multi(arm, V_upper) \\ multi(arm, V_lower)`; "
        "marginal J = Jaccard(fresh_bpe, fresh_unigram).\n",
        "| corpus | bnd | step | fresh BPE | fresh UL | shared | marginal J |",
        "|---|---|---|---|---|---|---|",
    ]
    lines.extend(
        f"| {s.corpus} | {s.boundary} | {s.v_lower}->{s.v_upper} | "
        f"{s.n_fresh_bpe} | {s.n_fresh_unigram} | {s.n_fresh_shared} | "
        f"{s.marginal_jaccard:.4f} |"
        for s in steps
    )
    md_path.write_text("\n".join(lines) + "\n")
    return str(MARGINAL_JACCARD_TABLE), str(md_path)


def _build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description=(__doc__ or "").split("\n\n", 1)[0])


def main(argv: list[str] | None = None) -> int:
    _build_parser().parse_args(argv)
    steps = compute_steps()
    if not steps:
        print("[marginal-j] no steps (artifacts missing?)", file=sys.stderr)
        return 1
    json_path, md_path = deposit(steps)
    worst = max(steps, key=lambda s: s.marginal_jaccard)
    for s in steps:
        print(
            f"[marginal-j] {s.corpus:10} {s.boundary} {s.v_lower}->{s.v_upper}: "
            f"J={s.marginal_jaccard:.4f} (fresh shared {s.n_fresh_shared})"
        )
    print(
        f"[marginal-j] max marginal J = {worst.marginal_jaccard:.4f} "
        f"({worst.corpus} {worst.boundary} {worst.v_lower}->{worst.v_upper})"
    )
    print(f"[marginal-j] → {json_path}")
    print(f"[marginal-j] → {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
