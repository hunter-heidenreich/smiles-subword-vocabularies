"""Measure the sensitivity battery and deposit the response-curve report.

Walks every sensitivity-battery cell (the OFAT ladders, the four interaction
grids, the shared anchor, and the off-anchor BPE references) plus COCONUT's
full-corpus headline BPE V=512 reference, measures each cell's vocabulary sets
(Jaccard) and held-out fertility, then assembles the per-knob response curves
and interaction surfaces into
``results/data/sensitivity/sensitivity_report.json`` --- the
payload the robustness-extras figures consume.

The per-cell measurements are cached alongside the report; a re-run without
``--rebuild`` reuses the cache and re-assembles the report instantly (useful
while iterating on the figures). ``--rebuild`` re-measures every cell.

Examples::

    uv run python scripts/measure/compute_sensitivity.py
    uv run python scripts/measure/compute_sensitivity.py --rebuild
    uv run python scripts/measure/compute_sensitivity.py --sample 100000
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import TYPE_CHECKING

from smiles_subword._io import atomic_write_json
from smiles_subword.tokenize.measure.supplementary.sensitivity import (
    CellMeasured,
    build_report,
)
from smiles_subword.tokenize.measure.supplementary.sensitivity.io import (
    SENSITIVITY_DIR,
    write_report,
)
from smiles_subword.tokenize.measure.supplementary.sensitivity.runner import (
    DEFAULT_FERTILITY_SAMPLE,
    cells_to_measure,
    measure_cell,
)

if TYPE_CHECKING:
    from smiles_subword.tokenize.measure.jaccard import GlyphTuple

_CACHE_PATH = SENSITIVITY_DIR / "_measured_cache.json"


def _measured_to_dict(m: CellMeasured) -> dict[str, object]:
    return {
        "cell_id": m.cell_id,
        "arm": m.arm,
        "corpus": m.corpus,
        "vocab_size": m.vocab_size,
        "boundary": m.boundary,
        "multi": sorted(list(t) for t in m.multi),
        "fertility": m.fertility,
    }


def _measured_from_dict(d: dict[str, object]) -> CellMeasured:
    def _tuples(key: str) -> frozenset[GlyphTuple]:
        return frozenset(tuple(t) for t in d[key])  # type: ignore[union-attr]

    return CellMeasured(
        cell_id=str(d["cell_id"]),
        arm=d["arm"],  # type: ignore[arg-type]
        corpus=str(d["corpus"]),
        vocab_size=int(d["vocab_size"]),  # type: ignore[arg-type]
        boundary=d["boundary"],  # type: ignore[arg-type]
        multi=_tuples("multi"),
        fertility=float(d["fertility"]),  # type: ignore[arg-type]
    )


def load_cache() -> dict[str, CellMeasured]:
    if not _CACHE_PATH.is_file():
        return {}
    raw = json.loads(_CACHE_PATH.read_text())
    return {cid: _measured_from_dict(d) for cid, d in raw.items()}


def write_cache(measured: dict[str, CellMeasured]) -> None:
    payload: dict[str, object] = {
        cid: _measured_to_dict(m) for cid, m in measured.items()
    }
    atomic_write_json(_CACHE_PATH, payload)


def measure_all(*, rebuild: bool, sample: int) -> dict[str, CellMeasured]:
    cache = {} if rebuild else load_cache()
    measured: dict[str, CellMeasured] = {}
    targets = cells_to_measure()
    for i, (cell_id, arm) in enumerate(targets, 1):
        if cell_id in cache:
            measured[cell_id] = cache[cell_id]
            continue
        print(f"[sensitivity] {i}/{len(targets)} measure {cell_id} ({arm})")
        measured[cell_id] = measure_cell(cell_id, arm, fertility_sample=sample)
    write_cache(measured)
    return measured


def _build_parser() -> argparse.ArgumentParser:
    description = (__doc__ or "").split("\n\n", 1)[0]
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="re-measure every cell, ignoring the per-cell measurement cache",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=DEFAULT_FERTILITY_SAMPLE,
        help=(
            "held-out molecules per cell for fertility "
            f"(default {DEFAULT_FERTILITY_SAMPLE})"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    measured = measure_all(rebuild=args.rebuild, sample=args.sample)
    report = build_report(measured)
    path = write_report(report)
    n_pts = sum(len(c.points) for c in report.ladders) + sum(
        len(g.points) for g in report.interactions
    )
    print(
        f"[sensitivity] {len(measured)} cells → {len(report.ladders)} ladders, "
        f"{len(report.interactions)} interactions, {n_pts} contrast points",
        file=sys.stderr,
    )
    print(f"[sensitivity] report → {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
