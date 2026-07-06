"""Render the cross-``V`` trend figure (one panel per measurement).

Pure on-disk read over the Jaccard / Fertility / Distribution ``*_table.json``
deposits under ``results/data/`` (via :mod:`figspec`) — retokenizes nothing —
emitting ``results/figures/cross_v_trends.pdf``. The non-trivial split-legend
rendering is shared with the interaction figure in :mod:`render`.

Needs the ``results`` optional-dependency extra (matplotlib);
``uv sync --all-extras`` installs it.

Usage::

    uv run python results/build/figure_cross_v_trends.py
"""

from __future__ import annotations

import sys

import figspec
import render

from smiles_subword.paths import RESULTS_FIGURES_DIR


def main() -> int:
    out = render.render_trend(
        figspec.cross_v_trend_spec(), RESULTS_FIGURES_DIR / "cross_v_trends.pdf"
    )
    print(f"[figure] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
