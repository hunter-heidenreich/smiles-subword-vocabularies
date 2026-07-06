"""Render the algorithm×boundary interaction figure (NMB − MB per cell).

Pure on-disk read over the measurement ``*_table.json`` deposits under
``results/data/`` (via :mod:`figspec`) — retokenizes nothing — emitting
``results/figures/algo_boundary_interaction.pdf``. The signed-bar rendering is
shared with the cross-``V`` trend figure in :mod:`render`.

Needs the ``results`` optional-dependency extra (matplotlib);
``uv sync --all-extras`` installs it.

Usage::

    uv run python results/build/figure_algo_boundary_interaction.py
"""

from __future__ import annotations

import sys

import figspec
import render

from smiles_subword.paths import RESULTS_FIGURES_DIR


def main() -> int:
    out = render.render_interaction(
        figspec.algo_boundary_interaction_spec(),
        RESULTS_FIGURES_DIR / "algo_boundary_interaction.pdf",
    )
    print(f"[figure] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
