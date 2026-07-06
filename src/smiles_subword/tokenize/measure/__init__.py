"""Measurements over the trained grid.

Each measurement is a self-contained package ``measure/<topic>/`` with a
uniform split:

- ``math.py`` — the pure measurement math (the package re-exports its API, so
  ``from measure.<topic> import <symbol>`` reaches it directly),
- ``runner.py`` — the per-cell (or per-pair) build that encodes the held-out
  split through the trained tokenizers,
- ``io.py`` — the idempotent JSON deposit + aggregate-table renderer.

The headline seven are ``deadzone`` (no runner — it joins deposited F95 audit
records), ``absorption``, ``scaffold``, ``fertility``, ``jaccard``,
``distribution``, and ``segmentation``. The paper-appendix measurements
``closure``, ``fg_alignment``, ``nestedness``, and ``noncanon`` follow the same
shape. Cross-cutting helpers live at this level: ``_cells`` (load a cell +
locate/fingerprint/stream its held-out split), ``_glyphmap`` (token-id glyph
maps), ``_bootstrap`` (percentile CIs), ``_pairing`` (cross-arm matchmaking),
``_cellmeta`` (resolve a cell's ``meta.yaml`` into deposit fields + freshness),
``_tables`` (``*_table.md`` cell formatting), and ``_deposit`` (the shared
deposit + table-join engine every ``io.py`` routes through). Further secondary
analyses live in ``measure/supplementary/``.

This package intentionally exports nothing at the top level: every consumer
imports the specific submodule it needs.
"""

from __future__ import annotations
