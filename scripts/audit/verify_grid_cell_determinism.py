"""Verify per-arm determinism for grid cells (the determinism audit).

Resolves a cell id (e.g. ``pubchem__smirk_gpe_v256_nmb``) — or ``--all``,
optionally narrowed by ``--tier`` — against the committed grid manifest,
retrains each trained cell once into a scratch directory, and asserts per-arm
determinism (BPE byte-identical; Unigram piece-set identical).
Each result is deposited under
``results/data/determinism/<cell_id>.json``. Cells not yet
trained are reported and skipped, so ``--all`` is safe against a partial grid.
``--table`` aggregates the deposited JSONs into the tokenizer-grid determinism
table.

``scripts/tokenize/dispatch_grid_cell.py --verify-determinism`` runs this immediately
after each cell trains; this script is the standalone / backfill entry point.

A BPE mismatch raises and crashes the sweep (deterministic by construction — a
bug). An *unexpected* Unigram piece-set jitter (one outside the expected
``V=1024 NMB`` set) makes the script exit non-zero so it cannot pass silently.
"""

from __future__ import annotations

import argparse
import sys

from smiles_subword.tokenize import dispatch
from smiles_subword.tokenize.audit.determinism_io import build_determinism_table
from smiles_subword.tokenize.audit.determinism_verify import (
    is_expected_unigram_jitter,
    verify_cell,
)
from smiles_subword.tokenize.grid import cells_for_tier


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cell",
        help="cell id, e.g. pubchem__smirk_gpe_v256_nmb (not with --all)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="verify every grid cell, skipping cells already verified",
    )
    parser.add_argument(
        "--tier",
        choices=("headline", "sensitivity", "anchor"),
        help="restrict --all / --list to one grid tier (e.g. headline)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the cell per cell; do not retrain, compare, or deposit",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-verify even cells whose determinism JSON is already on disk",
    )
    parser.add_argument(
        "--list",
        dest="list_cells",
        action="store_true",
        help="print every cell id (honors --tier) and exit",
    )
    parser.add_argument(
        "--table",
        action="store_true",
        help="rebuild determinism_table.{json,md} from deposited JSONs and exit",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.list_cells:
        for cell in cells_for_tier(args.tier):
            print(cell.cell_id)
        return 0

    if args.table:
        table_json, table_md = build_determinism_table()
        print(f"wrote {table_json}")
        print(f"wrote {table_md}")
        return 0

    if args.all and args.cell is not None:
        print("error: pass --cell or --all, not both", file=sys.stderr)
        return 2
    if not args.all and args.cell is None:
        print("error: --cell or --all is required (or pass --list)", file=sys.stderr)
        return 2
    if args.cell is not None and args.tier is not None:
        print("error: --tier applies to --all, not a single --cell", file=sys.stderr)
        return 2

    if args.all:
        cells = cells_for_tier(args.tier)
    else:
        assert args.cell is not None
        try:
            cells = [dispatch.resolve_cell(args.cell, cells_for_tier())]
        except FileNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    unexpected: list[str] = []
    for cell in cells:
        result = verify_cell(cell, force=args.force, dry_run=args.dry_run)
        if (
            result is not None
            and not result.deterministic
            and not is_expected_unigram_jitter(cell)
        ):
            unexpected.append(cell.cell_id)

    if unexpected:
        print(
            f"error: {len(unexpected)} cell(s) failed determinism unexpectedly: "
            f"{', '.join(unexpected)}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
