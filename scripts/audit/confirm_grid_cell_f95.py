"""Confirm F_{p,n} for grid cells (the embedding-tail confirmation).

Resolves a cell id (e.g. ``pubchem__smirk_gpe_v256_nmb``) — or ``--all`` —
against the committed grid manifest, computes the Gowda & May 2020
embedding-learnability metric over the cell's full training corpus, and
deposits ``results/data/f95/<cell_id>.json``. Cells not yet
trained are reported and skipped, so ``--all`` is safe to run against a partial
grid throughout the later audit and robustness-extras runs. ``--table``
aggregates the deposited JSONs into the tokenizer-grid learnability table.

``scripts/tokenize/dispatch_grid_cell.py --confirm-f95`` runs this right after
each cell trains; this script is the standalone / backfill entry point.
"""

from __future__ import annotations

import argparse
import sys

from smiles_subword.tokenize import dispatch
from smiles_subword.tokenize.audit.f95_confirm import confirm_cell
from smiles_subword.tokenize.audit.f95_io import build_f95_table
from smiles_subword.tokenize.grid import load_grid_manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cell",
        help="cell id, e.g. pubchem__smirk_gpe_v256_nmb (not with --all)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="confirm every grid cell, skipping cells already confirmed",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the cell + corpus per cell; do not compute or deposit",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-confirm even cells whose f95 JSON is already on disk",
    )
    parser.add_argument(
        "--list",
        dest="list_cells",
        action="store_true",
        help="print every cell id and exit",
    )
    parser.add_argument(
        "--table",
        action="store_true",
        help="rebuild the aggregate f95_table.{json,md} from deposited JSONs and exit",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.list_cells:
        for cell in load_grid_manifest():
            print(cell.cell_id)
        return 0

    if args.table:
        table_json, table_md = build_f95_table()
        print(f"wrote {table_json}")
        print(f"wrote {table_md}")
        return 0

    if args.all and args.cell is not None:
        print("error: pass --cell or --all, not both", file=sys.stderr)
        return 2
    if not args.all and args.cell is None:
        print("error: --cell or --all is required (or pass --list)", file=sys.stderr)
        return 2

    if args.all:
        cells = load_grid_manifest()
    else:
        assert args.cell is not None
        try:
            cells = [dispatch.resolve_cell(args.cell, load_grid_manifest())]
        except FileNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    for cell in cells:
        confirm_cell(cell, force=args.force, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
