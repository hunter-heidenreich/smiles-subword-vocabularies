"""Dispatch one grid cell.

Resolves a cell id (e.g. ``pubchem__smirk_gpe_v256_nmb``) against the committed
grid manifest ``configs/tokenizer/grid.yaml``, materializes its
``TokenizerConfig`` to a gitignored cache YAML, and shells out to
``scripts/tokenize/train_tokenizer.py``. ``--all`` walks every committed cell (the
44-cell grid plus the 2 conditional cells) and skips cells already
trained, so an interrupted run resumes cleanly.

The orchestration (resolve / skip-if-done / materialize / dispatch / audit hooks)
lives in :mod:`smiles_subword.tokenize.dispatch`; this driver binds the grid
seams and the ``--tier`` argparser.
"""

from __future__ import annotations

import argparse
import sys

from smiles_subword.paths import CONFIGS_DIR
from smiles_subword.tokenize import dispatch
from smiles_subword.tokenize.audit.determinism_verify import verify_cell
from smiles_subword.tokenize.audit.f95_confirm import confirm_cell
from smiles_subword.tokenize.grid import (
    GridCell,
    cells_for_tier,
    grid_cell_to_config,
    load_grid_manifest,
    write_grid_manifest,
)

GRID_SEAMS: dispatch.DispatchSeams[GridCell] = dispatch.DispatchSeams(
    to_config=grid_cell_to_config,
    confirm_hook=confirm_cell,
    verify_hook=verify_cell,
    cache_dir=CONFIGS_DIR / "tokenizer" / ".dispatch_cache",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cell",
        help="cell id, e.g. pubchem__smirk_gpe_v256_nmb (not with --all)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="dispatch every grid cell, skipping cells already trained",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the resolved config + command per cell; do not train",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-dispatch even cells whose artifact is already on disk",
    )
    parser.add_argument(
        "--tier",
        choices=("headline", "sensitivity", "anchor", "conditional"),
        help="restrict --all to one tier (headline, or one of the other three)",
    )
    parser.add_argument(
        "--confirm-f95",
        dest="confirm_f95",
        action="store_true",
        help="run the F_{p,n} confirmation immediately after each cell",
    )
    parser.add_argument(
        "--verify-determinism",
        dest="verify_determinism",
        action="store_true",
        help="retrain each cell once and assert per-arm determinism",
    )
    parser.add_argument(
        "--retrain-scaffold",
        dest="retrain_scaffold",
        action="store_true",
        help=(
            "supplementary retrain each BPE cell with scaffold "
            "logging on; assert byte-identity to the canonical artifact and "
            "materialize scaffold.jsonl in place (scaffold)"
        ),
    )
    parser.add_argument(
        "--list",
        dest="list_cells",
        action="store_true",
        help="print every cell id and exit",
    )
    parser.add_argument(
        "--write-manifest",
        dest="write_manifest",
        action="store_true",
        help="regenerate configs/tokenizer/grid.yaml from the grid rule and exit",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.write_manifest:
        print(f"wrote grid manifest: {write_grid_manifest()}")
        return 0

    if args.list_cells:
        for cell in load_grid_manifest():
            print(cell.cell_id)
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
            cells = [dispatch.resolve_cell(args.cell, load_grid_manifest())]
        except FileNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    dispatch.run_dispatch(
        cells,
        GRID_SEAMS,
        log=print,
        dry_run=args.dry_run,
        force=args.force,
        confirm_f95=args.confirm_f95,
        verify_determinism=args.verify_determinism,
        retrain_scaffold=args.retrain_scaffold,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
