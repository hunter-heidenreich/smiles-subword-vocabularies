"""Dispatch one robustness-extras cell.

Parallel to ``scripts/tokenize/dispatch_grid_cell.py`` but consumes the extras
manifest at ``configs/tokenizer/extras.yaml`` (subsample-redraws, size-sweep,
seed-cap probe, prune-schedule probe, and the merge-exhaustion cell). Resolves a
cell id, materializes its :class:`~smiles_subword.config.TokenizerConfig` to a
gitignored cache YAML, and shells out to ``scripts/tokenize/train_tokenizer.py``.
``--all`` walks every cell and skips ones already trained, so an interrupted
sweep resumes cleanly.

The orchestration (resolve / skip-if-done / materialize / dispatch / audit hooks)
lives in :mod:`smiles_subword.tokenize.dispatch`; this driver binds the extras
seams and the ``--extras-kind`` argparser.
"""

from __future__ import annotations

import argparse
import sys

from smiles_subword.paths import CONFIGS_DIR
from smiles_subword.tokenize import dispatch
from smiles_subword.tokenize.audit.determinism_verify import verify_extras_cell
from smiles_subword.tokenize.audit.f95_confirm import confirm_extras_cell
from smiles_subword.tokenize.extras import (
    ExtrasCell,
    ExtrasKind,
    cells_for_extras_kind,
    extras_cell_to_config,
    load_extras_manifest,
    write_extras_manifest,
)

EXTRAS_SEAMS: dispatch.DispatchSeams[ExtrasCell] = dispatch.DispatchSeams(
    to_config=extras_cell_to_config,
    confirm_hook=confirm_extras_cell,
    verify_hook=verify_extras_cell,
    cache_dir=CONFIGS_DIR / "tokenizer" / ".dispatch_cache_extras",
)

_EXTRAS_KIND_CHOICES: tuple[ExtrasKind, ...] = (
    "subsample_redraw",
    "size_sweep",
    "seed_cap",
    "prune_schedule",
    "merge_exhaustion",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cell",
        help="extras cell id (not with --all)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="dispatch every extras cell, skipping those already trained",
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
        "--extras-kind",
        dest="extras_kind",
        choices=_EXTRAS_KIND_CHOICES,
        help="restrict --all to one extras kind",
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
        help="print every extras cell id and exit",
    )
    parser.add_argument(
        "--write-manifest",
        dest="write_manifest",
        action="store_true",
        help="regenerate configs/tokenizer/extras.yaml from the spec and exit",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.write_manifest:
        print(f"wrote extras manifest: {write_extras_manifest()}")
        return 0

    if args.list_cells:
        for cell in load_extras_manifest():
            print(cell.cell_id)
        return 0

    if args.all and args.cell is not None:
        print("error: pass --cell or --all, not both", file=sys.stderr)
        return 2
    if not args.all and args.cell is None:
        print("error: --cell or --all is required (or pass --list)", file=sys.stderr)
        return 2
    if args.cell is not None and args.extras_kind is not None:
        print(
            "error: --extras-kind applies to --all, not a single --cell",
            file=sys.stderr,
        )
        return 2

    if args.all:
        cells = cells_for_extras_kind(args.extras_kind)
    else:
        assert args.cell is not None
        try:
            cells = [dispatch.resolve_cell(args.cell, load_extras_manifest())]
        except FileNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    dispatch.run_dispatch(
        cells,
        EXTRAS_SEAMS,
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
