"""Compute the cross-corpus transfer matrix (fertility + atom-OOV).

Reads each trained grid cell on every corpus's held-out test split and deposits
one per-cell JSON under ``results/data/transfer/``. The transfer-matrix table
(aggregate + ``.tex``) is rendered separately by
``results/build/table_transfer.py``. Lenient: cells whose artifact or split is
missing are skipped and reported.

Examples::

    uv run python scripts/measure/compute_transfer.py --limit 2000   # quick probe
    uv run python scripts/measure/compute_transfer.py                # full matrix
"""

from __future__ import annotations

import argparse
import sys

from smiles_subword.config import algo_to_engine_tag
from smiles_subword.tokenize.measure.supplementary.transfer.math import TRANSFER_DIR
from smiles_subword.tokenize.measure.supplementary.transfer.runner import (
    enumerate_transfer_cells,
    run_transfer,
    write_transfer_record,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n", 1)[0])
    parser.add_argument(
        "--sample",
        type=int,
        default=100_000,
        help="fixed-seed random subset of molecules per eval split (default 100k)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="take the first N molecules instead of sampling (debug)",
    )
    parser.add_argument(
        "--rebuild", action="store_true", help="recompute even cells already deposited"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    cells = enumerate_transfer_cells()
    done = skipped = 0
    for train, ev, arm, v, boundary in cells:
        key = f"{train}__{ev}__{algo_to_engine_tag(arm)}_v{v}_{boundary}"
        if not args.rebuild and (TRANSFER_DIR / f"{key}.json").is_file():
            continue
        try:
            rec = run_transfer(
                train_corpus=train,
                eval_corpus=ev,
                arm=arm,
                vocab_size=v,
                boundary=boundary,
                sample=None if args.limit is not None else args.sample,
                limit=args.limit,
            )
        except FileNotFoundError as exc:
            print(f"[transfer] skip {key}: {exc}", file=sys.stderr)
            skipped += 1
            continue
        write_transfer_record(rec)
        done += 1
        print(
            f"[transfer] {key}: fertility={rec.fertility_mean:.2f} "
            f"oov_tok={rec.oov_token_rate:.4f}"
        )
    print(f"[transfer] {done} deposited, {skipped} skipped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
