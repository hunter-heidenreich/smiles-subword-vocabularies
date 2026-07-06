"""Deposit Segmentation Unigram-segmentation-entropy records and rebuild the aggregator.

Computes each Unigram cell's exact segmentation entropy (forward DP over the
per-molecule lattice) on its corpus's deterministic held-out split, both raw
(nats/molecule) and per glyph, with 95% bootstrap CIs (Segmentation entropy).
BPE cells are zero by construction (verified, no encode). Deposits one
per-pair JSON under ``results/data/segmentation/`` and
rebuilds ``segmentation_table.{json,md}``.

Idempotent: re-running skips pairs whose deposited record is still fresh;
``--rebuild`` forces a full recompute.

Examples::

    uv run python scripts/measure/compute_segmentation.py
    uv run python scripts/measure/compute_segmentation.py --pair pubchem__v256_nmb
    uv run python scripts/measure/compute_segmentation.py --rebuild
    uv run python scripts/measure/compute_segmentation.py --table-only
"""

from __future__ import annotations

import argparse
import sys

from smiles_subword.tokenize.measure.segmentation import io as segmentation_io


def _build_parser() -> argparse.ArgumentParser:
    description = (__doc__ or "").split("\n\n", 1)[0]
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--pair",
        action="append",
        dest="pairs",
        metavar="PAIR_KEY",
        help="restrict to one or more pair_keys; may be repeated",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="ignore the fresh-record cache and recompute every selected pair",
    )
    parser.add_argument(
        "--table-only",
        action="store_true",
        help="skip per-pair deposition, just rebuild the aggregator table",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero if any selected pair has missing inputs",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    only_pair_keys: frozenset[str] | None = (
        frozenset(args.pairs) if args.pairs else None
    )

    if not args.table_only:
        deposited, pending = segmentation_io.deposit_all(
            rebuild=args.rebuild, only_pair_keys=only_pair_keys
        )
        for pair_key in deposited:
            print(f"[segmentation] {pair_key} ok")
        for pair_key, reason in pending:
            print(f"[segmentation] {pair_key} PENDING: {reason}", file=sys.stderr)
        print(
            f"[segmentation] {len(deposited)} deposited, {len(pending)} pending",
            file=sys.stderr,
        )
        if args.strict and pending:
            return 1

    table_json, table_md = segmentation_io.build_segmentation_table()
    print(f"[segmentation] aggregator → {table_json}")
    print(f"[segmentation] aggregator → {table_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
