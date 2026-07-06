"""Deposit dead-zone surplus (``ΔF``) records and aggregate the table.

Walks the committed grid + extras manifest, joins the deposited F95 JSONs
arm-by-arm at each matched ``(V, corpus, boundary)`` coordinate, and writes
one per-pair JSON under ``results/data/deadzone/``.
Single-arm coordinates (the ZINC-22 BPE ``V=2048`` conditional branch
and the four single-arm-knob extras) emit single-arm JSONs of the same
schema.

Idempotent: re-running skips pairs whose cached Deadzone record is fresh
(both underlying F95 ``training_corpus_sha`` values still match). Use
``--rebuild`` to force a full recompute.

Lenient by default — pairs whose F95 inputs are not on disk go to the
aggregator's ``pending`` list and the script exits 0. ``--strict`` exits
non-zero when any pair is pending.

Examples::

    uv run python scripts/measure/compute_deadzone.py
    uv run python scripts/measure/compute_deadzone.py --pair pubchem__v256_nmb
    uv run python scripts/measure/compute_deadzone.py --rebuild
    uv run python scripts/measure/compute_deadzone.py --table-only
"""

from __future__ import annotations

import argparse
import sys

from smiles_subword.tokenize.measure.deadzone import io as deadzone_io


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
        help="exit non-zero if any selected pair has a missing F95 input",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    only_pair_keys: frozenset[str] | None = (
        frozenset(args.pairs) if args.pairs else None
    )

    if not args.table_only:
        deposited, pending = deadzone_io.deposit_all(
            rebuild=args.rebuild, only_pair_keys=only_pair_keys
        )
        for pair_key in deposited:
            print(f"[deadzone] {pair_key} ok")
        for pair_key, reason in pending:
            print(f"[deadzone] {pair_key} PENDING: {reason}", file=sys.stderr)
        print(
            f"[deadzone] {len(deposited)} deposited, {len(pending)} pending",
            file=sys.stderr,
        )
        if args.strict and pending:
            return 1

    table_json, table_md = deadzone_io.build_deadzone_table()
    print(f"[deadzone] aggregator → {table_json}")
    print(f"[deadzone] aggregator → {table_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
