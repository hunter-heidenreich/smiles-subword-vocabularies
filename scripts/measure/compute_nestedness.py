"""Deposit Nestedness boundary-agreement records and aggregate the table.

For each matched ``(V, corpus, boundary)`` coordinate this loads both cell
tokenizers, streams the corpus's deterministic held-out test split once,
encodes every molecule through *both* arms, and compares their token
boundaries into the 2x2 (agree-cut / nest / conflict / agree-merge) plus the
per-class conflict localization, writing one per-pair JSON under
``results/data/nestedness/``. Single-arm coordinates (the ZINC-22 BPE ``V=2048``
conditional branch and the four single-arm-knob extras) deposit a metric-free
single-arm JSON, since boundary agreement is intrinsically cross-arm.

Idempotent: re-running skips pairs whose deposited record's
``training_corpus_sha`` (per arm ``meta.yaml``) and ``eval_split_sha`` (the
corpus's held-out test-split MANIFEST) all still match. ``--rebuild`` forces a
full recompute.

Lenient by default — pairs whose cell artifacts or splits are missing go to the
aggregator's ``pending`` list and the script exits 0. ``--strict`` exits
non-zero when any selected pair is pending.

Examples::

    uv run python scripts/measure/compute_nestedness.py
    uv run python scripts/measure/compute_nestedness.py --pair pubchem__v256_nmb
    uv run python scripts/measure/compute_nestedness.py --rebuild
    uv run python scripts/measure/compute_nestedness.py --table-only
"""

from __future__ import annotations

import argparse
import sys

from smiles_subword.tokenize.measure.nestedness import io as nestedness_io


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
        deposited, pending = nestedness_io.deposit_all(
            rebuild=args.rebuild, only_pair_keys=only_pair_keys
        )
        for pair_key in deposited:
            print(f"[nestedness] {pair_key} ok")
        for pair_key, reason in pending:
            print(f"[nestedness] {pair_key} PENDING: {reason}", file=sys.stderr)
        print(
            f"[nestedness] {len(deposited)} deposited, {len(pending)} pending",
            file=sys.stderr,
        )
        if args.strict and pending:
            return 1

    table_json, table_md = nestedness_io.build_nestedness_table()
    print(f"[nestedness] aggregator → {table_json}")
    print(f"[nestedness] aggregator → {table_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
