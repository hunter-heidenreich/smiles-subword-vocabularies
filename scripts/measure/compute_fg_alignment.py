"""Deposit functional-bond-locality records and aggregate the table.

For each matched ``(V, corpus, boundary)`` coordinate this dual-encodes the
corpus's held-out split through both cell tokenizers and asks, for every
multiply-bonded heteroatom (the ``=O`` of a carbonyl, the ``#N`` of a nitrile,
the ``=O`` on sulfur/phosphorus/nitrogen, ...), whether the arm kept the bond
inside a single token. It writes one per-pair JSON under
``results/data/fg_alignment/`` carrying each arm's overall locality (with a 95%
molecule-resampled bootstrap CI), the per-bond-class breakdown, and the
cross-arm gap. Single-arm coordinates still deposit a real reading for their
present arm, since locality is a within-arm property.

Idempotent: re-running skips pairs whose deposited record's per-arm
``training_corpus_sha`` (from each cell's ``meta.yaml``) and ``eval_split_sha``
(from the held-out test-split MANIFEST) still match. ``--rebuild`` forces a full
recompute.

Lenient by default — pairs whose cell artifacts are missing go to the
aggregator's ``pending`` list and the script exits 0. ``--strict`` exits
non-zero when any selected pair is pending.

Examples::

    uv run python scripts/measure/compute_fg_alignment.py
    uv run python scripts/measure/compute_fg_alignment.py --pair pubchem__v256_nmb
    uv run python scripts/measure/compute_fg_alignment.py --rebuild
    uv run python scripts/measure/compute_fg_alignment.py --table-only
"""

from __future__ import annotations

import argparse
import sys

from smiles_subword.tokenize.measure.fg_alignment import io as fg_alignment_io


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
        deposited, pending = fg_alignment_io.deposit_all(
            rebuild=args.rebuild, only_pair_keys=only_pair_keys
        )
        for pair_key in deposited:
            print(f"[fg_alignment] {pair_key} ok")
        for pair_key, reason in pending:
            print(f"[fg_alignment] {pair_key} PENDING: {reason}", file=sys.stderr)
        print(
            f"[fg_alignment] {len(deposited)} deposited, {len(pending)} pending",
            file=sys.stderr,
        )
        if args.strict and pending:
            return 1

    table_json, table_md = fg_alignment_io.build_fg_alignment_table()
    print(f"[fg_alignment] aggregator → {table_json}")
    print(f"[fg_alignment] aggregator → {table_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
