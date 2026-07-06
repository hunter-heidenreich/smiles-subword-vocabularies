"""Deposit non-canonicity records and aggregate the table.

For each matched ``(V, corpus, boundary)`` coordinate this builds each held-out
molecule's identity-preserving rewrite orbit (RDKit randomized SMILES, ring-digit
relabel, Kekule, all-explicit-H), encodes the canonical string and every variant
through both cell tokenizers, and measures how much each arm's segmentation moves
(fertility dispersion and piece-bag instability per axis, with bootstrap CIs),
plus the cross-arm fertility-gap survival. One per-pair JSON lands under
``results/data/noncanon/``. Single-arm coordinates still deposit a real reading.

The pass runs over a seeded subsample (the first ``MOLECULE_LIMIT`` molecules of
the held-out split), since each molecule carries a K-fold orbit. Idempotent:
re-running skips pairs whose per-arm ``training_corpus_sha`` and ``eval_split_sha``
still match. ``--rebuild`` forces recompute. Lenient by default; ``--strict``
exits non-zero on any pending pair.

Examples::

    uv run python scripts/measure/compute_noncanon.py
    uv run python scripts/measure/compute_noncanon.py --pair pubchem__v1024_nmb
    uv run python scripts/measure/compute_noncanon.py --rebuild
    uv run python scripts/measure/compute_noncanon.py --table-only
"""

from __future__ import annotations

import argparse
import sys

from smiles_subword.tokenize.measure.noncanon import io as noncanon_io


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
        deposited, pending = noncanon_io.deposit_all(
            rebuild=args.rebuild, only_pair_keys=only_pair_keys
        )
        for pair_key in deposited:
            print(f"[noncanon] {pair_key} ok")
        for pair_key, reason in pending:
            print(f"[noncanon] {pair_key} PENDING: {reason}", file=sys.stderr)
        print(
            f"[noncanon] {len(deposited)} deposited, {len(pending)} pending",
            file=sys.stderr,
        )
        if args.strict and pending:
            return 1

    table_json, table_md = noncanon_io.build_noncanon_table()
    print(f"[noncanon] aggregator → {table_json}")
    print(f"[noncanon] aggregator → {table_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
