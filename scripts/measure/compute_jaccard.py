"""Deposit Jaccard vocabulary-Jaccard records and aggregate the table.

For each matched ``(V, corpus, boundary)`` coordinate this loads both cell
tokenizers, builds each arm's multi-glyph subword set, partitions it into
structural / bracket-internal from the cached training-corpus chunk inventory,
streams the held-out split once per arm, and writes one per-pair JSON under
``results/data/jaccard/`` carrying the unweighted
Jaccard ``J``, the structural Jaccard ``J_struct``, and the frequency-weighted
``J_w`` with a 95% bootstrap CI (Membership). Single-arm
coordinates emit JSONs of the same schema with the cross-arm Jaccards null.

Idempotent: re-running skips pairs whose deposited Jaccard record's
``training_corpus_sha`` (from each cell's ``meta.yaml``) and ``eval_split_sha``
(from the corpus's held-out test-split MANIFEST) all still match. Use
``--rebuild`` to force a full recompute.

Lenient by default — pairs whose cell artifacts or splits are missing go to the
aggregator's ``pending`` list and the script exits 0. ``--strict`` exits
non-zero when any selected pair is pending.

Examples::

    uv run python scripts/measure/compute_jaccard.py
    uv run python scripts/measure/compute_jaccard.py --pair pubchem__v256_nmb
    uv run python scripts/measure/compute_jaccard.py --rebuild
    uv run python scripts/measure/compute_jaccard.py --table-only
"""

from __future__ import annotations

import argparse
import sys

from smiles_subword.tokenize.measure.jaccard import io as jaccard_io


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
        deposited, pending = jaccard_io.deposit_all(
            rebuild=args.rebuild, only_pair_keys=only_pair_keys
        )
        for pair_key in deposited:
            print(f"[jaccard] {pair_key} ok")
        for pair_key, reason in pending:
            print(f"[jaccard] {pair_key} PENDING: {reason}", file=sys.stderr)
        print(
            f"[jaccard] {len(deposited)} deposited, {len(pending)} pending",
            file=sys.stderr,
        )
        if args.strict and pending:
            return 1

    table_json, table_md = jaccard_io.build_jaccard_table()
    print(f"[jaccard] aggregator → {table_json}")
    print(f"[jaccard] aggregator → {table_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
