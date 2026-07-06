"""Deposit compositional-closure records and aggregate the table.

For each matched ``(V, corpus, boundary)`` coordinate this reads both cell
tokenizers' realized vocabularies (no corpus, no encoding pass) and counts the
three within-arm closure metrics — binary-split closure (``1.000`` for BPE by
construction, the empirical unknown for Unigram-LM), the orphan rate, and the
stronger full-substring closure — writing one per-pair JSON under
``results/data/closure/``. Single-arm coordinates (the ZINC-22 BPE ``V=2048``
conditional branch and the single-arm-knob extras) still deposit a real reading
for their present arm, since closure is a within-arm property.

Idempotent: re-running skips pairs whose deposited record's per-arm
``training_corpus_sha`` (from each cell's ``meta.yaml``) still matches. Closure
is vocabulary-only, so there is no held-out split SHA to track. ``--rebuild``
forces a full recompute.

Lenient by default — pairs whose cell artifacts are missing go to the
aggregator's ``pending`` list and the script exits 0. ``--strict`` exits
non-zero when any selected pair is pending.

Examples::

    uv run python scripts/measure/compute_closure.py
    uv run python scripts/measure/compute_closure.py --pair pubchem__v256_nmb
    uv run python scripts/measure/compute_closure.py --rebuild
    uv run python scripts/measure/compute_closure.py --table-only
"""

from __future__ import annotations

import argparse
import sys

from smiles_subword.tokenize.measure.closure import io as closure_io


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
        deposited, pending = closure_io.deposit_all(
            rebuild=args.rebuild, only_pair_keys=only_pair_keys
        )
        for pair_key in deposited:
            print(f"[closure] {pair_key} ok")
        for pair_key, reason in pending:
            print(f"[closure] {pair_key} PENDING: {reason}", file=sys.stderr)
        print(
            f"[closure] {len(deposited)} deposited, {len(pending)} pending",
            file=sys.stderr,
        )
        if args.strict and pending:
            return 1

    table_json, table_md = closure_io.build_closure_table()
    print(f"[closure] aggregator → {table_json}")
    print(f"[closure] aggregator → {table_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
