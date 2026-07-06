"""Deposit scaffold-token records and aggregate the table.

For each matched ``(V, corpus, boundary)`` coordinate this reads the
BPE arm's sidecar ``scaffold.jsonl`` (per-merge-step ``GpeTrainer``
scaffold instrumentation), applies the Lian-2024 criterion against the surviving
post-trim vocabulary, buckets the resulting scaffold tokens by surface
form, and writes one per-pair JSON under
``results/data/scaffold/``. Unigram arms emit a
``verified_by_construction=True`` zero record so the matched-pair
schema parallel to Deadzone/Absorption stays symmetric. Single-arm coordinates (the
ZINC-22 BPE ``V=2048`` conditional branch and the four single-arm-
knob extras) emit single-arm JSONs of the same schema.

Idempotent: re-running skips pairs whose deposited Scaffold record's
``training_corpus_sha`` (from each cell's ``meta.yaml``) and
``scaffold_log_sha`` (re-hashed from the on-disk sidecar) all still
match. Use ``--rebuild`` to force a full recompute.

Lenient by default — pairs whose cell artifacts or sidecar scaffold
logs are missing go to the aggregator's ``pending`` list and the
script exits 0. ``--strict`` exits non-zero when any selected pair is
pending. BPE cells without a sidecar ``scaffold.jsonl`` need the
``dispatch_grid_cell.py --retrain-scaffold`` (and the extras analog)
sweep to materialize one — supplementary retrain asserts byte-identity
to the canonical artifact, so existing tokenizers are untouched.

Examples::

    uv run python scripts/measure/compute_scaffold.py
    uv run python scripts/measure/compute_scaffold.py --pair pubchem__v256_nmb
    uv run python scripts/measure/compute_scaffold.py --rebuild
    uv run python scripts/measure/compute_scaffold.py --table-only
"""

from __future__ import annotations

import argparse
import sys

from smiles_subword.tokenize.measure.scaffold import io as scaffold_io


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
        deposited, pending = scaffold_io.deposit_all(
            rebuild=args.rebuild, only_pair_keys=only_pair_keys
        )
        for pair_key in deposited:
            print(f"[scaffold] {pair_key} ok")
        for pair_key, reason in pending:
            print(f"[scaffold] {pair_key} PENDING: {reason}", file=sys.stderr)
        print(
            f"[scaffold] {len(deposited)} deposited, {len(pending)} pending",
            file=sys.stderr,
        )
        if args.strict and pending:
            return 1

    table_json, table_md = scaffold_io.build_scaffold_table()
    print(f"[scaffold] aggregator → {table_json}")
    print(f"[scaffold] aggregator → {table_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
