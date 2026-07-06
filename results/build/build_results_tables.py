"""Render the results tables (.tex) from the deposited aggregators.

Pure on-disk read over the measurement ``*_table.json`` deposits under
``results/data/`` — retokenizes nothing — emitting the
``booktabs`` tables under ``results/tables/``: the seven measurements, the
cross-arm ΔF, the three Jaccards, the per-condition fertility / nestedness /
distribution / absorption detail, and the compact robustness-extras summary
(from the Deadzone extras rows + the ``data/audits/`` deposits).

Idempotent: skips the write when every required upstream aggregator and audit
still hashes the same (re-run a measurement to invalidate) unless ``--rebuild``.
Lenient by default — a missing upstream aggregator or audit is reported and the
script still emits what it can; ``--strict`` exits non-zero when any required
aggregator or audit is absent. The deposit-derived tables honour ``--include-extras``
(grid only by default); the robustness-extras table always summarises the extras.

Examples::

    uv run python results/build/build_results_tables.py
    uv run python results/build/build_results_tables.py --rebuild
    uv run python results/build/build_results_tables.py --include-extras --strict
"""

from __future__ import annotations

import argparse
import sys

import extract
import latex


def _build_parser() -> argparse.ArgumentParser:
    description = (__doc__ or "").split("\n\n", 1)[0]
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="ignore the fresh-artifact cache and re-render every table",
    )
    parser.add_argument(
        "--include-extras",
        action="store_true",
        help="also tabulate the robustness-extras rows (default: grid only)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero if any required upstream aggregator is missing",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    missing = extract.missing_tables()
    for name in missing:
        print(f"[tables] MISSING upstream aggregator: {name}.json", file=sys.stderr)
    missing_audits = extract.missing_audits()
    for name in missing_audits:
        print(f"[tables] MISSING robustness audit: audits/{name}.json", file=sys.stderr)
    if args.strict and (missing or missing_audits):
        return 1

    if not args.rebuild and latex.is_tables_fresh(include_extras=args.include_extras):
        print("[tables] fresh (skipped) — upstream aggregators unchanged")
        return 0

    written = latex.write_tables(include_extras=args.include_extras)
    for path in written:
        print(f"[tables] wrote {path}")
    print(f"[tables] manifest → {latex.RESULTS_MANIFEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
