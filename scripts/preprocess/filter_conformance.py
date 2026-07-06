"""conformant_v1 driver: canon_dedup_v1 -> base-conformant Parquet shards.

Drops molecules outside the Smirk base's domain (non-OpenSMILES atoms RDKit
emits, e.g. aromatic Si/Te), closing the corpus under the 158-glyph OpenSMILES
base by construction. Every dropped molecule is deposited to a JSONL sidecar for
inventory.

Example::

    uv run python scripts/preprocess/filter_conformance.py \
        --input-dir data/processed/pubchem/canon_dedup_v1_full \
        --output-dir data/processed/pubchem/conformant_v1_full
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from smiles_subword.preprocess.conformance import filter_conformant


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n", 1)[0])
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--deposit",
        type=Path,
        default=None,
        help="JSONL sidecar for dropped molecules "
        "(default: <output-dir>/dropped_nonconformant.jsonl)",
    )
    args = parser.parse_args(argv)

    deposit = args.deposit or args.output_dir / "dropped_nonconformant.jsonl"
    result = filter_conformant(args.input_dir, args.output_dir, deposit)
    print(
        f"kept {result.n_kept:,}/{result.n_input_rows:,} rows "
        f"({result.n_dropped:,} non-conformant dropped, {result.drop_rate:.6%}) "
        f"to {result.output_dir}; offenders -> {result.deposit_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
