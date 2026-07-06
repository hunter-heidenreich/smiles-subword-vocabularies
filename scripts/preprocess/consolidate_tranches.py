"""tranche_union driver: nested per-tranche raw_v1 -> flat raw_v1 Parquet directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from smiles_subword.config import TrancheUnionConfig
from smiles_subword.preprocess.tranche_union import consolidate_tranches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--no-verify-input-sha",
        action="store_true",
        help="Skip SHA256 verification of tranche shards against per-tranche MANIFESTs",
    )
    args = parser.parse_args()

    cfg = TrancheUnionConfig.from_yaml(args.config)
    result = consolidate_tranches(cfg, verify_input_sha=not args.no_verify_input_sha)
    print(
        f"consolidated {result.n_used}/{result.n_discovered} tranches, "
        f"{len(result.shards)} shards, {result.n_rows:,} rows "
        f"({len(result.excluded)} excluded) to {result.output_dir}"
    )


if __name__ == "__main__":
    main()
