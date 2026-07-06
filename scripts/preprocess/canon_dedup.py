"""canon_dedup_v1 driver: raw_v1 -> canon_dedup_v1 Parquet shards."""

from __future__ import annotations

import argparse
from pathlib import Path

from smiles_subword.config import CanonDedupConfig
from smiles_subword.preprocess.canon_dedup import canon_dedup


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--no-verify-input-sha",
        action="store_true",
        help="Skip SHA256 verification of input shards against source MANIFEST.yaml",
    )
    args = parser.parse_args()

    cfg = CanonDedupConfig.from_yaml(args.config)
    result = canon_dedup(cfg, verify_input_sha=not args.no_verify_input_sha)
    print(
        f"wrote {len(result.shards)} shards, "
        f"{result.n_output_rows:,}/{result.n_input_rows:,} rows kept "
        f"({result.n_rdkit_rejected:,} RDKit-rejected, "
        f"{result.rdkit_rejection_rate:.4%} rate; "
        f"{result.n_duplicates:,} duplicates removed) "
        f"to {result.output_dir} [rdkit {result.rdkit_version}]"
    )


if __name__ == "__main__":
    main()
