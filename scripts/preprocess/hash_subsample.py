"""hash_subsample driver: canon_dedup_v1 -> acceptance-band-subsampled shards."""

from __future__ import annotations

import argparse
from pathlib import Path

from smiles_subword.config import HashSubsampleConfig
from smiles_subword.preprocess.hash_subsample import hash_subsample


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--no-verify-input-sha",
        action="store_true",
        help="Skip SHA256 verification of input shards against source MANIFEST.yaml",
    )
    args = parser.parse_args()

    cfg = HashSubsampleConfig.from_yaml(args.config)
    result = hash_subsample(cfg, verify_input_sha=not args.no_verify_input_sha)
    print(
        f"wrote {len(result.shards)} shards, "
        f"{result.n_kept:,}/{result.n_input_rows:,} rows kept "
        f"(acceptance band {result.acceptance_fraction:.6f}, "
        f"target {result.target_n:,}) to {result.output_dir}"
    )


if __name__ == "__main__":
    main()
