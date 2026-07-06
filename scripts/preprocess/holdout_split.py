"""holdout_split driver: canon_dedup_v1 -> train/ + test/ Parquet subdirs."""

from __future__ import annotations

import argparse
from pathlib import Path

from smiles_subword.config import HoldoutSplitConfig
from smiles_subword.preprocess.holdout_split import split_train_test


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--no-verify-input-sha",
        action="store_true",
        help="Skip SHA256 verification of input shards against source MANIFEST.yaml",
    )
    args = parser.parse_args()

    cfg = HoldoutSplitConfig.from_yaml(args.config)
    result = split_train_test(cfg, verify_input_sha=not args.no_verify_input_sha)
    print(
        f"split {result.n_input_rows:,} rows -> "
        f"train {result.n_train:,} / test {result.n_test:,} "
        f"(threshold {result.effective_threshold:.6f}, "
        f"cap_bound={result.cap_bound}) to {result.output_dir}"
    )


if __name__ == "__main__":
    main()
