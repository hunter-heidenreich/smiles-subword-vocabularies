"""REAL-Space ingest driver: Enamine REAL `.cxsmiles` -> raw_v1 shards.

Globs the pre-staged Enamine REAL Sample `.cxsmiles` file set under the
config's `raw_dir` and streams it into raw_v1 Parquet shards. Acquisition is
out of band — stage the files before running.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from smiles_subword.config import RealSpaceCorpusConfig
from smiles_subword.ingest.real_space import ingest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg = RealSpaceCorpusConfig.from_yaml(args.config)
    result = ingest(cfg)
    print(
        f"wrote {result.n_shards} shards, {result.n_rows:,} rows to {result.output_dir}"
    )


if __name__ == "__main__":
    main()
