"""Stage 0 ZINC-22 ingest driver: smoke (one tranche) or multi-tranche (many).

Discriminates on the YAML payload's `tranches_path` key: if present, the
config is parsed as `Zinc22MultiTrancheConfig` and dispatched to
`ingest_multi_tranche`; otherwise it's parsed as `Zinc22CorpusConfig` and
dispatched to the single-tranche `ingest`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from smiles_subword.config import Zinc22CorpusConfig, Zinc22MultiTrancheConfig
from smiles_subword.ingest.zinc22 import ingest
from smiles_subword.ingest.zinc22_multi_tranche import ingest_multi_tranche


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--no-verify-sha",
        action="store_true",
        help="Skip SHA reconciliation against data/MANIFEST.yaml",
    )
    parser.add_argument(
        "--no-fetch",
        action="store_true",
        help="Skip download even if the local raw file is missing",
    )
    parser.add_argument(
        "--tranches-limit",
        type=int,
        default=None,
        help="Multi-tranche only: cap on the number of tranches processed "
        "(sizing run). Ignored for single-tranche configs.",
    )
    args = parser.parse_args()

    with args.config.open() as fh:
        payload = yaml.safe_load(fh)

    if "tranches_path" in payload:
        cfg = Zinc22MultiTrancheConfig.model_validate(payload)
        result = ingest_multi_tranche(
            cfg,
            verify_input_sha=not args.no_verify_sha,
            fetch=not args.no_fetch,
            limit=args.tranches_limit,
        )
        n_bytes = sum(t.n_bytes for t in result.tranches)
        print(
            f"multi-tranche: {result.n_tranches} tranches, "
            f"{result.n_shards} shards, {result.n_rows:,} rows, "
            f"{n_bytes / 2**30:.2f} GiB -> {result.output_root}"
        )
        if result.failures:
            print(f"  {len(result.failures)} failure(s):")
            for f in result.failures:
                print(f"    {f.tranche_id}: {f.reason}")
    else:
        cfg = Zinc22CorpusConfig.model_validate(payload)
        single = ingest(
            cfg,
            verify_input_sha=not args.no_verify_sha,
            fetch=not args.no_fetch,
        )
        print(
            f"single: {single.n_shards} shards, {single.n_rows:,} rows -> "
            f"{single.output_dir}"
        )


if __name__ == "__main__":
    main()
