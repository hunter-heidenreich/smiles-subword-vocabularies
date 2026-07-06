"""Stage 0 COCONUT ingest driver: natural-products CSV -> raw_v1 Parquet shards.

Cross-corpus probe corpus #5 (natural products, long-tail
scaffolds).
"""

from __future__ import annotations

from smiles_subword.ingest.csv_corpus import run_csv_ingest_cli

if __name__ == "__main__":
    print(run_csv_ingest_cli(__doc__))
