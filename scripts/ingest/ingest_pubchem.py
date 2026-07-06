"""Stage 0 PubChem ingest driver: CID-SMILES.gz -> raw_v1 Parquet shards."""

from __future__ import annotations

from smiles_subword.ingest.csv_corpus import run_csv_ingest_cli

if __name__ == "__main__":
    print(run_csv_ingest_cli(__doc__))
