"""Stage 0 tmQM ingest driver: tmQM CSV release -> raw_v1 Parquet shards.

Cross-corpus probe corpus #4 (transition-metal complexes;
no-UNK fallback stress test).
"""

from __future__ import annotations

from smiles_subword.ingest.csv_corpus import run_csv_ingest_cli

if __name__ == "__main__":
    print(run_csv_ingest_cli(__doc__))
