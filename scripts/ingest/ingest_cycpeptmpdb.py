"""Stage 0 CycPeptMPDB ingest driver: CSV release -> raw_v1 Parquet shards.

OOD-eval probe corpus (cyclic peptides; length / fertility stress test).
"""

from __future__ import annotations

from smiles_subword.ingest.csv_corpus import run_csv_ingest_cli

if __name__ == "__main__":
    print(run_csv_ingest_cli(__doc__))
