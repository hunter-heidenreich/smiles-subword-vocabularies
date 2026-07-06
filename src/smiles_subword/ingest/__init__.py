"""Stage 0 ingest: external corpora -> raw_v1 Parquet shards.

Per-corpus backends turn a downloaded source into a `raw_v1` Parquet directory:
the config-driven CSV/TSV driver (:mod:`~smiles_subword.ingest.csv_corpus`, for
PubChem / COCONUT / CycPeptMPDB / tmQM), the ZINC-22 single- and multi-tranche
fetchers (:mod:`~smiles_subword.ingest.zinc22`,
:mod:`~smiles_subword.ingest.zinc22_multi_tranche`), and the REAL-Space file-set
ingest (:mod:`~smiles_subword.ingest.real_space`). The shared `raw_v1` contract,
CSV reader, and shard/manifest writers live in
:mod:`~smiles_subword.ingest._common`. Import submodules directly; the package
re-exports nothing.
"""

from __future__ import annotations
