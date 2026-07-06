"""Derive tmQM's OpenSMILES eval set from raw_v1 dative-bond serialization.

tmQM ships ~93% dative-bond SMILES (``->`` / ``<-``), non-OpenSMILES and so
``[UNK]``-routing on the ``>`` glyph. This driver converts each via
:func:`smiles_subword.preprocess.dative.dative_to_opensmiles`, exact-string dedups,
and writes ``data/processed/tmqm/opensmiles_v1/`` (one shard + a deterministic
``MANIFEST.yaml``) for the OOD-eval runner. It deliberately does *not* route
through ``canon_dedup`` --- that stage's full sanitize would re-perceive the
metal coordination and restore the dative bonds.

Example::

    uv run python scripts/preprocess/derive_tmqm_opensmiles.py
"""

from __future__ import annotations

import sys

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from rdkit import Chem, RDLogger

from smiles_subword._hashing import sha256_file
from smiles_subword.paths import processed_corpus_dir
from smiles_subword.preprocess.dative import dative_to_opensmiles
from smiles_subword.tokenize._corpus import iter_smiles_from_parquet

RAW_DIR = processed_corpus_dir("tmqm") / "raw_v1"
OUT_DIR = processed_corpus_dir("tmqm") / "opensmiles_v1"


def main() -> int:
    """Convert, dedup, and deposit the OpenSMILES eval set; print a summary."""
    RDLogger.DisableLog("rdApp.*")  # pyright: ignore[reportAttributeAccessIssue]

    n_input = n_failed = 0
    seen: set[str] = set()
    for smiles in iter_smiles_from_parquet(RAW_DIR):
        n_input += 1
        out = dative_to_opensmiles(smiles)
        if out is None:
            n_failed += 1
            continue
        seen.add(out)

    kept = sorted(seen)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    shard = OUT_DIR / "opensmiles_v1-00000.parquet"
    pq.write_table(pa.table({"smiles": kept}), shard, compression="zstd")

    manifest = {
        "schema": "opensmiles_v1",
        "name": "tmqm",
        "derivation": "dative_to_opensmiles (DATIVE family -> SINGLE, "
        "UpdatePropertyCache, canonical isomeric SMILES)",
        "rdkit_version": Chem.rdBase.rdkitVersion,
        "n_input_rows": n_input,
        "n_rdkit_rejected": n_failed,
        "n_deduplicated": n_input - n_failed - len(kept),
        "n_output_rows": len(kept),
        "shards": [
            {
                "name": shard.name,
                "sha256": sha256_file(shard),
            }
        ],
    }
    (OUT_DIR / "MANIFEST.yaml").write_text(yaml.safe_dump(manifest, sort_keys=True))

    print(
        f"derived {len(kept):,}/{n_input:,} rows "
        f"({n_failed:,} RDKit-rejected, "
        f"{manifest['n_deduplicated']:,} duplicates removed) to {OUT_DIR} "
        f"[rdkit {manifest['rdkit_version']}]"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
