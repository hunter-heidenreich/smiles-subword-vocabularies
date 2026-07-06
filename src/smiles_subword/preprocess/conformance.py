"""Filter a canonicalized corpus to the Smirk base's domain.

The Smirk base (the 158-glyph OpenSMILES regex decomposition) covers any
OpenSMILES-conformant string with no ``[UNK]``. RDKit non-Kekulé canonicalization
can nonetheless emit off-OpenSMILES atoms — aromatic silicon and tellurium —
that decompose to off-base glyphs and route to ``[UNK]``; left in, they leak into
the trained vocabularies (asymmetrically: BPE retains them, Unigram-LM prunes
them), breaking the "no UNK on conformant input" premise.

This stage drops those molecules, closing the corpus under the base. The oracle
is the bare base tokenizer (:class:`smirk.SmirkTokenizerFast`, no merges): a
molecule is conformant iff its decomposition contains no ``[UNK]`` (id ``0``).
Frequency-independent and needs no trained tokenizer, so it runs before training
and catches even RDKit-unparseable offenders (which an RDKit-based check would
error on).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger
from smirk import SmirkTokenizerFast

from smiles_subword.preprocess._io import (
    ShardWriter,
    list_input_shards,
    shard_dicts,
    stage_run,
    write_manifest,
)
from smiles_subword.preprocess.types import ConformanceResult

if TYPE_CHECKING:
    from pathlib import Path

BASE_UNK_ID = 0
_SPECIALS = frozenset({"[UNK]", "[BOS]", "[EOS]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"})
_DEFAULT_BATCH = 20_000
_DEFAULT_TARGET_BYTES = 256 * 2**20


class BaseConformanceOracle:
    """Frequency-independent conformance test against the bare Smirk base.

    Wraps a merge-free :class:`SmirkTokenizerFast`; a molecule is non-conformant
    iff its glyph decomposition contains the base ``[UNK]`` (id ``0``).
    """

    def __init__(self) -> None:
        self._tok = SmirkTokenizerFast()

    def base_glyphs(self) -> set[str]:
        """The 158 chemistry glyphs (the base vocabulary minus special tokens)."""
        return set(self._tok.get_vocab()) - _SPECIALS

    def nonconformant_mask(self, smiles: list[str]) -> list[bool]:
        """Return per-molecule ``True`` where the base cannot cover the string."""
        encoded = cast(
            "list[list[int]]",
            self._tok(smiles, add_special_tokens=False)["input_ids"],
        )
        return [BASE_UNK_ID in ids for ids in encoded]


def offending_atoms(smiles: str, base_glyphs: set[str]) -> list[str]:
    """Coarse RDKit account of why ``smiles`` is non-conformant.

    Returns ``aromatic-<Sym>`` for each aromatic atom whose lowercased symbol is
    not a base glyph (the known class: aromatic Si, Te), ``rdkit-unparseable``
    when RDKit cannot read the string, or ``other`` when no aromatic offender is
    found. Characterization only; the conformance decision is the oracle's.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ["rdkit-unparseable"]
    bad = {
        f"aromatic-{a.GetSymbol()}"
        for a in mol.GetAtoms()
        if a.GetIsAromatic() and a.GetSymbol().lower() not in base_glyphs
    }
    return sorted(bad) if bad else ["other"]


def filter_conformant(
    input_dir: Path,
    output_dir: Path,
    deposit_path: Path,
    *,
    batch_size: int = _DEFAULT_BATCH,
    target_bytes: int = _DEFAULT_TARGET_BYTES,
    characterize: bool = True,
) -> ConformanceResult:
    """Stream ``input_dir`` Parquet, drop non-conformant rows, deposit offenders.

    Conformant rows are written to ``output_dir`` preserving the input schema
    (atomic staging-dir rename via :func:`stage_run`). Every dropped molecule is
    streamed to ``deposit_path`` as JSONL (``{smiles, offenders}``) so any deeper
    inventory is a cheap offline pass over that small file.

    Raises:
        FileNotFoundError: ``input_dir`` holds no Parquet shards.
    """
    RDLogger.DisableLog("rdApp.*")  # pyright: ignore[reportAttributeAccessIssue]
    shards = list_input_shards(input_dir)
    if not shards:
        raise FileNotFoundError(f"no Parquet shards under {input_dir}")
    schema = pq.ParquetFile(shards[0]).schema_arrow

    oracle = BaseConformanceOracle()
    base_glyphs = oracle.base_glyphs()
    n_input = n_dropped = 0

    deposit_path.parent.mkdir(parents=True, exist_ok=True)
    with stage_run(output_dir) as (staging_dir, started_ts):
        writer = ShardWriter(
            staging_dir,
            schema=schema,
            shard_prefix="conformant_v1",
            target_bytes=target_bytes,
        )
        with deposit_path.open("w") as sink:
            for shard in shards:
                for batch in pq.ParquetFile(shard).iter_batches(batch_size):
                    smiles = batch.column("smiles").to_pylist()
                    bad = oracle.nonconformant_mask(smiles)
                    n_input += len(smiles)
                    writer.write_batch(batch.filter(pa.array([not b for b in bad])))
                    for smi, is_bad in zip(smiles, bad, strict=True):
                        if not is_bad:
                            continue
                        n_dropped += 1
                        cats = offending_atoms(smi, base_glyphs) if characterize else []
                        sink.write(
                            json.dumps({"smiles": smi, "offenders": cats}) + "\n"
                        )
        writer.close_current()
        manifest = {
            "schema": "conformant_v1",
            "input_dir": str(input_dir),
            "n_base_glyphs": len(base_glyphs),
            "started_ts": started_ts.isoformat() + "Z",
            "n_input_rows": n_input,
            "n_dropped": n_dropped,
            "n_output_rows": n_input - n_dropped,
            "shards": shard_dicts(writer.shards),
        }
        write_manifest(staging_dir, manifest)

    return ConformanceResult(
        n_input_rows=n_input,
        n_kept=n_input - n_dropped,
        n_dropped=n_dropped,
        output_dir=output_dir,
        deposit_path=deposit_path,
    )


__all__ = [
    "BaseConformanceOracle",
    "ConformanceResult",
    "filter_conformant",
    "offending_atoms",
]
