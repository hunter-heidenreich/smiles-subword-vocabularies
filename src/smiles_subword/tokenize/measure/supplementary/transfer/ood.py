"""Out-of-distribution eval: read the PubChem generalist on adversarial corpora.

Extends the transfer matrix (:mod:`.math`) past the four in-grid corpora onto
deliberately adversarial, in-spec OOD chemistry (rare-but-valid OpenSMILES the
training corpora barely contain). There is no native tokenizer for these corpora,
so this is not a penalty matrix: it asks whether the cross-arm divergence (BPE
coarser, Unigram-LM finer) survives OOD, measured as the relative fertility gap,
with atom-level OOV as coverage sanity check.

Scope: the PubChem generalist at the headline ``V=1024``, ``nmb`` cell, both arms,
read on each OOD corpus's full ``canon_dedup_v1`` output (the whole canonicalized
set is the eval set — no train/test split to honor on a corpus we never trained
on). Fertility and its bootstrap CI reuse :func:`compute_transfer_record`
verbatim, so the OOD numbers are on the same footing as the transfer matrix.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from smiles_subword.paths import (
    RESULTS_DATA_DIR,
    processed_corpus_dir,
    tokenizer_artifact_dir,
)
from smiles_subword.tokenize._corpus import (
    iter_smiles_from_parquet,
    manifest_shard_fingerprint,
)
from smiles_subword.tokenize.measure._cells import load_cell_adapter
from smiles_subword.tokenize.measure._glyphmap import glyph_count_map
from smiles_subword.tokenize.measure.supplementary.transfer.math import (
    Arm,
    Boundary,
    TransferRecord,
    compute_transfer_record,
)
from smiles_subword.tokenize.measure.supplementary.transfer.runner import (
    ARMS,
    HEADLINE_BOUNDARY,
    HEADLINE_V,
    cell_name,
    count_per_molecule,
)

if TYPE_CHECKING:
    from pathlib import Path

GENERALIST_CORPUS = "pubchem"
"""The diverse-corpus tokenizer read on every OOD eval corpus."""

OOD_CORPORA: tuple[str, ...] = ("tmqm", "cycpeptmpdb")
"""OOD eval corpora, in display order. Extend as eval sets land."""

OOD_EVAL_DIR = RESULTS_DATA_DIR / "ood_eval"

_EVAL_STAGE = {"tmqm": "opensmiles_v1"}
"""Per-corpus eval-set stage; tmQM uses the dative-reset derivation (see
:mod:`smiles_subword.preprocess.dative`), others the standard ``canon_dedup_v1``."""


def ood_canon_dir(corpus: str) -> Path:
    """Return a corpus's full eval-set Parquet directory."""
    stage = _EVAL_STAGE.get(corpus, "canon_dedup_v1")
    return processed_corpus_dir(corpus) / stage


def eval_corpus_fingerprint(corpus: str) -> str:
    """Stable 32-hex fingerprint of an OOD eval set from its stage manifest.

    Fingerprints the eval stage's ``MANIFEST.yaml`` shard set via the shared
    :func:`smiles_subword.tokenize._corpus.manifest_shard_fingerprint` (sorted
    per-shard SHA256s, BLAKE2b-128) — the same recipe as
    :func:`smiles_subword.tokenize.measure._cells.eval_split_sha`, so a
    re-canonicalization that changes the shard set invalidates every downstream
    OOD record. Hashing only the shard SHAs (not the manifest text) keeps it
    stable across re-derivations of byte-identical data.
    """
    return manifest_shard_fingerprint(ood_canon_dir(corpus) / "MANIFEST.yaml")


def run_ood_eval(
    *,
    eval_corpus: str,
    arm: Arm,
    vocab_size: int = HEADLINE_V,
    boundary: Boundary = HEADLINE_BOUNDARY,
    limit: int | None = None,
) -> TransferRecord:
    """Read the PubChem ``(arm, V, boundary)`` cell on ``eval_corpus`` (full set).

    Mirrors :func:`.runner.run_transfer` but streams the eval corpus's whole
    ``canon_dedup_v1`` output rather than a held-out test split: an OOD corpus we
    never trained on has no contamination risk, so every molecule is fair game
    and more of them sharpens the fertility estimate. ``limit`` takes the first
    N molecules (tests / debug only).
    """
    name = cell_name(arm, vocab_size, boundary)
    adapter = load_cell_adapter(GENERALIST_CORPUS, name)
    glyph_map = glyph_count_map(tokenizer_artifact_dir(GENERALIST_CORPUS, name), arm)
    unk_id = getattr(adapter, "unk_id", None)

    smiles = iter_smiles_from_parquet(ood_canon_dir(eval_corpus))
    if limit is not None:
        smiles = (s for i, s in enumerate(smiles) if i < limit)

    per_molecule = count_per_molecule(adapter, glyph_map, unk_id, smiles)

    return compute_transfer_record(
        train_corpus=GENERALIST_CORPUS,
        eval_corpus=eval_corpus,
        arm=arm,
        vocab_size=vocab_size,
        boundary=boundary,
        per_molecule=per_molecule,
        train_corpus_sha="",
        eval_split_sha=eval_corpus_fingerprint(eval_corpus),
    )


__all__ = [
    "ARMS",
    "GENERALIST_CORPUS",
    "HEADLINE_BOUNDARY",
    "HEADLINE_V",
    "OOD_CORPORA",
    "OOD_EVAL_DIR",
    "eval_corpus_fingerprint",
    "ood_canon_dir",
    "run_ood_eval",
]
