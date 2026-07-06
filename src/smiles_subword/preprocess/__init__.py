"""Preprocessing for the subword-vocabulary study.

The corpus-prep pipeline that turns raw ingested shards into the trained-on and
held-out corpora the study measures: RDKit canonicalization + exact-string dedup
(:mod:`~smiles_subword.preprocess.canon_dedup`,
:mod:`~smiles_subword.preprocess.canonicalize_minimal`), the OpenSMILES-conformance
filter (:mod:`~smiles_subword.preprocess.conformance`), dative-bond rewriting
(:mod:`~smiles_subword.preprocess.dative`), ZINC-22 tranche consolidation
(:mod:`~smiles_subword.preprocess.tranche_union`), hash subsampling
(:mod:`~smiles_subword.preprocess.hash_subsample`), and the deterministic
held-out test split (:mod:`~smiles_subword.preprocess.holdout_split`). Import the
submodules directly; the package namespace deliberately re-exports nothing.
"""

from __future__ import annotations
