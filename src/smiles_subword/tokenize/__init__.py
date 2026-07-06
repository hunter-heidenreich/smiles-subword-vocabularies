"""Tokenizer protocol, concrete kinds, and registry."""

from __future__ import annotations

from typing import TYPE_CHECKING

from smiles_subword.tokenize._corpus import (
    ensure_smi_cache as _ensure_smi_cache,
)
from smiles_subword.tokenize._corpus import (
    iter_smiles_from_parquet,
    materialize_smiles_txt,
    training_corpus_sha,
)
from smiles_subword.tokenize.adapters.smirk import (
    ATOMIC_VOCAB_SIZE,
    SmirkAdapter,
)
from smiles_subword.tokenize.adapters.smirk_unigram import UnigramSmirkAdapter
from smiles_subword.tokenize.base import Tokenizer, TokenizerMeta
from smiles_subword.tokenize.grid import (
    GridCell,
    load_grid_manifest,
)

if TYPE_CHECKING:
    from smiles_subword.config import TokenizerConfig


def build_tokenizer(cfg: TokenizerConfig) -> Tokenizer:
    """Construct a fitted/loaded tokenizer from a config.

    The three implemented kinds are ``smirk_base``, ``smirk_gpe``, and
    ``smirk_unigram`` (the Unigram-LM trainer arm).
    """
    if cfg.kind == "smirk_base":
        return _build_smirk_base(cfg)
    if cfg.kind == "smirk_gpe":
        return _build_smirk_gpe(cfg)
    if cfg.kind == "smirk_unigram":
        return _build_smirk_unigram(cfg)
    raise NotImplementedError(f"unhandled tokenizer kind={cfg.kind!r}")


def _build_smirk_base(cfg: TokenizerConfig) -> SmirkAdapter:
    return SmirkAdapter.atomic(name=cfg.name)


def _build_smirk_gpe(cfg: TokenizerConfig) -> SmirkAdapter:
    if cfg.training_input is None:
        raise ValueError("smirk_gpe kind requires training_input")
    if cfg.vocab_size is None:
        raise ValueError("smirk_gpe kind requires vocab_size")
    sha = training_corpus_sha(cfg.training_input)
    corpus_txt = _ensure_smi_cache(cfg.training_input, sha=sha)
    ref = SmirkAdapter.load(cfg.ref_artifact_dir) if cfg.ref_artifact_dir else None
    scaffold_log_path = None
    if cfg.scaffold_log:
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        scaffold_log_path = cfg.output_dir / "scaffold.jsonl"
    return SmirkAdapter.train_gpe(
        [corpus_txt],
        name=cfg.name,
        vocab_size=cfg.vocab_size,
        min_frequency=cfg.min_frequency,
        ref=ref,
        training_corpus_sha=sha,
        merge_brackets=cfg.merge_brackets,
        split_structure=cfg.split_structure,
        scaffold_log_path=scaffold_log_path,
    )


def _build_smirk_unigram(cfg: TokenizerConfig) -> UnigramSmirkAdapter:
    from smiles_subword.tokenize.adapters.smirk_unigram import (
        DEFAULT_MAX_PIECE_LENGTH,
        DEFAULT_N_SUB_ITERATIONS,
        DEFAULT_SEED_SIZE,
        DEFAULT_SHRINKING_FACTOR,
    )

    if cfg.training_input is None:
        raise ValueError("smirk_unigram kind requires training_input")
    if cfg.vocab_size is None:
        raise ValueError("smirk_unigram kind requires vocab_size")
    sha = training_corpus_sha(cfg.training_input)
    corpus_txt = _ensure_smi_cache(cfg.training_input, sha=sha)
    return UnigramSmirkAdapter.train_unigram(
        [corpus_txt],
        name=cfg.name,
        vocab_size=cfg.vocab_size,
        min_frequency=cfg.min_frequency,
        training_corpus_sha=sha,
        merge_brackets=cfg.merge_brackets,
        split_structure=cfg.split_structure,
        seed_size=cfg.seed_size if cfg.seed_size is not None else DEFAULT_SEED_SIZE,
        max_piece_length=(
            cfg.max_piece_length
            if cfg.max_piece_length is not None
            else DEFAULT_MAX_PIECE_LENGTH
        ),
        n_sub_iterations=(
            cfg.n_sub_iterations
            if cfg.n_sub_iterations is not None
            else DEFAULT_N_SUB_ITERATIONS
        ),
        shrinking_factor=(
            cfg.shrinking_factor
            if cfg.shrinking_factor is not None
            else DEFAULT_SHRINKING_FACTOR
        ),
    )


__all__ = [
    "ATOMIC_VOCAB_SIZE",
    "GridCell",
    "SmirkAdapter",
    "Tokenizer",
    "TokenizerMeta",
    "UnigramSmirkAdapter",
    "build_tokenizer",
    "iter_smiles_from_parquet",
    "load_grid_manifest",
    "materialize_smiles_txt",
    "training_corpus_sha",
]
