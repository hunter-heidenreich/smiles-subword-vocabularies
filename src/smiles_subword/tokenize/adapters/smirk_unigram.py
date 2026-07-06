"""Smirk Unigram-LM adapter — wraps ``smirk.train_unigram`` for the protocol.

The Unigram-LM sibling of
:class:`~smiles_subword.tokenize.adapters.smirk.SmirkAdapter`: same
``smirk.SmirkTokenizerFast`` runtime and shared inference surface, but a
Unigram-LM model fitted by ``smirk.train_unigram`` rather than BPE ``train_gpe``.

The on-disk artifact contract:

    artifacts/tokenizer/<corpus>/<name>/
    ├── tokenizer.json          # via SmirkTokenizerFast.save_pretrained
    ├── tokenizer_config.json   # ditto
    ├── special_tokens_map.json # ditto
    └── meta.yaml               # TokenizerMeta serialized

Unigram-LM has no merge list, so — unlike ``smirk_gpe`` — there is no
``merges.txt`` and ``n_merges`` is None.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, Literal

from smirk import SmirkTokenizerFast, train_unigram

from smiles_subword.tokenize._corpus import write_meta_yaml
from smiles_subword.tokenize.adapters._smirk_runtime import SmirkBackedTokenizer
from smiles_subword.tokenize.base import TokenizerMeta

UNIGRAM_BASE_KIND: Final[Literal["smirk_unigram"]] = "smirk_unigram"

DEFAULT_SEED_SIZE: Final[int] = 1_000_000
"""Unigram seed-pool cap; the smirk-fork default used in the study."""

DEFAULT_MAX_PIECE_LENGTH: Final[int] = 16
"""SentencePiece's default Unigram max piece length; the headline value. The
former fork default of 128 is now only the top rung of the ``max_piece_length``
sensitivity ladder."""

DEFAULT_N_SUB_ITERATIONS: Final[int] = 2
"""Unigram EM sub-iterations; the smirk-fork default used in the study."""

DEFAULT_SHRINKING_FACTOR: Final[float] = 0.75
"""Unigram shrinking factor; the smirk-fork default used in the study."""


class UnigramSmirkAdapter(SmirkBackedTokenizer):
    """``Tokenizer``-protocol wrapper around a Smirk Unigram-LM tokenizer."""

    def __init__(
        self,
        *,
        name: str,
        tokenizer: SmirkTokenizerFast,
        training_corpus_sha: str | None = None,
        merge_brackets: bool | None = None,
        split_structure: bool | None = None,
        seed_size: int | None = None,
        max_piece_length: int | None = None,
        n_sub_iterations: int | None = None,
        shrinking_factor: float | None = None,
    ) -> None:
        super().__init__(
            name=name,
            tokenizer=tokenizer,
            training_corpus_sha=training_corpus_sha,
            merge_brackets=merge_brackets,
            split_structure=split_structure,
        )
        self.base_kind: Final[Literal["smirk_unigram"]] = UNIGRAM_BASE_KIND
        self._seed_size = seed_size
        self._max_piece_length = max_piece_length
        self._n_sub_iterations = n_sub_iterations
        self._shrinking_factor = shrinking_factor

    @classmethod
    def train_unigram(
        cls,
        corpus_files: list[str | Path],
        *,
        name: str,
        vocab_size: int,
        min_frequency: int = 0,
        merge_brackets: bool = False,
        split_structure: bool = True,
        seed_size: int = DEFAULT_SEED_SIZE,
        max_piece_length: int = DEFAULT_MAX_PIECE_LENGTH,
        n_sub_iterations: int = DEFAULT_N_SUB_ITERATIONS,
        shrinking_factor: float = DEFAULT_SHRINKING_FACTOR,
        training_corpus_sha: str | None = None,
    ) -> UnigramSmirkAdapter:
        """Train a Unigram-LM tokenizer at one vocab-size target.

        ``seed_size`` / ``max_piece_length`` / ``n_sub_iterations`` /
        ``shrinking_factor`` are the Unigram-LM knobs the smirk fork exposes;
        their defaults are the values used in the study. ``min_frequency`` is
        accepted for parity with ``smirk_gpe`` and forwarded to the Rust trainer,
        but has no effect on the Unigram-LM algorithm.
        """
        files = [str(p) for p in corpus_files]
        tok = train_unigram(
            files,
            min_frequency=min_frequency,
            vocab_size=vocab_size,
            merge_brackets=merge_brackets,
            split_structure=split_structure,
            seed_size=seed_size,
            max_piece_length=max_piece_length,
            n_sub_iterations=n_sub_iterations,
            shrinking_factor=shrinking_factor,
        )
        return cls(
            name=name,
            tokenizer=tok,
            training_corpus_sha=training_corpus_sha,
            merge_brackets=merge_brackets,
            split_structure=split_structure,
            seed_size=seed_size,
            max_piece_length=max_piece_length,
            n_sub_iterations=n_sub_iterations,
            shrinking_factor=shrinking_factor,
        )

    def save(self, path: Path) -> None:
        """Write tokenizer.json + the HF config sidecars + meta.yaml to ``path``."""
        path = self._persist_runtime(path)

        meta = TokenizerMeta(
            name=self.name,
            base_kind=self.base_kind,
            vocab_size=self.vocab_size,
            training_corpus_sha=self._training_corpus_sha,
            merge_brackets=self._merge_brackets,
            split_structure=self._split_structure,
            seed_size=self._seed_size,
            max_piece_length=self._max_piece_length,
            n_sub_iterations=self._n_sub_iterations,
            shrinking_factor=self._shrinking_factor,
        )
        write_meta_yaml(path / "meta.yaml", meta.model_dump())

    @classmethod
    def load(cls, path: Path) -> UnigramSmirkAdapter:
        """Reload a previously :meth:`save`-d Unigram artifact directory."""
        path = Path(path)
        tok, meta = cls._read_meta_and_tok(
            path, valid_kinds=(UNIGRAM_BASE_KIND,), artifact_label="Smirk Unigram"
        )
        return cls(
            name=meta.name,
            tokenizer=tok,
            training_corpus_sha=meta.training_corpus_sha,
            merge_brackets=meta.merge_brackets,
            split_structure=meta.split_structure,
            seed_size=meta.seed_size,
            max_piece_length=meta.max_piece_length,
            n_sub_iterations=meta.n_sub_iterations,
            shrinking_factor=meta.shrinking_factor,
        )


__all__ = [
    "DEFAULT_MAX_PIECE_LENGTH",
    "DEFAULT_N_SUB_ITERATIONS",
    "DEFAULT_SEED_SIZE",
    "DEFAULT_SHRINKING_FACTOR",
    "UNIGRAM_BASE_KIND",
    "UnigramSmirkAdapter",
]
