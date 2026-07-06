"""Shared base for the two smirk-backed tokenizer adapters.

``SmirkAdapter`` (``smirk_base`` / ``smirk_gpe``) and ``UnigramSmirkAdapter``
(``smirk_unigram``) both wrap the same ``smirk.SmirkTokenizerFast`` runtime,
differing only in how the vocabulary is *built* (BPE merges vs Unigram-LM
pruning) and which fields they persist. This base owns the identical part: the
``Tokenizer``-protocol inference surface plus save/load I/O mechanics. Separate
from ``base.py`` so the protocol/schema layer stays free of the ``smirk`` binding.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import yaml
from smirk import SmirkTokenizerFast

from smiles_subword.tokenize.base import TokenizerMeta

if TYPE_CHECKING:
    from collections.abc import Iterable


class SmirkBackedTokenizer:
    """``Tokenizer``-protocol inference surface shared by the smirk adapters.

    Subclasses add their training constructors and any kind-specific
    ``save``/``load`` fields (merge list + scaffold for BPE; the four
    Unigram-LM knobs for Unigram), and may set their own ``base_kind``.
    """

    def __init__(
        self,
        *,
        name: str,
        tokenizer: SmirkTokenizerFast,
        training_corpus_sha: str | None = None,
        merge_brackets: bool | None = None,
        split_structure: bool | None = None,
    ) -> None:
        self.name = name
        self._tok = tokenizer
        self._training_corpus_sha = training_corpus_sha
        self._merge_brackets = merge_brackets
        self._split_structure = split_structure

    @property
    def vocab_size(self) -> int:
        return self._tok.vocab_size

    def __len__(self) -> int:
        """Max-id+1 across base + added tokens.

        ``vocab_size`` reports the WordLevel base (e.g. 16389 for
        ``smirk_gpe_16k``) which excludes added specials placed at the
        tail; ``len(self)`` is the full embedding-axis extent (base plus
        the tail specials). The measurements index tokens by id over
        ``range(len(self))``, so they rely on this size — not
        ``vocab_size`` — to reach every id without a gap.
        """
        return len(self._tok)

    @property
    def bos_id(self) -> int:
        return cast("int", self._tok.bos_token_id)

    @property
    def eos_id(self) -> int:
        return cast("int", self._tok.eos_token_id)

    @property
    def pad_id(self) -> int:
        return cast("int", self._tok.pad_token_id)

    @property
    def unk_id(self) -> int | None:
        return cast("int | None", self._tok.unk_token_id)

    @property
    def hf_tokenizer(self) -> SmirkTokenizerFast:
        """The underlying HF-compatible tokenizer."""
        return self._tok

    def encode(self, smiles: str, *, add_special_tokens: bool = False) -> list[int]:
        ids = cast(
            "list[int]", self._tok(smiles, add_special_tokens=False)["input_ids"]
        )
        if add_special_tokens:
            return [self.bos_id, *ids, self.eos_id]
        return ids

    def encode_batch(
        self,
        smiles: list[str],
        *,
        add_special_tokens: bool = False,
    ) -> list[list[int]]:
        """Encode many SMILES via the underlying HF fast tokenizer.

        The HF ``__call__`` on a ``list[str]`` releases the GIL and dispatches
        to the rust-side rayon-parallel ``encode_batch``, faster than per-SMILES
        :meth:`encode` on smirk_gpe. Empty lists short-circuit so callers need
        not special-case the trailing chunk.
        """
        if not smiles:
            return []
        ids_lists = cast(
            "list[list[int]]",
            self._tok(smiles, add_special_tokens=False)["input_ids"],
        )
        if add_special_tokens:
            bos, eos = self.bos_id, self.eos_id
            return [[bos, *ids, eos] for ids in ids_lists]
        return ids_lists

    def decode(self, ids: list[int], *, skip_special_tokens: bool = True) -> str:
        """Decode ``ids`` to a SMILES string.

        Strips the inter-piece spaces ``SmirkTokenizerFast`` inserts at piece
        boundaries during decode, recovering the contiguous glyph string.
        """
        text = cast(
            "str", self._tok.decode(ids, skip_special_tokens=skip_special_tokens)
        )
        return text.replace(" ", "")

    def pretokenize_layer_b(self, smiles: str) -> list[tuple[str, tuple[int, int]]]:
        """Return the SMILES split into Layer-B chunks with character offsets.

        Read-only access to the shared Layer-B chunker. Chunk boundaries are
        independent of the trained model, the ``merge_brackets`` axis, and
        any merges — two cells at matched ``(corpus, V, boundary)`` see
        byte-identical chunks for the same SMILES. Offsets are character-
        based to align with HF ``return_offsets_mapping=True``.

        Requires the Layer-B chunker binding (smirk fork PR #7,
        ``c9e099d``), folded into the production pin ``vtc-2026-05-24``.
        """
        return cast(
            "list[tuple[str, tuple[int, int]]]",
            self._tok.pretokenize_layer_b(smiles),
        )

    def token_to_id(self, tok: str) -> int | None:
        return self._tok.get_vocab().get(tok)

    def id_to_token(self, idx: int) -> str:
        return cast("str", self._tok.convert_ids_to_tokens(idx))

    def _persist_runtime(self, path: Path) -> Path:
        """Make ``path`` and write the HF tokenizer sidecars; return the dir."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._tok.save_pretrained(str(path))
        return path

    @classmethod
    def _read_meta_and_tok(
        cls,
        path: Path,
        *,
        valid_kinds: Iterable[str],
        artifact_label: str,
    ) -> tuple[SmirkTokenizerFast, TokenizerMeta]:
        """Load + validate ``meta.yaml`` and reload the HF tokenizer.

        Shared ``load`` boilerplate; ``artifact_label`` only shapes the
        error message so each subclass keeps its own wording.
        """
        path = Path(path)
        meta = TokenizerMeta.model_validate(
            yaml.safe_load((path / "meta.yaml").read_text())
        )
        if meta.base_kind not in set(valid_kinds):
            raise ValueError(
                f"meta.yaml at {path} is not a {artifact_label} artifact "
                f"(base_kind={meta.base_kind!r})"
            )
        tok = SmirkTokenizerFast.from_pretrained(str(path))
        return tok, meta
