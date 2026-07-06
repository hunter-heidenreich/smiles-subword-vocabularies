"""Tokenizer protocol, its special-id helper, and on-disk meta schema."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from pathlib import Path


@runtime_checkable
class Tokenizer(Protocol):
    """Common surface across every concrete tokenizer kind.

    The ``vocab_size``/``bos_id``/``eos_id``/``pad_id``/``unk_id`` members are
    read-only properties so adapters may implement them as either ``@property``
    or plain attributes; both satisfy the structural check.
    """

    name: str

    @property
    def vocab_size(self) -> int: ...

    def __len__(self) -> int: ...

    @property
    def bos_id(self) -> int: ...

    @property
    def eos_id(self) -> int: ...

    @property
    def pad_id(self) -> int: ...

    @property
    def unk_id(self) -> int | None: ...

    def encode(self, smiles: str, *, add_special_tokens: bool = False) -> list[int]: ...

    def encode_batch(
        self,
        smiles: list[str],
        *,
        add_special_tokens: bool = False,
    ) -> list[list[int]]:
        """Encode a list of SMILES, returning one id-list per input.

        Should beat a per-SMILES :meth:`encode` loop where the engine has a real
        batch path (smirk adapters call the HF fast tokenizer's ``__call__``,
        dispatching to the rust rayon-parallel encoder); else fall back to a
        list comprehension.
        """
        ...

    def decode(self, ids: list[int], *, skip_special_tokens: bool = True) -> str: ...

    def token_to_id(self, tok: str) -> int | None: ...

    def id_to_token(self, idx: int) -> str: ...

    def save(self, path: Path) -> None: ...

    @classmethod
    def load(cls, path: Path) -> Tokenizer: ...


def collect_special_ids(tok: Tokenizer) -> frozenset[int]:
    """Special-token ids (``bos``/``eos``/``pad``/``unk``) to exclude from
    token-frequency counts. ``unk_id`` may be ``None`` (dropped); repeated ids
    collapse.
    """
    candidates = (tok.bos_id, tok.eos_id, tok.pad_id, tok.unk_id)
    return frozenset(c for c in candidates if c is not None)


class TokenizerMeta(BaseModel):
    """On-disk `meta.yaml` companion to every tokenizer artifact.

    `n_merges` is meaningful for GPE kinds only (None for Unigram).
    Corpus-conditional intrinsics are not stored here — they are computed per
    measurement over the held-out split and deposited under `results/data/`.

    `merge_brackets`/`split_structure` are the boundary-policy knobs to
    ``smirk.train_gpe``/``train_unigram``; ``seed_size``, ``max_piece_length``,
    ``n_sub_iterations``, ``shrinking_factor`` the Unigram-LM knobs for
    ``base_kind='smirk_unigram'``. All default to None so already-on-disk meta
    files validate unchanged.

    ``scaffold_log_sha`` is the SHA256 of the sidecar ``scaffold.jsonl``; set
    only for ``smirk_gpe`` cells trained with ``scaffold_log=True``, else None.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    base_kind: str
    vocab_size: int = Field(ge=1)
    training_corpus_sha: str | None = None
    n_merges: int | None = Field(default=None, ge=0)
    merge_brackets: bool | None = None
    split_structure: bool | None = None
    seed_size: int | None = Field(default=None, ge=1)
    max_piece_length: int | None = Field(default=None, ge=1)
    n_sub_iterations: int | None = Field(default=None, ge=1)
    shrinking_factor: float | None = Field(default=None, gt=0.0, lt=1.0)
    scaffold_log_sha: str | None = None
