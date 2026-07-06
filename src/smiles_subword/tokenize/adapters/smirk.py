"""Smirk adapter — wraps ``smirk.SmirkTokenizerFast`` for the project protocol.

Two construction paths share one class:

- **Atomic baseline** (``base_kind="smirk_base"``): the stock 159-token
  ``SmirkTokenizerFast()`` with no corpus pass — the no-merge floor of the V
  trajectory.
- **Merge-trained** (``base_kind="smirk_gpe"``): driven by ``smirk.train_gpe``.
  Chaining ``ref=`` from V₁→V₂→V₃ … captures merge-trajectory checkpoints, each
  call extending the prior vocabulary monotonically rather than re-training.

The shared inference surface lives on
:class:`~smiles_subword.tokenize.adapters._smirk_runtime.SmirkBackedTokenizer`;
this module adds only GPE training, the merge list, and scaffold logging.

The on-disk artifact contract:

    artifacts/tokenizer/<corpus>/<name>/
    ├── tokenizer.json          # via SmirkTokenizerFast.save_pretrained
    ├── tokenizer_config.json   # ditto
    ├── special_tokens_map.json # ditto
    ├── meta.yaml               # TokenizerMeta serialized
    └── merges.txt              # GPE only; ordered "tok1 tok2" merges
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Final, Literal, cast

from smirk import SmirkTokenizerFast, train_gpe

from smiles_subword._hashing import sha256_file
from smiles_subword.tokenize._corpus import write_meta_yaml
from smiles_subword.tokenize.adapters._smirk_runtime import SmirkBackedTokenizer
from smiles_subword.tokenize.base import TokenizerMeta

SmirkBaseKind = Literal["smirk_base", "smirk_gpe"]

ATOMIC_VOCAB_SIZE: Final[int] = 159
"""Stock ``SmirkTokenizerFast()`` reports vocab_size=159 (incl. ``[UNK]``)."""


class SmirkAdapter(SmirkBackedTokenizer):
    """``Tokenizer``-protocol wrapper around ``smirk.SmirkTokenizerFast``."""

    def __init__(
        self,
        *,
        name: str,
        base_kind: SmirkBaseKind,
        tokenizer: SmirkTokenizerFast,
        training_corpus_sha: str | None = None,
        merge_brackets: bool | None = None,
        split_structure: bool | None = None,
        scaffold_log_path: str | Path | None = None,
    ) -> None:
        if base_kind not in ("smirk_base", "smirk_gpe"):
            raise ValueError(
                f"base_kind must be 'smirk_base' or 'smirk_gpe', got {base_kind!r}"
            )
        super().__init__(
            name=name,
            tokenizer=tokenizer,
            training_corpus_sha=training_corpus_sha,
            merge_brackets=merge_brackets,
            split_structure=split_structure,
        )
        self.base_kind: SmirkBaseKind = base_kind
        self._scaffold_log_path = (
            Path(scaffold_log_path) if scaffold_log_path is not None else None
        )
        self._n_merges_cache: int | None = None

    @classmethod
    def atomic(cls, *, name: str = "smirk_base") -> SmirkAdapter:
        """Construct the stock atomic-baseline Smirk tokenizer (no corpus pass)."""
        return cls(
            name=name,
            base_kind="smirk_base",
            tokenizer=SmirkTokenizerFast(),
            training_corpus_sha=None,
        )

    @classmethod
    def train_gpe(
        cls,
        corpus_files: list[str | Path],
        *,
        name: str,
        vocab_size: int,
        min_frequency: int = 2,
        ref: SmirkAdapter | None = None,
        training_corpus_sha: str | None = None,
        merge_brackets: bool = False,
        split_structure: bool = True,
        scaffold_log_path: str | Path | None = None,
    ) -> SmirkAdapter:
        """Train a GPE tokenizer at one V checkpoint, optionally chained from ``ref``.

        Passing ``ref=<prior_adapter>`` extends that tokenizer's merge trajectory
        rather than starting fresh — required for multi-V checkpoint capture.

        ``merge_brackets`` / ``split_structure`` are smirk's boundary-policy knobs:

        - ``merge_brackets=True`` permits merges containing ``[`` / ``]`` (MB);
          ``False`` keeps the bracketed atom opaque (NMB).
        - ``split_structure=False`` permits merges to span the chemistry
          pre-tokenization boundaries (rings, branches).

        ``scaffold_log_path`` enables the logging-only ``GpeTrainer`` scaffold
        instrumentation: a per-merge-step JSONL log streams to that path. It does
        not alter merge selection, so the trained tokenizer is byte-identical to a
        run with the flag unset. The parent directory must already exist.
        """
        if vocab_size < ATOMIC_VOCAB_SIZE:
            raise ValueError(
                f"vocab_size must be >= atomic baseline ({ATOMIC_VOCAB_SIZE}), "
                f"got {vocab_size}"
            )
        files = [str(p) for p in corpus_files]
        ref_tok = ref._tok if ref is not None else None  # noqa: SLF001
        scaffold_path = (
            Path(scaffold_log_path) if scaffold_log_path is not None else None
        )
        tok = train_gpe(
            files,
            ref=ref_tok,
            min_frequency=min_frequency,
            vocab_size=vocab_size,
            merge_brackets=merge_brackets,
            split_structure=split_structure,
            scaffold_log_path=str(scaffold_path) if scaffold_path else None,
        )
        return cls(
            name=name,
            base_kind="smirk_gpe",
            tokenizer=tok,
            training_corpus_sha=training_corpus_sha,
            merge_brackets=merge_brackets,
            split_structure=split_structure,
            scaffold_log_path=scaffold_path,
        )

    @property
    def n_merges(self) -> int | None:
        """Number of BPE merge rules — None for ``smirk_base``.

        Counts only true BPE merges, not atomic-vocab additions GPE makes for
        corpus atoms the stock vocab lacks (e.g. aromatic-iodine ``i`` on
        PubChem). ``vocab_size`` counts both; ``len(model.merges)`` — what
        ``merges.txt`` enumerates — counts merges strictly. The Rust tokenizer
        doesn't expose its merge list via Python, so the count is read from the
        saved ``tokenizer.json`` (cached on first read).
        """
        if self.base_kind != "smirk_gpe":
            return None
        if self._n_merges_cache is None:
            self._n_merges_cache = _count_model_merges(self._tok)
        return self._n_merges_cache

    def save(self, path: Path) -> None:
        """Write tokenizer.json + meta.yaml (+ merges.txt for GPE) to ``path``.

        If trained with ``scaffold_log_path`` set, that ``scaffold.jsonl``'s
        SHA256 is recorded in ``meta.yaml`` and, when it lives elsewhere, copied
        into the artifact directory so the on-disk contract is self-contained.
        """
        path = self._persist_runtime(path)

        if self.base_kind == "smirk_gpe":
            merges = _read_merges(self._tok, path / "tokenizer.json")
            self._n_merges_cache = len(merges)
            _write_merges_txt(merges, path / "merges.txt")

        scaffold_sha = self._materialize_scaffold_log(path)

        meta = TokenizerMeta(
            name=self.name,
            base_kind=self.base_kind,
            vocab_size=self.vocab_size,
            training_corpus_sha=self._training_corpus_sha,
            n_merges=self.n_merges,
            merge_brackets=self._merge_brackets,
            split_structure=self._split_structure,
            scaffold_log_sha=scaffold_sha,
        )
        write_meta_yaml(path / "meta.yaml", meta.model_dump())

    def _materialize_scaffold_log(self, artifact_dir: Path) -> str | None:
        """Copy the scaffold log to ``artifact_dir`` if needed; return its SHA256."""
        log_path = self._scaffold_log_path
        if log_path is None:
            return None
        if not log_path.is_file():
            return None
        target = artifact_dir / "scaffold.jsonl"
        if log_path.resolve() != target.resolve():
            target.write_bytes(log_path.read_bytes())
        return sha256_file(target)

    @classmethod
    def load(cls, path: Path) -> SmirkAdapter:
        """Reload a previously :meth:`save`-d Smirk artifact directory."""
        path = Path(path)
        tok, meta = cls._read_meta_and_tok(
            path, valid_kinds=("smirk_base", "smirk_gpe"), artifact_label="Smirk"
        )
        scaffold_path = path / "scaffold.jsonl"
        return cls(
            name=meta.name,
            base_kind=meta.base_kind,  # type: ignore[arg-type]
            tokenizer=tok,
            training_corpus_sha=meta.training_corpus_sha,
            merge_brackets=meta.merge_brackets,
            split_structure=meta.split_structure,
            scaffold_log_path=scaffold_path if scaffold_path.is_file() else None,
        )


def _read_merges(
    tok: SmirkTokenizerFast, tokenizer_json: Path
) -> list[tuple[str, str]]:
    """Resolve the BPE merge list in ``tokenizer.json`` to ``(tok, tok)`` pairs.

    ``tokenizer.json`` stores merges as ``[id1, id2]`` pairs under
    ``model.merges``; this returns them as decoded token-string pairs in
    the order the tokenizer produced them.
    """
    payload = json.loads(tokenizer_json.read_text())
    raw_merges = payload.get("model", {}).get("merges", [])
    return [
        (
            cast("str", tok.convert_ids_to_tokens(int(a))),
            cast("str", tok.convert_ids_to_tokens(int(b))),
        )
        for a, b in raw_merges
    ]


def _count_model_merges(tok: SmirkTokenizerFast) -> int:
    """Read ``len(model.merges)`` by save-and-introspect.

    SmirkTokenizerFast doesn't expose ``backend_tokenizer.to_str()``, so the
    only path to the BPE merge list is the saved ``tokenizer.json``. Used
    for adapters that haven't yet been ``save``-d (the post-train_gpe meta
    case); save-time call sites use :func:`_read_merges` directly to avoid
    a second save.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tok.save_pretrained(tmp)
        return len(_read_merges(tok, Path(tmp) / "tokenizer.json"))


def _write_merges_txt(merges: list[tuple[str, str]], merges_path: Path) -> None:
    """Write resolved BPE merges to standard HF ``merges.txt`` form.

    Emits a ``#version: 0.2`` header followed by one ``tok1 tok2`` line per
    merge in the order produced by the tokenizer.
    """
    lines = ["#version: 0.2", *(f"{a} {b}" for a, b in merges)]
    merges_path.write_text("\n".join(lines) + "\n")


__all__ = [
    "ATOMIC_VOCAB_SIZE",
    "SmirkAdapter",
    "SmirkBaseKind",
]
