"""Cross-corpus generalization: the train x eval transfer matrix.

A tokenizer trained on corpus ``A`` is applied to corpus ``B``'s held-out test
split. The diagonal (``A == B``) is the on-domain reading from Fertility; the
off-diagonal cells are the generalization signal.

Two readings per cell:

* **fertility** — mean tokens per held-out molecule (and mean glyphs per token).
  smirk's shared 165-glyph base makes coverage essentially solved, so the
  cross-corpus penalty shows up as sequence-length inflation.
* **atom-level OOV** — the fraction of emitted tokens that are ``[UNK]`` (a glyph
  in ``B`` absent from ``A``'s base alphabet, id 0) and the fraction of molecules
  carrying at least one; near-zero except the ``-> PubChem`` direction, reported
  precisely so the near-absence of OOV is on the record.

Fertility carries a 95% percentile bootstrap CI over molecule-resamples, seeded
from the ``(train, eval, arm, V, boundary)`` key. Kept separate from the
matched-pair Fertility schema: transfer is a ``train != eval`` grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from smiles_subword.config import algo_to_engine_tag
from smiles_subword.paths import RESULTS_DATA_DIR
from smiles_subword.tokenize.measure._bootstrap import (
    CI_LEVEL,
    N_BOOTSTRAP_RESAMPLES,
    bootstrap_ratio_ci,
    bootstrap_seed,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

Arm = Literal["bpe", "unigram"]
Boundary = Literal["nmb", "mb"]

TRANSFER_DIR = RESULTS_DATA_DIR / "transfer"


@dataclass(frozen=True)
class PerMoleculeTransfer:
    """Per-molecule transfer counts for one (train, eval) tokenizer pass."""

    n_tokens: int
    n_glyphs: int
    n_unk: int


@dataclass(frozen=True)
class TransferRecord:
    """One (train_corpus, eval_corpus, arm, V, boundary) transfer reading."""

    train_corpus: str
    eval_corpus: str
    arm: Arm
    vocab_size: int
    boundary: Boundary
    n_molecules: int
    total_tokens: int
    fertility_mean: float
    fertility_ci: tuple[float, float]
    glyphs_per_token_mean: float
    oov_token_rate: float
    oov_molecule_rate: float
    train_corpus_sha: str
    eval_split_sha: str
    bootstrap_seed: int
    n_resamples: int

    @property
    def is_diagonal(self) -> bool:
        """True when the tokenizer is read on its own training corpus."""
        return self.train_corpus == self.eval_corpus

    @property
    def cell_key(self) -> str:
        """Filesystem-safe key for the per-cell deposit."""
        arm_tag = algo_to_engine_tag(self.arm)
        return (
            f"{self.train_corpus}__{self.eval_corpus}__"
            f"{arm_tag}_v{self.vocab_size}_{self.boundary}"
        )

    def as_dict(self) -> dict[str, object]:
        """JSON-ready payload."""
        return {
            "train_corpus": self.train_corpus,
            "eval_corpus": self.eval_corpus,
            "arm": self.arm,
            "vocab_size": self.vocab_size,
            "boundary": self.boundary,
            "is_diagonal": self.is_diagonal,
            "n_molecules": self.n_molecules,
            "total_tokens": self.total_tokens,
            "fertility_mean": self.fertility_mean,
            "fertility_ci": list(self.fertility_ci),
            "glyphs_per_token_mean": self.glyphs_per_token_mean,
            "oov_token_rate": self.oov_token_rate,
            "oov_molecule_rate": self.oov_molecule_rate,
            "train_corpus_sha": self.train_corpus_sha,
            "eval_split_sha": self.eval_split_sha,
            "bootstrap_seed": self.bootstrap_seed,
            "n_resamples": self.n_resamples,
        }


def compute_transfer_record(
    *,
    train_corpus: str,
    eval_corpus: str,
    arm: Arm,
    vocab_size: int,
    boundary: Boundary,
    per_molecule: Sequence[PerMoleculeTransfer],
    train_corpus_sha: str,
    eval_split_sha: str,
    n_resamples: int = N_BOOTSTRAP_RESAMPLES,
) -> TransferRecord:
    """Aggregate per-molecule transfer counts into a :class:`TransferRecord`.

    Pure over the per-molecule counts: fertility is the mean token count,
    glyphs-per-token the pooled ratio, and the two OOV rates are token-level
    and molecule-level. The fertility CI is a molecule-resample bootstrap.

    Raises:
        ValueError: ``per_molecule`` is empty (no held-out molecules).
    """
    n = len(per_molecule)
    if n == 0:
        raise ValueError("per_molecule is empty; no held-out molecules to read")

    token_counts = [m.n_tokens for m in per_molecule]
    total_tokens = sum(token_counts)
    total_glyphs = sum(m.n_glyphs for m in per_molecule)
    total_unk = sum(m.n_unk for m in per_molecule)
    n_mol_with_unk = sum(1 for m in per_molecule if m.n_unk > 0)

    key = f"{train_corpus}__{eval_corpus}__{arm}_v{vocab_size}_{boundary}"
    seed = bootstrap_seed(key)

    return TransferRecord(
        train_corpus=train_corpus,
        eval_corpus=eval_corpus,
        arm=arm,
        vocab_size=vocab_size,
        boundary=boundary,
        n_molecules=n,
        total_tokens=total_tokens,
        fertility_mean=total_tokens / n,
        fertility_ci=bootstrap_ratio_ci(
            token_counts, [1] * n, seed=seed, n_resamples=n_resamples
        ),
        glyphs_per_token_mean=(
            total_glyphs / total_tokens if total_tokens else float("nan")
        ),
        oov_token_rate=(total_unk / total_tokens if total_tokens else 0.0),
        oov_molecule_rate=n_mol_with_unk / n,
        train_corpus_sha=train_corpus_sha,
        eval_split_sha=eval_split_sha,
        bootstrap_seed=seed,
        n_resamples=n_resamples,
    )


__all__ = [
    "CI_LEVEL",
    "N_BOOTSTRAP_RESAMPLES",
    "TRANSFER_DIR",
    "Arm",
    "Boundary",
    "PerMoleculeTransfer",
    "TransferRecord",
    "compute_transfer_record",
]
