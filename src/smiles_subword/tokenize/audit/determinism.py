"""Per-arm determinism digests for a trained tokenizer artifact.

Every grid tokenizer is trained twice and its reproducibility asserted *per
arm*:

- **BPE** — the artifact must be byte-identical: same ``tokenizer.json`` and
  same ``merges.txt``.
- **Unigram** — the *piece set* must be identical; the raw ``tokenizer.json``
  is not byte-reproducible (process-level Rust ``HashMap`` order), so the
  glyph-sequence set is the load-bearing comparison.

Arm-agnostic computation only: digests one artifact directory and compares two
such digests. No training, subprocess, or deposition — :mod:`.determinism_verify`
orchestrates those.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from smiles_subword._hashing import sha256_bytes, sha256_file

if TYPE_CHECKING:
    from pathlib import Path

_ARMS: tuple[str, ...] = ("bpe", "unigram")


@dataclass(frozen=True)
class ArtifactDigest:
    """SHA256 fingerprints of one trained tokenizer artifact directory.

    ``tokenizer_json_sha`` and ``merges_txt_sha`` are raw file digests and
    define the BPE byte-identity check. The three glyph digests come from
    ``model.vocab`` for the Unigram arm only — ``piece_set_sha`` over the
    *sorted* glyph-sequence set is the load-bearing one; ``vocab_order_sha``
    and ``log_probs_sha`` are diagnostics (they jitter even when the piece set
    holds).
    """

    algo: str
    tokenizer_json_sha: str
    merges_txt_sha: str | None
    piece_set_sha: str | None
    vocab_order_sha: str | None
    log_probs_sha: str | None

    def as_dict(self) -> dict[str, object]:
        """JSON-ready payload."""
        return {
            "algo": self.algo,
            "tokenizer_json_sha": self.tokenizer_json_sha,
            "merges_txt_sha": self.merges_txt_sha,
            "piece_set_sha": self.piece_set_sha,
            "vocab_order_sha": self.vocab_order_sha,
            "log_probs_sha": self.log_probs_sha,
        }


@dataclass(frozen=True)
class DeterminismResult:
    """Outcome of comparing a canonical artifact against one rerun.

    ``rerun_spread`` is the symmetric-difference piece count for a Unigram
    mismatch (0 otherwise) — the integer the measurements consume
    in place of the assumed-zero ``J`` / ``J_struct`` variance.
    """

    arm: str
    deterministic: bool
    mismatch_kind: str | None
    rerun_spread: int
    canonical: ArtifactDigest
    rerun: ArtifactDigest

    def as_dict(self) -> dict[str, object]:
        """JSON-ready payload; the two digests nest under ``*_digest`` keys."""
        return {
            "arm": self.arm,
            "deterministic": self.deterministic,
            "mismatch_kind": self.mismatch_kind,
            "rerun_spread": self.rerun_spread,
            "canonical_digest": self.canonical.as_dict(),
            "rerun_digest": self.rerun.as_dict(),
        }


def _sha256(data: bytes) -> str:
    """SHA256 of in-memory bytes (the serialized glyph/score digests).

    Whole-file digests use :func:`sha256_file` instead.
    """
    return sha256_bytes(data)


def _unigram_vocab(tokenizer_json: Path) -> list[dict[str, object]]:
    """Return ``model.vocab`` of a Unigram ``tokenizer.json``.

    Raises:
        ValueError: the serialized model is not a Unigram model.
    """
    model = json.loads(tokenizer_json.read_text())["model"]
    if model.get("type") != "Unigram":
        raise ValueError(f"{tokenizer_json}: model type is not Unigram")
    vocab = model["vocab"]
    assert isinstance(vocab, list)
    return vocab


def unigram_glyph_set(artifact_dir: Path) -> frozenset[tuple[str, ...]]:
    """Return a Unigram artifact's piece set — each piece as a glyph tuple."""
    vocab = _unigram_vocab(artifact_dir / "tokenizer.json")
    return frozenset(tuple(cast("list[str]", entry["glyphs"])) for entry in vocab)


def digest_artifact(artifact_dir: Path, *, algo: str) -> ArtifactDigest:
    """Digest a trained tokenizer artifact directory.

    For ``bpe`` the determinism-defining digests are the raw SHA256 of
    ``tokenizer.json`` and ``merges.txt``. For ``unigram`` they are the glyph
    digests of ``model.vocab`` (the raw JSON is not byte-reproducible).

    Raises:
        ValueError: ``algo`` is not a known arm.
    """
    if algo not in _ARMS:
        raise ValueError(f"algo must be 'bpe' or 'unigram'; got {algo!r}")

    tokenizer_json = artifact_dir / "tokenizer.json"
    tokenizer_json_sha = sha256_file(tokenizer_json)

    if algo == "bpe":
        return ArtifactDigest(
            algo=algo,
            tokenizer_json_sha=tokenizer_json_sha,
            merges_txt_sha=sha256_file(artifact_dir / "merges.txt"),
            piece_set_sha=None,
            vocab_order_sha=None,
            log_probs_sha=None,
        )

    vocab = _unigram_vocab(tokenizer_json)
    glyphs = [cast("list[str]", entry["glyphs"]) for entry in vocab]
    scored = [
        (cast("list[str]", entry["glyphs"]), cast("float", entry["score"]))
        for entry in vocab
    ]
    return ArtifactDigest(
        algo=algo,
        tokenizer_json_sha=tokenizer_json_sha,
        merges_txt_sha=None,
        piece_set_sha=_sha256(json.dumps(sorted(glyphs)).encode("utf-8")),
        vocab_order_sha=_sha256(json.dumps(glyphs).encode("utf-8")),
        log_probs_sha=_sha256(json.dumps(sorted(scored)).encode("utf-8")),
    )


def compare_artifacts(
    canonical: ArtifactDigest,
    rerun: ArtifactDigest,
    *,
    pieces: tuple[frozenset[tuple[str, ...]], frozenset[tuple[str, ...]]] | None = None,
) -> DeterminismResult:
    """Compare a canonical artifact digest against one rerun digest.

    BPE is ``deterministic`` iff ``tokenizer.json`` and ``merges.txt`` are
    both byte-identical. Unigram is ``deterministic`` iff the piece set is
    identical; ``pieces`` — the canonical and rerun glyph sets — is required
    so a mismatch can be sized as the symmetric-difference piece count.

    Raises:
        ValueError: the two digests are for different arms, or a Unigram
            comparison was requested without ``pieces``.
    """
    if canonical.algo != rerun.algo:
        raise ValueError(
            f"cannot compare a {canonical.algo!r} digest against a "
            f"{rerun.algo!r} digest"
        )
    arm = canonical.algo

    if arm == "bpe":
        deterministic = (
            canonical.tokenizer_json_sha == rerun.tokenizer_json_sha
            and canonical.merges_txt_sha == rerun.merges_txt_sha
        )
        return DeterminismResult(
            arm=arm,
            deterministic=deterministic,
            mismatch_kind=None if deterministic else "bpe_byte",
            rerun_spread=0,
            canonical=canonical,
            rerun=rerun,
        )

    if pieces is None:
        raise ValueError("unigram comparison requires the canonical/rerun piece sets")
    canonical_pieces, rerun_pieces = pieces
    deterministic = canonical_pieces == rerun_pieces
    return DeterminismResult(
        arm=arm,
        deterministic=deterministic,
        mismatch_kind=None if deterministic else "unigram_piece_set",
        rerun_spread=len(canonical_pieces ^ rerun_pieces),
        canonical=canonical,
        rerun=rerun,
    )


__all__ = [
    "ArtifactDigest",
    "DeterminismResult",
    "compare_artifacts",
    "digest_artifact",
    "unigram_glyph_set",
]
