"""Scaffold fraction (scaffold-token share; mechanism diagnostic).

Per Lian 2024, a *scaffold token* is an intermediate BPE merge whose
end-of-training standalone frequency falls below the next-merge-candidate's
frequency — a token BPE produced only as scaffolding for later, more useful
merges. The threshold is operationalized as the ``candidate_freq`` of the last
committed merge (the lowest-frequency pair that still cleared the bar to merge).

The scaffold-instrumentation log is the per-merge-step JSONL emitted by the
``GpeTrainer`` when ``scaffold_log_path`` is set: one header line + one record
per committed merge. Records carry ``standalone: [(token_id, freq_delta)]``
deltas; a token's end-of-training standalone count is the cumulative sum of its
deltas across every record.

This module is the pure math: parse a log into per-step records, apply the
criterion against the surviving vocabulary (merge ids present in the saved
``tokenizer.json``), and bucket the scaffold set by surface form. Log reading +
adapter dispatch live in :mod:`.runner`.

Scaffold is BPE-only by construction (Unigram-LM produces no merge trace);
Unigram cells emit a ``verified_by_construction=True`` zero record so the
matched-pair schema stays symmetric.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

Boundary = Literal["nmb", "mb"]
Arm = Literal["bpe", "unigram"]
UnpairedReason = Literal["conditional_negative_branch", "extras_single_arm_knob"]
SurfaceClass = Literal["bracket_internal", "structural", "atomic"]

_SURFACE_CLASSES: tuple[SurfaceClass, ...] = (
    "bracket_internal",
    "structural",
    "atomic",
)

_STRUCTURAL_CHARS: frozenset[str] = frozenset("()=#./\\:%@+-0123456789")


@dataclass(frozen=True)
class ScaffoldRecord:
    """One per-merge-step log record; mirrors the ``smirk-scaffold-log/v1`` schema.

    ``standalone`` is the per-record delta in standalone counts (positive when a
    token's standalone use rose after a merge that produced it; negative when a
    later merge consumed it as a sub-unit).
    """

    step: int
    pair: tuple[int, int]
    new_id: int
    new_token: str
    candidate_freq: int
    standalone: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class ScaffoldLogHeader:
    """Header (line 0) of the smirk-scaffold-log JSONL."""

    format: str
    min_frequency: int
    vocab_size: int
    merge_brackets: bool
    base_alphabet: tuple[tuple[int, str], ...]

    def alphabet_dict(self) -> dict[int, str]:
        return dict(self.base_alphabet)


def classify_surface_form(token: str) -> SurfaceClass:
    """Bucket a token's surface form for the scaffold breakdown.

    Three mutually exclusive bins, deterministic per surface string:

    - ``bracket_internal``: contains ``[`` or ``]`` (SMILES bracket atoms —
      ``[NH3+]`` and friends, plus their sub-merges).
    - ``structural``: every character is a structural glyph (``()=#./\\:%@+-`` +
      digits 0-9) — SMILES topology / bond types / ring closures, not atom
      identity.
    - ``atomic``: everything else — tokens carrying at least one atom-glyph
      character.

    Self-contained on the surface form alone so Scaffold does not block on
    Jaccard's finer structural tagging.
    """
    if "[" in token or "]" in token:
        return "bracket_internal"
    if all(c in _STRUCTURAL_CHARS for c in token):
        return "structural"
    return "atomic"


def parse_scaffold_log(
    lines: Iterable[str],
) -> tuple[ScaffoldLogHeader, list[ScaffoldRecord]]:
    """Parse a ``smirk-scaffold-log/v1`` JSONL stream.

    Returns the header dataclass and the list of per-step records, in
    file order. Raises :class:`ValueError` when the header is missing,
    malformed, or carries an unexpected ``format`` tag.
    """
    iterator = iter(lines)
    try:
        header_line = next(iterator)
    except StopIteration as exc:
        raise ValueError("scaffold log is empty (no header line)") from exc
    header_payload = json.loads(header_line)
    fmt = header_payload.get("format")
    if fmt != "smirk-scaffold-log/v1":
        raise ValueError(
            f"unexpected scaffold log format {fmt!r}; expected 'smirk-scaffold-log/v1'"
        )
    header = ScaffoldLogHeader(
        format=str(fmt),
        min_frequency=int(header_payload["min_frequency"]),
        vocab_size=int(header_payload["vocab_size"]),
        merge_brackets=bool(header_payload["merge_brackets"]),
        base_alphabet=tuple(
            (int(tid), str(glyph)) for tid, glyph in header_payload["base_alphabet"]
        ),
    )
    records: list[ScaffoldRecord] = []
    for raw in iterator:
        if not raw.strip():
            continue
        payload = json.loads(raw)
        pair = payload["pair"]
        standalone = payload["standalone"]
        records.append(
            ScaffoldRecord(
                step=int(payload["step"]),
                pair=(int(pair[0]), int(pair[1])),
                new_id=int(payload["new_id"]),
                new_token=str(payload["new_token"]),
                candidate_freq=int(payload["candidate_freq"]),
                standalone=tuple((int(tid), int(delta)) for tid, delta in standalone),
            )
        )
    return header, records


def end_of_training_standalone(
    records: Sequence[ScaffoldRecord], *, only_ids: frozenset[int] | None = None
) -> dict[int, int]:
    """Sum standalone-count deltas across every record for each token id.

    The sum is the running standalone count at end of training. With ``only_ids``
    the result is restricted to those ids (each present, defaulting to 0 when no
    record mentions it); with ``only_ids=None`` every id appearing in any
    ``standalone`` delta is included.
    """
    totals: dict[int, int] = defaultdict(int)
    for rec in records:
        for tid, delta in rec.standalone:
            totals[tid] += delta
    if only_ids is None:
        return dict(totals)
    return {tid: totals.get(tid, 0) for tid in only_ids}


def scaffold_threshold(records: Sequence[ScaffoldRecord]) -> int:
    """The Lian-2024 "next-merge-candidate frequency" threshold.

    Operationalized as the ``candidate_freq`` of the last committed
    merge — the lowest-frequency pair that still cleared the bar to be
    merged. Raises :class:`ValueError` on an empty record list (no
    merges committed ⇒ no threshold defined).
    """
    if not records:
        raise ValueError("cannot derive scaffold threshold from zero records")
    return records[-1].candidate_freq


def classify_scaffolds(
    records: Sequence[ScaffoldRecord],
    *,
    surviving_ids: frozenset[int],
) -> frozenset[int]:
    """Apply Lian-2024 to ``records``; return scaffold-token ids in ``surviving_ids``.

    A token id ``t`` is a *scaffold* iff:

    1. ``t`` is among the committed merges present in the saved
       vocabulary (i.e., ``t in surviving_ids``); and
    2. ``t``'s end-of-training standalone count is strictly less than
       :func:`scaffold_threshold` for the log.

    A token id outside ``surviving_ids`` is never a scaffold — it is not
    in the saved vocabulary and so is not "in V" for the scaffold count.
    """
    threshold = scaffold_threshold(records)
    standalone = end_of_training_standalone(records, only_ids=surviving_ids)
    return frozenset(tid for tid, count in standalone.items() if count < threshold)


@dataclass(frozen=True)
class ArmScaffold:
    """Scaffold readings for one arm on one cell.

    BPE arms carry counts from the scaffold log; Unigram arms emit a structural
    zero with ``verified_by_construction=True`` so the cross-arm Δ stays definable.

    ``scaffold_fraction_of_v = scaffold_count / vocab_size``.
    ``surface_form_breakdown`` counts scaffold tokens per :data:`_SURFACE_CLASSES`
    bin (always all three keys). ``threshold`` is the Lian-2024 ``candidate_freq``
    of the final committed merge for BPE, ``None`` for Unigram.
    """

    cell_id: str
    arm: Arm
    boundary: Boundary
    vocab_size: int
    n_merges: int | None
    scaffold_count: int
    scaffold_fraction_of_v: float
    surface_form_breakdown: dict[str, int]
    threshold: int | None
    verified_by_construction: bool
    training_corpus_sha: str
    scaffold_log_sha: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "cell_id": self.cell_id,
            "arm": self.arm,
            "boundary": self.boundary,
            "vocab_size": self.vocab_size,
            "n_merges": self.n_merges,
            "scaffold_count": self.scaffold_count,
            "scaffold_fraction_of_v": self.scaffold_fraction_of_v,
            "surface_form_breakdown": dict(self.surface_form_breakdown),
            "threshold": self.threshold,
            "verified_by_construction": self.verified_by_construction,
            "training_corpus_sha": self.training_corpus_sha,
            "scaffold_log_sha": self.scaffold_log_sha,
        }


@dataclass(frozen=True)
class MatchedPairScaffold:
    """Scaffold for one matched-arm ``(corpus, V, boundary)`` coordinate."""

    pair_key: str
    tier: str
    corpus: str
    vocab_size: int
    boundary: Boundary
    extras_kind: str | None
    extras_label: str | None
    bpe: ArmScaffold
    unigram: ArmScaffold
    delta_scaffold_fraction: float

    @property
    def pair_status(self) -> Literal["matched"]:
        return "matched"

    def as_dict(self) -> dict[str, object]:
        return {
            "pair_key": self.pair_key,
            "pair_status": self.pair_status,
            "tier": self.tier,
            "corpus": self.corpus,
            "vocab_size": self.vocab_size,
            "boundary": self.boundary,
            "extras_kind": self.extras_kind,
            "extras_label": self.extras_label,
            "bpe": self.bpe.as_dict(),
            "unigram": self.unigram.as_dict(),
            "delta_scaffold_fraction": self.delta_scaffold_fraction,
            "missing_arm": None,
            "unpaired_reason": None,
        }


@dataclass(frozen=True)
class UnpairedScaffold:
    """Scaffold for one structurally single-arm coordinate; cross-arm Δ undefined."""

    pair_key: str
    tier: str
    corpus: str
    vocab_size: int
    boundary: Boundary
    extras_kind: str | None
    extras_label: str | None
    present_arm: ArmScaffold
    missing_arm: Arm
    unpaired_reason: UnpairedReason

    @property
    def pair_status(self) -> Literal["single_arm"]:
        return "single_arm"

    def as_dict(self) -> dict[str, object]:
        arm_blocks: dict[str, object] = {
            self.present_arm.arm: self.present_arm.as_dict(),
            self.missing_arm: None,
        }
        return {
            "pair_key": self.pair_key,
            "pair_status": self.pair_status,
            "tier": self.tier,
            "corpus": self.corpus,
            "vocab_size": self.vocab_size,
            "boundary": self.boundary,
            "extras_kind": self.extras_kind,
            "extras_label": self.extras_label,
            "bpe": arm_blocks["bpe"],
            "unigram": arm_blocks["unigram"],
            "delta_scaffold_fraction": None,
            "missing_arm": self.missing_arm,
            "unpaired_reason": self.unpaired_reason,
        }


def empty_surface_breakdown() -> dict[str, int]:
    """Zero counts for every :data:`_SURFACE_CLASSES` bin."""
    return dict.fromkeys(_SURFACE_CLASSES, 0)


def bucket_by_surface_form(
    scaffold_ids: Iterable[int], records: Sequence[ScaffoldRecord]
) -> dict[str, int]:
    """Count scaffold tokens per :func:`classify_surface_form` bin.

    Records are scanned to recover each id's surface form (``new_token``
    on the record that created it). Ids absent from ``records`` are
    skipped — they can never be scaffolds by construction
    (:func:`classify_scaffolds` only returns committed-merge ids).
    """
    surface_by_id: dict[int, str] = {rec.new_id: rec.new_token for rec in records}
    counts = empty_surface_breakdown()
    for tid in scaffold_ids:
        if tid not in surface_by_id:
            continue
        counts[classify_surface_form(surface_by_id[tid])] += 1
    return counts


def compute_bpe_arm_scaffold(
    records: Sequence[ScaffoldRecord],
    *,
    cell_id: str,
    boundary: Boundary,
    vocab_size: int,
    n_merges: int,
    atomic_vocab_size: int,
    training_corpus_sha: str,
    scaffold_log_sha: str,
) -> ArmScaffold:
    """Compute Scaffold for one BPE arm from its scaffold log.

    ``n_merges`` is the count of merges in the saved
    ``tokenizer.json``; surviving merge ids are
    ``range(atomic_vocab_size, atomic_vocab_size + n_merges)``. Any log
    records for merges beyond that range (fired during training but absent
    from the saved vocabulary) are still consulted for the threshold +
    standalone deltas — they shaped the running counts — but only
    surviving ids are eligible to count as scaffolds.
    """
    surviving = frozenset(range(atomic_vocab_size, atomic_vocab_size + n_merges))
    scaffold_ids = classify_scaffolds(records, surviving_ids=surviving)
    breakdown = bucket_by_surface_form(scaffold_ids, records)
    threshold = scaffold_threshold(records)
    fraction = (len(scaffold_ids) / vocab_size) if vocab_size > 0 else 0.0
    return ArmScaffold(
        cell_id=cell_id,
        arm="bpe",
        boundary=boundary,
        vocab_size=vocab_size,
        n_merges=n_merges,
        scaffold_count=len(scaffold_ids),
        scaffold_fraction_of_v=fraction,
        surface_form_breakdown=breakdown,
        threshold=threshold,
        verified_by_construction=False,
        training_corpus_sha=training_corpus_sha,
        scaffold_log_sha=scaffold_log_sha,
    )


def compute_unigram_arm_scaffold(
    *,
    cell_id: str,
    boundary: Boundary,
    vocab_size: int,
    training_corpus_sha: str,
) -> ArmScaffold:
    """Unigram cells produce no merge trace; emit a zero record by construction.

    Unigram-LM does not do merge-greedy training, so the scaffold notion is
    inapplicable; satisfied structurally with ``scaffold_count=0`` and
    ``verified_by_construction=True``.
    """
    return ArmScaffold(
        cell_id=cell_id,
        arm="unigram",
        boundary=boundary,
        vocab_size=vocab_size,
        n_merges=None,
        scaffold_count=0,
        scaffold_fraction_of_v=0.0,
        surface_form_breakdown=empty_surface_breakdown(),
        threshold=None,
        verified_by_construction=True,
        training_corpus_sha=training_corpus_sha,
        scaffold_log_sha=None,
    )


def compute_matched_pair_scaffold(
    bpe: ArmScaffold,
    unigram: ArmScaffold,
    *,
    pair_key: str,
    tier: str,
    corpus: str,
    vocab_size: int,
    boundary: Boundary,
    extras_kind: str | None = None,
    extras_label: str | None = None,
) -> MatchedPairScaffold:
    """Join the two :class:`ArmScaffold` arms into a matched-pair record.

    ``delta_scaffold_fraction = bpe.scaffold_fraction_of_v −
    unigram.scaffold_fraction_of_v``; positive ⇒ BPE produces more
    scaffolding than Unigram-LM. For verified-by-construction Unigram
    arms this is just the BPE fraction.

    Raises:
        ValueError: arm-tag mismatch or boundary disagreement.
    """
    if bpe.arm != "bpe":
        raise ValueError(f"first argument must be the BPE arm; got arm={bpe.arm!r}")
    if unigram.arm != "unigram":
        raise ValueError(
            f"second argument must be the Unigram arm; got arm={unigram.arm!r}"
        )
    if bpe.boundary != unigram.boundary or bpe.boundary != boundary:
        raise ValueError(
            "arm boundaries must match the matched-pair boundary; got "
            f"bpe={bpe.boundary!r}, unigram={unigram.boundary!r}, pair={boundary!r}"
        )
    delta = bpe.scaffold_fraction_of_v - unigram.scaffold_fraction_of_v
    return MatchedPairScaffold(
        pair_key=pair_key,
        tier=tier,
        corpus=corpus,
        vocab_size=vocab_size,
        boundary=boundary,
        extras_kind=extras_kind,
        extras_label=extras_label,
        bpe=bpe,
        unigram=unigram,
        delta_scaffold_fraction=delta,
    )


def compute_unpaired_scaffold(
    arm_record: ArmScaffold,
    *,
    pair_key: str,
    tier: str,
    corpus: str,
    vocab_size: int,
    boundary: Boundary,
    extras_kind: str | None,
    extras_label: str | None,
    missing_arm: Arm,
    unpaired_reason: UnpairedReason,
) -> UnpairedScaffold:
    """Wrap one present arm as an :class:`UnpairedScaffold` record."""
    if missing_arm == arm_record.arm:
        raise ValueError(
            f"missing_arm={missing_arm!r} cannot equal the present arm "
            f"{arm_record.arm!r}"
        )
    if arm_record.boundary != boundary:
        raise ValueError(
            f"arm boundary {arm_record.boundary!r} disagrees with pair {boundary!r}"
        )
    return UnpairedScaffold(
        pair_key=pair_key,
        tier=tier,
        corpus=corpus,
        vocab_size=vocab_size,
        boundary=boundary,
        extras_kind=extras_kind,
        extras_label=extras_label,
        present_arm=arm_record,
        missing_arm=missing_arm,
        unpaired_reason=unpaired_reason,
    )


__all__ = [
    "Arm",
    "ArmScaffold",
    "Boundary",
    "MatchedPairScaffold",
    "ScaffoldLogHeader",
    "ScaffoldRecord",
    "SurfaceClass",
    "UnpairedScaffold",
    "bucket_by_surface_form",
    "classify_scaffolds",
    "classify_surface_form",
    "compute_bpe_arm_scaffold",
    "compute_matched_pair_scaffold",
    "compute_unigram_arm_scaffold",
    "compute_unpaired_scaffold",
    "empty_surface_breakdown",
    "end_of_training_standalone",
    "parse_scaffold_log",
    "scaffold_threshold",
]
