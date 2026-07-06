"""Per-pair non-canonicity pass over a subsample of the held-out split.

For each held-out molecule, build its identity-preserving rewrite orbit (K
randomized SMILES, a ring-digit relabel, a Kekule form, an all-explicit-H form,
the OpenBabel canonical SMILES), encode the canonical string and every variant
through each arm, and reduce to the per-molecule fertility-dispersion and
bag-instability the :mod:`noncanon` math aggregates. Variant generation is
arm-independent — built once per pair, reused for both arms.

The ``obcanon`` axis needs OpenBabel (the optional ``crosstoolkit`` extra): it is
imported lazily and, when absent, the axis is skipped.

The pass runs over the first ``MOLECULE_LIMIT`` molecules of the held-out split
(order fixed, tied to ``eval_split_sha``) — a seeded subsample large enough for
tight molecule-resampled CIs. RDKit's randomizer is the restricted,
augmentation-realistic distribution (Arus-Pous et al. 2019).
"""

from __future__ import annotations

import logging
from collections import Counter
from functools import cache
from typing import TYPE_CHECKING, Any

from rdkit import Chem, RDLogger

from smiles_subword.tokenize.measure._cells import (
    eval_split_sha,
    iter_test_split,
)
from smiles_subword.tokenize.measure.noncanon.math import (
    ArmNoncanon,
    Boundary,
    MatchedPairNoncanon,
    PerMoleculeNoncanon,
    compute_arm_noncanon,
    compute_matched_pair_noncanon,
)

if TYPE_CHECKING:
    from smiles_subword.tokenize.adapters.smirk import SmirkAdapter
    from smiles_subword.tokenize.adapters.smirk_unigram import UnigramSmirkAdapter

RDLogger.DisableLog("rdApp.*")  # type: ignore[attr-defined]

logger = logging.getLogger(__name__)

MOLECULE_LIMIT = 10000
K_RANDOM = 4
SINGLE_AXES = ("ringperm", "kekule", "explicitH", "obcanon")

__all__ = [
    "K_RANDOM",
    "MOLECULE_LIMIT",
    "build_variant_orbit",
    "openbabel_available",
    "openbabel_canon",
    "per_molecule_readings",
    "run_pair_noncanon",
    "run_single_arm_noncanon",
]


@cache
def _pybel() -> Any:
    """Lazily load OpenBabel's pybel, or None when the crosstoolkit extra is absent.

    Memoized so the missing-dependency warning fires at most once per process.
    """
    try:
        from openbabel import pybel
    except ImportError:
        logger.warning(
            "openbabel not installed; the obcanon (cross-toolkit canonical) axis "
            "is skipped. Install it with: uv sync --extra crosstoolkit"
        )
        return None
    pybel.ob.obErrorLog.SetOutputLevel(0)  # silence OpenBabel's stderr chatter
    return pybel


def openbabel_available() -> bool:
    """True iff OpenBabel is importable (the obcanon axis can be generated)."""
    return _pybel() is not None


def openbabel_canon(canon: str) -> str | None:
    """OpenBabel's canonical SMILES for ``canon``, gated to identity-preservation.

    Returns None when OpenBabel is unavailable, emits nothing, or its output fails
    to round-trip back through RDKit to the same canonical molecule. When the two
    toolkits agree the canonical string is returned unchanged --- a genuine zero
    the cross-toolkit axis must average in, not a skip.
    """
    pybel = _pybel()
    if pybel is None:
        return None
    try:
        ob = pybel.readstring("smi", canon).write("can").strip().split("\t")[0]
    except Exception:  # noqa: BLE001 - OpenBabel parse/convert failure: skip molecule
        return None
    if not ob:
        return None
    m = Chem.MolFromSmiles(ob)
    if m is None or Chem.MolToSmiles(m) != canon:
        return None
    return ob


def randomized(mol: Chem.Mol, n: int, seed: int) -> list[str]:
    """n reproducible randomized SMILES (RDKit's restricted/augmentation-realistic)."""
    try:
        return list(Chem.MolToRandomSmilesVect(mol, n, randomSeed=seed))
    except Exception:  # noqa: BLE001 - older RDKit: fall back to per-call doRandom
        return [Chem.MolToSmiles(mol, canonical=False, doRandom=True) for _ in range(n)]


def kekulized(mol: Chem.Mol) -> str | None:
    """Explicit-Kekule SMILES, or None when the molecule has no aromaticity."""
    if not any(a.GetIsAromatic() for a in mol.GetAtoms()):
        return None
    try:
        km = Chem.Mol(mol)
        Chem.Kekulize(km, clearAromaticFlags=True)
        return Chem.MolToSmiles(km, kekuleSmiles=True)
    except Exception:  # noqa: BLE001 - skip unkekulizable molecules
        return None


def explicit_h(mol: Chem.Mol) -> str | None:
    """All-hydrogens-explicit SMILES (``C`` -> ``[CH4]``)."""
    try:
        return Chem.MolToSmiles(mol, allHsExplicit=True)
    except Exception:  # noqa: BLE001
        return None


def perturb_rings(canon: str) -> str | None:
    """Cyclic-shift ring-closure digits at bracket depth 0, verified same molecule.

    Returns None for molecules with no single-digit ring closures, ``%nn``
    two-digit closures (skipped), or a relabel that fails to round-trip to the
    identical molecule. The token count is invariant by construction, so this is
    the measurement's mild calibration floor.
    """
    if "%" in canon:
        return None
    out: list[str] = []
    depth = 0
    touched = False
    for ch in canon:
        if ch == "[":
            depth += 1
            out.append(ch)
        elif ch == "]":
            depth = max(0, depth - 1)
            out.append(ch)
        elif depth == 0 and ch.isdigit():
            out.append(str((int(ch) % 9) + 1))
            touched = True
        else:
            out.append(ch)
    if not touched:
        return None
    perturbed = "".join(out)
    m = Chem.MolFromSmiles(perturbed)
    if m is None or Chem.MolToSmiles(m) != canon:
        return None
    return perturbed


def _multiset_jaccard(a: list[int], b: list[int]) -> float:
    ca, cb = Counter(a), Counter(b)
    inter = sum((ca & cb).values())
    union = sum((ca | cb).values())
    return inter / union if union else 1.0


def build_variant_orbit(
    corpus: str, *, limit: int = MOLECULE_LIMIT, k_random: int = K_RANDOM
) -> tuple[list[str], list[tuple[int, str]]]:
    """Build the flat (string, key) orbit over the first ``limit`` held-out molecules.

    ``key`` is ``(molecule_index, axis)`` with ``axis`` one of ``canonical`` /
    ``random`` (x ``k_random``) / ``ringperm`` / ``kekule`` / ``explicitH`` /
    ``obcanon``; a variant is omitted when the molecule does not support that axis
    (or, for obcanon, when OpenBabel is absent or the round-trip gate fails).
    """
    flat: list[str] = []
    keys: list[tuple[int, str]] = []
    stream = iter_test_split(corpus, limit_molecules=limit)
    for i, smi in enumerate(stream):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        canon = Chem.MolToSmiles(mol)
        flat.append(canon)
        keys.append((i, "canonical"))
        for rs in randomized(mol, k_random, seed=(i % (2**31 - 1)) + 1):
            flat.append(rs)
            keys.append((i, "random"))
        for axis, v in (
            ("ringperm", perturb_rings(canon)),
            ("kekule", kekulized(mol)),
            ("explicitH", explicit_h(mol)),
            ("obcanon", openbabel_canon(canon)),
        ):
            if v is not None:
                flat.append(v)
                keys.append((i, axis))
    return flat, keys


def per_molecule_readings(
    ids: list[list[int]], keys: list[tuple[int, str]]
) -> list[PerMoleculeNoncanon]:
    """Reduce one arm's encoded orbit into per-molecule non-canonicity readings."""
    grouped: dict[int, dict[str, list[tuple[int, list[int]]]]] = {}
    for (mi, axis), tid in zip(keys, ids, strict=True):
        grouped.setdefault(mi, {}).setdefault(axis, []).append((len(tid), tid))

    out: list[PerMoleculeNoncanon] = []
    for d in grouped.values():
        canon = d.get("canonical")
        if not canon:
            continue
        f0, ids0 = canon[0]
        if f0 == 0:
            continue
        axis_dfert: dict[str, float] = {}
        axis_bag: dict[str, float] = {}
        rand_ferts: list[int] = []
        for axis in ("random", *SINGLE_AXES):
            variants = d.get(axis)
            if not variants:
                continue
            axis_dfert[axis] = sum(abs(fv - f0) / f0 for fv, _ in variants) / len(
                variants
            )
            axis_bag[axis] = sum(
                1 - _multiset_jaccard(ids0, idv) for _, idv in variants
            ) / len(variants)
            if axis == "random":
                rand_ferts = [fv for fv, _ in variants]
        out.append(
            PerMoleculeNoncanon(
                canon_fert=f0,
                rand_fert_mean=(
                    sum(rand_ferts) / len(rand_ferts) if rand_ferts else float(f0)
                ),
                axis_dfert=axis_dfert,
                axis_bag=axis_bag,
            )
        )
    return out


def run_pair_noncanon(
    bpe: SmirkAdapter | UnigramSmirkAdapter,
    unigram: SmirkAdapter | UnigramSmirkAdapter,
    *,
    pair_key: str,
    tier: str,
    corpus: str,
    vocab_size: int,
    boundary: Boundary,
    bpe_cell_id: str,
    unigram_cell_id: str,
    bpe_training_corpus_sha: str,
    unigram_training_corpus_sha: str,
    extras_kind: str | None = None,
    extras_label: str | None = None,
    eval_split_sha_value: str | None = None,
    limit_molecules: int = MOLECULE_LIMIT,
) -> MatchedPairNoncanon:
    """Build the rewrite orbit once and aggregate non-canonicity for both arms."""
    sha = eval_split_sha_value or eval_split_sha(corpus)
    flat, keys = build_variant_orbit(corpus, limit=limit_molecules)

    bpe_arm = compute_arm_noncanon(
        per_molecule_readings(bpe.encode_batch(flat, add_special_tokens=False), keys),
        cell_id=bpe_cell_id,
        arm="bpe",
        boundary=boundary,
        training_corpus_sha=bpe_training_corpus_sha,
        eval_split_sha=sha,
    )
    ul_arm = compute_arm_noncanon(
        per_molecule_readings(
            unigram.encode_batch(flat, add_special_tokens=False), keys
        ),
        cell_id=unigram_cell_id,
        arm="unigram",
        boundary=boundary,
        training_corpus_sha=unigram_training_corpus_sha,
        eval_split_sha=sha,
    )
    return compute_matched_pair_noncanon(
        bpe_arm,
        ul_arm,
        pair_key=pair_key,
        tier=tier,
        corpus=corpus,
        vocab_size=vocab_size,
        boundary=boundary,
        extras_kind=extras_kind,
        extras_label=extras_label,
    )


def run_single_arm_noncanon(
    adapter: SmirkAdapter | UnigramSmirkAdapter,
    *,
    arm: str,
    corpus: str,
    cell_id: str,
    boundary: Boundary,
    training_corpus_sha: str,
    eval_split_sha_value: str | None = None,
    limit_molecules: int = MOLECULE_LIMIT,
) -> ArmNoncanon:
    """Build the rewrite orbit and aggregate non-canonicity for one arm."""
    sha = eval_split_sha_value or eval_split_sha(corpus)
    flat, keys = build_variant_orbit(corpus, limit=limit_molecules)
    return compute_arm_noncanon(
        per_molecule_readings(
            adapter.encode_batch(flat, add_special_tokens=False), keys
        ),
        cell_id=cell_id,
        arm="bpe" if arm == "bpe" else "unigram",
        boundary=boundary,
        training_corpus_sha=training_corpus_sha,
        eval_split_sha=sha,
    )
