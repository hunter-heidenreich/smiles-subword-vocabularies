"""Per-pair functional-bond-locality pass over the held-out split.

Streams every held-out molecule once, encodes through both arms, and asks for
each multiply-bonded heteroatom whether the arm kept the bond token-local (the
:mod:`fg_alignment` math). Single-arm coordinates run the same pass through one
adapter (locality is within-arm). The chemistry — bond enumeration,
heteroatom/partner classification, atom-to-glyph-span mapping — is RDKit-side
and lives here.

The atom-span mapping leans on RDKit indexing atoms in SMILES parse order, so
the k-th atom-glyph (left to right) is atom index ``k``; a molecule whose
recovered atom-span count disagrees with the parsed atom count, or whose encoded
glyph length disagrees with the segmenter, is dropped and counted rather than
guessed at.
"""

from __future__ import annotations

from itertools import accumulate
from typing import TYPE_CHECKING

from rdkit import Chem, RDLogger

from smiles_subword.paths import tokenizer_artifact_dir
from smiles_subword.tokenize._batched import ENCODE_BATCH_SIZE
from smiles_subword.tokenize.measure._cells import (
    eval_split_sha,
    iter_test_split,
)
from smiles_subword.tokenize.measure._glyphmap import (
    glyph_count_map,
    glyph_tuple_map,
)
from smiles_subword.tokenize.measure.fg_alignment.math import (
    FUNCTIONAL_CLASSES,
    ArmFgAlignment,
    Boundary,
    MatchedPairFgAlignment,
    PerMoleculeFgLocality,
    compute_arm_fg_alignment,
    compute_matched_pair_fg_alignment,
)

GlyphTuple = tuple[str, ...]

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from smiles_subword.tokenize.adapters.smirk import SmirkAdapter
    from smiles_subword.tokenize.adapters.smirk_unigram import UnigramSmirkAdapter

    Segmenter = Callable[[str], GlyphTuple | None]

RDLogger.DisableLog("rdApp.*")  # type: ignore[attr-defined]

# OpenSMILES organic-subset atom glyphs that stand alone outside a bracket
# (aromatic lowercase included); every other element is bracket-only.
ATOM_GLYPHS: frozenset[str] = frozenset("BCNOPSFIbcnops") | {"Cl", "Br"}
_BOND_OP = {Chem.BondType.DOUBLE: "=", Chem.BondType.TRIPLE: "#"}

__all__ = [
    "ATOM_GLYPHS",
    "atom_spans",
    "build_glyph_segmenter",
    "mappable_functional_bonds",
    "molecule_arms_locality",
    "run_pair_fg_alignment",
    "run_single_arm_fg_alignment",
]


def build_glyph_segmenter(glyphs: frozenset[str]) -> Segmenter:
    """Bracket-aware longest-match segmenter over the recovered glyph alphabet.

    Only the organic-subset ``Cl`` / ``Br`` are multi-character atoms outside a
    bracket; every other multi-char glyph occurs only inside ``[...]``. Honoring
    bracket depth disambiguates a bare ``Cn`` piece into ``C`` + aromatic ``n``
    rather than copernicium. Returns ``None`` for an out-of-alphabet character.

    Matching is length-bounded (longest candidate first, up to the longest glyph
    in the alphabet) against a set, so it is ``O(maxlen)`` per character rather
    than ``O(|alphabet|)`` — the held-out splits run to ~10^6 molecules.
    """
    glyph_set = glyphs
    max_len = max((len(g) for g in glyph_set), default=1)
    always_multi = {"Cl", "Br"}

    def seg(s: str) -> GlyphTuple | None:
        out: list[str] = []
        i = 0
        depth = 0
        n = len(s)
        while i < n:
            matched: str | None = None
            for length in range(min(max_len, n - i), 0, -1):
                candidate = s[i : i + length]
                if length > 1 and depth == 0 and candidate not in always_multi:
                    continue
                if candidate in glyph_set:
                    matched = candidate
                    break
            if matched is None:
                return None
            out.append(matched)
            i += len(matched)
            if matched == "[":
                depth += 1
            elif matched == "]":
                depth = max(0, depth - 1)
        return tuple(out)

    return seg


def atom_spans(glyphs: GlyphTuple) -> list[tuple[int, int]] | None:
    """Glyph span of each atom in string order (= RDKit atom-index order).

    A bracket atom ``[...]`` is one atom spanning all its glyphs; a bare
    atom-glyph is one atom of one glyph; structural glyphs are glue. Returns
    ``None`` on an unbalanced bracket.
    """
    spans: list[tuple[int, int]] = []
    i = 0
    n = len(glyphs)
    while i < n:
        g = glyphs[i]
        if g == "[":
            j = i + 1
            while j < n and glyphs[j] != "]":
                j += 1
            if j >= n:
                return None
            spans.append((i, j + 1))
            i = j + 1
        elif g in ATOM_GLYPHS:
            spans.append((i, i + 1))
            i += 1
        else:
            i += 1
    return spans


def _classify_bond(bond: Chem.Bond, op: str) -> tuple[str, int] | None:
    """Return ``(class_label, salient_atom_idx)`` for one multiple bond.

    The salient atom is the heteroatom (non-carbon); when both endpoints are
    heteroatoms (nitro ``N=O``, sulfonyl ``S=O``) it is the terminal one (lower
    degree). A pure carbon double/triple bond (``C=C`` / ``C#C``) is not a
    functional heteroatom bond and returns ``None``.
    """
    a, b = bond.GetBeginAtom(), bond.GetEndAtom()
    a_c = a.GetSymbol() == "C"
    b_c = b.GetSymbol() == "C"
    if a_c and b_c:
        return None
    if a_c != b_c:
        hetero, partner = (b, a) if a_c else (a, b)
    elif a.GetDegree() <= b.GetDegree():
        hetero, partner = a, b
    else:
        hetero, partner = b, a
    label = f"{partner.GetSymbol()}{op}{hetero.GetSymbol()}"
    if label not in FUNCTIONAL_CLASSES:
        label = "other"
    return label, hetero.GetIdx()


def mappable_functional_bonds(
    mol: Chem.Mol, spans: list[tuple[int, int]], glyphs: GlyphTuple
) -> list[tuple[str, int]]:
    """Return ``(class_label, test_position)`` for each locatable functional bond.

    ``test_position`` is the inter-glyph position between the salient heteroatom
    and its adjacent ``=`` / ``#`` bond glyph; the arm is local on this bond iff
    that position is not a token boundary. A bond whose op glyph is not adjacent
    to the heteroatom's span (rare; e.g. a bracketed heteroatom written apart
    from its bond) is dropped — arm-independently, so the two arms stay
    comparable.
    """
    out: list[tuple[str, int]] = []
    n = len(glyphs)
    for bond in mol.GetBonds():
        op = _BOND_OP.get(bond.GetBondType())
        if op is None:
            continue
        classified = _classify_bond(bond, op)
        if classified is None:
            continue
        label, hidx = classified
        h0, h1 = spans[hidx]
        if h0 - 1 >= 0 and glyphs[h0 - 1] == op:
            out.append((label, h0))
        elif h1 < n and glyphs[h1] == op:
            out.append((label, h1))
    return out


def _token_cuts(ids: list[int], glyph_len: dict[int, int]) -> tuple[set[int], int]:
    """Interior cut positions and total glyph length from per-token glyph counts."""
    prefix = list(accumulate(glyph_len.get(t, 1) for t in ids))
    if not prefix:
        return set(), 0
    return set(prefix[:-1]), prefix[-1]


def _count_arm(
    mappable: list[tuple[str, int]], cuts: set[int]
) -> PerMoleculeFgLocality:
    class_bonds: dict[str, int] = {}
    class_local: dict[str, int] = {}
    n_local = 0
    for label, pos in mappable:
        class_bonds[label] = class_bonds.get(label, 0) + 1
        if pos not in cuts:
            n_local += 1
            class_local[label] = class_local.get(label, 0) + 1
    return PerMoleculeFgLocality(
        n_bonds=len(mappable),
        n_local=n_local,
        class_bonds=class_bonds,
        class_local=class_local,
    )


def molecule_arms_locality(
    smi: str,
    arm_ids: list[list[int]],
    arm_glyph_len: list[dict[int, int]],
    seg: Segmenter,
) -> list[PerMoleculeFgLocality] | None:
    """Per-molecule functional-bond locality for one or more arms.

    ``arm_ids`` / ``arm_glyph_len`` are parallel lists (one entry per arm) of the
    molecule's encoded ids and that arm's ``{id: glyph_count}`` map. Returns one
    :class:`PerMoleculeFgLocality` per arm, or ``None`` if the molecule is
    dropped (unparseable, unsegmentable, atom-count mismatch, or an arm's encoded
    glyph length disagreeing with the segmenter).
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    glyphs = seg(smi)
    if glyphs is None:
        return None
    spans = atom_spans(glyphs)
    if spans is None or len(spans) != mol.GetNumAtoms():
        return None
    arm_cuts: list[set[int]] = []
    for ids, glyph_len in zip(arm_ids, arm_glyph_len, strict=True):
        cuts, total = _token_cuts(ids, glyph_len)
        if total != len(glyphs):
            return None
        arm_cuts.append(cuts)
    mappable = mappable_functional_bonds(mol, spans, glyphs)
    return [_count_arm(mappable, cuts) for cuts in arm_cuts]


def _glyph_alphabet(artifact_dirs: Iterable, arms: Iterable[str]) -> frozenset[str]:
    glyphs: set[str] = set()
    for artifact_dir, arm in zip(artifact_dirs, arms, strict=True):
        tmap = glyph_tuple_map(artifact_dir, arm)  # type: ignore[arg-type]
        glyphs.update(g for t in tmap.values() if len(t) == 1 for g in t)
    return frozenset(glyphs)


def run_pair_fg_alignment(
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
    bpe_name: str,
    unigram_name: str,
    bpe_training_corpus_sha: str,
    unigram_training_corpus_sha: str,
    extras_kind: str | None = None,
    extras_label: str | None = None,
    eval_split_sha_value: str | None = None,
    limit_molecules: int | None = None,
    batch_size: int = ENCODE_BATCH_SIZE,
) -> MatchedPairFgAlignment:
    """Dual-encode the held-out split and aggregate functional-bond locality."""
    sha = eval_split_sha_value or eval_split_sha(corpus)
    bpe_dir = tokenizer_artifact_dir(corpus, bpe_name)
    ul_dir = tokenizer_artifact_dir(corpus, unigram_name)
    bpe_counts = glyph_count_map(bpe_dir, "bpe")
    ul_counts = glyph_count_map(ul_dir, "unigram")
    seg = build_glyph_segmenter(_glyph_alphabet((bpe_dir, ul_dir), ("bpe", "unigram")))

    smiles_iter = iter_test_split(corpus, limit_molecules=limit_molecules)

    bpe_pm: list[PerMoleculeFgLocality] = []
    ul_pm: list[PerMoleculeFgLocality] = []
    batch: list[str] = []

    def _flush() -> None:
        if not batch:
            return
        bids = bpe.encode_batch(batch, add_special_tokens=False)
        uids = unigram.encode_batch(batch, add_special_tokens=False)
        for smi, bi, ui in zip(batch, bids, uids, strict=True):
            result = molecule_arms_locality(smi, [bi, ui], [bpe_counts, ul_counts], seg)
            if result is not None:
                bpe_pm.append(result[0])
                ul_pm.append(result[1])
        batch.clear()

    for smi in smiles_iter:
        batch.append(smi)
        if len(batch) >= batch_size:
            _flush()
    _flush()

    bpe_arm = compute_arm_fg_alignment(
        bpe_pm,
        cell_id=bpe_cell_id,
        arm="bpe",
        boundary=boundary,
        training_corpus_sha=bpe_training_corpus_sha,
        eval_split_sha=sha,
    )
    ul_arm = compute_arm_fg_alignment(
        ul_pm,
        cell_id=unigram_cell_id,
        arm="unigram",
        boundary=boundary,
        training_corpus_sha=unigram_training_corpus_sha,
        eval_split_sha=sha,
    )
    return compute_matched_pair_fg_alignment(
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


def run_single_arm_fg_alignment(
    adapter: SmirkAdapter | UnigramSmirkAdapter,
    *,
    arm: str,
    corpus: str,
    name: str,
    cell_id: str,
    boundary: Boundary,
    training_corpus_sha: str,
    eval_split_sha_value: str | None = None,
    limit_molecules: int | None = None,
    batch_size: int = ENCODE_BATCH_SIZE,
) -> ArmFgAlignment:
    """Encode the held-out split through one arm and aggregate its locality."""
    sha = eval_split_sha_value or eval_split_sha(corpus)
    artifact_dir = tokenizer_artifact_dir(corpus, name)
    counts = glyph_count_map(artifact_dir, arm)  # type: ignore[arg-type]
    seg = build_glyph_segmenter(_glyph_alphabet((artifact_dir,), (arm,)))

    smiles_iter = iter_test_split(corpus, limit_molecules=limit_molecules)

    per_molecule: list[PerMoleculeFgLocality] = []
    batch: list[str] = []

    def _flush() -> None:
        if not batch:
            return
        ids = adapter.encode_batch(batch, add_special_tokens=False)
        for smi, bi in zip(batch, ids, strict=True):
            result = molecule_arms_locality(smi, [bi], [counts], seg)
            if result is not None:
                per_molecule.append(result[0])
        batch.clear()

    for smi in smiles_iter:
        batch.append(smi)
        if len(batch) >= batch_size:
            _flush()
    _flush()

    return compute_arm_fg_alignment(
        per_molecule,
        cell_id=cell_id,
        arm="bpe" if arm == "bpe" else "unigram",
        boundary=boundary,
        training_corpus_sha=training_corpus_sha,
        eval_split_sha=sha,
    )
