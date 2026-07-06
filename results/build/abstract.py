"""Map tokenizer pieces back onto molecule atoms for the graphical abstract.

Pure, backend-free logic (regex only — no RDKit, no matplotlib), split from the
matplotlib drawing in ``results/build/figure_graphical_abstract.py`` (its sole
consumer). Given a
SMILES and the in-order piece surfaces a tokenizer emits (whose concatenation is
the SMILES), :func:`assign_atoms_to_pieces` returns, for each atom in RDKit's
indexing order, the index of the piece that covers it. The driver turns that into
per-atom / per-bond highlight colours, so a multi-glyph piece (a learned merge)
paints one contiguous coloured region and the eye reads segmentation granularity:
BPE paints a few large pieces, Unigram-LM many small ones.

The atom-order assumption — that the n-th atom token in the SMILES string is
RDKit atom index ``n`` — holds because ``Chem.MolFromSmiles`` indexes atoms in
input-string order; the driver asserts the recovered count against
``mol.GetNumAtoms()`` at draw time.
"""

from __future__ import annotations

import re

# SMILES atom-level token regex (organic subset + bracket atoms + structure). We
# only need to separate atom tokens from structural ones (bonds, ring digits,
# branches) and recover atom order, so the alternation is ordered so two-letter
# organic elements (Cl, Br) match before their single-letter prefixes.
_SMILES_TOKEN = re.compile(
    r"\[[^\]]+\]|Br|Cl|[BCNOSPFIbcnosp]|\(|\)|\.|=|#|-|\+|\\|/|:|~|@|\?|>|\*|\$"
    r"|%[0-9]{2}|[0-9]"
)
_ORGANIC_ATOMS = frozenset(
    ("B", "C", "N", "O", "S", "P", "F", "I", "b", "c", "n", "o", "s", "p", "Br", "Cl")
)


def _is_atom(token: str) -> bool:
    """Whether a SMILES token denotes an atom (bracket atom, organic, wildcard)."""
    return token.startswith("[") or token in _ORGANIC_ATOMS or token == "*"


def atom_char_starts(smiles: str) -> list[int]:
    """Char offset of each atom token, in RDKit atom-index (input-string) order."""
    starts: list[int] = []
    pos = 0
    while pos < len(smiles):
        m = _SMILES_TOKEN.match(smiles, pos)
        if m is None:
            msg = f"cannot tokenize SMILES at offset {pos}: {smiles[pos:]!r}"
            raise ValueError(msg)
        if _is_atom(m.group(0)):
            starts.append(m.start())
        pos = m.end()
    return starts


def assign_atoms_to_pieces(smiles: str, pieces: list[str]) -> list[int]:
    """Return, per atom (RDKit index order), the index of the covering piece.

    ``pieces`` is the in-order surface sequence the tokenizer emits; its
    concatenation must equal ``smiles`` (a segmentation). The returned list has
    one entry per atom, giving the index into ``pieces`` of the piece whose char
    span contains that atom.
    """
    spans: list[tuple[int, int, int]] = []
    cursor = 0
    for i, piece in enumerate(pieces):
        spans.append((cursor, cursor + len(piece), i))
        cursor += len(piece)
    if cursor != len(smiles):
        msg = "pieces do not concatenate to the SMILES"
        raise ValueError(msg)

    assigned: list[int] = []
    for start in atom_char_starts(smiles):
        for lo, hi, idx in spans:
            if lo <= start < hi:
                assigned.append(idx)
                break
        else:  # pragma: no cover - every atom lies inside some piece span
            msg = f"atom at offset {start} fell outside all pieces"
            raise ValueError(msg)
    return assigned
