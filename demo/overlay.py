"""Paint a molecule's atoms by which token covers them (per arm).

The visual from the paper's graphical abstract: the same molecule, tokenized by
each arm, with every multi-glyph piece painting one contiguous coloured region.
BPE paints a few large blobs, Unigram-LM many small ones, so the eye reads
segmentation granularity directly.

The atom-locating regex (``atom_char_starts``) is vendored from
``results/build/abstract.py`` so the demo stays self-contained (that module is
not shipped to the Space). We map atoms to pieces via the ``Piece`` char offsets
the segmenter already produces, which also sidesteps the rare ``[UNK]`` piece
whose surface differs from the glyphs it covers.

The atom-order assumption (n-th atom token in the string is RDKit atom index n)
holds because ``Chem.MolFromSmiles`` indexes atoms in input-string order; we
assert the recovered atom count against ``mol.GetNumAtoms()`` before drawing and
fall back to a plain depiction on any mismatch.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from segmenter import Piece, Segmenter

try:
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D

    RDKIT = True
except Exception:  # pragma: no cover
    RDKIT = False

# Atom-level SMILES token regex (two-letter organics before single-letter).
_SMILES_TOKEN = re.compile(
    r"\[[^\]]+\]|Br|Cl|[BCNOSPFIbcnosp]|\(|\)|\.|=|#|-|\+|\\|/|:|~|@|\?|>|\*|\$"
    r"|%[0-9]{2}|[0-9]"
)
_ORGANIC_ATOMS = frozenset(
    ("B", "C", "N", "O", "S", "P", "F", "I", "b", "c", "n", "o", "s", "p", "Br", "Cl")
)

# Light tints of the two arm hues (RGB 0..1). Cycled across a molecule's
# multi-glyph pieces so neighbouring pieces stay visually distinct; light enough
# that RDKit's atom labels stay readable on top.
BPE_TINTS = [(0.60, 0.80, 0.92), (0.40, 0.66, 0.85), (0.75, 0.88, 0.96)]
UNI_TINTS = [(0.97, 0.78, 0.60), (0.93, 0.63, 0.42), (0.99, 0.86, 0.73)]


def _is_atom(token: str) -> bool:
    return token.startswith("[") or token in _ORGANIC_ATOMS or token == "*"


def atom_char_starts(smiles: str) -> list[int]:
    """Char offset of each atom token, in RDKit atom-index (string) order."""
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


def _atom_piece_index(normalized: str, pieces: list[Piece]) -> list[int]:
    """Per atom (RDKit order): index into ``pieces`` of the covering piece."""
    out: list[int] = []
    for start in atom_char_starts(normalized):
        for i, p in enumerate(pieces):
            if p.start <= start < p.end:
                out.append(i)
                break
        else:  # pragma: no cover - every atom lies in some piece span
            msg = f"atom at offset {start} outside all pieces"
            raise ValueError(msg)
    return out


def svg(seg: Segmenter, smiles: str, arm: str, *, size: int = 300) -> str:
    """Return an SVG (wrapped in a light card) of ``smiles`` coloured by ``arm``
    segmentation, or an empty string if RDKit is absent or parsing fails."""
    if not RDKIT:
        return ""
    normalized = seg.normalize(smiles)
    mol = Chem.MolFromSmiles(normalized)
    if mol is None:
        return _card("", "could not parse")
    pieces = seg.segment(smiles)
    try:
        atom_piece = _atom_piece_index(normalized, pieces)
    except ValueError:
        atom_piece = []
    if len(atom_piece) != mol.GetNumAtoms():  # fall back to a plain depiction
        atom_piece = []

    tints = BPE_TINTS if arm == "bpe" else UNI_TINTS
    # Assign a tint to each multi-glyph piece, cycling in order of appearance so
    # adjacent pieces differ; single-glyph pieces stay uncoloured.
    piece_color: dict[int, tuple[float, float, float]] = {}
    for i, p in enumerate(pieces):
        if p.is_multiglyph:
            piece_color[i] = tints[len(piece_color) % len(tints)]

    hi_atoms: list[int] = []
    atom_colors: dict[int, tuple[float, float, float]] = {}
    for a, pidx in enumerate(atom_piece):
        if pidx in piece_color:
            hi_atoms.append(a)
            atom_colors[a] = piece_color[pidx]

    hi_bonds: list[int] = []
    bond_colors: dict[int, tuple[float, float, float]] = {}
    if atom_piece:
        for bond in mol.GetBonds():
            a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if atom_piece[a] == atom_piece[b] and atom_piece[a] in piece_color:
                hi_bonds.append(bond.GetIdx())
                bond_colors[bond.GetIdx()] = piece_color[atom_piece[a]]

    d = rdMolDraw2D.MolDraw2DSVG(size, int(size * 0.78))
    opts = d.drawOptions()
    opts.clearBackground = False
    opts.highlightBondWidthMultiplier = 12
    rdMolDraw2D.PrepareAndDrawMolecule(
        d,
        mol,
        highlightAtoms=hi_atoms,
        highlightAtomColors=atom_colors,
        highlightBonds=hi_bonds,
        highlightBondColors=bond_colors,
    )
    d.FinishDrawing()
    return _card(d.GetDrawingText(), "")


def _card(inner_svg: str, note: str) -> str:
    body = inner_svg or f'<span style="color:#999;font-size:13px">{note}</span>'
    return (
        '<div style="background:#ffffff;border-radius:10px;padding:8px 10px;'
        'display:inline-block;border:1px solid #e0e0e0;text-align:center">'
        f"{body}</div>"
    )
