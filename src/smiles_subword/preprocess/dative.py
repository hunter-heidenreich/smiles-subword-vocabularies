"""Derive an OpenSMILES-conformant encoding from dative-bond organometallics.

tmQM and other organometallic releases serialize metal coordination with dative
bonds (``->`` / ``<-``), which are not OpenSMILES and tokenize to ``[UNK]`` on
the ``>`` glyph. This converts the dative bond family to single bonds so the
bracketed metal atoms (the rare-token lever) survive while the non-conformant
arrows do not.

Full :func:`Chem.SanitizeMol` would re-perceive metal coordination and convert
the single bonds *back* to dative, so we recompute valence and implicit-H with
the lighter :meth:`UpdatePropertyCache` instead, then emit canonical isomeric
SMILES. This is a derivation, not an upstream reference encoding: the
dative-to-single conversion shifts implicit-H and valence bookkeeping on
metal-bound atoms, and a small fraction of inputs do not survive it (recorded as
a drop rate by the driver).
"""

from __future__ import annotations

from rdkit import Chem

DATIVE_BOND_TYPES = frozenset(
    {
        Chem.BondType.DATIVE,
        Chem.BondType.DATIVEONE,
        Chem.BondType.DATIVEL,
        Chem.BondType.DATIVER,
    }
)
"""Every dative bond subtype RDKit may assign when parsing ``->`` / ``<-``."""


def dative_to_opensmiles(smiles: str) -> str | None:
    """Convert dative bonds to single bonds and emit canonical isomeric SMILES.

    Returns ``None`` when RDKit cannot parse the input or write the converted
    molecule. Inputs with no dative bonds are passed through the same canonical
    writer, so the function is a uniform canonicalizer for an organometallic
    corpus.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if not any(b.GetBondType() in DATIVE_BOND_TYPES for b in mol.GetBonds()):
        return Chem.MolToSmiles(mol, isomericSmiles=True)

    rwmol = Chem.RWMol(mol)
    for bond in rwmol.GetBonds():
        if bond.GetBondType() in DATIVE_BOND_TYPES:
            bond.SetBondType(Chem.BondType.SINGLE)
    converted = rwmol.GetMol()
    converted.UpdatePropertyCache(strict=False)
    try:
        return Chem.MolToSmiles(converted, isomericSmiles=True)
    except (Chem.AtomValenceException, Chem.KekulizeException, RuntimeError):
        return None


__all__ = ["DATIVE_BOND_TYPES", "dative_to_opensmiles"]
