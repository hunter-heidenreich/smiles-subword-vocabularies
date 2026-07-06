"""Render the graphical abstract: same molecules, two segmentations.

For two pinned molecules (nicotine and serotonin), draws each with RDKit twice,
once segmented by Smirk-GPE (BPE) and once by Unigram-LM. Hue encodes the *arm*
(the paper-wide palette: BPE blue, Unigram-LM warm vermillion), and within each
hue the shade depth encodes the *size* of the piece covering an atom (atoms per
piece) on a shared size scale, so "darker = larger piece" reads the same in both
columns. BPE paints a few large, dark regions (whole rings, chains) while
Unigram-LM stays near-atomic, a scatter of small beads. The eye reads the
fertility gap at a glance: same atoms, different pieces. Hue marks the arm, not
piece identity, so same-hue pieces across rows are not the same vocabulary piece
(the two vocabularies are near-disjoint).
Emits ``results/figures/graphical_abstract.pdf``.

Molecules are canonicalised (RDKit) to match the corpus form the tokenizers saw.
Pinned to the committed ``V=1024`` merge-brackets tokenizers, so re-running
reproduces the figure from the deposited artifacts. Needs the ``figures`` extra
(matplotlib) and RDKit; if RDKit or the trained tokenizers are absent, the step
skips cleanly (returns 0) so ``make figures`` still completes the other panels.

Usage::

    uv run python results/build/figure_graphical_abstract.py
"""

from __future__ import annotations

import io
import sys
from collections import Counter

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

try:
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D
except ImportError:  # rdkit is heavy and platform-gated; degrade to a clean skip
    Chem = None
    rdMolDraw2D = None  # noqa: N816  # mirrors the rdkit import name for the None fallback

import style
from abstract import assign_atoms_to_pieces

from smiles_subword.config import cell_artifact_name
from smiles_subword.paths import RESULTS_FIGURES_DIR, tokenizer_artifact_dir
from smiles_subword.tokenize.adapters.smirk import SmirkAdapter
from smiles_subword.tokenize.adapters.smirk_unigram import UnigramSmirkAdapter

BPE_DIR = tokenizer_artifact_dir("pubchem", cell_artifact_name("bpe", 1024, "mb"))
UL_DIR = tokenizer_artifact_dir("pubchem", cell_artifact_name("unigram", 1024, "mb"))
OUT = RESULTS_FIGURES_DIR / "graphical_abstract.pdf"

MOLECULES = (
    ("Nicotine", "CN1CCCC1c1cccnc1"),
    ("Serotonin", "NCCc1c[nH]c2ccc(O)cc12"),
)

# Hue encodes the *arm* (the paper-wide convention: BPE blue, Unigram-LM warm
# vermillion; see style.ARM_COLOR), and intensity within each hue encodes piece
# size. This makes the two columns read as two distinct colours rather than one
# bland blue, ties the figure to the arm palette used in every other panel, and
# lets the warm Unigram side stay visible where pale blue washed out. Piece
# boundaries survive same-colour neighbours because inter-piece bonds are left
# unhighlighted (a visible seam), and atom labels are drawn black (B/W palette)
# so they stay legible across the ramp.
_BPE_HUE = style.BPE_COLOR
_UL_HUE = style.UNIGRAM_COLOR
# Raster the RDKit panels well above the embedded print size and save at high
# DPI so the molecules stay crisp (RDKit emits raster only; no SVG->PDF tool is
# available here). The text labels are matplotlib vector regardless.
_PANEL_W = 1600  # landscape canvas: both molecules are wider than tall, so a
_PANEL_H = 920  # non-square canvas trims the vertical whitespace around them
_SAVE_DPI = 600
style.apply_base_style()
_SAVE_METADATA = {"CreationDate": None}


def _tint(hex_color: str, weight: float = 0.55) -> tuple[float, float, float]:
    """Blend a hue toward white so highlights are pastel (atom labels readable)."""
    r, g, b = to_rgb(hex_color)
    return (
        1 - weight * (1 - r),
        1 - weight * (1 - g),
        1 - weight * (1 - b),
    )


def _size_cmap(hue: str) -> mpl.colors.LinearSegmentedColormap:
    """Pale-to-saturated ramp in one arm's hue: pale = singleton, dark = large.

    The floor is lifted well above white so even size-1 pieces read as a clearly
    tinted bead: on the near-atomic Unigram-LM side every atom is its own piece,
    and at a near-white floor those beads blended into the white inter-piece
    seams, leaving that panel looking faintly uncoloured rather than finely
    chopped. A visible floor makes "many small beads + white seams" the legible
    signal; the dark ceiling keeps BPE's few large pieces standing out.
    """
    return mpl.colors.LinearSegmentedColormap.from_list(
        f"piece_size_{hue}", [_tint(hue, 0.34), _tint(hue, 0.82)]
    )


def _canon(smiles: str) -> str:
    """Canonicalise to the form the tokenizers saw at training (corpus prep)."""
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


def _surface(adapter: SmirkAdapter | UnigramSmirkAdapter, smiles: str) -> list[str]:
    return [adapter.id_to_token(i) for i in adapter.encode(smiles)]


def _panel_png(
    smiles: str,
    atom_piece: list[int],
    piece_color: dict[int, tuple[float, float, float]],
) -> bytes:
    """Draw the molecule with each piece shaded by its size (``piece_color``)."""
    mol = Chem.MolFromSmiles(smiles)
    if len(atom_piece) != mol.GetNumAtoms():
        msg = (
            f"atom-mapping mismatch for {smiles!r}: "
            f"{len(atom_piece)} mapped vs {mol.GetNumAtoms()} atoms"
        )
        raise ValueError(msg)

    atoms = list(range(mol.GetNumAtoms()))
    # Every atom takes its piece colour, so the disc shade honours the size
    # encoding (lightening labelled atoms would read as a false seam in a solid
    # piece and would fade the heteroatom beads on the near-atomic Unigram side).
    # Heteroatom legibility comes instead from a larger font and a ramp ceiling
    # kept below pure-dark, so black labels still read on the largest pieces.
    atom_colors = {a: piece_color[atom_piece[a]] for a in atoms}
    bonds: list[int] = []
    bond_colors: dict[int, tuple[float, float, float]] = {}
    for bond in mol.GetBonds():
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if atom_piece[a1] == atom_piece[a2]:  # intra-piece bond -> share colour
            bonds.append(bond.GetIdx())
            bond_colors[bond.GetIdx()] = piece_color[atom_piece[a1]]

    drawer = rdMolDraw2D.MolDraw2DCairo(_PANEL_W, _PANEL_H)
    opts = drawer.drawOptions()
    opts.clearBackground = False
    opts.useBWAtomPalette()  # black atom labels stay legible over the colour ramp
    opts.bondLineWidth = 7  # bold skeleton bonds, so the chemistry reads through
    # Slightly larger atom labels so heteroatoms (N, O, H) read clearly: nudge the
    # pixel cap up from its default 40 (fully uncapping it makes the letters
    # gigantic and collide with the skeleton).
    opts.baseFontSize = 1.2
    opts.maxFontSize = 90
    opts.padding = 0.02  # let the molecule fill more of the panel
    # Sharpen the piece boundaries. The seam between two pieces is the
    # unhighlighted bond between them, so a piece reads as crisp only if its
    # shaded region stops short of that bond. Two levers do that: a tighter atom
    # highlight radius (0.24 vs the 0.3 default) keeps boundary-atom discs from
    # bleeding across the seam, so the near-atomic Unigram-LM side reads as a
    # scatter of distinct beads rather than a pale wash; and a thinner highlight
    # ribbon (5x bondLineWidth, down from 7x) lets the bolder black skeleton show
    # through at every seam, so adjacent same-size pieces (identical hue, since
    # shade encodes size) stay separated by a visible break. BPE's multi-atom
    # pieces still read as solid contiguous regions.
    opts.highlightRadius = 0.24
    opts.highlightBondWidthMultiplier = 5
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        mol,
        highlightAtoms=atoms,
        highlightAtomColors=atom_colors,
        highlightBonds=bonds,
        highlightBondColors=bond_colors,
    )
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def main() -> int:
    if Chem is None:
        print("[abstract] skipped: rdkit is not installed (optional for this figure)")
        return 0
    if not (BPE_DIR.exists() and UL_DIR.exists()):
        print(
            f"[abstract] skipped: trained tokenizers not found under "
            f"{BPE_DIR.parent} (deposited to Zenodo, not committed)"
        )
        return 0
    bpe = SmirkAdapter.load(BPE_DIR)
    ul = UnigramSmirkAdapter.load(UL_DIR)
    arms = (("Smirk-GPE (BPE)", bpe), ("Unigram-LM", ul))

    # First pass: segment every (molecule, arm) and record piece sizes (atoms
    # per piece) so the colour scale can be shared across all four panels.
    panels: dict[tuple[int, int], dict] = {}
    vmax = 1
    for r, (_mol_label, raw_smiles) in enumerate(MOLECULES):
        smiles = _canon(raw_smiles)
        for c, (_arm_label, adapter) in enumerate(arms):
            pieces = _surface(adapter, smiles)
            atom_piece = assign_atoms_to_pieces(smiles, pieces)
            size_by_pid = Counter(atom_piece)
            vmax = max(vmax, *size_by_pid.values())
            panels[(r, c)] = {
                "smiles": smiles,
                "pieces": pieces,
                "atom_piece": atom_piece,
                "size_by_pid": size_by_pid,
            }

    # One ramp per arm hue, sharing a single size norm so intensity means the
    # same piece size in both columns; hue alone distinguishes the arms.
    arm_hue = (_BPE_HUE, _UL_HUE)
    cmaps = [_size_cmap(hue) for hue in arm_hue]
    norm = mpl.colors.Normalize(vmin=1, vmax=vmax)

    def color_for(size: int, col: int) -> tuple[float, float, float]:
        return cmaps[col](norm(size))[:3]

    # Molecules are rows, algorithms are columns: BPE and Unigram-LM sit
    # side-by-side per molecule for direct comparison, and two rows keep the
    # figure compact enough to float beside the abstract.
    nrows, ncols = len(MOLECULES), len(arms)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.1 * ncols, 2.2 * nrows), squeeze=False
    )
    fig.subplots_adjust(
        left=0.055, right=0.985, top=0.955, bottom=0.13, wspace=0.02, hspace=0.05
    )
    for r, (mol_label, _raw) in enumerate(MOLECULES):
        for c, (arm_label, _adapter) in enumerate(arms):
            panel = panels[(r, c)]
            piece_color = {
                pid: color_for(sz, c) for pid, sz in panel["size_by_pid"].items()
            }
            png = _panel_png(panel["smiles"], panel["atom_piece"], piece_color)
            ax = axes[r][c]
            ax.imshow(plt.imread(io.BytesIO(png), format="png"))
            ax.set_axis_off()
            if r == 0:
                # Colour the column title in its arm hue so hue=arm is legible
                # before the reader reaches the key.
                ax.set_title(
                    arm_label,
                    fontsize=11,
                    fontweight="bold",
                    pad=6,
                    color=_tint(arm_hue[c], 0.95),
                )
            n_tok = len(panel["pieces"])
            if c == 0:
                count_label = f"{n_tok} tokens"
            else:  # Unigram column: show the fertility gap vs the BPE baseline.
                delta = n_tok - len(panels[(r, 0)]["pieces"])
                count_label = f"{n_tok} tokens (+{delta})"
            ax.text(
                0.5,
                -0.02,
                count_label,
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=11,
                fontweight="bold",
                color="#222222",
            )
        axes[r][0].text(
            -0.05,
            0.5,
            mol_label,
            transform=axes[r][0].transAxes,
            ha="right",
            va="center",
            rotation=90,
            fontsize=11,
            fontweight="bold",
        )

    # Shared size key (qualitative): two slim horizontal gradients stacked, one
    # per arm hue, smaller to larger, centred under the panels. Stacking the two
    # ramps shows the size scale applies within each arm; the arm-coloured column
    # titles already carry which hue is which, so the key needs no arm labels. No
    # numeric ticks: the message is the gradient, and the exact max is incidental
    # to these two molecules.
    key_x, key_w = 0.40, 0.24
    for i, hue in enumerate(arm_hue):  # i=0 (BPE) on top, i=1 (Unigram) below
        cax = fig.add_axes((key_x, 0.072 - i * 0.022, key_w, 0.015))
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=_size_cmap(hue))
        cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cbar.set_ticks([])
        cbar.outline.set_visible(False)
        if i == 0:
            cax.set_title("piece size", fontsize=8, pad=2)
    end_kw = {"y": 0.064, "va": "center", "fontsize": 8, "color": "#444444"}
    fig.text(key_x - 0.01, s="smaller", ha="right", **end_kw)
    fig.text(key_x + key_w + 0.01, s="larger", ha="left", **end_kw)

    fig.savefig(OUT, metadata=_SAVE_METADATA, dpi=_SAVE_DPI)
    plt.close(fig)
    print(f"[abstract] wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
