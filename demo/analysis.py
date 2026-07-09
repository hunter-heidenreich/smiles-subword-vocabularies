"""Demo presentation layer: load cells, contrast the two arms, render HTML.

Separate from ``segmenter.py`` (the faithful core) so the reproduction logic
stays free of any display concerns. Reads only the shipped ``demo/tokenizers/``
bundle: no smirk, no ``artifacts/``.

The view mirrors the paper's three contrasts (Heidenreich, "Where to cut, how
deep") on one molecule at a time:

* **Membership** - the two arms learn near-disjoint multi-glyph vocabularies
  (cross-arm Jaccard is small).
* **Granularity** - Unigram-LM emits more tokens and stays near-atomic; BPE packs
  more glyphs per token.
* **Compatibility** - the arms agree on *where* to cut but differ in *how deeply*:
  the disagreement is nesting (Unigram cuts where BPE merges), not crossing, so
  BPE's parse is a strict coarsening of Unigram-LM's on the large majority of
  molecules. The nesting diagram draws exactly this.
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from segmenter import Piece, Segmenter

TOKENIZERS = Path(__file__).resolve().parent / "tokenizers"

# Paper-wide arm palette (results/build/style.py): BPE blue, Unigram-LM
# vermillion; overlap teal for pieces both arms learned.
C_BPE = "#0072B2"
C_UNI = "#D55E00"
C_SHARED = "#44AA99"

CORPUS_LABELS = {"pubchem": "PubChem", "zinc22": "ZINC-22", "coconut": "COCONUT"}
BOUNDARY_LABELS = {
    "nmb": "NMB - bracket atoms kept opaque",
    "mb": "MB - merges may enter bracket atoms",
}

# The first three are the paper's own segmentation-figure molecules (drug-like,
# a stereocentre-bearing natural product with strong nesting, and a charged
# salt); the rest are recognisable follow-ons.
EXAMPLES = {
    "Aspirin (drug-like)": "CC(=O)Oc1ccccc1C(=O)O",
    "Natural product (lactone)": "CC(=O)CCC[C@@H]1CC=CC(=O)O1",
    "Sodium acetate (salt)": "CC(=O)[O-].[Na+]",
    "Nicotine": "CN1CCCC1c1cccnc1",
    "Serotonin": "NCCc1c[nH]c2ccc(O)cc12",
    "Caffeine": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "Penicillin G": "CC1(C)S[C@@H]2[C@H](NC(=O)Cc3ccccc3)C(=O)N2[C@H]1C(=O)O",
    "Glucose": "OC[C@@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",
    "Ibuprofen": "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
}


def manifest() -> dict:
    return json.loads((TOKENIZERS / "manifest.json").read_text())


@lru_cache(maxsize=64)
def load(corpus: str, algo: str, vocab_size: int, boundary: str) -> Segmenter:
    name = f"smirk_{algo}_v{vocab_size}_{boundary}"
    return Segmenter.from_dir(TOKENIZERS / corpus / name)


def _glyph_spans(
    glyphs: list[Piece], pieces: list[Piece]
) -> list[tuple[int, int, Piece]]:
    """Map each piece to its [glyph_start, glyph_end) index span."""
    starts = {g.start: i for i, g in enumerate(glyphs)}
    ends = {g.end: i + 1 for i, g in enumerate(glyphs)}
    return [(starts[p.start], ends[p.end], p) for p in pieces]


@dataclass
class Contrast:
    """Both arms' segmentation of one molecule, aligned on the glyph stream."""

    glyphs: list[Piece]
    bpe: list[tuple[int, int, Piece]]  # (glyph_start, glyph_end, piece)
    uni: list[tuple[int, int, Piece]]
    bpe_vocab: set[str]  # multi-glyph pieces the BPE arm learned
    uni_vocab: set[str]  # multi-glyph pieces the Unigram arm learned

    # -- membership --
    @property
    def jaccard(self) -> float:
        union = self.bpe_vocab | self.uni_vocab
        return len(self.bpe_vocab & self.uni_vocab) / len(union) if union else 0.0

    # -- granularity --
    @property
    def n_glyphs(self) -> int:
        return len(self.glyphs)

    def n_tokens(self, arm: str) -> int:
        return len(self.bpe if arm == "bpe" else self.uni)

    def compression(self, arm: str) -> float:
        n = self.n_tokens(arm)
        return self.n_glyphs / n if n else 0.0

    # -- compatibility (nesting) --
    def _cuts(self, arm: str) -> set[int]:
        spans = self.bpe if arm == "bpe" else self.uni
        return {end for _, end, _ in spans[:-1]}

    @property
    def agree(self) -> int:
        return len(self._cuts("bpe") & self._cuts("uni"))

    @property
    def nest(self) -> int:
        return len(self._cuts("uni") - self._cuts("bpe"))

    @property
    def conflict(self) -> int:
        return len(self._cuts("bpe") - self._cuts("uni"))

    @property
    def is_coarsening(self) -> bool:
        """True when every BPE cut is also a Unigram cut (conflict == 0)."""
        return self.conflict == 0


def contrast(corpus: str, vocab_size: int, boundary: str, smiles: str) -> Contrast:
    bpe = load(corpus, "gpe", vocab_size, boundary)
    uni = load(corpus, "unigram", vocab_size, boundary)
    glyphs = bpe.glyphs(smiles)  # glyph base is identical for both arms
    return Contrast(
        glyphs=glyphs,
        bpe=_glyph_spans(glyphs, bpe.segment(smiles)),
        uni=_glyph_spans(glyphs, uni.segment(smiles)),
        bpe_vocab=bpe.multiglyph_pieces,
        uni_vocab=uni.multiglyph_pieces,
    )


# ---- HTML rendering --------------------------------------------------------
def _boxes(
    spans: list[tuple[int, int, Piece]], row: int, arm_color: str, other_vocab: set[str]
) -> str:
    """One arm's token boxes: filled when a learned multi-glyph piece (teal if
    shared with the other arm, else arm hue), hollow outline when a base glyph."""
    out = []
    for gs, ge, p in spans:
        if not p.is_multiglyph:
            style = f"border:1px solid {arm_color};opacity:.45;"
            title = f"{p.text}  (single glyph)"
        else:
            fill = C_SHARED if p.text in other_vocab else arm_color
            shared = "shared by both arms" if p.text in other_vocab else "arm-exclusive"
            style = f"border:1.5px solid {arm_color};background:{fill};opacity:.85;"
            title = f"{p.text}  (learned piece, {shared})"
        out.append(
            f'<div class="nbox" style="grid-column:{gs + 1}/{ge + 1};'
            f'grid-row:{row};{style}" title="{html.escape(title)}"></div>'
        )
    return "".join(out)


def _glyph_cells(c: Contrast) -> str:
    bpe_starts = {gs for gs, _, _ in c.bpe if gs > 0}
    cells = []
    for i, g in enumerate(c.glyphs):
        cut = " cut" if i in bpe_starts else ""
        cells.append(
            f'<div class="gly{cut}" style="grid-column:{i + 1};grid-row:2">'
            f"{html.escape(g.text)}</div>"
        )
    return "".join(cells)


def _stat(value: str, label: str, color: str = "") -> str:
    col = f"color:{color}" if color else ""
    return f'<div class="stat"><b style="{col}">{value}</b><span>{label}</span></div>'


def render(c: Contrast) -> str:
    n = c.n_glyphs
    uni_boxes = _boxes(c.uni, 1, C_UNI, c.bpe_vocab)
    bpe_boxes = _boxes(c.bpe, 3, C_BPE, c.uni_vocab)
    glyph_cells = _glyph_cells(c)

    coarse = (
        f'<b style="color:{C_BPE}">Yes</b> - BPE only merges further along '
        "Unigram-LM's cuts"
        if c.is_coarsening
        else '<b style="color:#c0392b">No</b> - the arms cross (a rare case)'
    )
    return f"""
<style>
  .nest-wrap {{ overflow-x:auto; padding:4px 0 2px; }}
  .nest {{ display:grid; grid-auto-rows:min-content; gap:0 0;
           grid-template-rows:26px 26px 26px; align-items:center; width:max-content; }}
  .nbox {{ height:22px; border-radius:6px; margin:0 1px; }}
  .gly {{ text-align:center; font-family:ui-monospace,Menlo,monospace; font-size:14px;
          line-height:26px; }}
  .gly.cut {{ box-shadow:inset 1px 0 0 0 #b0b0b0; }}
  .rowlab {{ font-size:11px; text-transform:uppercase; letter-spacing:.04em;
             opacity:.7; white-space:nowrap; padding-right:10px; }}
  .legend {{ display:flex; flex-wrap:wrap; gap:16px; font-size:12.5px;
             margin-top:10px; opacity:.85; }}
  .legend span b {{ display:inline-block; width:11px; height:11px; border-radius:3px;
                    margin-right:5px; vertical-align:middle; }}
  .stats {{ display:flex; flex-wrap:wrap; gap:26px; margin-top:16px;
            padding-top:12px; border-top:1px solid var(--pill-border); }}
  .stat b {{ display:block; font-size:19px; font-weight:600; }}
  .stat span {{ opacity:.7; font-size:12px; }}
  .compat {{ margin-top:14px; font-size:13px; }}
  :root {{ --pill-border:#cfcfcf; }}
  @media (prefers-color-scheme: dark) {{ :root {{ --pill-border:#555; }} }}
</style>
<div style="display:flex;align-items:center">
  <div>
    <div class="rowlab" style="color:{C_UNI}">Unigram-LM &#9650;</div>
    <div class="rowlab" style="height:26px">&nbsp;</div>
    <div class="rowlab" style="color:{C_BPE}">BPE &#9660;</div>
  </div>
  <div class="nest-wrap" style="flex:1">
    <div class="nest" style="grid-template-columns:repeat({n},1.55em)">
      {uni_boxes}{glyph_cells}{bpe_boxes}
    </div>
  </div>
</div>
<div class="legend">
  <span><b style="background:{C_BPE}"></b>BPE-only learned piece</span>
  <span><b style="background:{C_UNI}"></b>Unigram-only learned piece</span>
  <span><b style="background:{C_SHARED}"></b>learned by both</span>
  <span><b style="background:transparent;border:1px solid var(--pill-border)"></b>single glyph (shared base)</span>
</div>
<div class="stats">
  {_stat(str(c.n_tokens("bpe")), f"BPE tokens &middot; {c.compression('bpe'):.2f} glyphs/tok", C_BPE)}
  {_stat(str(c.n_tokens("uni")), f"Unigram tokens &middot; {c.compression('uni'):.2f} glyphs/tok", C_UNI)}
  {_stat(str(c.agree), "shared cuts (agree where)")}
  {_stat(str(c.nest), "nesting cuts (Unigram deeper)")}
  {_stat(str(c.conflict), "conflicting cuts")}
  {_stat(f"{c.jaccard:.1%}", "vocab Jaccard (near-disjoint)")}
</div>
<div class="compat">Is BPE a strict coarsening of Unigram-LM here? {coarse}.</div>
"""
