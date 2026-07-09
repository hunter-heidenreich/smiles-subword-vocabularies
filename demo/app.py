"""Local interactive demo: how BPE and Unigram-LM carve up the same SMILES.

    python demo/app.py            # serve locally at http://127.0.0.1:7860
    uv run --with gradio --with rdkit python demo/app.py

Runs on CPU (no smirk, no torch): the tokenizers are re-implemented in pure
Python (``segmenter.py``) and validated byte-faithful against smirk. This Gradio
app adds the RDKit molecule overlay and is meant for local use. For a free,
zero-server deployment, ``demo/web/`` builds a static Pyodide site (see
``build_web.py``) that runs the same segmenter in the browser.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import analysis as A
import gradio as gr
import overlay as O

_RDKIT = O.RDKIT

MANIFEST = A.manifest()
CORPORA = list(MANIFEST)
DEFAULT_V = 1024
# "mb" matches the paper's segmentation figure (Unigram-LM forms its own pieces).
DEFAULT_BOUNDARY = "mb"
DEFAULT_EXAMPLE = next(iter(A.EXAMPLES))

INTRO = """
# Where to cut, how deep: BPE vs Unigram-LM on SMILES

Two tokenizers trained on the **same** molecules, with the **same** OpenSMILES
glyph base and the **same** target vocabulary size, learn *near-disjoint*
multi-glyph vocabularies (the **Jaccard** below). Yet they largely agree on
**where** to cut and differ in **how deeply**: Unigram-LM stays near-atomic and
emits more tokens, while BPE merges further along the *same* cut skeleton, so
BPE's segmentation is usually a **strict coarsening** of Unigram-LM's.

The diagram aligns both arms on the shared glyph stream (Unigram-LM above, BPE
below). A filled box is a learned multi-glyph piece, a hollow box a single base
glyph. Where BPE draws one box over several Unigram boxes, that is nesting: BPE
cutting less deeply, not differently. The structure panels paint the same
picture on the molecule.
"""


def _overlays_html(corpus: str, vocab_size: int, boundary: str, smiles: str) -> str:
    """Two stacked structure panels: the molecule painted by each arm's pieces.

    A multi-glyph piece paints one contiguous region, so BPE shows a few large
    blobs and Unigram-LM many small ones (usually near-atomic): the granularity
    gap, made visible. The token counts under each panel explain why one looks
    busy and the other plain.
    """
    if not _RDKIT:
        return ""
    bpe = A.load(corpus, "gpe", vocab_size, boundary)
    uni = A.load(corpus, "unigram", vocab_size, boundary)
    panels = [
        ("BPE", A.C_BPE, O.svg(bpe, smiles, "bpe"), len(bpe.segment(smiles))),
        (
            "Unigram-LM",
            A.C_UNI,
            O.svg(uni, smiles, "unigram"),
            len(uni.segment(smiles)),
        ),
    ]
    return "".join(
        f'<div style="text-align:center;margin-bottom:10px">'
        f'<div style="font-size:12px;text-transform:uppercase;letter-spacing:.03em;'
        f'opacity:.75;margin-bottom:3px"><b style="color:{color}">{name}</b> '
        f"&middot; {ntok} tokens</div>{svg_html}</div>"
        for name, color, svg_html, ntok in panels
    )


def _v_choices(corpus: str) -> list[int]:
    return MANIFEST[corpus]["vocab_sizes"]


def _boundary_choices(corpus: str) -> list[str]:
    return MANIFEST[corpus]["boundaries"]


def _default_v(corpus: str) -> int:
    vs = _v_choices(corpus)
    return DEFAULT_V if DEFAULT_V in vs else vs[-1]


def _default_boundary(corpus: str) -> str:
    bs = _boundary_choices(corpus)
    return DEFAULT_BOUNDARY if DEFAULT_BOUNDARY in bs else bs[0]


def run(smiles: str, corpus: str, vocab_size: int, boundary: str):
    smiles = (smiles or "").strip()
    if not smiles:
        return "<p style='opacity:.6'>Enter a SMILES string above.</p>", ""
    try:
        c = A.contrast(corpus, int(vocab_size), boundary, smiles)
    except Exception as e:  # malformed cell selection, etc.
        return f"<p style='color:#c0392b'>Could not tokenize: {e}</p>", ""
    return A.render(c), _overlays_html(corpus, int(vocab_size), boundary, smiles)


def _on_corpus_change(corpus: str):
    vs = _v_choices(corpus)
    bs = _boundary_choices(corpus)
    return (
        gr.update(choices=vs, value=_default_v(corpus)),
        gr.update(
            choices=[(A.BOUNDARY_LABELS[b], b) for b in bs],
            value=_default_boundary(corpus),
        ),
    )


def build() -> gr.Blocks:
    with gr.Blocks(title="SMILES subword vocabularies", theme=gr.themes.Soft()) as app:
        gr.Markdown(INTRO)
        with gr.Row():
            with gr.Column(scale=3):
                example = gr.Dropdown(
                    label="Example molecule",
                    choices=list(A.EXAMPLES),
                    value=DEFAULT_EXAMPLE,
                )
                smiles = gr.Textbox(
                    label="SMILES",
                    value=A.EXAMPLES[DEFAULT_EXAMPLE],
                    lines=1,
                )
            with gr.Column(scale=2):
                corpus = gr.Dropdown(
                    label="Training corpus",
                    choices=[(A.CORPUS_LABELS[c], c) for c in CORPORA],
                    value=CORPORA[0],
                )
                vocab_size = gr.Dropdown(
                    label="Vocabulary size (V)",
                    choices=_v_choices(CORPORA[0]),
                    value=_default_v(CORPORA[0]),
                )
                boundary = gr.Radio(
                    label="Bracket-atom boundary mode",
                    choices=[
                        (A.BOUNDARY_LABELS[b], b) for b in _boundary_choices(CORPORA[0])
                    ],
                    value=_default_boundary(CORPORA[0]),
                )
        with gr.Row():
            out = gr.HTML(elem_id="tok-out")
            with gr.Column(scale=0, min_width=320):
                structure = gr.HTML(visible=_RDKIT)

        inputs = [smiles, corpus, vocab_size, boundary]

        example.change(lambda name: A.EXAMPLES[name], example, smiles)
        corpus.change(_on_corpus_change, corpus, [vocab_size, boundary])
        for comp in inputs:
            comp.change(run, inputs, [out, structure])
        app.load(run, inputs, [out, structure])
    return app


if __name__ == "__main__":
    build().launch()
