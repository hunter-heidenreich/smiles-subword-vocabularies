# smiles-subword-vocabularies

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21228245.svg)](https://doi.org/10.5281/zenodo.21228245)
[![arXiv](https://img.shields.io/badge/arXiv-2607.05691-b31b1b.svg)](https://arxiv.org/abs/2607.05691)

<!-- ChemRxiv badge to be added once that identifier is assigned. -->

Code and paper for a controlled comparison of how subword tokenizers build their
**vocabularies** over chemistry SMILES. Holding the corpus, the OpenSMILES glyph
base, and the target size fixed, BPE and Unigram-LM are shown to construct
**near-disjoint** multi-glyph subword vocabularies — the same atoms, almost none of
the same pieces.

The study trains **no language models**. Every result is a property of the
tokenizers and the corpora; the comparison is therefore exactly controlled.

## Layout

```
src/smiles_subword/ # the measurement + pipeline package
  ingest/           # corpus ingest (PubChem, ZINC-22, COCONUT, REAL-Space, tmQM, CycPeptMPDB)
  preprocess/       # canonicalization, dedup, OpenSMILES-conformance filter, dative handling
  tokenize/         # grid training (BPE / Unigram-LM), measurements, audit
scripts/            # CLI drivers, grouped by stage: ingest/ preprocess/ tokenize/ audit/ measure/
results/            # the generation surface (standalone; decoupled from the manuscript)
  data/             # measurement deposits (per-cell JSONs + aggregated tables)
  build/            # table/figure renderers (not part of the package)
  tables/           # rendered LaTeX tables
  figures/          # rendered vector-PDF figures
tests/              # unit + property tests for the above
paper/              # paper.tex, refs.bib (downstream consumer; `make paper` syncs results/ in)
```

## Setup

```bash
make setup   # uv sync --extra figures --extra crosstoolkit --dev — smirk builds from a pinned source fork (needs Rust)
```

## Reproduce

The `Makefile` is the runnable reference for the pipeline; `make help` lists
every target. The reproducible deliverable is the **experiment pipeline** — the
`scripts/` drivers that ingest, preprocess, train the grid, and compute the
measurement deposits — each stage a `make` target run in order:

```
prep  ->  train  ->  audit  ->  measure
```

Re-running it from the raw, SHA-pinned corpora deterministically regenerates the
per-condition measurement data in `results/data/`.

Rendering the manuscript itself (the LaTeX tables, figures, and PDF) is
author-side convenience, not a deliverable; those scripts live under
`results/build/`, outside the package, and read the `results/data/` deposits:

```bash
make reproduce   # re-render the manuscript's tables/figures/PDF from results/data/
```

Corpus prep is per-corpus and config-driven (`configs/preprocess/`); ZINC-22
first assembles its enumerated tranche set (`make tranche-union`). The `Makefile`
header documents the exact per-corpus command sequence; corpus provenance is in
`data/MANIFEST.yaml`.

## Artifacts

Trained tokenizers and full measurement deposits are archived on Zenodo ([10.5281/zenodo.21228245](https://doi.org/10.5281/zenodo.21228245)).
Preprint: [arXiv:2607.05691](https://arxiv.org/abs/2607.05691) (ChemRxiv TBD).

## Citation

If you use this work, please cite the paper:

```bibtex
@misc{heidenreich2026neardisjoint,
  title        = {{Where to cut, how deep: BPE and Unigram-LM on chemistry SMILES}},
  author       = {Heidenreich, Hunter},
  year         = {2026},
  eprint       = {2607.05691},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
  doi          = {10.5281/zenodo.21228245},
  url          = {https://arxiv.org/abs/2607.05691}
}
```

`CITATION.cff` carries the same metadata for GitHub's "Cite this repository"
button.

## License

Code is released under the [MIT License](LICENSE); the paper text, figures, and
measurement data are CC-BY-4.0. The pinned Smirk fork this depends on remains
Apache-2.0.
