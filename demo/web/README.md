---
title: SMILES Subword Vocabularies
emoji: ⚗️
colorFrom: blue
colorTo: red
sdk: static
app_file: index.html
pinned: false
short_description: How BPE and Unigram-LM carve up chemistry SMILES differently
---

# SMILES subword vocabularies: BPE vs Unigram-LM

An interactive companion to *"Where to cut, how deep: BPE and Unigram-LM on
chemistry SMILES"* ([arXiv:2607.05691](https://arxiv.org/abs/2607.05691)).

Pick a molecule and see both tokenizers segment it, aligned on the shared glyph
stream. It surfaces the paper's three contrasts live: **membership** (near-disjoint
vocabularies, small Jaccard), **granularity** (Unigram-LM stays near-atomic and
emits more tokens; BPE packs more glyphs per token), and **compatibility** (they
agree on *where* to cut but differ in *how deeply*, so BPE's parse is usually a
strict coarsening of Unigram-LM's).

**Runs entirely in your browser** via [Pyodide](https://pyodide.org): a static
page, no server, no cost. The tokenizer logic is a pure-Python reimplementation
(`segmenter.py`) validated byte-for-byte against
[smirk](https://github.com/hunter-heidenreich/smirk); the shipped `tokenizers/`
bundle holds only each cell's `tokenizer.json` + `meta.yaml`.
