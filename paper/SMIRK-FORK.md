# The Smirk fork

**Status:** complete — landed at the production pin `vtc-2026-05-24` (commit
`c2ee070`). This is the version recorded in `pyproject.toml` and
`data/MANIFEST.yaml` and used for every measurement in the paper.
**Owner:** Hunter Heidenreich

The study needs one tokenizer engine that hosts **both** trainer arms — BPE
(Smirk's stock `GpeTrainer`) and Unigram-LM — behind an identical
pre-tokenization front-end, so the comparison is controlled at everything except
the training algorithm. Stock Smirk ships only the BPE arm, so the study runs on
a source fork of Smirk `v0.2.0`. This file records what the fork adds, how it is
built, and the exact pin.

## What the fork is

A source fork of `BattModels/Smirk` `v0.2.0`. `smirk`'s own version stays
`0.2.0` (this repo's bound is `smirk>=0.2.0,<0.3`); all changes are source-level.
Over stock `v0.2.0` it adds, in `src/`:

1. **Shared pre-tokenization front-end** (`shared.rs`) — `compute_alphabet` and
   `tokenize_words` (Smirk's Layer A/B/C glyph-id pipeline) were private methods
   of `GpeTrainer`; they are lifted out as free functions so both trainer arms
   consume the identical front-end. Pure code move — BPE output stays
   byte-identical to stock (`test/test_byte_identity.py`).
2. **Unigram-LM sibling trainer** (`unigram/`) — `UnigramModel` (Layer-A split
   then Viterbi-argmax over glyph pieces) + `UnigramTrainer`, mirroring
   `GpeTrainer`'s knobs and sharing the front-end. It **delegates the EM/pruning
   loop to HuggingFace `tokenizers`' `UnigramTrainer`** (see "The two forks"
   below). Python entry point `smirk.train_unigram()`, mirroring `train_gpe()`.
3. **PUA bridge** (`pua.rs`) — glyph id `g` ↔ `U+E000 + g`, so glyph-id
   sequences become the `char` strings HuggingFace's char-based trainer
   requires. One Layer-B chunk → one HuggingFace `Sentence`. PUA exists only
   transiently in training; no PUA codepoint reaches a saved artifact (asserted
   in `test/test_train_unigram.py`).
4. **Scaffold instrumentation of `GpeTrainer`** — an opt-in `scaffold_log_path`;
   when set, `do_train` streams a per-merge-step JSONL log (each committed
   merge's selected-candidate frequency + the running standalone frequencies it
   touches). Purely additive reads + a side-file write ⇒ BPE output unchanged
   when off (the default). Feeds the scaffold-fraction measurement;
   `test/test_scaffold_log.py`.
5. **Layer-B chunker binding** — `pretokenize_layer_b(smi) -> [(chunk, (start,
   end))]` on `SmirkTokenizer` / `SmirkTokenizerFast`, a read-only wrapper over
   the shared `split_structure` chunker so the absorption measurement can
   obtain Layer-B chunks at inference. Training behavior unchanged;
   `test/test_pretokenize_layer_b.py`.
6. **Settable Unigram knobs** — `seed_size`, `max_piece_length`,
   `n_sub_iterations`, `shrinking_factor` were baked Rust `const`s; they are
   promoted to `UnigramTrainer` builder fields (and surfaced on
   `train_unigram()`) so the robustness probes can vary them. Defaults are
   unchanged (`1e6` / `128` / `2` / `0.75`).

## The two forks

The production pin chains two forks:

| | |
|---|---|
| Smirk fork | [`hunter-heidenreich/smirk`](https://github.com/hunter-heidenreich/smirk) @ `vtc-2026-05-24` = `c2ee070` |
| ↳ built against | [`hunter-heidenreich/tokenizers`](https://github.com/hunter-heidenreich/tokenizers) @ `81cca59` |
| Smirk upstream base | `BattModels/Smirk` `v0.2.0` = `f7dd001c9e02fcafa4468b80587c2fc80426ecc5` |

`smirk`'s `Cargo.toml` pins `tokenizers` to the personal fork
(`hunter-heidenreich/tokenizers` @ `81cca59`), which is HuggingFace
`tokenizers` **0.23** plus a single change — `fix/unigram-prune-per-piece-alternatives`,
i.e. **HF PR #2070**, the per-piece-alternatives prune fix. The fork exists only
until #2070 lands in a stock `tokenizers` release, at which point `smirk` can
revert to the crates.io crate. No other `Cargo.toml` pins are changed
(`pyo3 = "^0.27"`, …).

The Unigram arm delegates its EM/pruning to HuggingFace's `UnigramTrainer`
rather than a native reimplementation: HuggingFace `tokenizers` is a faithful
SentencePiece port whose only behavioral prune divergence (per-piece
alternatives, HF PR #2070) is inert on SMILES, so a native trainer would carry
maintenance cost for no observable benefit on Smirk's domain. The settable knobs
forward to the HuggingFace builder.

This repo pins the Smirk fork in `pyproject.toml`
(`[tool.uv.sources]` git `tag = "vtc-2026-05-24"`, bound `smirk>=0.2.0,<0.3`)
and records the commit/tag/upstream in `data/MANIFEST.yaml` (`software:`).
Because the fork builds from source via maturin/pyo3, `uv sync` needs a Rust
toolchain.

## Test contract at the pin

- **BPE / GPE is byte-identical to stock `v0.2.0`** — `test_byte_identity.py`
  trains the four `merge_brackets` × `split_structure` boundary configs at two
  vocab sizes and diffs the serialized tokenizer against goldens captured from
  stock. The goldens are unchanged across the `tokenizers` 0.23 bump.
- **Unigram determinism is piece-set equality**, not byte-identity (the HF
  delegation does not guarantee serialization-order stability); tests pin
  `TOKENIZERS_PARALLELISM=false`.
- Scaffold log, Layer-B binding, and no-PUA-in-artifacts each have a dedicated
  test (above). The fork's gate is the **local** run of `cargo test` +
  `uv run pytest test/` + `pre-commit` — fork CI/CD is left disabled (the `CD.yml`
  wheel-publish workflow fires on any tag push, so Actions are off by design).

## Build / dev loop

The fork is a maturin/pyo3 Rust extension (`build-backend = "maturin"`),
verified on macOS (darwin, arm64):

| Tool | Version |
|---|---|
| rustc / cargo | 1.95.0 |
| maturin | 1.13.3 |
| Python | 3.13.12 |
| uv | 0.11.7 |
| `tokenizers` crate (resolved) | 0.23 (fork @ `81cca59`) |

```sh
git clone https://github.com/hunter-heidenreich/smirk && cd smirk
uv sync                 # builds the pyo3 extension + dev deps (needs Rust)
cargo test              # Rust unit tests
uv run pytest test/     # Python suite incl. test_byte_identity.py
uv run pre-commit run --all-files
```
