"""Concrete tokenizer engine adapters wrapping the pinned smirk Rust fork.

Each adapter satisfies the :class:`~smiles_subword.tokenize.base.Tokenizer`
protocol and is constructed via :func:`smiles_subword.tokenize.build_tokenizer`:

- :class:`~smiles_subword.tokenize.adapters.smirk.SmirkAdapter` — the
  ``smirk_base`` and ``smirk_gpe`` (BPE-over-glyphs) kinds.
- :class:`~smiles_subword.tokenize.adapters.smirk_unigram.UnigramSmirkAdapter`
  — the ``smirk_unigram`` (Unigram-LM) kind.

Both share the runtime in ``_smirk_runtime``. This subpackage is only the
engines; the protocol (``base``), build registry (package ``__init__``), grid
(``grid`` / ``extras``), and measurements live one level up.
"""
