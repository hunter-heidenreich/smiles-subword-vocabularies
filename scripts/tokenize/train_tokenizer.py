"""Stage 5 driver: TokenizerConfig -> trained tokenizer artifact directory.

Both arms ship the natural artifact their trainer produces — no post-train
trimming. A ``smirk_gpe`` artifact therefore lands at ``len(tok) ==
vocab_size + 6`` (the six tail specials above the WordLevel base), and a
``smirk_unigram`` artifact at the size its pruning loop settled on. The
driver reloads the saved artifact only to report its realized ``len(tok)``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from smiles_subword.config import TokenizerConfig
from smiles_subword.tokenize import build_tokenizer
from smiles_subword.tokenize.adapters.smirk import SmirkAdapter
from smiles_subword.tokenize.adapters.smirk_unigram import UnigramSmirkAdapter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg = TokenizerConfig.from_yaml(args.config)
    tok = build_tokenizer(cfg)
    tok.save(cfg.output_dir)

    final_len = _reload_len(cfg.output_dir, kind=cfg.kind)
    print(
        f"saved {cfg.kind} tokenizer {cfg.name!r} "
        f"(vocab_size={tok.vocab_size}, len={final_len}) to {cfg.output_dir}"
    )


def _reload_len(artifact_dir: Path, *, kind: str) -> int:
    if kind == "smirk_unigram":
        return len(UnigramSmirkAdapter.load(artifact_dir))
    return len(SmirkAdapter.load(artifact_dir))


if __name__ == "__main__":
    main()
