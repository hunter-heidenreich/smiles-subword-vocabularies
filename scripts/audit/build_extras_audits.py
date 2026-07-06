"""Write the robustness-extras audit summaries to ``results/data/audits/``.

Derives the three bespoke audit deposits the results table reads
(``seed_cap.json``, ``prune_schedule.json``, ``merge_exhaustion.json``) from the
trained extras tokenizers under ``artifacts/tokenizer/``: the seed-cap and
prune-schedule multi-glyph Jaccards of a one-armed Unigram-LM probe against its
reference-default arm, and the BPE arm's natural merge-exhaustion terminal
vocabulary. These feed the robustness-extras summary table; see
:mod:`smiles_subword.tokenize.measure.supplementary.extras_audits`.

Pure vocabulary-set computation over the committed tokenizer artifacts — no
corpus pass — so it runs anywhere the extras tokenizers and their reference-arm
baselines are present. Run after ``train-extras`` and before ``results``.

Examples::

    uv run python scripts/audit/build_extras_audits.py
"""

from __future__ import annotations

import sys

from smiles_subword.tokenize.measure.supplementary.extras_audits import (
    write_extras_audits,
)


def main() -> int:
    for path in write_extras_audits():
        print(f"[extras-audits] wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
