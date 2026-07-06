"""Compute the OOD-eval extension: PubChem generalist on adversarial corpora.

Reads the PubChem ``V=1024`` ``nmb`` tokenizer (both arms) on each OOD corpus's
full ``canon_dedup_v1`` output and deposits one per-cell JSON under
``results/data/ood_eval/``. The table is rendered separately by
``results/build/table_ood_eval.py``.

Examples::

    uv run python scripts/measure/compute_ood_eval.py                 # full run
    uv run python scripts/measure/compute_ood_eval.py --limit 2000    # quick probe
"""

from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING

from smiles_subword._io import atomic_write_json
from smiles_subword.config import algo_to_engine_tag
from smiles_subword.tokenize.measure.supplementary.transfer.ood import (
    ARMS,
    OOD_CORPORA,
    OOD_EVAL_DIR,
    run_ood_eval,
)

if TYPE_CHECKING:
    from smiles_subword.tokenize.measure.supplementary.transfer.math import (
        TransferRecord,
    )


def _deposit(record: TransferRecord) -> None:
    atomic_write_json(OOD_EVAL_DIR / f"{record.cell_key}.json", record.as_dict())


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n", 1)[0])
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="take the first N molecules per eval corpus instead of all (debug)",
    )
    parser.add_argument(
        "--rebuild", action="store_true", help="recompute even cells already deposited"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    for corpus in OOD_CORPORA:
        for arm in ARMS:
            arm_tag = algo_to_engine_tag(arm)
            deposit = OOD_EVAL_DIR / f"pubchem__{corpus}__{arm_tag}_v1024_nmb.json"
            if not args.rebuild and deposit.is_file():
                continue
            rec = run_ood_eval(eval_corpus=corpus, arm=arm, limit=args.limit)
            _deposit(rec)
            print(
                f"[ood] {corpus}__{arm}: fertility={rec.fertility_mean:.2f} "
                f"oov_tok={rec.oov_token_rate:.5f} n={rec.n_molecules:,}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
