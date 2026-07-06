"""Render the OOD-eval table from the deposited records.

Reads the per-cell ``results/data/ood_eval/`` deposits (the PubChem ``V=1024``
``nmb`` generalist read on each adversarial OOD corpus, both arms) and emits
``results/tables/ood_eval.tex`` — per-arm fertility, the cross-arm relative
fertility gap, and atom-level OOV (the coverage sanity check). Pure on-disk read
over the deposits; the measurement itself is
``scripts/measure/compute_ood_eval.py``.

Usage::

    uv run python results/build/table_ood_eval.py
"""

from __future__ import annotations

import json
import sys

from smiles_subword.paths import RESULTS_TABLES_DIR
from smiles_subword.tokenize.measure.fertility import relative_fertility_gap
from smiles_subword.tokenize.measure.supplementary.transfer.ood import (
    OOD_CORPORA,
    OOD_EVAL_DIR,
)

_CORPUS_LABEL = {
    "tmqm": "tmQM (metals)",
    "cycpeptmpdb": "CycPeptMPDB (macrocycles)",
}


def _load_records() -> dict[tuple[str, str], dict[str, object]]:
    if not OOD_EVAL_DIR.is_dir():
        return {}
    out: dict[tuple[str, str], dict[str, object]] = {}
    for path in sorted(OOD_EVAL_DIR.glob("*__*__*.json")):
        r = json.loads(path.read_text())
        out[(str(r["eval_corpus"]), str(r["arm"]))] = r
    return out


def write_tex_table() -> str:
    """Render the OOD-eval table to ``results/tables/ood_eval.tex``."""
    records = _load_records()
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Out-of-distribution generalization: the PubChem diverse-corpus "
        r"generalist ($V{=}1024$, NMB) read on adversarial in-spec chemistry it was "
        r"not fit to. Fertility is mean tokens per molecule; bootstrap CIs "
        r"negligible at these $n$. rel$|\Delta f|$ is the cross-arm relative "
        r"fertility gap (Eq.~\ref{eq:fertility}). The BPE-coarser / Unigram-finer "
        r"divergence persists off-distribution, and atom-level OOV stays negligible "
        r"(shared $165$-token base), so the contrast is genuine granularity, not "
        r"coverage.}",
        r"  \label{tab:ood-eval}",
        r"  \small",
        r"  \begin{tabular}{@{}l r rr r rr@{}}",
        r"    \toprule",
        r"    & & \multicolumn{2}{c}{fertility} & & \multicolumn{2}{c}{atom OOV}\\",
        r"    \cmidrule(lr){3-4}\cmidrule(lr){6-7}",
        r"    corpus & $n$ & BPE & Unigram-LM & rel$|\Delta f|$ & BPE & Unigram-LM \\",
        r"    \midrule",
    ]
    for corpus in OOD_CORPORA:
        bpe = records.get((corpus, "bpe"))
        ul = records.get((corpus, "unigram"))
        if bpe is None or ul is None:
            continue
        f_bpe = float(bpe["fertility_mean"])  # type: ignore[arg-type]
        f_ul = float(ul["fertility_mean"])  # type: ignore[arg-type]
        gap = relative_fertility_gap(f_bpe, f_ul)
        label = _CORPUS_LABEL.get(corpus, corpus)
        n = int(bpe["n_molecules"])  # type: ignore[arg-type]
        oov_bpe = float(bpe["oov_token_rate"])  # type: ignore[arg-type]
        oov_ul = float(ul["oov_token_rate"])  # type: ignore[arg-type]
        lines.append(
            f"    {label} & {n:,} & {f_bpe:.1f} & {f_ul:.1f} & "
            f"{gap * 100:.1f}\\% & {oov_bpe * 100:.3f}\\% & "
            f"{oov_ul * 100:.3f}\\% \\\\"
        )
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]

    RESULTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_TABLES_DIR / "ood_eval.tex"
    path.write_text("\n".join(lines))
    return str(path)


def main() -> int:
    print(f"[ood-table] wrote {write_tex_table()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
