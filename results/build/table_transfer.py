"""Render the cross-corpus transfer-matrix table from the deposited records.

Reads the per-cell ``results/data/transfer/`` deposits, aggregates the headline
``V=1024`` cells into ``results/data/transfer/_transfer_matrix.{json,md}`` (a
human-readable matrix of fertility + off-domain penalty per (train, eval)), and
renders ``results/tables/transfer_matrix.tex``. Pure on-disk read over the
deposits; the per-cell measurement is ``scripts/measure/compute_transfer.py``.

Usage::

    uv run python results/build/table_transfer.py
"""

from __future__ import annotations

import json
import sys

from _corpora import CORPUS_LABEL

from smiles_subword._io import atomic_write_json
from smiles_subword.paths import RESULTS_TABLES_DIR
from smiles_subword.tokenize.measure.supplementary.transfer.math import TRANSFER_DIR
from smiles_subword.tokenize.measure.supplementary.transfer.runner import CORPORA

_HEADLINE_V = 1024
_HEADLINE_BOUNDARY = "nmb"
_FERTILITY_DIR = TRANSFER_DIR.parent / "fertility"


def _load_records() -> list[dict[str, object]]:
    if not TRANSFER_DIR.is_dir():
        return []
    return [
        json.loads(path.read_text())
        for path in sorted(TRANSFER_DIR.glob("*__*__*.json"))
    ]


def _fertility_native_fertility(eval_corpus: str, arm: str) -> float | None:
    """On-domain fertility for ``eval_corpus``/``arm`` from the Fertility deposit.

    The transfer-penalty normalizer; reused rather than recomputed (the transfer
    diagonal would be identical to this Fertility number).
    """
    path = _FERTILITY_DIR / f"{eval_corpus}__v{_HEADLINE_V}_{_HEADLINE_BOUNDARY}.json"
    if not path.is_file():
        return None
    block = json.loads(path.read_text()).get(arm)
    if not isinstance(block, dict):
        return None
    value = block.get("fertility_mean")
    return float(value) if isinstance(value, (int, float)) else None


def build_table() -> tuple[str, str]:
    """Aggregate deposited records into the transfer-matrix table (json + md)."""
    records = _load_records()
    by_arm: dict[str, dict[tuple[str, str], dict[str, object]]] = {
        "bpe": {},
        "unigram": {},
    }
    for r in records:
        if r["vocab_size"] != _HEADLINE_V:
            continue
        by_arm[str(r["arm"])][(str(r["train_corpus"]), str(r["eval_corpus"]))] = r

    payload: dict[str, object] = {"headline_v": _HEADLINE_V, "arms": {}}
    md_lines = [
        f"# Transfer matrix (V={_HEADLINE_V}, nmb)\n",
        "Off-diagonal = tokenizer trained on `train`, read on `eval`'s held-out "
        "split. Diagonal (native) reused from fertility. Penalty = off-domain "
        "fertility / native.\n",
    ]
    for arm, cells in by_arm.items():
        native = {ev: _fertility_native_fertility(ev, arm) for ev in CORPORA}
        matrix: dict[str, dict[str, object]] = {}
        md_lines.append(f"\n## {arm} — fertility (penalty vs native)\n")
        md_lines.append("train\\eval | " + " | ".join(CORPORA))
        md_lines.append("|".join(["---"] * (len(CORPORA) + 1)))
        for train in CORPORA:
            row_cells: dict[str, object] = {}
            md_cells: list[str] = []
            for ev in CORPORA:
                if train == ev:
                    nat = native[ev]
                    md_cells.append("native" if nat is None else f"{nat:.2f} (1.00x)")
                    continue
                rec = cells.get((train, ev))
                if rec is None:
                    md_cells.append("—")
                    continue
                fert = float(rec["fertility_mean"])  # type: ignore[arg-type]
                nat = native[ev]
                penalty = fert / nat if nat else float("nan")
                row_cells[ev] = {
                    "fertility": fert,
                    "penalty_vs_native": penalty,
                    "oov_token_rate": rec["oov_token_rate"],
                }
                md_cells.append(f"{fert:.2f} ({penalty:.2f}x)")
            matrix[train] = row_cells
            md_lines.append(f"{train} | " + " | ".join(md_cells))
        payload["arms"][arm] = matrix  # type: ignore[index]

    json_path = TRANSFER_DIR / "_transfer_matrix.json"
    md_path = TRANSFER_DIR / "_transfer_matrix.md"
    atomic_write_json(json_path, payload)
    md_path.write_text("\n".join(md_lines) + "\n")
    return str(json_path), str(md_path)


def write_tex_table() -> str:
    """Render the cross-corpus transfer-penalty table to ``results/tables/``.

    A single 4x4 of (train, eval) cells, each the ``BPE\\,/\\,Unigram-LM``
    off-domain penalty (fertility relative to the eval corpus's native tokenizer
    of the same arm). The diagonal (self-transfer, trivially $1.00$) renders as a
    dash; the single largest Unigram-LM penalty is bold.
    """
    records = _load_records()
    by_arm: dict[str, dict[tuple[str, str], dict[str, object]]] = {
        "bpe": {},
        "unigram": {},
    }
    for r in records:
        if r["vocab_size"] != _HEADLINE_V:
            continue
        by_arm[str(r["arm"])][(str(r["train_corpus"]), str(r["eval_corpus"]))] = r

    native = {
        arm: {ev: _fertility_native_fertility(ev, arm) for ev in CORPORA}
        for arm in ("bpe", "unigram")
    }

    def _penalty(arm: str, train: str, ev: str) -> float | None:
        rec = by_arm[arm].get((train, ev))
        nat = native[arm][ev]
        if rec is None or not nat:
            return None
        return float(rec["fertility_mean"]) / nat  # type: ignore[arg-type]

    # Bold the single largest Unigram-LM off-domain penalty (the headline cell).
    ul_pen = {
        (train, ev): _penalty("unigram", train, ev)
        for train in CORPORA
        for ev in CORPORA
        if train != ev
    }
    best = max(
        ((pen, cell) for cell, pen in ul_pen.items() if pen is not None),
        default=None,
    )
    bold_cell = best[1] if best is not None else None

    header = " & ".join(CORPUS_LABEL[c] for c in CORPORA)
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Cross-corpus transfer: off-domain fertility ($V{=}1024$, NMB), "
        r"each cell BPE\,/\,Unigram-LM. Each value is the fertility of the "
        r"\emph{train}-corpus tokenizer on the \emph{eval} corpus's held-out split, "
        r"divided by the eval corpus's native tokenizer of the same arm; "
        r"self-transfer on the diagonal is omitted (trivially $1.00$). BPE transfers "
        r"at near-native fertility everywhere; Unigram-LM is modestly "
        r"domain-sensitive, largest where the combinatorial REAL-Space specialist "
        r"meets natural products (REAL-Space$\to$COCONUT, in bold). Atom-level OOV "
        r"is below $0.01\%$ in every cell (shared $165$-token base).}",
        r"  \label{tab:transfer}",
        r"  \small",
        r"  \begin{tabular}{@{}l cccc@{}}",
        r"    \toprule",
        rf"    train $\downarrow$ / eval $\rightarrow$ & {header} \\",
        r"    \midrule",
    ]
    for train in CORPORA:
        cells: list[str] = []
        for ev in CORPORA:
            pb, pu = _penalty("bpe", train, ev), _penalty("unigram", train, ev)
            if train == ev or pb is None or pu is None:
                cells.append("---")
                continue
            ul = rf"\textbf{{{pu:.2f}}}" if (train, ev) == bold_cell else f"{pu:.2f}"
            cells.append(rf"{pb:.2f}\,/\,{ul}")
        lines.append(rf"    {CORPUS_LABEL[train]} & " + " & ".join(cells) + r" \\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]

    RESULTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_TABLES_DIR / "transfer_matrix.tex"
    path.write_text("\n".join(lines))
    return str(path)


def main() -> int:
    json_path, md_path = build_table()
    print(f"[transfer-table] wrote {json_path}")
    print(f"[transfer-table] wrote {md_path}")
    print(f"[transfer-table] wrote {write_tex_table()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
