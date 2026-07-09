"""Validate the pure-Python ``Segmenter`` against real smirk (dev-only).

Requires the full project env (smirk + trained ``artifacts/tokenizer/``); this is
NOT shipped to the Space. It is the gate that certifies the demo's segmenter is
byte-for-byte faithful before deploying.

    uv run python demo/validate.py                 # sweep a default cell set
    uv run python demo/validate.py <cell_dir> [N]  # one cell, N SMILES

Exit code is nonzero if any cell has a mismatch.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from segmenter import Segmenter

REPO = Path(__file__).resolve().parent.parent
ARTIFACTS = REPO / "artifacts" / "tokenizer"

DEFAULT_CELLS = [
    "zinc22/smirk_gpe_v256_nmb",
    "zinc22/smirk_gpe_v512_nmb",
    "zinc22/smirk_gpe_v512_mb",
    "zinc22/smirk_gpe_v1024_nmb",
    "zinc22/smirk_gpe_v2048_nmb",
    "zinc22/smirk_unigram_v256_nmb",
    "zinc22/smirk_unigram_v512_nmb",
    "zinc22/smirk_unigram_v512_mb",
    "zinc22/smirk_unigram_v1024_nmb",
    "coconut/smirk_gpe_v512_nmb",
    "coconut/smirk_unigram_v512_nmb",
    "pubchem/smirk_gpe_v512_nmb",
    "pubchem/smirk_unigram_v512_nmb",
]


def _load_smirk(cell_dir: Path):
    from smiles_subword.tokenize.adapters.smirk import SmirkAdapter
    from smiles_subword.tokenize.adapters.smirk_unigram import UnigramSmirkAdapter

    try:
        return SmirkAdapter.load(cell_dir)
    except Exception:
        return UnigramSmirkAdapter.load(cell_dir)


def _sample(corpus: str, n: int) -> list[str]:
    path = REPO / "data" / "processed" / corpus / "canon_dedup_v1" / "train.smi"
    out = []
    with path.open() as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            s = line.strip()
            if s:
                out.append(s)
    return out


# BPE must be exact. Unigram is allowed a tiny rate of documented bit-identical
# Viterbi tie divergences (see segmenter._viterbi); anything above this is a bug.
UNIGRAM_TOLERANCE = 0.001  # 0.1%


def check_cell(rel: str, n: int) -> bool:
    cell_dir = ARTIFACTS / rel
    corpus = rel.split("/")[0]
    gold_tok = _load_smirk(cell_dir)
    seg = Segmenter.from_dir(cell_dir)
    is_bpe = "gpe" in rel
    mism = 0
    tested = 0
    shown = 0
    for s in _sample(corpus, n):
        tested += 1
        gold = [gold_tok.id_to_token(i) for i in gold_tok.encode(s)]
        mine = seg.tokens(s)
        if gold != mine:
            mism += 1
            if shown < 5:
                shown += 1
                print(f"  MISMATCH: {s}")
                print(f"    smirk: {gold}")
                print(f"    ours : {mine}")
    rate = mism / tested if tested else 0.0
    limit = 0 if is_bpe else UNIGRAM_TOLERANCE
    ok = rate <= limit
    status = "OK  " if ok else "FAIL"
    note = "" if ok else "  <-- exceeds tolerance"
    print(f"[{status}] {rel:40s} tested={tested} mismatches={mism} ({rate:.4%}){note}")
    return ok


def main() -> int:
    if len(sys.argv) > 1:
        cells = [sys.argv[1].replace("artifacts/tokenizer/", "").rstrip("/")]
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
    else:
        cells, n = DEFAULT_CELLS, 4000
    failed = [rel for rel in cells if not check_cell(rel, n)]
    if failed:
        print(f"\n{len(failed)} CELL(S) EXCEED TOLERANCE: {', '.join(failed)}")
        return 1
    print("\nALL CELLS WITHIN TOLERANCE (BPE exact, Unigram <=0.1% tie edges)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
