"""Copy the trained cells the demo needs into ``demo/tokenizers/`` (dev-only).

The demo ships only ``tokenizer.json`` + ``meta.yaml`` per cell (tens of KB
each) - enough for the pure-Python ``Segmenter`` - so the Space needs neither the
full ``artifacts/`` tree nor smirk. Run this once (with the project artifacts
present) before deploying:

    uv run python demo/export_cells.py

It exports every matched BPE/Unigram grid cell that exists for the target
corpora, and writes ``demo/tokenizers/manifest.json`` listing what's available.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ARTIFACTS = REPO / "artifacts" / "tokenizer"
OUT = Path(__file__).resolve().parent / "tokenizers"

CORPORA = ["pubchem", "zinc22", "coconut"]
# Only plain grid cells: <algo>_v<V>_<nmb|mb>. No robustness-extras cells.
CELL_RE = re.compile(r"^smirk_(gpe|unigram)_v(\d+)_(nmb|mb)$")
SHIP_FILES = ("tokenizer.json", "meta.yaml")


def _grid_cells(corpus: str) -> dict[tuple[int, str, str], str]:
    """Return {(V, boundary, algo): cell_name} for one corpus."""
    corp_dir = ARTIFACTS / corpus
    found: dict[tuple[int, str, str], str] = {}
    if not corp_dir.is_dir():
        return found
    for cell in corp_dir.iterdir():
        m = CELL_RE.match(cell.name)
        if m:
            algo, v, boundary = m.group(1), int(m.group(2)), m.group(3)
            found[(v, boundary, algo)] = cell.name
    return found


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    manifest: dict[str, dict] = {}
    exported = 0
    for corpus in CORPORA:
        cells = _grid_cells(corpus)
        # keep only (V, boundary) combos where BOTH arms exist - the demo always
        # shows a matched pair.
        combos = sorted(
            {(v, b) for (v, b, algo) in cells}
            & {(v, b) for (v, b, algo) in cells if (v, b, "unigram") in cells}
            & {(v, b) for (v, b, algo) in cells if (v, b, "gpe") in cells}
        )
        if not combos:
            continue
        manifest[corpus] = {
            "vocab_sizes": sorted({v for v, _ in combos}),
            "boundaries": sorted({b for _, b in combos}),
        }
        for v, boundary in combos:
            for algo in ("gpe", "unigram"):
                name = cells[(v, boundary, algo)]
                src = ARTIFACTS / corpus / name
                dst = OUT / corpus / name
                dst.mkdir(parents=True, exist_ok=True)
                for fname in SHIP_FILES:
                    if (src / fname).is_file():
                        shutil.copy2(src / fname, dst / fname)
                exported += 1
                print(f"  exported {corpus}/{name}")

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    print(f"\nExported {exported} cells across {len(manifest)} corpora.")
    print(f"Total shipped size: {size / 1024:.0f} KB -> {OUT}")
    print(f"Manifest: {json.dumps(manifest, indent=2)}")


if __name__ == "__main__":
    main()
