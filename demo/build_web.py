"""Assemble the static (Pyodide) site into an output directory.

Gathers the browser entry point + the pure-Python modules it runs in-browser +
the tokenizer bundle into one folder, ready to serve locally or upload to a free
static host (Hugging Face static Space, GitHub Pages, ...).

    uv run python demo/build_web.py [OUTPUT_DIR]   # default: demo/_site

The output is self-contained and standalone; nothing native, no server.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

DEMO = Path(__file__).resolve().parent
WEB = DEMO / "web"
# Pure-Python modules that run inside Pyodide (no gradio/rdkit/smirk).
MODULES = ["segmenter.py", "analysis.py"]


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else DEMO / "_site"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    shutil.copy2(WEB / "index.html", out / "index.html")
    shutil.copy2(WEB / "README.md", out / "README.md")
    for m in MODULES:
        shutil.copy2(DEMO / m, out / m)
    shutil.copytree(
        DEMO / "tokenizers",
        out / "tokenizers",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )

    files = sum(1 for _ in out.rglob("*") if _.is_file())
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"assembled {files} files ({size / 1024:.0f} KB) -> {out}")


if __name__ == "__main__":
    main()
