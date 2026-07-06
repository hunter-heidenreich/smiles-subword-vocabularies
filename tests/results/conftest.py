"""Put ``results/build/`` on ``sys.path`` so its plain-script modules import.

The result renderers are not a package and not installed (they live outside the
wheel); they import each other by bare name (``from _corpora import ...``). The
build directory therefore has to be on the path before any ``import extract``
in this subtree is collected. conftest runs before sibling test modules are
imported, so the insertion lands in time.
"""

from __future__ import annotations

import sys

from smiles_subword.paths import REPO_ROOT

_BUILD_DIR = str(REPO_ROOT / "results" / "build")
if _BUILD_DIR not in sys.path:
    sys.path.insert(0, _BUILD_DIR)
