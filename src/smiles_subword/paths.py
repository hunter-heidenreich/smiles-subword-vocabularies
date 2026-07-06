"""Repo-anchored filesystem constants."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
DATA_DIR = REPO_ROOT / "data"

# Shared by the grid/extras dispatchers and F95/scaffold orchestrators; single
# source so a rename can't drift.
TRAIN_TOKENIZER_SCRIPT = REPO_ROOT / "scripts" / "tokenize" / "train_tokenizer.py"

# Generation surface: measurement I/O deposits JSON into ``results/data``;
# renderers (``results/build``) emit tables/figures into ``results/{tables,figures}``.
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DATA_DIR = RESULTS_DIR / "data"
RESULTS_FIGURES_DIR = RESULTS_DIR / "figures"
RESULTS_TABLES_DIR = RESULTS_DIR / "tables"


def processed_corpus_dir(corpus: str) -> Path:
    """Return ``data/processed/<corpus>/``; callers append the stage subdir."""
    return DATA_DIR / "processed" / corpus


def tokenizer_artifact_dir(corpus: str, name: str) -> Path:
    """Return ``artifacts/tokenizer/<corpus>/<name>/``."""
    return ARTIFACTS_DIR / "tokenizer" / corpus / name


def audit_path(name: str) -> Path:
    """Return the on-disk path of a robustness-extras audit ``{name}.json``."""
    return RESULTS_DATA_DIR / "audits" / f"{name}.json"
