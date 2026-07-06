"""Tests for the ZINC-22 config schemas (the config-side siblings of the
ZINC-22 ingest-driver tests in test_zinc22.py / test_zinc22_multi_tranche.py).

Two schemas live here: `Zinc22CorpusConfig` (single tranche) — whose
`resolved_transport` is the one piece of branching logic, pinned over its three
outcomes since the driver tests never vary it — and `Zinc22MultiTrancheConfig`,
whose YAML round-trip is pinned here rather than inside the driver test file.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import yaml

from smiles_subword.config import Zinc22CorpusConfig, Zinc22MultiTrancheConfig

if TYPE_CHECKING:
    from pathlib import Path


def _config(**overrides: object) -> Zinc22CorpusConfig:
    payload: dict[str, object] = {
        "name": "zinc22",
        "manifest_id": "zinc22-tranche",
        "tranche_id": "AAAA",
        "tranche_url": "https://example.org/AAAA.smi.gz",
        "raw_path": "data/raw/zinc22/AAAA.smi.gz",
        "output_dir": "data/processed/zinc22/AAAA/raw_v1",
    }
    payload.update(overrides)
    return Zinc22CorpusConfig.model_validate(payload)


def test_explicit_transport_overrides_url_scheme() -> None:
    # An explicit transport wins even when the URL scheme would imply otherwise.
    cfg = _config(transport="rsync", tranche_url="https://example.org/AAAA.smi.gz")
    assert cfg.resolved_transport() == "rsync"


def test_auto_infers_rsync_from_url_scheme() -> None:
    cfg = _config(transport="auto", tranche_url="rsync://example.org/AAAA.smi.gz")
    assert cfg.resolved_transport() == "rsync"


def test_auto_falls_back_to_curl() -> None:
    cfg = _config(transport="auto", tranche_url="https://example.org/AAAA.smi.gz")
    assert cfg.resolved_transport() == "curl"


def test_multi_tranche_config_yaml_roundtrip(tmp_path: Path) -> None:
    payload = {
        "name": "zinc22-multi-tranche",
        "source": "zinc22",
        "tranches_path": "configs/corpus/zinc22_tranches.tsv",
        "raw_root": "data/raw/zinc22/tranches",
        "output_root": "data/processed/zinc22/raw_v1",
        "concurrency": 4,
        "transport": "auto",
        "shard_target_bytes": 268435456,
        "rows_per_batch": 131072,
    }
    cfg_path = tmp_path / "zinc22.yaml"
    cfg_path.write_text(yaml.safe_dump(payload))

    cfg = Zinc22MultiTrancheConfig.from_yaml(cfg_path)

    assert cfg.name == "zinc22-multi-tranche"
    assert cfg.concurrency == 4
    assert cfg.tranches_path.is_absolute()
