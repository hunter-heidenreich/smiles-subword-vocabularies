"""Fixtures for Stage 0 ingest tests."""

from __future__ import annotations

import gzip
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from smiles_subword.config import (
    CorpusConfig,
    Zinc22CorpusConfig,
    Zinc22MultiTrancheConfig,
)

if TYPE_CHECKING:
    from collections.abc import Iterable


@pytest.fixture
def shard_paths() -> Callable[[Path], list[Path]]:
    """Reader: a stage output dir -> its sorted raw_v1 Parquet shards.

    Hoisted here because every ingest backend test enumerated its shards the
    same way; injected as a callable so call sites read `shard_paths(out_dir)`.
    """

    def _paths(output_dir: Path) -> list[Path]:
        return sorted(output_dir.glob("raw_v1-*.parquet"))

    return _paths


@pytest.fixture
def shard_columns() -> Callable[[Path], dict[str, list]]:
    """Reader: a stage output dir -> {column_name: values} over all its shards."""

    def _columns(output_dir: Path) -> dict[str, list]:
        table = pa.concat_tables(
            pq.read_table(s) for s in sorted(output_dir.glob("raw_v1-*.parquet"))
        )
        return {name: table.column(name).to_pylist() for name in table.column_names}

    return _columns


_MINI_SMILES: tuple[str, ...] = (
    "1",
    "C",
    "CC",
    "CCO",
    "CCN",
    "c1ccccc1",
    "CC(=O)O",
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "C(C(=O)O)N",
    "OC[C@@H](O)[C@H](O)[C@H](O)CO",
)


def _build_rows(n: int) -> Iterable[tuple[int, str]]:
    for i in range(n):
        yield i + 1, _MINI_SMILES[i % len(_MINI_SMILES)]


def _write_gzip_tsv(path: Path, rows: Iterable[tuple[int, str]]) -> None:
    with gzip.open(path, "wt", encoding="ascii") as fh:
        for cid, smiles in rows:
            fh.write(f"{cid}\t{smiles}\n")


@pytest.fixture
def mini_input(tmp_path: Path) -> Path:
    path = tmp_path / "mini.tsv.gz"
    _write_gzip_tsv(path, _build_rows(1000))
    return path


@pytest.fixture
def empty_input(tmp_path: Path) -> Path:
    path = tmp_path / "empty.tsv.gz"
    _write_gzip_tsv(path, [])
    return path


@pytest.fixture
def mini_corpus_config(tmp_path: Path, mini_input: Path) -> CorpusConfig:
    return CorpusConfig(
        name="pubchem",
        source="pubchem",
        manifest_id="pubchem-cid-smiles",
        raw_path=mini_input,
        output_dir=tmp_path / "out",
        shard_target_bytes=2**20,
        rows_per_batch=1024,
    )


@pytest.fixture
def multi_shard_corpus_config(tmp_path: Path, mini_input: Path) -> CorpusConfig:
    return CorpusConfig(
        name="pubchem",
        source="pubchem",
        manifest_id="pubchem-cid-smiles",
        raw_path=mini_input,
        output_dir=tmp_path / "out",
        shard_target_bytes=2**12,
        rows_per_batch=128,
    )


_ZINC22_FIXTURE_ROWS: tuple[tuple[str, str], ...] = (
    ("CC(C)C[C@@H]1C[C@H]1Nc1nccc(C(=O)O)n1", "ZINChq0000066mDY"),
    ("CCN(Cc1nocc1C(C)C)[C@@H](C)C(=O)O", "ZINChq0000066mEr"),
    ("CN(C(=O)c1ccsc1CC(=O)O)C1(C)CC1", "ZINChq0000066mEK"),
    ("CN(OCC(=O)O)C(=O)c1ccc(Cl)cc1I", "ZINChq0000066mEO"),
    ("COc1nc(F)ccc1-c1ncc(C(=O)O)s1", "ZINChq0000066mF5"),
)


def _write_zinc22_smi_gz(path: Path, rows: Iterable[tuple[str, str]]) -> None:
    with gzip.open(path, "wt", encoding="ascii") as fh:
        for smiles, zinc_id in rows:
            fh.write(f"{smiles}\t{zinc_id}\n")


@pytest.fixture
def zinc22_smi_gz(tmp_path: Path) -> Path:
    path = tmp_path / "raw" / "zinc22-smoke.smi.gz"
    path.parent.mkdir(parents=True)
    _write_zinc22_smi_gz(path, _ZINC22_FIXTURE_ROWS)
    return path


@pytest.fixture
def zinc22_empty_smi_gz(tmp_path: Path) -> Path:
    path = tmp_path / "raw" / "zinc22-empty.smi.gz"
    path.parent.mkdir(parents=True)
    _write_zinc22_smi_gz(path, [])
    return path


@pytest.fixture
def zinc22_manifest(tmp_path: Path) -> Path:
    path = tmp_path / "MANIFEST.yaml"
    path.write_text(yaml.safe_dump({"version": 1, "artifacts": []}))
    return path


@pytest.fixture
def zinc22_smoke_config(tmp_path: Path, zinc22_smi_gz: Path) -> Zinc22CorpusConfig:
    return Zinc22CorpusConfig(
        name="zinc22-smoke",
        source="zinc22",
        manifest_id="zinc22-test-fixture",
        tranche_id="test-fixture",
        tranche_url="https://example.invalid/test-fixture.smi.gz",
        transport="curl",
        raw_path=zinc22_smi_gz,
        output_dir=tmp_path / "out",
        smiles_column="smiles",
        id_column="zinc_id",
        has_header=False,
        delim="\t",
        file_compression="gzip",
        shard_target_bytes=2**20,
        rows_per_batch=128,
    )


_MULTI_TRANCHE_FIXTURE_ROWS: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
    (
        "zinc-22x-H17P200",
        (
            ("CC(=O)Nc1ccc(O)cc1", "ZINChq000000aaa1"),
            ("CCOc1ccc(N)cc1", "ZINChq000000aaa2"),
        ),
    ),
    (
        "zinc-22x-H18P210",
        (
            ("Clc1ccc(N)cc1", "ZINChq000000bbb1"),
            ("Brc1ccc(N)cc1", "ZINChq000000bbb2"),
            ("Fc1ccc(N)cc1", "ZINChq000000bbb3"),
        ),
    ),
    (
        "zinc-22f-H19P220",
        (("CN(C)c1ccc(C(=O)O)cc1", "ZINChq000000ccc1"),),
    ),
)


@pytest.fixture
def zinc22_multi_tranche_raw_root(tmp_path: Path) -> Path:
    """Stage one synthetic .smi.gz per tranche under raw_root/."""
    raw_root = tmp_path / "raw" / "tranches"
    raw_root.mkdir(parents=True)
    for tranche_id, rows in _MULTI_TRANCHE_FIXTURE_ROWS:
        path = raw_root / f"{tranche_id}.smi.gz"
        _write_zinc22_smi_gz(path, rows)
    return raw_root


@pytest.fixture
def zinc22_multi_tranche_tranches_tsv(tmp_path: Path) -> Path:
    """Tranche TSV pointing at the synthetic smi.gz fixtures."""
    path = tmp_path / "tranches.tsv"
    header = "tranche_id\tgeneration\theavy_atom_bin\tlogp_bin\turl\texpected_bytes\n"
    rows = []
    for tranche_id, content in _MULTI_TRANCHE_FIXTURE_ROWS:
        parts = tranche_id.split("-")
        gen = parts[1][2:]
        h = int(parts[2][1:3])
        p = int(parts[2].split("P")[-1])
        url = f"https://example.invalid/{tranche_id}.smi.gz"
        rows.append(f"{tranche_id}\t{gen}\t{h}\t{p}\t{url}\t{len(content)}\n")
    path.write_text(header + "".join(rows))
    return path


@pytest.fixture
def zinc22_multi_tranche_config(
    tmp_path: Path,
    zinc22_multi_tranche_tranches_tsv: Path,
    zinc22_multi_tranche_raw_root: Path,
) -> Zinc22MultiTrancheConfig:
    return Zinc22MultiTrancheConfig(
        name="zinc22-multi-tranche-test",
        source="zinc22",
        tranches_path=zinc22_multi_tranche_tranches_tsv,
        raw_root=zinc22_multi_tranche_raw_root,
        output_root=tmp_path / "out",
        concurrency=2,
        transport="curl",
        smiles_column="smiles",
        id_column="zinc_id",
        has_header=False,
        delim="\t",
        file_compression="gzip",
        shard_target_bytes=2**20,
        rows_per_batch=128,
    )
