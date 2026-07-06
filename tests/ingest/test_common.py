"""Direct unit tests for `ingest._common`.

The shared raw_v1 helpers are exercised end-to-end by every per-corpus backend
test; this module pins the pieces that round-trip coverage hits only coarsely —
defensive branches unreachable through a trusted config, plus the config-driven
SQL builder whose full branch matrix no single corpus covers:

* `ingest_timestamp` returning a naive, second-truncated stamp (the
  reproducibility contract behind the `timestamp[us]` cast and byte-identical
  same-second reruns), which integration coverage hits only timing-dependently;
* `relative_to_repo` falling back to the input path when it lives outside the
  repo tree;
* `stream_arrow_batches` closing its reader + connection when iteration raises;
* `build_csv_select_sql` emitting the right projection and `read_csv(...)` args
  across the positional/named branch matrix (and raising before interpolating a
  non-identifier column);
* `stream_csv_path` binding parameters in the exact order the generated `?`
  placeholders expect.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import duckdb
import pyarrow as pa
import pytest

from smiles_subword.ingest._common import (
    RAW_V1_SCHEMA,
    build_csv_select_sql,
    ingest_timestamp,
    open_duckdb,
    relative_to_repo,
    run_single_file_ingest,
    stream_arrow_batches,
    stream_csv_path,
)
from smiles_subword.paths import REPO_ROOT

if TYPE_CHECKING:
    from collections.abc import Iterator
    from datetime import datetime
    from pathlib import Path

    from smiles_subword.config import CorpusConfig


def test_ingest_timestamp_is_naive_and_second_truncated() -> None:
    ts = ingest_timestamp()
    assert ts.tzinfo is None  # naive, so it casts to the schema's timestamp[us]
    assert ts.microsecond == 0  # truncated, so same-second reruns hash identically


def test_relative_to_repo_strips_repo_prefix() -> None:
    inside = REPO_ROOT / "data" / "processed" / "x.parquet"
    assert relative_to_repo(inside).as_posix() == "data/processed/x.parquet"


def test_relative_to_repo_returns_path_unchanged_when_outside_tree(
    tmp_path,  # noqa: ANN001 - pytest tmp_path fixture, outside REPO_ROOT
) -> None:
    outside = tmp_path / "elsewhere" / "x.parquet"
    assert relative_to_repo(outside) == outside


def test_stream_arrow_batches_closes_connection_when_cast_fails() -> None:
    con = open_duckdb()
    # The single projected column cannot cast to the 4-field RAW_V1_SCHEMA, so
    # the first `batch.cast(...)` raises mid-iteration and the `finally` must
    # still close the connection.
    gen = stream_arrow_batches(con, "SELECT 1 AS x", [], 10)

    with pytest.raises(ValueError, match=r"schema|field|cast"):
        list(gen)

    # Connection was closed by the generator's `finally`; reusing it now raises.
    with pytest.raises(duckdb.ConnectionException):
        con.execute("SELECT 1")


def test_raw_v1_cast_rejects_null_source_id() -> None:
    # The raw_v1 contract declares every field non-nullable. `smiles` has
    # explicit null handling upstream (`drop_null_smiles` / `coalesce`), but
    # `source_id` has none, so the schema cast in `stream_arrow_batches` is the
    # sole enforcer of "no null source_id". A null id must fail loudly rather
    # than silently land a null in a column the contract promises is non-null.
    con = open_duckdb()
    sql = """
        SELECT * FROM (VALUES
            ('a',  'C',  'x', TIMESTAMP '2026-01-01 00:00:00'),
            (NULL, 'CC', 'x', TIMESTAMP '2026-01-01 00:00:00')
        ) AS t(source_id, smiles, source, ingest_ts)
    """
    with pytest.raises(ValueError, match="non-nullable"):
        list(stream_arrow_batches(con, sql, [], 10))


def test_ingest_failure_leaves_prior_output_dir_intact(
    mini_corpus_config: CorpusConfig,
) -> None:
    # Staging-into-`.tmp` + atomic rename exists so a crash mid-ingest can't
    # corrupt a good prior `output_dir`. The rename is the last step, so a
    # failure during streaming must leave the existing directory byte-for-byte.
    out = mini_corpus_config.output_dir
    out.mkdir(parents=True)
    prior = out / "good.parquet"
    prior.write_bytes(b"prior-good-data")

    def _boom(_cfg: CorpusConfig, ingest_ts: datetime) -> Iterator[pa.RecordBatch]:
        yield pa.record_batch(
            {
                "source_id": pa.array(["a"], pa.string()),
                "smiles": pa.array(["C"], pa.string()),
                "source": pa.array(["x"], pa.string()),
                "ingest_ts": pa.array([ingest_ts], pa.timestamp("us")),
            }
        ).cast(RAW_V1_SCHEMA)
        raise RuntimeError("simulated mid-stream failure")

    with pytest.raises(RuntimeError, match="mid-stream"):
        run_single_file_ingest(
            mini_corpus_config,
            stream_batches=_boom,
            ingest_ts=ingest_timestamp(),
            input_sha256="0" * 64,
            manifest_id="pubchem-cid-smiles",
        )

    assert prior.exists()
    assert prior.read_bytes() == b"prior-good-data"


# --- build_csv_select_sql / stream_csv_path branch matrix -----------------


@dataclass(frozen=True)
class _FakeCsvCfg:
    """Minimal stand-in satisfying the `_CsvReadConfig` protocol.

    Each SQL-generation axis is an independent field, so a single branch can be
    toggled without standing up a real corpus that happens to exercise it.
    """

    source: str = "testsrc"
    smiles_column: str = "smiles"
    id_column: str = "id"
    id_column_type: Literal["VARCHAR", "BIGINT"] = "VARCHAR"
    delim: str = "\t"
    has_header: bool = False
    file_compression: Literal["gzip", "zstd", "none"] = "gzip"
    csv_read_mode: Literal["positional", "named"] = "positional"
    positional_id_first: bool = False
    normalize_names: bool = False
    drop_null_smiles: bool = False
    coalesce_null_smiles: bool = False
    disable_quoting: bool = False
    rows_per_batch: int = 1024


def _norm(sql: str) -> str:
    """Collapse whitespace so assertions ignore the SQL template's alignment."""
    return " ".join(sql.split())


def test_build_sql_positional_smiles_first_column_order() -> None:
    sql = build_csv_select_sql(_FakeCsvCfg(id_column_type="BIGINT"))
    assert "columns={'smiles': 'VARCHAR', 'id': 'BIGINT'}" in sql
    assert "auto_detect=False" in sql


def test_build_sql_positional_id_first_column_order() -> None:
    sql = build_csv_select_sql(
        _FakeCsvCfg(id_column_type="BIGINT", positional_id_first=True)
    )
    assert "columns={'id': 'BIGINT', 'smiles': 'VARCHAR'}" in sql


def test_build_sql_positional_varchar_id_is_not_cast() -> None:
    # An already-VARCHAR positional id is projected raw; a redundant CAST would
    # make DuckDB's Arrow encoding nondeterministic across runs.
    sql = build_csv_select_sql(_FakeCsvCfg(id_column_type="VARCHAR"))
    assert '"id" AS source_id' in _norm(sql)
    assert 'CAST("id"' not in sql


def test_build_sql_positional_bigint_id_is_cast() -> None:
    sql = build_csv_select_sql(_FakeCsvCfg(id_column_type="BIGINT"))
    assert 'CAST("id" AS VARCHAR) AS source_id' in _norm(sql)


def test_build_sql_named_mode_autodetects_and_casts_id() -> None:
    sql = build_csv_select_sql(_FakeCsvCfg(csv_read_mode="named"))
    assert "auto_detect=True" in sql
    assert "header=True" in sql
    assert 'CAST("id" AS VARCHAR) AS source_id' in _norm(sql)


@pytest.mark.parametrize("normalize", [True, False])
def test_build_sql_named_mode_normalize_names_toggles(normalize: bool) -> None:
    sql = build_csv_select_sql(
        _FakeCsvCfg(csv_read_mode="named", normalize_names=normalize)
    )
    assert ("normalize_names=True" in sql) == normalize


@pytest.mark.parametrize("coalesce", [True, False])
def test_build_sql_coalesce_wraps_smiles(coalesce: bool) -> None:
    sql = _norm(build_csv_select_sql(_FakeCsvCfg(coalesce_null_smiles=coalesce)))
    if coalesce:
        assert "COALESCE(\"smiles\", '') AS smiles" in sql
    else:
        assert '"smiles" AS smiles' in sql
        assert "COALESCE" not in sql


@pytest.mark.parametrize("disable", [True, False])
def test_build_sql_disable_quoting_adds_quote_escape(disable: bool) -> None:
    sql = build_csv_select_sql(_FakeCsvCfg(disable_quoting=disable))
    assert ("quote=''" in sql) == disable
    assert ("escape=''" in sql) == disable


@pytest.mark.parametrize("drop", [True, False])
def test_build_sql_drop_null_smiles_adds_where(drop: bool) -> None:
    sql = build_csv_select_sql(_FakeCsvCfg(drop_null_smiles=drop))
    if drop:
        assert 'WHERE "smiles" IS NOT NULL AND length("smiles") > 0' in _norm(sql)
    else:
        assert "WHERE" not in sql


@pytest.mark.parametrize("field", ["id_column", "smiles_column"])
def test_build_sql_rejects_injection_in_column_name(field: str) -> None:
    # The positional branch interpolates column names raw into `columns={...}`;
    # the only guard is `quote_ident` raising before that interpolation.
    # `field` is a plain str, so `**{field: ...}` would otherwise make the
    # checker assume the value could feed any of _FakeCsvCfg's params; typing
    # the splat as Any keeps it assignable to all of them.
    overrides: dict[str, Any] = {field: "smiles; DROP TABLE t"}
    with pytest.raises(ValueError, match="invalid SQL identifier"):
        build_csv_select_sql(_FakeCsvCfg(**overrides))


@pytest.fixture
def captured_csv_params(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Patch the DuckDB layer so `stream_csv_path` records its bind params."""
    captured: dict[str, object] = {}

    def _capture(
        _con: object, _sql: str, params: list[object], _rows: int
    ) -> Iterator[object]:
        captured["params"] = params
        return iter(())

    monkeypatch.setattr("smiles_subword.ingest._common.stream_arrow_batches", _capture)
    monkeypatch.setattr("smiles_subword.ingest._common.open_duckdb", lambda **_: None)
    return captured


def test_stream_csv_path_positional_binds_header_and_compression(
    captured_csv_params: dict[str, object], tmp_path: Path
) -> None:
    cfg = _FakeCsvCfg(
        csv_read_mode="positional",
        source="src",
        delim="\t",
        has_header=True,
        file_compression="gzip",
    )
    ts = ingest_timestamp()
    path = tmp_path / "x.smi"

    list(stream_csv_path(cfg, path, ts))

    assert captured_csv_params["params"] == ["src", ts, str(path), "\t", True, "gzip"]


def test_stream_csv_path_named_mode_omits_header_and_compression(
    captured_csv_params: dict[str, object], tmp_path: Path
) -> None:
    cfg = _FakeCsvCfg(csv_read_mode="named", source="src", delim=",")
    ts = ingest_timestamp()
    path = tmp_path / "x.csv"

    list(stream_csv_path(cfg, path, ts))

    assert captured_csv_params["params"] == ["src", ts, str(path), ","]
