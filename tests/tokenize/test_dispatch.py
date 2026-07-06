"""Tests for ``smiles_subword.tokenize.dispatch`` (shared grid/extras orchestration).

Pin cell resolution, the artifact-reload + skip-if-done freshness check, config
materialization, command building, and the per-cell / sweep dispatch (skip,
dry-run, subprocess, the three post-train hooks, scaffold aggregation) without
training a tokenizer. The ``scripts/`` drivers (train/audit dispatch) are thin
argparse bindings over these functions and carry no dedicated tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Never

import pytest
import yaml

from smiles_subword.config import TokenizerConfig
from smiles_subword.tokenize import dispatch

if TYPE_CHECKING:
    from collections.abc import Callable

A_CELL = "pubchem__smirk_gpe_v256_nmb"


@dataclass(frozen=True)
class _FakeCell:
    """Minimal stand-in satisfying the ``DispatchCell`` protocol (a ``cell_id``)."""

    cell_id: str


def _gpe_config(out: Path, corpus_dir: Path) -> TokenizerConfig:
    return TokenizerConfig(
        name="smirk_gpe_v256_nmb",
        kind="smirk_gpe",
        vocab_size=256,
        corpus="pubchem",
        training_input=corpus_dir,
        output_dir=out,
    )


def _to_config(out: Path, corpus_dir: Path) -> Callable[[object], TokenizerConfig]:
    return lambda _cell: _gpe_config(out, corpus_dir)


def _write_artifact(out: Path, *, sha: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "tokenizer.json").write_text("{}")
    (out / "merges.txt").write_text("")
    (out / "meta.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "smirk_gpe_v256_nmb",
                "base_kind": "smirk_gpe",
                "vocab_size": 256,
                "training_corpus_sha": sha,
            }
        )
    )


def _seams(
    to_config: Callable[[object], TokenizerConfig],
    *,
    cache_dir: Path,
    confirm_hook: Callable[..., object] | None = None,
    verify_hook: Callable[..., object] | None = None,
) -> dispatch.DispatchSeams[_FakeCell]:
    return dispatch.DispatchSeams(
        to_config=to_config,
        confirm_hook=confirm_hook or (lambda *_a, **_k: None),
        verify_hook=verify_hook or (lambda *_a, **_k: None),
        cache_dir=cache_dir,
    )


class TestResolveCell:
    def test_returns_the_matching_cell(self) -> None:
        cells = [_FakeCell("a"), _FakeCell("b")]

        assert dispatch.resolve_cell("b", cells).cell_id == "b"

    def test_unknown_cell_raises_with_the_valid_list(self) -> None:
        with pytest.raises(FileNotFoundError, match="unknown cell 'bogus'"):
            dispatch.resolve_cell("bogus", [_FakeCell("a")])


class TestBuildCommand:
    def test_runs_train_tokenizer_via_uv(self) -> None:
        cmd = dispatch.build_command(Path("/tmp/x.yaml"))

        assert cmd[:3] == ["uv", "run", "python"]
        assert cmd[-2:] == ["--config", "/tmp/x.yaml"]
        assert "train_tokenizer.py" in " ".join(cmd)


class TestArtifactReloads:
    def test_true_when_the_loader_succeeds(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(dispatch.SmirkAdapter, "load", lambda _dir: object())

        assert dispatch.artifact_reloads(tmp_path, kind="smirk_gpe") is True

    def test_false_when_the_loader_raises(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        def boom(_dir: Path) -> Never:
            raise ValueError("corrupt tokenizer.json")

        monkeypatch.setattr(dispatch.SmirkAdapter, "load", boom)

        assert dispatch.artifact_reloads(tmp_path, kind="smirk_gpe") is False


class TestIsCellDone:
    def test_done_when_artifact_complete_and_sha_matches(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        out = tmp_path / "art"
        _write_artifact(out, sha="corpus-sha")
        monkeypatch.setattr(dispatch, "artifact_reloads", lambda _d, *, kind: True)
        monkeypatch.setattr(dispatch, "training_corpus_sha", lambda _p: "corpus-sha")

        assert (
            dispatch.is_cell_done(
                _FakeCell(A_CELL), to_config=_to_config(out, tmp_path)
            )
            is True
        )

    def test_not_done_when_tokenizer_json_missing(self, tmp_path: Path) -> None:
        out = tmp_path / "art"
        out.mkdir()

        assert (
            dispatch.is_cell_done(
                _FakeCell(A_CELL), to_config=_to_config(out, tmp_path)
            )
            is False
        )

    def test_not_done_when_the_artifact_does_not_reload(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        out = tmp_path / "art"
        _write_artifact(out, sha="corpus-sha")
        monkeypatch.setattr(dispatch, "artifact_reloads", lambda _d, *, kind: False)

        assert (
            dispatch.is_cell_done(
                _FakeCell(A_CELL), to_config=_to_config(out, tmp_path)
            )
            is False
        )

    def test_not_done_when_training_corpus_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        out = tmp_path / "art"
        _write_artifact(out, sha="corpus-sha")
        monkeypatch.setattr(dispatch, "artifact_reloads", lambda _d, *, kind: True)
        absent = tmp_path / "missing"

        assert (
            dispatch.is_cell_done(_FakeCell(A_CELL), to_config=_to_config(out, absent))
            is False
        )

    def test_not_done_when_corpus_sha_drifted(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        out = tmp_path / "art"
        _write_artifact(out, sha="stale-sha")
        monkeypatch.setattr(dispatch, "artifact_reloads", lambda _d, *, kind: True)
        monkeypatch.setattr(dispatch, "training_corpus_sha", lambda _p: "fresh-sha")

        assert (
            dispatch.is_cell_done(
                _FakeCell(A_CELL), to_config=_to_config(out, tmp_path)
            )
            is False
        )


class TestMaterializeConfig:
    def test_writes_a_parseable_config_to_the_cache(self, tmp_path: Path) -> None:
        cache = tmp_path / "cache"

        out_path = dispatch.materialize_config(
            _FakeCell(A_CELL),
            to_config=_to_config(tmp_path / "art", tmp_path),
            cache_dir=cache,
        )

        assert out_path.parent == cache
        assert out_path.name == f"{A_CELL}.yaml"
        cfg = TokenizerConfig.from_yaml(out_path)
        assert cfg.name == "smirk_gpe_v256_nmb"
        assert cfg.kind == "smirk_gpe"


class TestDispatchOne:
    def test_dry_run_logs_command_and_skips_subprocess(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(dispatch, "is_cell_done", lambda *_a, **_k: False)

        def fake_run(*_args: object, **_kwargs: object) -> Never:
            raise AssertionError("subprocess.run must not run on dry_run")

        monkeypatch.setattr(dispatch.subprocess, "run", fake_run)
        logs: list[str] = []

        dispatch.dispatch_one(
            _FakeCell("c1"),
            _seams(
                _to_config(tmp_path / "art", tmp_path), cache_dir=tmp_path / "cache"
            ),
            log=logs.append,
            dry_run=True,
            force=False,
            confirm_f95=False,
            verify_determinism=False,
            retrain_scaffold=False,
        )

        assert "dispatch c1" in logs
        assert any("train_tokenizer.py" in line for line in logs)

    def test_real_run_invokes_subprocess_with_check_true(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(dispatch, "is_cell_done", lambda *_a, **_k: False)
        captured: dict[str, object] = {}

        def fake_run(cmd: list[str], **kwargs: object) -> None:
            captured["cmd"] = cmd
            captured["kwargs"] = kwargs

        monkeypatch.setattr(dispatch.subprocess, "run", fake_run)

        dispatch.dispatch_one(
            _FakeCell("c1"),
            _seams(
                _to_config(tmp_path / "art", tmp_path), cache_dir=tmp_path / "cache"
            ),
            log=lambda _m: None,
            dry_run=False,
            force=False,
            confirm_f95=False,
            verify_determinism=False,
            retrain_scaffold=False,
        )

        assert captured["kwargs"] == {"check": True}
        assert "train_tokenizer.py" in " ".join(captured["cmd"])  # type: ignore[arg-type]

    def test_skips_when_already_done(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(dispatch, "is_cell_done", lambda *_a, **_k: True)

        def no_train(*_a: object, **_k: object) -> Never:
            raise AssertionError("an already-trained cell must not be re-trained")

        monkeypatch.setattr(dispatch.subprocess, "run", no_train)
        logs: list[str] = []

        dispatch.dispatch_one(
            _FakeCell("c1"),
            _seams(
                _to_config(tmp_path / "art", tmp_path), cache_dir=tmp_path / "cache"
            ),
            log=logs.append,
            dry_run=False,
            force=False,
            confirm_f95=False,
            verify_determinism=False,
            retrain_scaffold=False,
        )

        assert logs == ["skip c1 (already trained)"]

    def test_hooks_run_after_an_already_trained_cell(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(dispatch, "is_cell_done", lambda *_a, **_k: True)
        monkeypatch.setattr(dispatch.subprocess, "run", lambda *_a, **_k: None)
        confirmed: list[str] = []
        verified: list[str] = []
        seams = _seams(
            _to_config(tmp_path / "art", tmp_path),
            cache_dir=tmp_path / "cache",
            confirm_hook=lambda cell, **_k: confirmed.append(cell.cell_id),
            verify_hook=lambda cell, **_k: verified.append(cell.cell_id),
        )

        dispatch.dispatch_one(
            _FakeCell("c1"),
            seams,
            log=lambda _m: None,
            dry_run=False,
            force=False,
            confirm_f95=True,
            verify_determinism=True,
            retrain_scaffold=False,
        )

        assert confirmed == ["c1"]
        assert verified == ["c1"]

    def test_hooks_do_not_run_without_their_flags(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(dispatch, "is_cell_done", lambda *_a, **_k: False)
        monkeypatch.setattr(dispatch.subprocess, "run", lambda *_a, **_k: None)

        def must_not_fire(*_a: object, **_k: object) -> Never:
            raise AssertionError("hook must not run without its flag")

        seams = _seams(
            _to_config(tmp_path / "art", tmp_path),
            cache_dir=tmp_path / "cache",
            confirm_hook=must_not_fire,
            verify_hook=must_not_fire,
        )

        dispatch.dispatch_one(
            _FakeCell("c1"),
            seams,
            log=lambda _m: None,
            dry_run=False,
            force=False,
            confirm_f95=False,
            verify_determinism=False,
            retrain_scaffold=False,
        )


class TestRunDispatch:
    def test_skips_done_cells_and_dispatches_the_rest(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        cells = [_FakeCell(f"c{i}") for i in range(8)]
        done = {f"c{i}" for i in range(5)}
        monkeypatch.setattr(
            dispatch, "is_cell_done", lambda cell, **_k: cell.cell_id in done
        )
        logs: list[str] = []

        dispatch.run_dispatch(
            cells,
            _seams(
                _to_config(tmp_path / "art", tmp_path), cache_dir=tmp_path / "cache"
            ),
            log=logs.append,
            dry_run=True,
            force=False,
            confirm_f95=False,
            verify_determinism=False,
            retrain_scaffold=False,
        )

        skipped = {ln.split()[1] for ln in logs if ln.startswith("skip ")}
        dispatched = {ln.split()[1] for ln in logs if ln.startswith("dispatch ")}
        assert skipped == done
        assert dispatched == {f"c{i}" for i in range(5, 8)}

    def test_force_redispatches_done_cells(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(dispatch, "is_cell_done", lambda *_a, **_k: True)
        logs: list[str] = []

        dispatch.run_dispatch(
            [_FakeCell("c1")],
            _seams(
                _to_config(tmp_path / "art", tmp_path), cache_dir=tmp_path / "cache"
            ),
            log=logs.append,
            dry_run=True,
            force=True,
            confirm_f95=False,
            verify_determinism=False,
            retrain_scaffold=False,
        )

        assert any(ln.startswith("dispatch c1") for ln in logs)
        assert not any(ln.startswith("skip ") for ln in logs)

    def test_scaffold_audit_deposited_once_over_all_cells(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(dispatch, "is_cell_done", lambda *_a, **_k: True)
        monkeypatch.setattr(
            dispatch, "retrain_with_scaffold_log", lambda **_k: object()
        )
        calls: list[int] = []

        def fake_record(results: list[object]) -> tuple[Path, Path]:
            calls.append(len(results))
            return tmp_path / "audit.json", tmp_path / "audit.md"

        monkeypatch.setattr(dispatch, "record_results", fake_record)
        logs: list[str] = []

        dispatch.run_dispatch(
            [_FakeCell("c1"), _FakeCell("c2")],
            _seams(
                _to_config(tmp_path / "art", tmp_path), cache_dir=tmp_path / "cache"
            ),
            log=logs.append,
            dry_run=True,
            force=False,
            confirm_f95=False,
            verify_determinism=False,
            retrain_scaffold=True,
        )

        assert calls == [2]
        assert any("audit" in ln for ln in logs)
