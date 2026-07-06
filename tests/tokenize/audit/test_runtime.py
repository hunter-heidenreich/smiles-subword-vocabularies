"""Tests for ``smiles_subword.tokenize.audit._runtime`` (shared orchestration).

The cell plumbing the grid + extras confirm/verify drivers share: a tagged
logger, loading a trained artifact, retrain-and-compare, and the
grid/extras-agnostic :func:`_runtime.run_confirm` / :func:`_runtime.run_verify`
cores. (The drivers themselves are thin wrappers binding their two-or-three
seams onto these.)
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from smiles_subword.config import TokenizerConfig
from smiles_subword.tokenize.audit import _runtime
from smiles_subword.tokenize.audit.determinism import ArtifactDigest, DeterminismResult
from smiles_subword.tokenize.audit.f95 import F95Result
from smiles_subword.tokenize.grid import GridCell, grid_cell_to_config

if TYPE_CHECKING:
    from typing import Never

BPE_CELL = GridCell(
    algo="bpe", vocab_size=256, corpus="pubchem", boundary="nmb", tier="headline"
)
UNI_CELL = GridCell(
    algo="unigram", vocab_size=256, corpus="pubchem", boundary="nmb", tier="headline"
)
EXPECTED_CELL = GridCell(
    algo="unigram", vocab_size=1024, corpus="pubchem", boundary="nmb", tier="headline"
)


def _digest(algo: str) -> ArtifactDigest:
    return ArtifactDigest(
        algo=algo,
        tokenizer_json_sha="t",
        merges_txt_sha="m" if algo == "bpe" else None,
        piece_set_sha=None if algo == "bpe" else "p",
        vocab_order_sha=None,
        log_probs_sha=None,
    )


def _result(
    *,
    algo: str = "bpe",
    deterministic: bool = True,
    mismatch_kind: str | None = None,
    spread: int = 0,
) -> DeterminismResult:
    digest = _digest(algo)
    return DeterminismResult(
        arm=algo,
        deterministic=deterministic,
        mismatch_kind=mismatch_kind,
        rerun_spread=spread,
        canonical=digest,
        rerun=digest,
    )


def _f95_result(*, unsafe: bool = False) -> F95Result:
    headline = 0.70 if unsafe else 1.0
    return F95Result(
        arm="bpe",
        v_observed=256,
        n_non_atomic=90,
        n_corpus_tokens=10,
        n_corpus_molecules=2,
        fp_thresholds=[],
        clearance_by_n={100: headline},
        headline_clearance=headline,
        embedding_tail_unsafe=unsafe,
    )


def _to_cfg(_cell: object) -> SimpleNamespace:
    """A stand-in TokenizerConfig — only ``training_input``/``output_dir`` read."""
    return SimpleNamespace(training_input="corpus.parquet", output_dir="/tmp/out")


class TestLoadTrained:
    def test_returns_the_loaded_tokenizer(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        sentinel = object()
        monkeypatch.setattr(_runtime.SmirkAdapter, "load", lambda _d: sentinel)

        assert _runtime.load_trained(BPE_CELL, tmp_path) is sentinel

    def test_returns_none_when_the_loader_raises(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        def boom(_d: Path) -> Never:
            raise ValueError("truncated tokenizer.json")

        monkeypatch.setattr(_runtime.UnigramSmirkAdapter, "load", boom)

        assert _runtime.load_trained(UNI_CELL, tmp_path) is None


class TestTrainInto:
    def test_writes_a_scratch_config_and_shells_train_tokenizer(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        cfg = grid_cell_to_config(BPE_CELL)
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        captured: dict[str, object] = {}

        def fake_run(cmd: list[str], **kwargs: object) -> None:
            captured["cmd"] = cmd
            captured["kwargs"] = kwargs

        monkeypatch.setattr(_runtime.subprocess, "run", fake_run)

        _runtime.train_into(cfg, scratch)

        config_path = scratch / "rerun_config.yaml"
        assert config_path.is_file()
        rerun_cfg = TokenizerConfig.from_yaml(config_path)
        assert rerun_cfg.output_dir == scratch
        assert captured["kwargs"] == {"check": True}
        assert "train_tokenizer.py" in " ".join(captured["cmd"])  # type: ignore[arg-type]

    def test_extra_update_overlays_onto_the_rerun_config(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # The scaffold audit retrains through this seam with scaffold_log=True;
        # the overlay must land in the written rerun config, not just output_dir.
        cfg = grid_cell_to_config(BPE_CELL)
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        monkeypatch.setattr(_runtime.subprocess, "run", lambda _cmd, **_k: None)

        _runtime.train_into(cfg, scratch, extra_update={"scaffold_log": True})

        rerun_cfg = TokenizerConfig.from_yaml(scratch / "rerun_config.yaml")
        assert rerun_cfg.output_dir == scratch
        assert rerun_cfg.scaffold_log is True


class TestRetrainAndCompare:
    def test_deletes_the_scratch_dir_even_when_training_raises(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        monkeypatch.setattr(_runtime.tempfile, "mkdtemp", lambda **_k: str(scratch))

        def boom(_cfg: object, _scratch: Path) -> Never:
            raise RuntimeError("training crashed")

        monkeypatch.setattr(_runtime, "train_into", boom)

        with pytest.raises(RuntimeError, match=r"training crashed"):
            _runtime.retrain_and_compare(
                BPE_CELL, grid_cell_to_config(BPE_CELL), prefix="det-"
            )

        assert not scratch.exists()

    def test_compares_canonical_against_the_rerun(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        monkeypatch.setattr(_runtime.tempfile, "mkdtemp", lambda **_k: str(scratch))
        monkeypatch.setattr(_runtime, "train_into", lambda _cfg, _s: None)
        monkeypatch.setattr(
            _runtime, "digest_artifact", lambda _d, *, algo: _digest(algo)
        )
        sentinel = _result()
        monkeypatch.setattr(_runtime, "compare_artifacts", lambda *_a, **_k: sentinel)

        result = _runtime.retrain_and_compare(
            BPE_CELL, grid_cell_to_config(BPE_CELL), prefix="det-"
        )

        assert result is sentinel
        assert not scratch.exists()

    def test_unigram_branch_passes_glyph_pieces_to_compare(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # The BPE path passes pieces=None; only the Unigram arm wires the
        # canonical/rerun glyph sets through to compare_artifacts.
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        monkeypatch.setattr(_runtime.tempfile, "mkdtemp", lambda **_k: str(scratch))
        monkeypatch.setattr(_runtime, "train_into", lambda _cfg, _s: None)
        monkeypatch.setattr(
            _runtime, "digest_artifact", lambda _d, *, algo: _digest(algo)
        )
        canonical_pieces = frozenset({("C",)})
        rerun_pieces = frozenset({("O",)})
        monkeypatch.setattr(
            _runtime,
            "unigram_glyph_set",
            lambda d: (
                canonical_pieces
                if d == grid_cell_to_config(UNI_CELL).output_dir
                else rerun_pieces
            ),
        )
        captured: dict[str, object] = {}

        def fake_compare(
            _can: object, _re: object, *, pieces: object
        ) -> DeterminismResult:
            captured["pieces"] = pieces
            return _result(algo="unigram")

        monkeypatch.setattr(_runtime, "compare_artifacts", fake_compare)

        _runtime.retrain_and_compare(
            UNI_CELL, grid_cell_to_config(UNI_CELL), prefix="det-"
        )

        assert captured["pieces"] == (canonical_pieces, rerun_pieces)


def _resolves(_cell: object, _tok: object) -> frozenset[str]:
    return frozenset({"C"})


def _unresolved(_cell: object, _tok: object) -> None:
    return None


class TestRunConfirm:
    """The grid/extras-agnostic F_{p,n} confirmation core."""

    def test_dry_run_returns_none_without_loading(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        def must_not_load(_c: object, _d: object) -> Never:
            raise AssertionError("dry-run must not load a tokenizer")

        monkeypatch.setattr(_runtime, "load_trained", must_not_load)

        assert (
            _runtime.run_confirm(
                BPE_CELL,
                to_config=_to_cfg,
                resolve_atomic_tokens=_resolves,
                log=_runtime.make_logger("f95"),
                dry_run=True,
            )
            is None
        )
        assert "dry-run" in capsys.readouterr().err

    def test_skips_when_cell_not_trained(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(_runtime, "load_trained", lambda _c, _d: None)

        assert (
            _runtime.run_confirm(
                BPE_CELL,
                to_config=_to_cfg,
                resolve_atomic_tokens=_resolves,
                log=_runtime.make_logger("f95"),
            )
            is None
        )
        assert "not trained yet" in capsys.readouterr().err

    def test_skips_when_already_confirmed(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(_runtime, "load_trained", lambda _c, _d: object())
        monkeypatch.setattr(_runtime, "training_corpus_sha", lambda _p: "sha-x")
        monkeypatch.setattr(_runtime, "is_f95_done", lambda _c, **_k: True)

        def must_not_compute(*_a: object, **_k: object) -> Never:
            raise AssertionError("an already-confirmed cell must not recompute")

        monkeypatch.setattr(_runtime, "compute_f95", must_not_compute)

        assert (
            _runtime.run_confirm(
                BPE_CELL,
                to_config=_to_cfg,
                resolve_atomic_tokens=_resolves,
                log=_runtime.make_logger("f95"),
                force=False,
            )
            is None
        )
        assert "already confirmed" in capsys.readouterr().err

    def test_force_recomputes_an_already_confirmed_cell(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # is_f95_done is True, but force bypasses the freshness gate.
        monkeypatch.setattr(_runtime, "load_trained", lambda _c, _d: object())
        monkeypatch.setattr(_runtime, "training_corpus_sha", lambda _p: "sha-x")
        monkeypatch.setattr(_runtime, "is_f95_done", lambda _c, **_k: True)
        monkeypatch.setattr(_runtime, "iter_smiles_from_parquet", lambda _p: iter(()))
        monkeypatch.setattr(_runtime, "compute_f95", lambda _t, _s, **_k: _f95_result())
        monkeypatch.setattr(
            _runtime, "write_f95_json", lambda *_a, **_k: Path("/tmp/x.json")
        )

        result = _runtime.run_confirm(
            BPE_CELL,
            to_config=_to_cfg,
            resolve_atomic_tokens=_resolves,
            log=_runtime.make_logger("f95"),
            force=True,
        )

        assert isinstance(result, F95Result)

    def test_skips_when_glyph_alphabet_unresolved(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(_runtime, "load_trained", lambda _c, _d: object())
        monkeypatch.setattr(_runtime, "training_corpus_sha", lambda _p: "sha-x")
        monkeypatch.setattr(_runtime, "is_f95_done", lambda _c, **_k: False)

        assert (
            _runtime.run_confirm(
                UNI_CELL,
                to_config=_to_cfg,
                resolve_atomic_tokens=_unresolved,
                log=_runtime.make_logger("f95"),
            )
            is None
        )
        assert "matched BPE" in capsys.readouterr().err

    def test_computes_deposits_and_passes_arm(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorder: dict[str, object] = {}
        monkeypatch.setattr(_runtime, "load_trained", lambda _c, _d: object())
        monkeypatch.setattr(_runtime, "training_corpus_sha", lambda _p: "sha-x")
        monkeypatch.setattr(_runtime, "is_f95_done", lambda _c, **_k: False)
        monkeypatch.setattr(_runtime, "iter_smiles_from_parquet", lambda _p: iter(()))
        monkeypatch.setattr(
            _runtime,
            "compute_f95",
            lambda _t, _s, **kw: (
                recorder.__setitem__("compute_kwargs", kw),
                _f95_result(),
            )[1],
        )

        def fake_write(cell: object, result: object, **kwargs: object) -> Path:
            recorder["written"] = (cell, result, kwargs)
            return Path("/tmp/f95.json")

        monkeypatch.setattr(_runtime, "write_f95_json", fake_write)

        result = _runtime.run_confirm(
            BPE_CELL,
            to_config=_to_cfg,
            resolve_atomic_tokens=_resolves,
            log=_runtime.make_logger("f95"),
        )

        assert isinstance(result, F95Result)
        cell, written_result, kwargs = recorder["written"]  # type: ignore[misc]
        assert cell == BPE_CELL
        assert written_result is result
        assert kwargs == {"training_corpus_sha": "sha-x"}
        assert recorder["compute_kwargs"] == {  # type: ignore[comparison-overlap]
            "arm": "bpe",
            "atomic_tokens": frozenset({"C"}),
        }

    def test_logs_the_flag_when_embedding_tail_unsafe(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(_runtime, "load_trained", lambda _c, _d: object())
        monkeypatch.setattr(_runtime, "training_corpus_sha", lambda _p: "sha-x")
        monkeypatch.setattr(_runtime, "is_f95_done", lambda _c, **_k: False)
        monkeypatch.setattr(_runtime, "iter_smiles_from_parquet", lambda _p: iter(()))
        monkeypatch.setattr(
            _runtime, "compute_f95", lambda _t, _s, **_k: _f95_result(unsafe=True)
        )
        monkeypatch.setattr(
            _runtime, "write_f95_json", lambda *_a, **_k: Path("/tmp/x.json")
        )

        _runtime.run_confirm(
            BPE_CELL,
            to_config=_to_cfg,
            resolve_atomic_tokens=_resolves,
            log=_runtime.make_logger("f95"),
        )

        assert "EMBEDDING-TAIL-UNSAFE" in capsys.readouterr().err


class TestRunVerify:
    """The grid/extras-agnostic determinism verification core."""

    def test_dry_run_returns_none_without_loading(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        def must_not_load(_c: object, _d: object) -> Never:
            raise AssertionError("dry-run must not load a tokenizer")

        monkeypatch.setattr(_runtime, "load_trained", must_not_load)

        assert (
            _runtime.run_verify(
                BPE_CELL,
                to_config=_to_cfg,
                is_expected_jitter=lambda _c: False,
                log=_runtime.make_logger("det"),
                prefix="det-",
                dry_run=True,
            )
            is None
        )
        assert "dry-run" in capsys.readouterr().err

    def test_skips_when_cell_not_trained(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(_runtime, "load_trained", lambda _c, _d: None)

        assert (
            _runtime.run_verify(
                BPE_CELL,
                to_config=_to_cfg,
                is_expected_jitter=lambda _c: False,
                log=_runtime.make_logger("det"),
                prefix="det-",
            )
            is None
        )
        assert "not trained yet" in capsys.readouterr().err

    def test_skips_when_already_verified(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(_runtime, "load_trained", lambda _c, _d: object())
        monkeypatch.setattr(_runtime, "training_corpus_sha", lambda _p: "sha-x")
        monkeypatch.setattr(_runtime, "is_determinism_done", lambda _c, **_k: True)

        def must_not_retrain(*_a: object, **_k: object) -> Never:
            raise AssertionError("an already-verified cell must not retrain")

        monkeypatch.setattr(_runtime, "retrain_and_compare", must_not_retrain)

        assert (
            _runtime.run_verify(
                BPE_CELL,
                to_config=_to_cfg,
                is_expected_jitter=lambda _c: False,
                log=_runtime.make_logger("det"),
                prefix="det-",
                force=False,
            )
            is None
        )
        assert "already verified" in capsys.readouterr().err

    def test_force_reverifies_an_already_verified_cell(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # is_determinism_done is True, but force bypasses the freshness gate.
        monkeypatch.setattr(_runtime, "load_trained", lambda _c, _d: object())
        monkeypatch.setattr(_runtime, "training_corpus_sha", lambda _p: "sha-x")
        monkeypatch.setattr(_runtime, "is_determinism_done", lambda _c, **_k: True)
        monkeypatch.setattr(
            _runtime, "retrain_and_compare", lambda _c, _cfg, **_k: _result()
        )
        monkeypatch.setattr(
            _runtime, "write_determinism_json", lambda *_a, **_k: Path("/tmp/x.json")
        )

        result = _runtime.run_verify(
            BPE_CELL,
            to_config=_to_cfg,
            is_expected_jitter=lambda _c: False,
            log=_runtime.make_logger("det"),
            prefix="det-",
            force=True,
        )

        assert isinstance(result, DeterminismResult)

    def test_retrains_deposits_and_logs_holds(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        recorder: dict[str, object] = {}
        monkeypatch.setattr(_runtime, "load_trained", lambda _c, _d: object())
        monkeypatch.setattr(_runtime, "training_corpus_sha", lambda _p: "sha-x")
        monkeypatch.setattr(_runtime, "is_determinism_done", lambda _c, **_k: False)
        monkeypatch.setattr(
            _runtime, "retrain_and_compare", lambda _c, _cfg, **_k: _result()
        )

        def fake_write(cell: object, result: object, **kwargs: object) -> Path:
            recorder["written"] = (cell, result, kwargs)
            return Path("/tmp/det.json")

        monkeypatch.setattr(_runtime, "write_determinism_json", fake_write)

        result = _runtime.run_verify(
            BPE_CELL,
            to_config=_to_cfg,
            is_expected_jitter=lambda _c: False,
            log=_runtime.make_logger("det"),
            prefix="det-",
        )

        assert isinstance(result, DeterminismResult)
        cell, written, kwargs = recorder["written"]  # type: ignore[misc]
        assert cell == BPE_CELL
        assert written is result
        assert kwargs == {"training_corpus_sha": "sha-x", "expected_failure": False}
        assert "determinism holds" in capsys.readouterr().err

    def test_passes_the_scratch_prefix_through(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, object] = {}

        def fake_retrain(_c: object, _cfg: object, *, prefix: str) -> DeterminismResult:
            captured["prefix"] = prefix
            return _result()

        monkeypatch.setattr(_runtime, "load_trained", lambda _c, _d: object())
        monkeypatch.setattr(_runtime, "training_corpus_sha", lambda _p: "sha-x")
        monkeypatch.setattr(_runtime, "is_determinism_done", lambda _c, **_k: False)
        monkeypatch.setattr(_runtime, "retrain_and_compare", fake_retrain)
        monkeypatch.setattr(
            _runtime, "write_determinism_json", lambda *_a, **_k: Path("/tmp/x.json")
        )

        _runtime.run_verify(
            BPE_CELL,
            to_config=_to_cfg,
            is_expected_jitter=lambda _c: False,
            log=_runtime.make_logger("det"),
            prefix="det-abc-",
        )

        assert captured["prefix"] == "det-abc-"

    def test_bpe_mismatch_deposits_evidence_then_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorder: dict[str, object] = {}
        monkeypatch.setattr(_runtime, "load_trained", lambda _c, _d: object())
        monkeypatch.setattr(_runtime, "training_corpus_sha", lambda _p: "sha-x")
        monkeypatch.setattr(_runtime, "is_determinism_done", lambda _c, **_k: False)
        monkeypatch.setattr(
            _runtime,
            "retrain_and_compare",
            lambda _c, _cfg, **_k: _result(
                deterministic=False, mismatch_kind="bpe_byte"
            ),
        )
        monkeypatch.setattr(
            _runtime,
            "write_determinism_json",
            lambda *_a, **_k: recorder.__setitem__("written", True),
        )

        with pytest.raises(RuntimeError, match=r"NOT byte-identical"):
            _runtime.run_verify(
                BPE_CELL,
                to_config=_to_cfg,
                is_expected_jitter=lambda _c: False,
                log=_runtime.make_logger("det"),
                prefix="det-",
            )

        assert recorder["written"] is True

    def test_expected_jitter_flags_without_raising(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        written: dict[str, object] = {}
        monkeypatch.setattr(_runtime, "load_trained", lambda _c, _d: object())
        monkeypatch.setattr(_runtime, "training_corpus_sha", lambda _p: "sha-x")
        monkeypatch.setattr(_runtime, "is_determinism_done", lambda _c, **_k: False)
        monkeypatch.setattr(
            _runtime,
            "retrain_and_compare",
            lambda _c, _cfg, **_k: _result(
                algo="unigram",
                deterministic=False,
                mismatch_kind="unigram_piece_set",
                spread=2,
            ),
        )
        monkeypatch.setattr(
            _runtime,
            "write_determinism_json",
            lambda _c, _r, **kw: written.update(kw) or Path("/tmp/x.json"),
        )

        result = _runtime.run_verify(
            EXPECTED_CELL,
            to_config=_to_cfg,
            is_expected_jitter=lambda _c: True,
            log=_runtime.make_logger("det"),
            prefix="det-",
        )

        assert result is not None
        assert not result.deterministic
        assert written["expected_failure"] is True
        assert "FLAGGED" in capsys.readouterr().err

    def test_unexpected_jitter_flags_loudly_without_raising(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        written: dict[str, object] = {}
        monkeypatch.setattr(_runtime, "load_trained", lambda _c, _d: object())
        monkeypatch.setattr(_runtime, "training_corpus_sha", lambda _p: "sha-x")
        monkeypatch.setattr(_runtime, "is_determinism_done", lambda _c, **_k: False)
        monkeypatch.setattr(
            _runtime,
            "retrain_and_compare",
            lambda _c, _cfg, **_k: _result(
                algo="unigram",
                deterministic=False,
                mismatch_kind="unigram_piece_set",
                spread=2,
            ),
        )
        monkeypatch.setattr(
            _runtime,
            "write_determinism_json",
            lambda _c, _r, **kw: written.update(kw) or Path("/tmp/x.json"),
        )

        result = _runtime.run_verify(
            UNI_CELL,
            to_config=_to_cfg,
            is_expected_jitter=lambda _c: False,
            log=_runtime.make_logger("det"),
            prefix="det-",
        )

        assert result is not None
        assert written["expected_failure"] is False
        assert "UNEXPECTED" in capsys.readouterr().err
