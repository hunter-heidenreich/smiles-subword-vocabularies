"""Tests for ``smiles_subword.tokenize.audit.scaffold_retrain``.

The subprocess training step is stubbed via ``monkeypatch`` so the
fixture exercises the orchestration (byte-identity assertion, log copy,
meta patch) without invoking the real ``train_tokenizer.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from smiles_subword.config import TokenizerConfig
from smiles_subword.tokenize.audit import scaffold_retrain
from smiles_subword.tokenize.audit.scaffold_retrain import retrain_with_scaffold_log


def _make_config(
    *,
    kind: str,
    output_dir: Path,
    training_input: Path | None = None,
    vocab_size: int = 256,
) -> TokenizerConfig:
    training = training_input if training_input is not None else output_dir / "in"
    if training_input is None:
        training.mkdir(parents=True, exist_ok=True)
    return TokenizerConfig(
        name="cell",
        kind=kind,  # type: ignore[arg-type]
        vocab_size=vocab_size if kind in ("smirk_gpe", "smirk_unigram") else None,
        training_input=training,
        output_dir=output_dir,
    )


def _write_canonical(
    canonical_dir: Path, *, tokenizer_bytes: bytes, merges_bytes: bytes
) -> None:
    canonical_dir.mkdir(parents=True, exist_ok=True)
    (canonical_dir / "tokenizer.json").write_bytes(tokenizer_bytes)
    (canonical_dir / "merges.txt").write_bytes(merges_bytes)
    meta = {
        "name": "cell",
        "base_kind": "smirk_gpe",
        "vocab_size": 256,
        "training_corpus_sha": "abc",
        "merge_brackets": False,
        "split_structure": True,
        "n_merges": 97,
    }
    (canonical_dir / "meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))


def _fake_train(
    monkeypatch: pytest.MonkeyPatch,
    *,
    tokenizer_bytes: bytes,
    merges_bytes: bytes,
    scaffold_bytes: bytes,
) -> list[dict[str, Any]]:
    """Replace subprocess.run with a stub that writes the requested artifacts."""
    calls: list[dict[str, Any]] = []

    def fake_run(cmd: list[str], *, check: bool = True) -> None:
        del check
        config_path = Path(cmd[cmd.index("--config") + 1])
        cfg_payload = yaml.safe_load(config_path.read_text())
        scratch = Path(cfg_payload["output_dir"])
        scratch.mkdir(parents=True, exist_ok=True)
        (scratch / "tokenizer.json").write_bytes(tokenizer_bytes)
        (scratch / "merges.txt").write_bytes(merges_bytes)
        (scratch / "scaffold.jsonl").write_bytes(scaffold_bytes)
        calls.append({"cmd": cmd, "config": cfg_payload})

    monkeypatch.setattr(scaffold_retrain._runtime.subprocess, "run", fake_run)
    return calls


class TestSkipsNonBpe:
    def test_unigram_kind_returns_skipped(self, tmp_path: Path) -> None:
        canonical = tmp_path / "canonical"
        canonical.mkdir()
        (canonical / "tokenizer.json").write_text("anything")
        cfg = _make_config(kind="smirk_unigram", output_dir=canonical)

        result = retrain_with_scaffold_log(
            cell_id="x", canonical_dir=canonical, base_config=cfg
        )

        assert result.status == "skipped"
        assert result.reason == "not-bpe"


class TestCanonicalArtifactMissing:
    def test_missing_tokenizer_json_returns_failed(self, tmp_path: Path) -> None:
        # A BPE cell whose canonical artifact is absent — the sole failed-status
        # path that is not a byte mismatch; it returns rather than raising.
        canonical = tmp_path / "canonical"
        canonical.mkdir()
        cfg = _make_config(kind="smirk_gpe", output_dir=canonical)

        result = retrain_with_scaffold_log(
            cell_id="absent", canonical_dir=canonical, base_config=cfg
        )

        assert result.status == "failed"
        assert result.reason == "canonical_artifact_missing"
        assert result.scaffold_log_sha is None


class TestTrainIntoEmitsLog:
    def test_raises_when_the_fork_does_not_emit_the_scaffold_log(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # _runtime.train_into runs but no scaffold.jsonl lands in scratch — a
        # broken fork contract; _train_into must surface it, not return silently.
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        monkeypatch.setattr(
            scaffold_retrain._runtime, "train_into", lambda _cfg, _s, **_k: None
        )
        cfg = _make_config(kind="smirk_gpe", output_dir=tmp_path / "out")

        with pytest.raises(RuntimeError, match="did not emit the log"):
            scaffold_retrain._train_into(cfg, scratch)


class TestSuccessfulRetrain:
    def test_writes_scaffold_log_and_patches_meta(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        canonical = tmp_path / "canonical"
        tokenizer_bytes = b'{"tokenizer":"json"}'
        merges_bytes = b"#version: 0.2\nCC CC\n"
        scaffold_bytes = b'{"format":"smirk-scaffold-log/v1"}\n{"step":0}\n'
        _write_canonical(
            canonical, tokenizer_bytes=tokenizer_bytes, merges_bytes=merges_bytes
        )
        _fake_train(
            monkeypatch,
            tokenizer_bytes=tokenizer_bytes,
            merges_bytes=merges_bytes,
            scaffold_bytes=scaffold_bytes,
        )
        cfg = _make_config(kind="smirk_gpe", output_dir=canonical)

        result = retrain_with_scaffold_log(
            cell_id="cell-x", canonical_dir=canonical, base_config=cfg
        )

        assert result.status == "ok"
        assert (canonical / "scaffold.jsonl").read_bytes() == scaffold_bytes
        meta = yaml.safe_load((canonical / "meta.yaml").read_text())
        assert isinstance(meta["scaffold_log_sha"], str)
        assert len(meta["scaffold_log_sha"]) == 64
        assert result.scaffold_log_sha == meta["scaffold_log_sha"]

    def test_canonical_tokenizer_bytes_untouched(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        canonical = tmp_path / "canonical"
        tokenizer_bytes = b'{"tokenizer":"json"}'
        merges_bytes = b"#version: 0.2\nCC CC\n"
        _write_canonical(
            canonical, tokenizer_bytes=tokenizer_bytes, merges_bytes=merges_bytes
        )
        _fake_train(
            monkeypatch,
            tokenizer_bytes=tokenizer_bytes,
            merges_bytes=merges_bytes,
            scaffold_bytes=b"hdr\nrec\n",
        )
        cfg = _make_config(kind="smirk_gpe", output_dir=canonical)

        retrain_with_scaffold_log(
            cell_id="cell", canonical_dir=canonical, base_config=cfg
        )

        assert (canonical / "tokenizer.json").read_bytes() == tokenizer_bytes
        assert (canonical / "merges.txt").read_bytes() == merges_bytes


class TestByteIdentityMismatch:
    def test_raises_on_tokenizer_json_drift(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        canonical = tmp_path / "canonical"
        _write_canonical(
            canonical,
            tokenizer_bytes=b"canonical-tok",
            merges_bytes=b"canonical-merges",
        )
        _fake_train(
            monkeypatch,
            tokenizer_bytes=b"DIFFERENT-tok",
            merges_bytes=b"canonical-merges",
            scaffold_bytes=b"x",
        )
        cfg = _make_config(kind="smirk_gpe", output_dir=canonical)

        with pytest.raises(RuntimeError, match="contract violation"):
            retrain_with_scaffold_log(
                cell_id="bad-cell", canonical_dir=canonical, base_config=cfg
            )

    def test_raises_on_merges_txt_drift(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        canonical = tmp_path / "canonical"
        _write_canonical(
            canonical,
            tokenizer_bytes=b"canonical-tok",
            merges_bytes=b"canonical-merges",
        )
        _fake_train(
            monkeypatch,
            tokenizer_bytes=b"canonical-tok",
            merges_bytes=b"DIFFERENT-merges",
            scaffold_bytes=b"x",
        )
        cfg = _make_config(kind="smirk_gpe", output_dir=canonical)

        with pytest.raises(RuntimeError, match=r"merges\.txt_mismatch"):
            retrain_with_scaffold_log(
                cell_id="cell", canonical_dir=canonical, base_config=cfg
            )


class TestIdempotency:
    def test_already_done_skips_retrain(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import hashlib

        canonical = tmp_path / "canonical"
        _write_canonical(canonical, tokenizer_bytes=b"tok", merges_bytes=b"merges")
        scaffold_body = b'{"format":"smirk-scaffold-log/v1"}\n'
        (canonical / "scaffold.jsonl").write_bytes(scaffold_body)
        sha = hashlib.sha256(scaffold_body).hexdigest()
        meta = yaml.safe_load((canonical / "meta.yaml").read_text())
        meta["scaffold_log_sha"] = sha
        (canonical / "meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
        calls = _fake_train(
            monkeypatch,
            tokenizer_bytes=b"tok",
            merges_bytes=b"merges",
            scaffold_bytes=b"new",
        )
        cfg = _make_config(kind="smirk_gpe", output_dir=canonical)

        result = retrain_with_scaffold_log(
            cell_id="cell", canonical_dir=canonical, base_config=cfg
        )

        assert result.status == "already_done"
        assert result.scaffold_log_sha == sha
        assert calls == []

    def test_stale_sha_retrains_without_force(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # scaffold.jsonl is present but its SHA no longer matches the meta record
        # (a stale sidecar): the freshness check fails, so it retrains even
        # without force.
        canonical = tmp_path / "canonical"
        _write_canonical(canonical, tokenizer_bytes=b"tok", merges_bytes=b"merges")
        (canonical / "scaffold.jsonl").write_bytes(b"stale-body")
        meta = yaml.safe_load((canonical / "meta.yaml").read_text())
        meta["scaffold_log_sha"] = "deadbeef" * 8  # does not match the body
        (canonical / "meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
        calls = _fake_train(
            monkeypatch,
            tokenizer_bytes=b"tok",
            merges_bytes=b"merges",
            scaffold_bytes=b"FRESH",
        )
        cfg = _make_config(kind="smirk_gpe", output_dir=canonical)

        result = retrain_with_scaffold_log(
            cell_id="cell", canonical_dir=canonical, base_config=cfg
        )

        assert result.status == "ok"
        assert (canonical / "scaffold.jsonl").read_bytes() == b"FRESH"
        assert calls != []

    def test_force_retrains_even_when_fresh(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        canonical = tmp_path / "canonical"
        _write_canonical(canonical, tokenizer_bytes=b"tok", merges_bytes=b"merges")
        (canonical / "scaffold.jsonl").write_bytes(b"old")
        meta = yaml.safe_load((canonical / "meta.yaml").read_text())
        meta["scaffold_log_sha"] = "stale" * 12
        (canonical / "meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
        _fake_train(
            monkeypatch,
            tokenizer_bytes=b"tok",
            merges_bytes=b"merges",
            scaffold_bytes=b"NEW",
        )
        cfg = _make_config(kind="smirk_gpe", output_dir=canonical)

        result = retrain_with_scaffold_log(
            cell_id="cell", canonical_dir=canonical, base_config=cfg, force=True
        )

        assert result.status == "ok"
        assert (canonical / "scaffold.jsonl").read_bytes() == b"NEW"


class TestDryRun:
    def test_returns_skipped_without_subprocess_invocation(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        canonical = tmp_path / "canonical"
        _write_canonical(canonical, tokenizer_bytes=b"tok", merges_bytes=b"merges")
        calls = _fake_train(
            monkeypatch,
            tokenizer_bytes=b"tok",
            merges_bytes=b"merges",
            scaffold_bytes=b"sl",
        )
        cfg = _make_config(kind="smirk_gpe", output_dir=canonical)

        result = retrain_with_scaffold_log(
            cell_id="cell", canonical_dir=canonical, base_config=cfg, dry_run=True
        )

        assert result.status == "skipped"
        assert result.reason == "dry_run"
        assert calls == []
        assert not (canonical / "scaffold.jsonl").exists()
