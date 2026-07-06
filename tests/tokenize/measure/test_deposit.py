"""Tests for ``measure._deposit``'s freshness gate (``standard_arm_block_fresh``).

The deposit/aggregate engine itself (``deposit_all`` / ``build_table`` and the
per-pair write/read) is exercised transitively through every topic's ``*_io``
tests — direct tests there would just duplicate them. This pins the one pure,
contract-bearing piece that is otherwise covered only at a distance: the per-arm
freshness predicate, including its eval-split-missing branch which no topic io
test reaches.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from smiles_subword.tokenize.measure import _deposit

if TYPE_CHECKING:
    import pytest


class TestStandardArmBlockFresh:
    """training_corpus_sha + eval_split_sha must both still match."""

    def test_non_dict_block_is_stale(self) -> None:
        assert _deposit.standard_arm_block_fresh(None) is False
        assert _deposit.standard_arm_block_fresh("not-a-block") is False

    def test_training_sha_drift_is_stale(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # cell_training_sha_fresh returns None when the meta sha has drifted.
        monkeypatch.setattr(_deposit, "cell_training_sha_fresh", lambda _c, _s: None)

        assert (
            _deposit.standard_arm_block_fresh(
                {"cell_id": "pubchem__x", "training_corpus_sha": "T"}
            )
            is False
        )

    def test_eval_split_sha_mismatch_is_stale(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            _deposit, "cell_training_sha_fresh", lambda _c, _s: ("pubchem", "x")
        )
        monkeypatch.setattr(_deposit, "eval_split_sha", lambda _c: "CURRENT")

        assert (
            _deposit.standard_arm_block_fresh(
                {"cell_id": "pubchem__x", "eval_split_sha": "DEPOSITED"}
            )
            is False
        )

    def test_missing_eval_split_manifest_is_stale(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The held-out split was deleted / not yet materialized: eval_split_sha
        # raises, and an unresolvable split means stale, not a crash.
        monkeypatch.setattr(
            _deposit, "cell_training_sha_fresh", lambda _c, _s: ("pubchem", "x")
        )

        def boom(_c: str) -> str:
            raise FileNotFoundError

        monkeypatch.setattr(_deposit, "eval_split_sha", boom)

        assert (
            _deposit.standard_arm_block_fresh(
                {"cell_id": "pubchem__x", "eval_split_sha": "DEPOSITED"}
            )
            is False
        )

    def test_empty_eval_split_manifest_is_stale(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The other eval_split_sha failure mode: a manifest with no shards raises
        # ValueError (from manifest_shard_fingerprint) — also stale.
        monkeypatch.setattr(
            _deposit, "cell_training_sha_fresh", lambda _c, _s: ("pubchem", "x")
        )

        def boom(_c: str) -> str:
            raise ValueError("no shards listed")

        monkeypatch.setattr(_deposit, "eval_split_sha", boom)

        assert (
            _deposit.standard_arm_block_fresh(
                {"cell_id": "pubchem__x", "eval_split_sha": "DEPOSITED"}
            )
            is False
        )

    def test_matching_shas_are_fresh(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            _deposit, "cell_training_sha_fresh", lambda _c, _s: ("pubchem", "x")
        )
        monkeypatch.setattr(_deposit, "eval_split_sha", lambda _c: "SAME")

        assert (
            _deposit.standard_arm_block_fresh(
                {"cell_id": "pubchem__x", "eval_split_sha": "SAME"}
            )
            is True
        )
