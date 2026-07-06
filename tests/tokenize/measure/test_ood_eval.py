"""Tests for ``smiles_subword.tokenize.measure.supplementary.transfer.ood``."""

from __future__ import annotations

import math

from smiles_subword.tokenize.measure.fertility import relative_fertility_gap
from smiles_subword.tokenize.measure.supplementary.transfer.ood import (
    OOD_CORPORA,
    ood_canon_dir,
)


class TestRelativeFertilityGap:
    def test_matches_eq_fertility(self) -> None:
        assert relative_fertility_gap(10.0, 14.0) == 4.0 / 12.0

    def test_symmetric_in_arms(self) -> None:
        assert relative_fertility_gap(14.0, 10.0) == relative_fertility_gap(10.0, 14.0)

    def test_zero_when_equal(self) -> None:
        assert relative_fertility_gap(7.0, 7.0) == 0.0

    def test_nan_on_degenerate_zero_mean(self) -> None:
        assert math.isnan(relative_fertility_gap(0.0, 0.0))


class TestOodCanonDir:
    def test_tmqm_uses_dative_reset_derivation(self) -> None:
        assert ood_canon_dir("tmqm").name == "opensmiles_v1"

    def test_other_corpora_use_standard_canon(self) -> None:
        assert ood_canon_dir("cycpeptmpdb").name == "canon_dedup_v1"

    def test_path_is_under_processed_corpus(self) -> None:
        path = ood_canon_dir("tmqm")
        assert path.parent.name == "tmqm"
        assert path.parent.parent.name == "processed"


def test_tmqm_is_an_ood_corpus() -> None:
    assert "tmqm" in OOD_CORPORA
