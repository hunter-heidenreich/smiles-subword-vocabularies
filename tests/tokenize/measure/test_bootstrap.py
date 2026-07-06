"""Tests for the shared bootstrap-CI recipe (``measure._bootstrap``).

This module is the single home for the percentile-bootstrap machinery that
every measurement reduces through; pinning it here pins the estimator and the
seed recipe for all of them at once.
"""

from __future__ import annotations

import math

import pytest

from smiles_subword.tokenize.measure._bootstrap import (
    CI_LEVEL,
    N_BOOTSTRAP_RESAMPLES,
    bootstrap_ratio_ci,
    bootstrap_seed,
    percentile_ci,
)


class TestConstants:
    def test_frozen_bootstrap_constants(self) -> None:
        assert N_BOOTSTRAP_RESAMPLES == 1000
        assert CI_LEVEL == 0.95


class TestPercentileCi:
    """The lower-order-statistic percentile estimator (no interpolation)."""

    def test_empty_sample_is_nan_nan(self) -> None:
        lo, hi = percentile_ci([])
        assert math.isnan(lo)
        assert math.isnan(hi)

    def test_single_sample_collapses_to_the_value(self) -> None:
        assert percentile_ci([3.5]) == (3.5, 3.5)

    def test_picks_0indexed_25_and_974_for_1000_samples(self) -> None:
        # The documented contract: at level=0.95 over 1000 sorted samples the
        # 0-indexed 25th and 974th entries bracket the CI (lower order
        # statistic, no interpolation between neighbours).
        samples = [float(i) for i in range(1000)]
        assert percentile_ci(samples) == (25.0, 974.0)

    def test_is_order_independent(self) -> None:
        ascending = [float(i) for i in range(100)]
        shuffled = ascending[::-1]
        assert percentile_ci(shuffled) == percentile_ci(ascending)

    def test_level_widens_the_interval(self) -> None:
        samples = [float(i) for i in range(1000)]
        narrow = percentile_ci(samples, level=0.5)
        wide = percentile_ci(samples, level=0.99)
        assert wide[0] <= narrow[0]
        assert wide[1] >= narrow[1]


class TestBootstrapRatioCi:
    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="equal length"):
            bootstrap_ratio_ci([1, 2], [1], seed=0)

    def test_empty_is_nan_nan(self) -> None:
        lo, hi = bootstrap_ratio_ci([], [], seed=0)
        assert math.isnan(lo)
        assert math.isnan(hi)

    def test_all_zero_denominator_is_nan_nan(self) -> None:
        # Every resample sums to a zero denominator -> the ratio is undefined.
        lo, hi = bootstrap_ratio_ci([1, 1, 1], [0, 0, 0], seed=7)
        assert math.isnan(lo)
        assert math.isnan(hi)

    def test_constant_ratio_brackets_the_value(self) -> None:
        # Every molecule has ratio 2/1, so every resample does too.
        lo, hi = bootstrap_ratio_ci([2] * 50, [1] * 50, seed=7)
        assert lo == pytest.approx(2.0)
        assert hi == pytest.approx(2.0)

    def test_is_deterministic_under_fixed_seed(self) -> None:
        a = bootstrap_ratio_ci([1, 2, 3, 4], [1, 1, 1, 1], seed=123)
        b = bootstrap_ratio_ci([1, 2, 3, 4], [1, 1, 1, 1], seed=123)
        assert a == b


class TestSeedRecipe:
    def test_golden_value(self) -> None:
        # Frozen against the pinned recipe (BLAKE2b, digest_size=8, big-endian,
        # masked to uint32). A deliberate change to any of those must update this
        # and re-fingerprint every deposited CI — the range/distinct/stable tests
        # below would all still pass under a swapped recipe, so this is the only
        # guard against a silent CI shift.
        assert bootstrap_seed("pubchem__smirk_gpe_v256_nmb") == 1830740745

    def test_seed_in_uint32_range(self) -> None:
        seed = bootstrap_seed("pubchem__smirk_gpe_v256_nmb")
        assert 0 <= seed < 2**32

    def test_distinct_keys_get_distinct_seeds(self) -> None:
        assert bootstrap_seed("pubchem__smirk_gpe_v256_nmb") != bootstrap_seed(
            "pubchem__smirk_unigram_v256_nmb"
        )

    def test_is_stable(self) -> None:
        assert bootstrap_seed("x") == bootstrap_seed("x")

    def test_is_key_agnostic(self) -> None:
        # One recipe over any key string: a cell_id and a pair_key are just
        # different strings, so the same string yields the same seed regardless
        # of what it denotes.
        assert bootstrap_seed("pubchem__v256_nmb") == bootstrap_seed(
            "pubchem__v256_nmb"
        )
