"""Tests for ``smiles_subword.tokenize.intrinsics``.

The module hosts the three frequency-distribution formulas the distribution
measurement evaluates on held-out token frequencies. Each is tested directly
against an explicit frequency vector: a uniform distribution, a known skew with
its closed form, the degenerate guards, and resampled-float / numpy inputs.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from smiles_subword.tokenize.intrinsics import (
    normalized_entropy,
    renyi_efficiency,
    token_imbalance,
)


class TestNormalizedEntropy:
    """Shannon entropy of the frequency distribution, rebased to [0, 1]."""

    def test_uniform_distribution_yields_one(self) -> None:
        # Five equally-used tokens over v_effective=5 → maximal entropy.
        assert normalized_entropy([4, 4, 4, 4, 4], total=20, v_effective=5) == (
            pytest.approx(1.0)
        )

    def test_two_token_skew_matches_closed_form(self) -> None:
        freqs = [1000, 1]
        total = 1001
        v = 5
        p_hot, p_rare = 1000 / 1001, 1 / 1001
        expected_h = -(p_hot * math.log2(p_hot) + p_rare * math.log2(p_rare))
        assert normalized_entropy(freqs, total, v) == pytest.approx(
            expected_h / math.log2(5), rel=1e-9
        )

    def test_single_live_token_yields_zero(self) -> None:
        assert normalized_entropy([42], total=42, v_effective=5) == pytest.approx(0.0)

    def test_dead_tokens_ignored(self) -> None:
        # Zero-frequency entries contribute nothing; result tracks the live set.
        with_dead = normalized_entropy([6, 2, 0, 0], total=8, v_effective=4)
        without_dead = normalized_entropy([6, 2], total=8, v_effective=4)
        assert with_dead == pytest.approx(without_dead)

    def test_degenerate_guards_return_zero(self) -> None:
        assert normalized_entropy([5], total=0, v_effective=4) == 0.0
        assert normalized_entropy([5], total=5, v_effective=1) == 0.0


class TestRenyiEfficiency:
    """α=2.5 Rényi efficiency tracks the head of the distribution."""

    def test_uniform_distribution_yields_one(self) -> None:
        assert renyi_efficiency([5, 5, 5, 5, 5], total=25, v_effective=5) == (
            pytest.approx(1.0)
        )

    def test_two_token_skew_matches_closed_form(self) -> None:
        freqs = [1000, 1]
        total = 1001
        v = 5
        alpha = 2.5
        p_hot, p_rare = 1000 / 1001, 1 / 1001
        p_alpha_sum = p_hot**alpha + p_rare**alpha
        h_alpha = math.log2(p_alpha_sum) / (1.0 - alpha)
        assert renyi_efficiency(freqs, total, v) == pytest.approx(
            h_alpha / math.log2(5), rel=1e-9
        )

    def test_skew_drives_efficiency_below_normalized_entropy(self) -> None:
        freqs = [1000, 1]
        assert renyi_efficiency(freqs, 1001, 5) < normalized_entropy(freqs, 1001, 5)

    def test_alpha_one_falls_back_to_shannon(self) -> None:
        # H_α has a removable singularity at α=1 (→ Shannon); the implementation
        # special-cases it by delegating to normalized_entropy.
        freqs = [1000, 50, 5, 1]
        total = sum(freqs)
        v = 8
        assert renyi_efficiency(freqs, total, v, alpha=1.0) == pytest.approx(
            normalized_entropy(freqs, total, v)
        )

    def test_degenerate_guards_return_zero(self) -> None:
        assert renyi_efficiency([5], total=0, v_effective=4) == 0.0
        assert renyi_efficiency([5], total=5, v_effective=1) == 0.0


def _brute_force_d(full_freq_vec: list[float], v_effective: int) -> float:
    """``D`` from an explicit length-``v_effective`` frequency vector (zeros = dead)."""
    assert len(full_freq_vec) == v_effective
    total = sum(full_freq_vec)
    inv_v = 1.0 / v_effective
    return 0.5 * sum(abs(f / total - inv_v) for f in full_freq_vec)


class TestTokenImbalance:
    """The headline statistic ``D = ½ Σ|p_i − 1/|V||`` (Gowda 2020)."""

    def test_uniform_distribution_yields_zero(self) -> None:
        assert token_imbalance([5, 5, 5, 5], total=20, v_effective=4) == pytest.approx(
            0.0
        )

    def test_single_live_token_yields_v_minus_one_over_v(self) -> None:
        v = 10
        assert token_imbalance([42], total=42, v_effective=v) == pytest.approx(
            (v - 1) / v
        )

    def test_matches_brute_force_over_padded_vector(self) -> None:
        v = 8
        live = [60.0, 30.0, 10.0]
        full = live + [0.0] * (v - len(live))
        assert token_imbalance(live, total=sum(live), v_effective=v) == pytest.approx(
            _brute_force_d(full, v)
        )

    def test_dead_in_both_arms_cancels_in_delta(self) -> None:
        v = 10
        arm_a = [70.0, 30.0]
        arm_b = [40.0, 60.0]
        d_a = token_imbalance(arm_a, total=sum(arm_a), v_effective=v)
        d_b = token_imbalance(arm_b, total=sum(arm_b), v_effective=v)

        inv_v = 1.0 / v
        live_only_a = 0.5 * sum(abs(f / sum(arm_a) - inv_v) for f in arm_a)
        live_only_b = 0.5 * sum(abs(f / sum(arm_b) - inv_v) for f in arm_b)
        assert (d_a - d_b) == pytest.approx(live_only_a - live_only_b)

    def test_accepts_float_resampled_counts(self) -> None:
        v = 5
        d = token_imbalance([12.5, 7.5, 0.0, 4.0], total=24.0, v_effective=v)
        assert d == pytest.approx(_brute_force_d([12.5, 7.5, 4.0, 0.0, 0.0], v))

    def test_degenerate_guards_return_zero(self) -> None:
        assert token_imbalance([5], total=0, v_effective=4) == 0.0
        assert token_imbalance([5], total=5, v_effective=1) == 0.0


class TestNumpyInputs:
    """All three formulas accept numpy arrays and return a builtin float."""

    def test_returns_builtin_float_on_numpy_input(self) -> None:
        freqs = np.asarray([70.0, 30.0], dtype=np.float64)
        total = float(freqs.sum())

        assert type(token_imbalance(freqs, total, 10)) is float
        assert type(normalized_entropy(freqs, total, 10)) is float
        assert type(renyi_efficiency(freqs, total, 10)) is float
