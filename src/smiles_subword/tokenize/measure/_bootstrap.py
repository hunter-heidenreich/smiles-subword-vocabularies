"""Shared recipe: 95% percentile bootstrap CIs over 1000 molecule resamples.

Single home for the resample count, CI level, percentile estimator
(:func:`percentile_ci`), stdlib ratio bootstrap (:func:`bootstrap_ratio_ci`), and
stable per-cell/per-pair seed derivation, so no arm drifts. High-scale arms
(nestedness, fg_alignment, ~10^6 molecules) keep their own numpy resampler and
reduce through :func:`percentile_ci`, so every arm shares the *one* percentile
estimator. Leaf module (stdlib only): importable without a cycle.
"""

from __future__ import annotations

import hashlib
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

N_BOOTSTRAP_RESAMPLES = 1000
"""1000 bootstrap resamples for the held-out-evaluated M's."""

CI_LEVEL = 0.95
"""95% percentile bootstrap CI."""


def bootstrap_seed(key: str) -> int:
    """Stable bootstrap seed in ``[0, 2**32)`` from a BLAKE2b digest of ``key``.

    Re-runs reproduce byte-identical CIs; distinct keys get distinct seeds.
    ``key`` is the ``cell_id`` (per-cell) or the ``pair_key`` (cross-arm, e.g.
    nestedness, whose resampled quantity is intrinsically a pair).
    """
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") & 0xFFFFFFFF


def percentile_ci(
    samples: list[float], *, level: float = CI_LEVEL
) -> tuple[float, float]:
    """Return the ``(low, high)`` percentile-interval bounds at ``level``.

    Percentile method via the lower order statistic (no interpolation): with
    1000 samples and level=0.95 the 0-indexed 25th and 974th sorted entries
    bracket the CI. Falls back to ``(min, max)`` for ``len(samples) <= 1`` and
    ``(nan, nan)`` for an empty sample. Callers pass an already-finite sample
    list (NaN resamples filtered out upstream). The non-interpolated convention
    is immaterial here: on the ~10^6-molecule splits it agrees with the
    interpolating estimators to ~1e-6, well below the bootstrap's seed-to-seed
    Monte-Carlo spread, itself far below the CI half-width. Seeds keep intervals
    reproducible.
    """
    if not samples:
        return (float("nan"), float("nan"))
    if len(samples) == 1:
        return (samples[0], samples[0])
    sorted_samples = sorted(samples)
    tail = (1.0 - level) / 2.0
    lo_idx = int(tail * len(sorted_samples))
    hi_idx = int((1.0 - tail) * len(sorted_samples)) - 1
    lo_idx = max(0, min(lo_idx, len(sorted_samples) - 1))
    hi_idx = max(0, min(hi_idx, len(sorted_samples) - 1))
    return (sorted_samples[lo_idx], sorted_samples[hi_idx])


def bootstrap_ratio_ci(
    numerators: Sequence[int],
    denominators: Sequence[int],
    *,
    seed: int,
    n_resamples: int = N_BOOTSTRAP_RESAMPLES,
    level: float = CI_LEVEL,
) -> tuple[float, float]:
    """Percentile bootstrap CI for ``Σ num / Σ denom`` over molecule resamples.

    ``denominators`` all ``1`` gives a per-molecule mean; per-molecule token
    counts give the glyphs-per-token ratio. Resamples molecule indices with
    replacement off the per-cell ``seed``. Returns ``(nan, nan)`` when there are
    no molecules or every resampled denominator is zero (ratio undefined).
    """
    if len(numerators) != len(denominators):
        raise ValueError("numerators and denominators must have equal length")
    n = len(numerators)
    if n == 0:
        return (float("nan"), float("nan"))

    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(n_resamples):
        picks = [rng.randrange(n) for _ in range(n)]
        num_sum = sum(numerators[i] for i in picks)
        denom_sum = sum(denominators[i] for i in picks)
        if denom_sum == 0:
            samples.append(float("nan"))
            continue
        samples.append(num_sum / denom_sum)
    finite = [s for s in samples if s == s]
    if not finite:
        return (float("nan"), float("nan"))
    return percentile_ci(finite, level=level)


__all__ = [
    "CI_LEVEL",
    "N_BOOTSTRAP_RESAMPLES",
    "bootstrap_ratio_ci",
    "bootstrap_seed",
    "percentile_ci",
]
