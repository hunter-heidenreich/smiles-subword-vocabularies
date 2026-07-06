"""Frequency-distribution intrinsic formulas for the token-usage measurements.

Three closed-form metrics over an empirical token-frequency distribution, shared
by :mod:`smiles_subword.tokenize.measure.distribution`, which evaluates them on
molecule-resampled held-out frequencies — hence all three accept non-integer
``freqs``:

- **normalized_entropy** — Shannon entropy (log2) divided by
  ``log2(v_effective)``, in [0, 1]. Closer to 1 = more uniform usage.
- **renyi_efficiency** (α=2.5, Zouhar 2023a) — order-α Rényi entropy divided by
  ``log2(v_effective)``, in [0, 1]. Within-family diagnostic only, **not** a
  cross-family validator: Cognetta et al. 2024 show BPE modifications inflate
  Rényi without improving downstream performance (cross-family needs n-gram CE
  or downstream NLL).
- **token_imbalance** — ``D = ½ Σ|p_i − 1/|V||`` (Gowda 2020), reported as the
  cross-arm ``|ΔD|`` contrast; dead tokens shared by both arms cancel in ``ΔD``.

``v_effective`` = vocab size minus distinct special-token ids; normalizing
against it keeps the entropy metrics in [0, 1] regardless of how many specials a
tokenizer reserves. All three return 0.0 in the degenerate ``total <= 0`` /
``v_effective <= 1`` cases.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable


_RENYI_ALPHA = 2.5


def normalized_entropy(freqs: Iterable[float], total: float, v_effective: int) -> float:
    """Shannon entropy of the frequency distribution, rebased to [0, 1].

    ``-Σ p_i log₂ p_i`` over the nonzero ``freqs`` (``p_i = f_i / total``),
    divided by ``log₂(v_effective)``. Returns 0.0 when ``total <= 0`` or
    ``v_effective <= 1``.
    """
    if total <= 0 or v_effective <= 1:
        return 0.0
    entropy_bits = 0.0
    for f in freqs:
        if f <= 0:
            continue
        p = f / total
        entropy_bits -= p * math.log2(p)
    return float(entropy_bits / math.log2(v_effective))


def renyi_efficiency(
    freqs: Iterable[float],
    total: float,
    v_effective: int,
    *,
    alpha: float = _RENYI_ALPHA,
) -> float:
    """Order-α Rényi efficiency: H_α(P) / log₂(v_effective).

    Formula (Zouhar 2023a, eq. 7):

        H_α(P) = (1 / (1 - α)) · log₂(Σ p_i^α)

    Returns 0.0 when ``total <= 0`` / ``v_effective <= 1``. ``alpha`` defaults to
    2.5; exposed to verify the convention, not for tuning (Rényi is
    cross-family-gameable — see module docstring).
    """
    if total <= 0 or v_effective <= 1:
        return 0.0
    if alpha == 1.0:
        return normalized_entropy(freqs, total, v_effective)
    p_alpha_sum = 0.0
    for f in freqs:
        if f <= 0:
            continue
        p_alpha_sum += (f / total) ** alpha
    if p_alpha_sum <= 0:
        return 0.0
    h_alpha = math.log2(p_alpha_sum) / (1.0 - alpha)
    return float(h_alpha / math.log2(v_effective))


def token_imbalance(freqs: Iterable[float], total: float, v_effective: int) -> float:
    """Token-imbalance ``D = ½ Σ_i |p_i − 1/|V||`` (Gowda 2020), ``|V| =
    v_effective``.

    The sum runs over the whole vocabulary: each *dead* token (zero held-out
    frequency) contributes ``1/|V|`` identically, so the closed form
    ``½[Σ_live |p_i − 1/|V|| + (|V| − n_live)/|V|]`` avoids enumerating them.
    Returns 0.0 when ``total <= 0`` or ``v_effective <= 1``.
    """
    if total <= 0 or v_effective <= 1:
        return 0.0
    inv_v = 1.0 / v_effective
    n_live = 0
    live_sum = 0.0
    for f in freqs:
        if f <= 0:
            continue
        n_live += 1
        live_sum += abs(f / total - inv_v)
    return float(0.5 * (live_sum + (v_effective - n_live) * inv_v))


__all__ = [
    "normalized_entropy",
    "renyi_efficiency",
    "token_imbalance",
]
