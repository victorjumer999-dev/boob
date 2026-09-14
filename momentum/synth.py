"""Synthetic price panels with a *known* amount of momentum.

Used by the test suite for the one check a real-data backtest can never
provide: if I plant an edge of known sign and size, does the engine recover
it, and if I plant none, does the engine report none? Without this, a
backtester that is subtly wrong and a strategy that has no edge are
indistinguishable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from momentum.data import PERIODS_PER_YEAR


def random_walk_panel(
    n_months: int = 360,
    n_assets: int = 8,
    ann_vol: float = 0.18,
    ann_drift: float = 0.07,
    seed: int = 0,
    start: str = "1996-01-31",
) -> pd.DataFrame:
    """Zero-momentum control: i.i.d. returns, no serial structure at all.

    Any strategy that shows a significant edge on this panel is reading its
    own look-ahead, not a signal.
    """
    rng = np.random.default_rng(seed)
    mu = (1 + ann_drift) ** (1 / PERIODS_PER_YEAR) - 1
    sigma = ann_vol / np.sqrt(PERIODS_PER_YEAR)
    shocks = rng.normal(mu, sigma, size=(n_months, n_assets))
    index = pd.date_range(start, periods=n_months, freq="ME")
    columns = [f"A{i:02d}" for i in range(n_assets)]
    prices = 100 * np.cumprod(1 + shocks, axis=0)
    return pd.DataFrame(prices, index=index, columns=columns)


def momentum_panel(
    n_months: int = 360,
    n_assets: int = 8,
    ann_vol: float = 0.18,
    ann_drift: float = 0.07,
    persistence: float = 0.15,
    lookback: int = 12,
    seed: int = 0,
    start: str = "1996-01-31",
) -> pd.DataFrame:
    """Panel with a planted cross-sectional momentum effect.

    Each month, an asset's return gets a nudge proportional to its own
    demeaned trailing `lookback`-month return:

        r_{i,t} = mu + persistence * (m_{i,t-1} - mean_j m_{j,t-1}) + eps

    `persistence` is the strength knob; 0 reduces exactly to the random walk
    above. The effect is deliberately cross-sectional (demeaned across
    assets) so it shows up in a rank strategy rather than in the market.
    """
    if persistence < 0:
        raise ValueError("persistence must be non-negative")
    rng = np.random.default_rng(seed)
    mu = (1 + ann_drift) ** (1 / PERIODS_PER_YEAR) - 1
    sigma = ann_vol / np.sqrt(PERIODS_PER_YEAR)

    rets = np.zeros((n_months, n_assets))
    for t in range(n_months):
        shock = rng.normal(mu, sigma, size=n_assets)
        if t >= lookback:
            trailing = rets[t - lookback : t].sum(axis=0)
            tilt = trailing - trailing.mean()
            shock = shock + persistence * tilt / lookback
        rets[t] = shock

    index = pd.date_range(start, periods=n_months, freq="ME")
    columns = [f"A{i:02d}" for i in range(n_assets)]
    prices = 100 * np.cumprod(1 + rets, axis=0)
    return pd.DataFrame(prices, index=index, columns=columns)


def trending_panel(
    n_months: int = 240, n_assets: int = 4, seed: int = 0, start: str = "2000-01-31"
) -> pd.DataFrame:
    """Long, clean regimes -- the best case for time-series momentum.

    Each asset alternates between up and down drifts of ~24 months. A
    trend-follower should make money here; if it does not, the wiring is
    wrong, not the market.
    """
    rng = np.random.default_rng(seed)
    sigma = 0.18 / np.sqrt(PERIODS_PER_YEAR)
    rets = np.zeros((n_months, n_assets))
    for a in range(n_assets):
        t = 0
        sign = 1.0 if a % 2 == 0 else -1.0
        while t < n_months:
            length = int(rng.integers(18, 31))
            drift = sign * 0.012
            span = min(length, n_months - t)
            rets[t : t + span, a] = rng.normal(drift, sigma, size=span)
            t += span
            sign *= -1.0
    index = pd.date_range(start, periods=n_months, freq="ME")
    columns = [f"T{i:02d}" for i in range(n_assets)]
    prices = 100 * np.cumprod(1 + rets, axis=0)
    return pd.DataFrame(prices, index=index, columns=columns)
