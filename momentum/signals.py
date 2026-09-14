"""Momentum signal construction.

Every function here maps information available *up to and including* bar t
onto a value indexed at bar t. Turning a signal into a position (and hence
shifting it forward one bar) is the backtester's job, not the signal's --
keeping that boundary sharp is what makes the look-ahead tests meaningful.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from momentum.data import PERIODS_PER_YEAR


class SignalError(ValueError):
    """Raised for a signal configuration that cannot produce a real reading."""


def _check_window(lookback: int, skip: int, n_obs: int, name: str) -> None:
    if lookback <= 0:
        raise SignalError(f"{name}: lookback must be positive, got {lookback}")
    if skip < 0:
        raise SignalError(f"{name}: skip must be non-negative, got {skip}")
    if skip >= lookback:
        raise SignalError(
            f"{name}: skip ({skip}) must be smaller than lookback ({lookback}); "
            "otherwise the formation window is empty"
        )
    if n_obs < lookback:
        # The catastrophic version of this is a guard that silently returns a
        # neutral value, leaving a whole signal layer dead. Fail loudly.
        raise SignalError(
            f"{name}: need at least {lookback} observations, got {n_obs}. "
            "Shorten the lookback or lengthen the sample."
        )


def cumulative_momentum(
    returns: pd.DataFrame | pd.Series, lookback: int = 12, skip: int = 1
) -> pd.DataFrame | pd.Series:
    """M_t = sum of the last `lookback` returns, excluding the last `skip`.

    The poster writes M_t = sum_{i=1..L} R_{t-i}. `skip=1` implements the
    standard Jegadeesh-Titman 12-1 convention: the most recent month is left
    out because short-horizon reversal works against momentum there. Set
    skip=0 to reproduce the poster's formula literally.
    """
    _check_window(lookback, skip, len(returns), "cumulative_momentum")
    window = lookback - skip
    rolled = returns.rolling(window=window, min_periods=window).sum()
    return rolled.shift(skip)


def compound_momentum(
    returns: pd.DataFrame | pd.Series, lookback: int = 12, skip: int = 1
) -> pd.DataFrame | pd.Series:
    """Geometric version: prod(1+R) - 1 over the same window.

    Preferred over the arithmetic sum for anything that is meant to be a
    return; the sum is only a first-order approximation and drifts badly at
    12-month horizons in volatile assets.
    """
    _check_window(lookback, skip, len(returns), "compound_momentum")
    window = lookback - skip
    log1p = np.log1p(returns)
    rolled = log1p.rolling(window=window, min_periods=window).sum()
    return np.expm1(rolled).shift(skip)


def price_to_sma(prices: pd.DataFrame | pd.Series, lookback: int = 12) -> pd.DataFrame | pd.Series:
    """M_t = P_t / SMA_{t,L} - 1, the poster's price-momentum formula."""
    _check_window(lookback, 0, len(prices), "price_to_sma")
    sma = prices.rolling(window=lookback, min_periods=lookback).mean()
    return prices / sma - 1.0


def rolling_zscore(
    signal: pd.DataFrame | pd.Series, window: int = 36, min_periods: int | None = None
) -> pd.DataFrame | pd.Series:
    """Standardise a signal against its own trailing distribution.

    This is the "Rolling 12-Month Momentum Signal (AAPL)" panel on the
    poster: the z-score is what the +/-1 long/short bands are drawn on.
    The mean and standard deviation both use trailing data only.
    """
    if window < 3:
        raise SignalError(f"rolling_zscore: window must be >= 3, got {window}")
    min_periods = window if min_periods is None else min_periods
    if min_periods < 3:
        raise SignalError("rolling_zscore: min_periods must be >= 3")
    mean = signal.rolling(window=window, min_periods=min_periods).mean()
    std = signal.rolling(window=window, min_periods=min_periods).std(ddof=1)
    # A zero trailing std means the signal never moved in the window; the
    # z-score is undefined there, so emit NaN (which the backtester treats
    # as "no position") rather than inf or a fabricated 0.
    std = std.where(std > 0)
    return (signal - mean) / std


def cross_sectional_rank(signal: pd.DataFrame, pct: bool = True) -> pd.DataFrame:
    """Rank each row across assets. 1.0 = strongest momentum that month."""
    if not isinstance(signal, pd.DataFrame):
        raise SignalError("cross_sectional_rank needs a wide DataFrame")
    if signal.shape[1] < 2:
        raise SignalError("cross-sectional ranking needs at least 2 assets")
    return signal.rank(axis=1, pct=pct, na_option="keep")


def quantile_weights(
    signal: pd.DataFrame,
    top_q: float = 1 / 3,
    bottom_q: float = 1 / 3,
    long_only: bool = False,
    min_names: int = 1,
) -> pd.DataFrame:
    """Equal-weight long the top quantile, short the bottom quantile.

    Weights are normalised per leg so the long book always sums to +1 and
    the short book to -1 (0 when long_only). Rows with fewer than
    `min_names` rankable assets get flat weights rather than a concentrated
    bet on whatever happened to survive.
    """
    if not isinstance(signal, pd.DataFrame):
        raise SignalError("quantile_weights needs a wide DataFrame of assets")
    if signal.shape[1] < 2:
        # Without at least two assets there is no cross-section to rank, and
        # the per-row guard below would quietly skip every date -- handing
        # back an all-zero book that looks like a flat strategy rather than
        # an impossible one. Refuse instead.
        raise SignalError(
            f"cross-sectional weights need at least 2 assets, got {signal.shape[1]}. "
            "For a single instrument use time-series momentum."
        )
    if not 0 < top_q <= 1 or not 0 <= bottom_q < 1:
        raise SignalError("quantiles must satisfy 0 < top_q <= 1 and 0 <= bottom_q < 1")
    if top_q + bottom_q > 1:
        raise SignalError("top_q + bottom_q must not exceed 1 (legs would overlap)")

    weights = pd.DataFrame(0.0, index=signal.index, columns=signal.columns)
    for date, row in signal.iterrows():
        valid = row.dropna()
        if len(valid) < min_names or len(valid) < 2:
            continue
        n_long = max(1, int(np.floor(len(valid) * top_q)))
        n_short = 0 if long_only else max(1, int(np.floor(len(valid) * bottom_q)))
        if n_long + n_short > len(valid):
            # Not enough names to fill both legs without overlap: shrink the
            # short leg first, then skip the month entirely if still stuck.
            n_short = max(0, len(valid) - n_long)
            if n_short == 0 and not long_only:
                continue
        order = valid.sort_values(ascending=False)
        longs = order.index[:n_long]
        weights.loc[date, longs] = 1.0 / n_long
        if n_short > 0:
            shorts = order.index[-n_short:]
            weights.loc[date, shorts] = -1.0 / n_short
    return weights


def timeseries_signal(
    signal: pd.DataFrame | pd.Series, threshold: float = 0.0, allow_short: bool = True
) -> pd.DataFrame | pd.Series:
    """+1 / -1 (or +1 / 0) on the sign of the signal versus a threshold.

    This is `np.where(momentum > 0, 1, -1)` from the poster's code block,
    with two corrections: NaN stays NaN instead of collapsing to -1 (numpy's
    comparison makes NaN falsy, which would put the book maximally short
    during every warm-up window), and shorting can be switched off.
    """
    mask_up = signal > threshold
    mask_down = signal <= threshold
    out = signal.where(pd.isna(signal), 0.0)
    out = out.mask(mask_up, 1.0)
    out = out.mask(mask_down, -1.0 if allow_short else 0.0)
    return out


def volatility_target_scalar(
    returns: pd.Series,
    target_vol: float = 0.10,
    window: int = 12,
    max_leverage: float = 2.0,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> pd.Series:
    """Leverage multiplier that steers realised vol toward `target_vol`.

    The poster's "volatility targeting" risk control. Uses only trailing
    realised vol, so the scalar at t is investable at t+1 once shifted.
    """
    if target_vol <= 0:
        raise SignalError("target_vol must be positive")
    if max_leverage <= 0:
        raise SignalError("max_leverage must be positive")
    realised = returns.rolling(window=window, min_periods=window).std(ddof=1) * np.sqrt(
        periods_per_year
    )
    realised = realised.where(realised > 0)
    return (target_vol / realised).clip(upper=max_leverage)
