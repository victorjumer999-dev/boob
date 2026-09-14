"""Performance statistics.

Annualisation constant lives in momentum.data so it cannot drift between
modules -- a classic source of Sharpe ratios that are wrong by sqrt(21).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from momentum.data import PERIODS_PER_YEAR


class MetricError(ValueError):
    pass


def _clean(returns: pd.Series) -> pd.Series:
    if not isinstance(returns, pd.Series):
        raise MetricError("expected a Series of periodic returns")
    cleaned = returns.dropna()
    if cleaned.empty:
        raise MetricError("no observations after dropping NaN")
    return cleaned


def _is_degenerate(spread: float, sample: pd.Series) -> bool:
    """True when a dispersion measure is zero up to floating-point noise.

    `float(pd.Series([0.01] * 30).std(ddof=1))` is 1.76e-18, not 0.0. An
    exact `== 0` guard therefore never fires, and the Sharpe of a constant
    return stream comes back as ~2e16: a finite, plausible-looking number
    that no downstream check would flag. Scale the tolerance to the data.
    """
    scale = float(sample.abs().max())
    return spread <= max(scale, 1e-12) * 1e-10


def cagr(returns: pd.Series, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    r = _clean(returns)
    growth = float((1.0 + r).prod())
    if growth <= 0:
        return -1.0  # the book was wiped out; CAGR is undefined below this
    years = len(r) / periods_per_year
    return growth ** (1.0 / years) - 1.0


def annual_vol(returns: pd.Series, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    r = _clean(returns)
    if len(r) < 2:
        raise MetricError("need at least 2 observations for a volatility")
    return float(r.std(ddof=1)) * np.sqrt(periods_per_year)


def sharpe(returns: pd.Series, risk_free: float = 0.0,
           periods_per_year: int = PERIODS_PER_YEAR) -> float:
    """E[R - Rf] / sigma_p, annualised -- the poster's Sharpe formula.

    `risk_free` is an *annual* rate; it is de-annualised geometrically before
    being subtracted, not divided by 12.
    """
    r = _clean(returns)
    if len(r) < 2:
        raise MetricError("need at least 2 observations for a Sharpe ratio")
    rf_period = (1.0 + risk_free) ** (1.0 / periods_per_year) - 1.0
    excess = r - rf_period
    sd = float(excess.std(ddof=1))
    if _is_degenerate(sd, excess):
        raise MetricError("zero volatility: Sharpe is undefined, not infinite")
    return float(excess.mean()) / sd * np.sqrt(periods_per_year)


def sortino(returns: pd.Series, risk_free: float = 0.0,
            periods_per_year: int = PERIODS_PER_YEAR) -> float:
    r = _clean(returns)
    rf_period = (1.0 + risk_free) ** (1.0 / periods_per_year) - 1.0
    excess = r - rf_period
    downside = excess[excess < 0]
    if downside.empty:
        raise MetricError("no losing periods: Sortino is undefined")
    dd = float(np.sqrt((downside ** 2).mean()))
    if _is_degenerate(dd, excess):
        raise MetricError("zero downside deviation: Sortino is undefined")
    return float(excess.mean()) / dd * np.sqrt(periods_per_year)


def drawdown_series(returns: pd.Series) -> pd.Series:
    r = _clean(returns)
    equity = (1.0 + r).cumprod()
    # The running peak must include the starting capital of 1.0. Without it
    # the peak on the first bar is the post-loss equity, so a strategy whose
    # worst loss happens immediately reports a max drawdown of 0.00%.
    peak = pd.concat([pd.Series([1.0], index=[None]), equity]).cummax().iloc[1:]
    peak.index = equity.index
    return equity / peak - 1.0


def max_drawdown(returns: pd.Series) -> float:
    return float(drawdown_series(returns).min())


def calmar(returns: pd.Series, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    mdd = abs(max_drawdown(returns))
    if mdd == 0:
        raise MetricError("no drawdown: Calmar is undefined")
    return cagr(returns, periods_per_year) / mdd


def hit_rate(returns: pd.Series) -> float:
    r = _clean(returns)
    return float((r > 0).mean())


def skew_kurt(returns: pd.Series) -> tuple[float, float]:
    r = _clean(returns)
    return float(r.skew()), float(r.kurt())


def worst_drawdown_window(returns: pd.Series) -> dict:
    dd = drawdown_series(returns)
    trough = dd.idxmin()
    peak = dd.loc[:trough]
    peak = peak[peak >= 0]
    peak_date = peak.index[-1] if len(peak) else dd.index[0]
    after = dd.loc[trough:]
    recovered = after[after >= -1e-12]
    return {
        "peak": peak_date,
        "trough": trough,
        "recovery": recovered.index[0] if len(recovered) else None,
        "depth": float(dd.min()),
        "months_to_trough": int(len(dd.loc[peak_date:trough]) - 1),
    }


def summarise(returns: pd.Series, name: str = "", risk_free: float = 0.0,
              periods_per_year: int = PERIODS_PER_YEAR) -> dict:
    """One row of headline statistics, safe to build a table from."""
    r = _clean(returns)
    row = {
        "strategy": name,
        "bars": len(r),
        "years": round(len(r) / periods_per_year, 2),
        "start": r.index[0].date(),
        "end": r.index[-1].date(),
        "total_return_%": 100 * (float((1 + r).prod()) - 1),
        "cagr_%": 100 * cagr(r, periods_per_year),
        "vol_%": 100 * annual_vol(r, periods_per_year),
        "max_dd_%": 100 * max_drawdown(r),
        "hit_rate_%": 100 * hit_rate(r),
        "best_bar_%": 100 * float(r.max()),
        "worst_bar_%": 100 * float(r.min()),
    }
    # A degenerate series has no defined Sharpe/Sortino/Calmar. In a summary
    # table that is a NaN cell, not a reason to abort the whole report -- but
    # the underlying functions still raise, so a caller asking for one metric
    # in isolation is never handed a fabricated number.
    try:
        row["sharpe"] = sharpe(r, risk_free, periods_per_year)
    except MetricError:
        row["sharpe"] = float("nan")
    try:
        row["sortino"] = sortino(r, risk_free, periods_per_year)
    except MetricError:
        row["sortino"] = float("nan")
    try:
        row["calmar"] = calmar(r, periods_per_year)
    except MetricError:
        row["calmar"] = float("nan")
    row["skew"], row["kurtosis"] = skew_kurt(r)
    return row


def summary_table(results: dict[str, pd.Series], risk_free: float = 0.0,
                  periods_per_year: int = PERIODS_PER_YEAR) -> pd.DataFrame:
    rows = [
        summarise(series, name=name, risk_free=risk_free, periods_per_year=periods_per_year)
        for name, series in results.items()
    ]
    return pd.DataFrame(rows).set_index("strategy")
