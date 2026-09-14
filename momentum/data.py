"""Loading and validation of monthly adjusted-close panels."""

from __future__ import annotations

import os
from typing import Iterable

import numpy as np
import pandas as pd

# Default frequency for the library is monthly; every function that
# annualises takes `periods_per_year` so a daily study cannot silently
# inherit the monthly constant. Getting this wrong does not raise -- it just
# scales every Sharpe by sqrt(252/12) = 4.6x, which is why it is a parameter
# and not a module-level assumption.
PERIODS_PER_YEAR = 12

BARS_PER_YEAR = {
    "monthly": 12,
    "weekly": 52,
    # Weekday (Mon-Fri) bars, not the 252-day US equity convention: a spot
    # FX/metal series quotes every weekday including US holidays, so it runs
    # ~261 bars/yr. Measured from the data, not assumed -- see infer_bars_per_year.
    "daily": 261,
}


def infer_bars_per_year(index: pd.DatetimeIndex) -> float:
    """Observed bars per calendar year for a DatetimeIndex.

    Used to check a declared frequency against the data. A series that is
    supposed to be weekday-daily but returns 365 is carrying weekend bars.
    """
    if len(index) < 3:
        raise DataQualityError("need at least 3 bars to infer a frequency")
    span_years = (index[-1] - index[0]).days / 365.25
    if span_years <= 0:
        raise DataQualityError("index spans no time")
    return len(index) / span_years


def drop_weekends(panel):
    """Remove Saturday and Sunday bars.

    Spot metals and FX do not trade at the weekend, but some vendors emit a
    calendar-day series anyway. Those bars are not tradeable, and in the
    XAUUSD feed used here the Saturday bar carries a full weekday's
    volatility (1.09% vs 1.06%) -- real price movement stamped onto a closed
    market. Dropping them loses nothing: the Friday-to-Monday return then
    spans the weekend, which is exactly what a holder experiences.
    """
    if not isinstance(panel.index, pd.DatetimeIndex):
        raise DataQualityError("need a DatetimeIndex to drop weekends")
    return panel[panel.index.dayofweek < 5]


class DataQualityError(ValueError):
    """Raised when a price panel fails a sanity check.

    Deliberately an exception rather than a warning: a silently-dropped or
    silently-patched price series is exactly the class of bug that makes a
    backtest look fine while measuring something else.
    """


def load_symbol(path: str) -> pd.Series:
    """Read one `date,adj_close` CSV into a date-ascending float Series."""
    frame = pd.read_csv(path)
    missing = {"date", "adj_close"} - set(frame.columns)
    if missing:
        raise DataQualityError(f"{path}: missing column(s) {sorted(missing)}")

    frame["date"] = pd.to_datetime(frame["date"])
    series = (
        frame.set_index("date")["adj_close"].astype(float).sort_index()
    )
    if series.index.has_duplicates:
        dupes = series.index[series.index.duplicated()].tolist()
        raise DataQualityError(f"{path}: duplicate dates {dupes[:5]}")
    if not np.isfinite(series.to_numpy()).all():
        raise DataQualityError(f"{path}: non-finite prices present")
    if (series <= 0).any():
        raise DataQualityError(f"{path}: non-positive prices present")
    return series


def load_panel(directory: str, symbols: Iterable[str]) -> pd.DataFrame:
    """Load several symbols into one wide frame indexed by month-end."""
    symbols = list(symbols)
    if not symbols:
        raise DataQualityError("no symbols requested")
    columns = {}
    for symbol in symbols:
        path = os.path.join(directory, f"{symbol}.csv")
        if not os.path.exists(path):
            raise DataQualityError(f"no cached data for {symbol} at {path}")
        columns[symbol] = load_symbol(path)
    panel = pd.DataFrame(columns)
    return to_month_end(panel)


def to_month_end(panel: pd.DataFrame) -> pd.DataFrame:
    """Normalise the index to period-end timestamps.

    Vendor month-end dates are the last *trading* day, which differs per symbol
    (e.g. 2010-10-29 for one, 2010-10-31 for another). Left as-is those
    become separate rows and the panel fills with holes that look like
    missing data. Collapsing to the calendar month period fixes the join.
    """
    if not isinstance(panel.index, pd.DatetimeIndex):
        raise DataQualityError("panel index must be a DatetimeIndex")
    periods = panel.index.to_period("M")
    if periods.has_duplicates:
        raise DataQualityError("two bars in the same calendar month")
    out = panel.copy()
    out.index = periods.to_timestamp("M")
    out.index.name = "date"
    return out.sort_index()


def drop_incomplete_last_month(panel: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    """Drop a final bar whose calendar month has not finished yet.

    The vendor reports a partial month as if it were a completed bar (the
    last row is simply the latest trading day). Keeping it puts a stub
    return of arbitrary length into the sample; at monthly frequency that is
    a material distortion of the most recent signal and of the final P&L.
    """
    if panel.empty:
        return panel
    asof = pd.Timestamp(asof)
    last_period = panel.index[-1].to_period("M")
    # Compare `asof` against the END of the bar's calendar month, not against
    # the bar's own timestamp. The bar is stamped with the last trading day
    # so far (2026-09-11), so `asof < bar` is false on 2026-09-14 and the
    # partial month slipped through the guard entirely.
    month_end = last_period.to_timestamp("M")
    if last_period == asof.to_period("M") and asof < month_end:
        return panel.iloc[:-1]
    return panel


def align_panel(panel: pd.DataFrame, min_coverage: float = 1.0) -> pd.DataFrame:
    """Trim to rows where at least `min_coverage` of columns are present.

    With min_coverage=1.0 this is the common sample: the strictest and the
    default, because a cross-sectional rank computed over a changing number
    of assets silently changes what "top tercile" means.
    """
    if not 0 < min_coverage <= 1:
        raise DataQualityError("min_coverage must be in (0, 1]")
    coverage = panel.notna().mean(axis=1)
    kept = panel.loc[coverage >= min_coverage]
    if kept.empty:
        raise DataQualityError("no rows meet the coverage requirement")
    return kept


def simple_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """R_t = (P_t - P_{t-1}) / P_{t-1}, the poster's first formula."""
    if isinstance(prices, pd.Series):
        if prices.isna().any():
            raise DataQualityError("price series contains NaN")
    returns = prices.pct_change()
    return returns.iloc[1:]


def audit(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-symbol summary used to eyeball a feed before trusting it."""
    empty = [c for c in panel.columns if panel[c].notna().sum() == 0]
    if empty:
        raise DataQualityError(
            f"column(s) with no observations: {empty}. A column that is all-NaN "
            "usually means an index mismatch when the panel was assembled "
            "(e.g. month-start dates joined against month-end dates)."
        )
    rows = []
    for symbol in panel.columns:
        series = panel[symbol].dropna()
        rets = series.pct_change().dropna()
        rows.append(
            {
                "symbol": symbol,
                "bars": len(series),
                "start": series.index[0].date(),
                "end": series.index[-1].date(),
                "first_px": round(float(series.iloc[0]), 4),
                "last_px": round(float(series.iloc[-1]), 4),
                "ann_ret_%": round(
                    100 * ((series.iloc[-1] / series.iloc[0]) ** (PERIODS_PER_YEAR / (len(series) - 1)) - 1), 2
                ),
                "ann_vol_%": round(100 * float(rets.std(ddof=1)) * np.sqrt(PERIODS_PER_YEAR), 2),
                "min_ret_%": round(100 * float(rets.min()), 2),
                "max_ret_%": round(100 * float(rets.max()), 2),
                "gaps": int(panel[symbol].isna().sum()),
            }
        )
    return pd.DataFrame(rows).set_index("symbol")
