"""Loading and validation of monthly adjusted-close panels."""

from __future__ import annotations

import os
from typing import Iterable

import numpy as np
import pandas as pd

# Monthly bars. Everything that annualises uses this single constant.
PERIODS_PER_YEAR = 12


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
