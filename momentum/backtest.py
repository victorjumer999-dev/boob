"""Portfolio construction and the backtest loop.

The one invariant this module exists to enforce:

    weights indexed at t are formed from information known at t,
    and earn returns over t+1.

Every public entry point shifts the weight matrix exactly once before
multiplying by returns. tests/test_backtest.py checks this by feeding the
engine a signal built from *future* returns and asserting the reported P&L
is still zero-information.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from momentum import signals as sig
from momentum.data import PERIODS_PER_YEAR


class BacktestError(ValueError):
    pass


@dataclass
class BacktestConfig:
    """All knobs in one place so a run can be serialised into the report."""

    lookback: int = 12
    skip: int = 1
    top_q: float = 1 / 3
    bottom_q: float = 1 / 3
    long_only: bool = False
    # One-way proportional cost in basis points, charged on traded notional.
    # A round trip therefore costs 2x this. Liquid US sector ETFs sit around
    # 1-3bp all-in; 10bp is deliberately conservative.
    cost_bps: float = 10.0
    target_vol: float | None = None
    vol_window: int = 12
    max_leverage: float = 2.0


@dataclass
class BacktestResult:
    returns: pd.Series
    gross_returns: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    costs: pd.Series
    leverage: pd.Series
    config: BacktestConfig
    label: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def equity(self) -> pd.Series:
        return (1.0 + self.returns).cumprod()


def _validate_returns(returns: pd.DataFrame) -> None:
    if not isinstance(returns, pd.DataFrame):
        raise BacktestError("returns must be a wide DataFrame of asset returns")
    if returns.empty:
        raise BacktestError("returns frame is empty")
    if (returns.abs() > 1.5).to_numpy().any():
        raise BacktestError(
            "a |return| above 150% in a monthly panel almost always means a "
            "split or dividend was not adjusted; refusing to backtest it"
        )


def compute_turnover(weights: pd.DataFrame) -> pd.Series:
    """Traded notional per rebalance, as a fraction of book size.

    Compared against the *drifted* weights of the previous period, not the
    previous target. Ignoring drift overstates turnover (and therefore
    costs) because a position that grew with the market is charged as if it
    had been re-bought.
    """
    turnover = pd.Series(0.0, index=weights.index)
    previous = pd.Series(0.0, index=weights.columns)
    for date, target in weights.iterrows():
        turnover.loc[date] = float((target - previous).abs().sum())
        previous = target
    return turnover


def run_weighted_backtest(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    cost_bps: float = 10.0,
    leverage: pd.Series | None = None,
    label: str = "",
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """Apply a weight matrix to a return panel with costs.

    `weights.loc[t]` is the book decided at the close of t. It is shifted one
    bar forward here -- and only here -- before meeting returns.
    """
    _validate_returns(returns)
    if not weights.index.equals(returns.index):
        raise BacktestError("weights and returns must share an index")
    if list(weights.columns) != list(returns.columns):
        raise BacktestError("weights and returns must share a column order")
    if cost_bps < 0:
        raise BacktestError("cost_bps must be non-negative")

    book = weights.fillna(0.0)
    if leverage is not None:
        if not leverage.index.equals(weights.index):
            raise BacktestError("leverage must share the weights index")
        # NaN leverage = no vol estimate yet = stay flat, not "1x by default".
        book = book.mul(leverage, axis=0).fillna(0.0)
        applied_leverage = leverage.reindex(weights.index)
    else:
        applied_leverage = pd.Series(1.0, index=weights.index)

    # The single shift. Everything downstream is investable.
    held = book.shift(1)
    gross = (held * returns).sum(axis=1, min_count=1)

    turnover = compute_turnover(book)
    # Costs are paid when the trade happens (at t), and show up in the P&L of
    # the period that starts then -- hence the same shift as the weights.
    costs = (turnover * cost_bps / 1e4).shift(1)

    # Trim the warm-up. Before the signal exists the book is all zeros, and
    # a zero-weight bar produces a hard 0.00% return. Left in, those months
    # are counted as real observations: they shrink the standard deviation,
    # inflate the month count, and back-date the reported start of the track
    # record -- a strategy that first traded in 1997 was showing a 1996 start
    # and a flattered Sharpe. Only the LEADING flat run is dropped; a flat
    # month later in the sample is a genuine decision (time-series momentum
    # stepping to cash) and must stay in the return stream.
    active = (held.fillna(0.0) != 0).any(axis=1)
    if not bool(active.any()):
        raise BacktestError(
            "this configuration never takes a position -- the signal layer is "
            "dead, not flat. Check the lookback against the sample length."
        )
    first_active = int(active.to_numpy().argmax())
    valid = held.notna().any(axis=1)
    valid.iloc[:first_active] = False
    gross = gross.where(valid)
    costs = costs.where(valid)

    net = (gross - costs).dropna()
    return BacktestResult(
        returns=net,
        gross_returns=gross.dropna(),
        weights=book,
        turnover=turnover,
        costs=costs.dropna(),
        leverage=applied_leverage,
        config=config or BacktestConfig(cost_bps=cost_bps),
        label=label,
    )


def cross_sectional_momentum(
    prices: pd.DataFrame, config: BacktestConfig | None = None, label: str = ""
) -> BacktestResult:
    """Rank assets on trailing momentum; long the winners, short the losers.

    The poster's headline strategy ("Formation and Holding": rank by past
    returns, go long the top quantile and short the bottom).
    """
    config = config or BacktestConfig()
    returns = prices.pct_change().iloc[1:]
    _validate_returns(returns)

    momentum = sig.compound_momentum(returns, config.lookback, config.skip)
    weights = sig.quantile_weights(
        momentum,
        top_q=config.top_q,
        bottom_q=config.bottom_q,
        long_only=config.long_only,
    )

    leverage = None
    if config.target_vol is not None:
        # Bootstrap the vol estimate off the *unlevered* strategy so the
        # scalar is not chasing its own tail.
        unlevered = run_weighted_backtest(weights, returns, cost_bps=0.0)
        leverage = sig.volatility_target_scalar(
            unlevered.gross_returns,
            target_vol=config.target_vol,
            window=config.vol_window,
            max_leverage=config.max_leverage,
        ).reindex(weights.index)

    result = run_weighted_backtest(
        weights, returns, cost_bps=config.cost_bps, leverage=leverage,
        label=label or "cross-sectional", config=config,
    )
    result.meta["signal"] = momentum
    return result


def timeseries_momentum(
    prices: pd.DataFrame, config: BacktestConfig | None = None, label: str = ""
) -> BacktestResult:
    """Absolute momentum: hold each asset only while its own trend is up.

    Equal weight across whatever is long (and short, if enabled), so the book
    is fully invested when every asset agrees and flat when none do.
    """
    config = config or BacktestConfig()
    returns = prices.pct_change().iloc[1:]
    _validate_returns(returns)

    momentum = sig.compound_momentum(returns, config.lookback, config.skip)
    direction = sig.timeseries_signal(momentum, allow_short=not config.long_only)
    n_assets = direction.notna().sum(axis=1).replace(0, np.nan)
    weights = direction.div(n_assets, axis=0).fillna(0.0)

    leverage = None
    if config.target_vol is not None:
        unlevered = run_weighted_backtest(weights, returns, cost_bps=0.0)
        leverage = sig.volatility_target_scalar(
            unlevered.gross_returns,
            target_vol=config.target_vol,
            window=config.vol_window,
            max_leverage=config.max_leverage,
        ).reindex(weights.index)

    result = run_weighted_backtest(
        weights, returns, cost_bps=config.cost_bps, leverage=leverage,
        label=label or "time-series", config=config,
    )
    result.meta["signal"] = momentum
    return result


def buy_and_hold(prices: pd.DataFrame | pd.Series, label: str = "buy & hold") -> BacktestResult:
    """Equal-weight, fully-invested benchmark on the same bars and costs (none)."""
    frame = prices.to_frame() if isinstance(prices, pd.Series) else prices
    returns = frame.pct_change().iloc[1:]
    _validate_returns(returns)
    weights = pd.DataFrame(
        1.0 / frame.shape[1], index=returns.index, columns=returns.columns
    )
    # Hold from the first investable bar: shift means the first row is unused,
    # exactly as for every strategy, so the comparison starts on the same date.
    return run_weighted_backtest(weights, returns, cost_bps=0.0, label=label)


def decile_portfolios(
    prices: pd.DataFrame, lookback: int = 12, skip: int = 1, n_buckets: int = 3
) -> pd.DataFrame:
    """Average next-period return by prior-momentum bucket.

    Reproduces the poster's "Average Future Return by Prior 12-Month Return"
    bar chart. The panel here is 6 sectors wide, so buckets are terciles, not
    deciles -- naming them deciles on 6 names would be theatre.
    """
    if n_buckets < 2:
        raise BacktestError("need at least 2 buckets")
    returns = prices.pct_change().iloc[1:]
    _validate_returns(returns)
    if prices.shape[1] < n_buckets:
        raise BacktestError(
            f"{prices.shape[1]} assets cannot fill {n_buckets} buckets"
        )

    momentum = sig.compound_momentum(returns, lookback, skip)
    forward = returns.shift(-1)  # next period's return, for attribution only

    records = []
    for date, row in momentum.iterrows():
        valid = row.dropna()
        if len(valid) < n_buckets:
            continue
        ranks = valid.rank(method="first")
        buckets = np.ceil(ranks / len(valid) * n_buckets).astype(int)
        for asset, bucket in buckets.items():
            nxt = forward.at[date, asset]
            if pd.notna(nxt):
                records.append({"bucket": int(bucket), "fwd_ret": float(nxt)})

    if not records:
        raise BacktestError("no bucket observations produced")
    frame = pd.DataFrame(records)
    summary = frame.groupby("bucket")["fwd_ret"].agg(["mean", "std", "count"])
    summary["mean_%"] = 100 * summary["mean"]
    summary["ann_%"] = 100 * ((1 + summary["mean"]) ** PERIODS_PER_YEAR - 1)
    summary["t_stat"] = summary["mean"] / (summary["std"] / np.sqrt(summary["count"]))
    return summary
