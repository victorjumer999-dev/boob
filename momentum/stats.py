"""Inference: HAC standard errors, block bootstrap, predictive regression.

Two traps this module exists to avoid, both of which manufacture
significance out of nothing:

  * Newey-West with the automatic bandwidth on *overlapping* h-period
    returns. Overlap induces MA(h-1) dependence; the bandwidth must be at
    least h-1 or the standard errors are too small.
  * An i.i.d. bootstrap on a statistic that measures serial dependence. The
    resample destroys the very structure being tested, so every replicate
    looks like white noise and the confidence interval collapses onto the
    null. Use the moving-block version.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from momentum.data import PERIODS_PER_YEAR


class StatsError(ValueError):
    pass


def newey_west_tstat(x: pd.Series, lags: int | None = None, overlap: int = 1) -> dict:
    """Mean of `x` with a HAC standard error, and its t-statistic.

    `overlap` is the number of periods an observation shares with its
    neighbour (1 = non-overlapping). The bandwidth is floored at
    `overlap - 1` regardless of what the automatic rule suggests.
    """
    values = pd.Series(x).dropna().to_numpy(dtype=float)
    n = len(values)
    if n < 8:
        raise StatsError(f"need at least 8 observations, got {n}")
    if overlap < 1:
        raise StatsError("overlap must be >= 1")

    auto = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    lags = auto if lags is None else int(lags)
    lags = max(lags, overlap - 1)
    lags = min(lags, n - 1)

    mean = values.mean()
    resid = values - mean
    gamma0 = float(resid @ resid) / n
    variance = gamma0
    for lag in range(1, lags + 1):
        gamma = float(resid[lag:] @ resid[:-lag]) / n
        weight = 1.0 - lag / (lags + 1.0)  # Bartlett kernel
        variance += 2.0 * weight * gamma
    if variance <= 0:
        raise StatsError("non-positive HAC variance; sample is degenerate")

    se = np.sqrt(variance / n)
    return {
        "mean": float(mean),
        "hac_se": float(se),
        "t_stat": float(mean / se),
        "lags_used": int(lags),
        "n": int(n),
    }


def moving_block_bootstrap(
    returns: pd.Series,
    statistic,
    n_boot: int = 2000,
    block: int | None = None,
    seed: int = 0,
) -> dict:
    """Percentile CI for a statistic, preserving short-range dependence.

    Block length defaults to n^(1/3), the usual rule of thumb, and is never
    shorter than 3 -- below that the blocks stop carrying the autocorrelation
    the whole exercise is meant to retain.
    """
    series = pd.Series(returns).dropna()
    n = len(series)
    if n < 24:
        raise StatsError(f"need at least 24 observations to bootstrap, got {n}")
    block = max(3, int(round(n ** (1 / 3)))) if block is None else int(block)
    if block >= n:
        raise StatsError("block length must be shorter than the sample")

    rng = np.random.default_rng(seed)
    values = series.to_numpy(dtype=float)
    n_blocks = int(np.ceil(n / block))
    max_start = n - block

    draws = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        sample = np.concatenate([values[s : s + block] for s in starts])[:n]
        draws[i] = statistic(pd.Series(sample, index=series.index))

    point = float(statistic(series))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {
        "point": point,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_gt_zero": float((draws > 0).mean()),
        "block": block,
        "n_boot": n_boot,
    }


def ols_hac(
    y: np.ndarray | pd.Series,
    regressors: dict[str, np.ndarray | pd.Series],
    overlap: int = 1,
) -> dict:
    """OLS with Newey-West standard errors and the same overlap floor.

    Rows must already be aligned and ordered by date -- the HAC lag
    structure is meaningless otherwise. Returns per-coefficient estimate,
    standard error and t-statistic, plus R-squared.
    """
    y = np.asarray(y, dtype=float)
    names = list(regressors)
    cols = [np.asarray(regressors[k], dtype=float) for k in names]
    if any(len(c) != len(y) for c in cols):
        raise StatsError("regressors and y must be the same length")
    n = len(y)
    if n < 30:
        raise StatsError(f"only {n} usable observations")

    design = np.column_stack([np.ones(n)] + cols)
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coef

    lags = max(int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0))), overlap - 1)
    xtx_inv = np.linalg.inv(design.T @ design)
    scores = design * resid[:, None]
    meat = scores.T @ scores
    for lag in range(1, lags + 1):
        gamma = scores[lag:].T @ scores[:-lag]
        weight = 1.0 - lag / (lags + 1.0)
        meat += weight * (gamma + gamma.T)
    cov = xtx_inv @ meat @ xtx_inv
    se = np.sqrt(np.diag(cov))

    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    out = {
        "n_obs": int(n),
        "hac_lags": int(lags),
        "r_squared": 1.0 - ss_res / ss_tot,
        "alpha": float(coef[0]),
        "alpha_t": float(coef[0] / se[0]),
    }
    for i, name in enumerate(names, start=1):
        out[name] = float(coef[i])
        out[f"{name}_se"] = float(se[i])
        out[f"{name}_t"] = float(coef[i] / se[i])
    return out


def predictive_regression(
    momentum_signal: pd.DataFrame | pd.Series,
    returns: pd.DataFrame | pd.Series,
    overlap: int = 1,
) -> dict:
    """R_{t+1} = alpha + beta * M_t + eps, pooled, with HAC inference.

    The third formula on the poster. Panel form: every (asset, month) pair
    with both a signal and a next-month return contributes one observation.
    Standard errors are HAC with the overlap floor, because a signal built
    from an L-month window at every month is itself overlapping.
    """
    if isinstance(momentum_signal, pd.Series):
        momentum_signal = momentum_signal.to_frame("asset")
    if isinstance(returns, pd.Series):
        returns = returns.to_frame("asset")
    if list(momentum_signal.columns) != list(returns.columns):
        raise StatsError("signal and return columns must match")

    forward = returns.shift(-1)
    pairs = pd.DataFrame(
        {
            "m": momentum_signal.stack(),
            "r": forward.stack(),
        }
    ).dropna()
    if len(pairs) < 30:
        raise StatsError(f"only {len(pairs)} usable observations")

    # Ordered by date so the HAC lag structure is meaningful.
    fit = ols_hac(pairs["r"].to_numpy(), {"beta": pairs["m"].to_numpy()}, overlap=overlap)
    return {
        "alpha": fit["alpha"],
        "alpha_t": fit["alpha_t"],
        "beta": fit["beta"],
        "beta_se": fit["beta_se"],
        "beta_t": fit["beta_t"],
        "r_squared": fit["r_squared"],
        "n_obs": fit["n_obs"],
        "hac_lags": fit["hac_lags"],
    }


def deflated_threshold(n_trials: int, alpha: float = 0.05) -> float:
    """Bonferroni-corrected two-sided critical t for `n_trials` tests.

    Quoted next to every raw t-stat in the report so a 2.0 found after
    scanning a 40-cell parameter grid is not mistaken for evidence.
    """
    if n_trials < 1:
        raise StatsError("n_trials must be >= 1")
    from scipy import stats as sps

    return float(sps.norm.ppf(1 - alpha / (2 * n_trials)))


def annualised_mean(returns: pd.Series, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    return float(returns.mean()) * periods_per_year
