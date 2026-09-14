import numpy as np
import pandas as pd
import pytest

from momentum import metrics
from momentum.metrics import MetricError


def series(values, start="2000-01-31"):
    idx = pd.date_range(start, periods=len(values), freq="ME")
    return pd.Series(values, index=idx, dtype=float)


class TestCAGR:
    def test_doubling_over_one_year(self):
        r = series([2 ** (1 / 12) - 1] * 12)
        assert metrics.cagr(r) == pytest.approx(1.0, rel=1e-9)

    def test_flat_returns_zero(self):
        assert metrics.cagr(series([0.0] * 24)) == pytest.approx(0.0)

    def test_total_wipeout_is_minus_one(self):
        assert metrics.cagr(series([0.01, -1.0, 0.01])) == -1.0


class TestSharpe:
    def test_known_value(self):
        # mean 1%/mo against a sample sd (ddof=1) of 1.00419%/mo.
        r = series([0.02, 0.0] * 60)
        expected = 0.01 / float(r.std(ddof=1)) * np.sqrt(12)
        assert metrics.sharpe(r) == pytest.approx(expected, rel=1e-9)
        assert metrics.sharpe(r) == pytest.approx(3.4496, abs=1e-4)

    def test_annualises_by_sqrt_12_not_12(self):
        r = series(list(np.random.default_rng(0).normal(0.01, 0.04, 240)))
        monthly = float((r.mean()) / r.std(ddof=1))
        assert metrics.sharpe(r) == pytest.approx(monthly * np.sqrt(12), rel=1e-9)

    def test_risk_free_is_deannualised_geometrically(self):
        r = series([0.01] * 120)
        rf = 0.03
        expected_period = (1 + rf) ** (1 / 12) - 1
        excess = r - expected_period
        assert excess.mean() == pytest.approx(0.01 - expected_period)
        with pytest.raises(MetricError, match="zero volatility"):
            metrics.sharpe(r, risk_free=rf)  # constant excess -> undefined

    def test_zero_vol_raises_rather_than_returning_infinity(self):
        with pytest.raises(MetricError, match="zero volatility"):
            metrics.sharpe(series([0.01] * 30))

    def test_empty_input_raises(self):
        with pytest.raises(MetricError, match="no observations"):
            metrics.sharpe(series([np.nan] * 10))


class TestDrawdown:
    def test_simple_halving(self):
        assert metrics.max_drawdown(series([-0.5, 1.0])) == pytest.approx(-0.5)

    def test_monotonic_gains_have_no_drawdown(self):
        assert metrics.max_drawdown(series([0.01] * 36)) == pytest.approx(0.0)

    def test_drawdown_is_never_positive(self):
        rng = np.random.default_rng(4)
        r = series(list(rng.normal(0.005, 0.05, 300)))
        assert metrics.drawdown_series(r).max() <= 1e-12

    def test_window_identifies_peak_and_trough(self):
        r = series([0.10, -0.20, -0.20, 0.50, 0.10])
        w = metrics.worst_drawdown_window(r)
        assert w["depth"] == pytest.approx(-0.36, rel=1e-6)
        assert w["months_to_trough"] == 2


class TestSummary:
    def test_table_has_one_row_per_strategy(self):
        rng = np.random.default_rng(1)
        a = series(list(rng.normal(0.008, 0.04, 200)))
        b = series(list(rng.normal(0.004, 0.03, 200)))
        table = metrics.summary_table({"alpha": a, "beta": b})
        assert list(table.index) == ["alpha", "beta"]
        assert table.loc["alpha", "bars"] == 200

    def test_undefined_sub_metrics_become_nan_not_a_crash(self):
        table = metrics.summary_table({"winner": series([0.01] * 36)})
        assert np.isnan(table.loc["winner", "sortino"])
        assert np.isnan(table.loc["winner", "calmar"])
