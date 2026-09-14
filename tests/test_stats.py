import numpy as np
import pandas as pd
import pytest

from momentum import metrics, stats, synth
from momentum.stats import StatsError


def series(values, start="2000-01-31"):
    idx = pd.date_range(start, periods=len(values), freq="ME")
    return pd.Series(values, index=idx, dtype=float)


class TestNeweyWest:
    def test_matches_the_plain_t_stat_on_iid_data_at_zero_lags(self):
        rng = np.random.default_rng(0)
        x = series(list(rng.normal(0.01, 0.05, 400)))
        nw = stats.newey_west_tstat(x, lags=0)
        plain = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
        assert nw["t_stat"] == pytest.approx(plain, rel=0.02)

    def test_positive_autocorrelation_widens_the_standard_error(self):
        rng = np.random.default_rng(1)
        n = 400
        eps = rng.normal(0, 0.04, n)
        ar = np.zeros(n)
        for t in range(1, n):
            ar[t] = 0.6 * ar[t - 1] + eps[t]
        x = series(list(ar + 0.01))
        naive = stats.newey_west_tstat(x, lags=0)["hac_se"]
        hac = stats.newey_west_tstat(x, lags=12)["hac_se"]
        assert hac > naive * 1.3

    def test_overlap_floors_the_bandwidth(self):
        rng = np.random.default_rng(2)
        x = series(list(rng.normal(0.01, 0.05, 200)))
        auto = stats.newey_west_tstat(x, overlap=1)["lags_used"]
        floored = stats.newey_west_tstat(x, overlap=12)["lags_used"]
        assert floored == 11
        assert floored > auto

    def test_tiny_sample_raises(self):
        with pytest.raises(StatsError, match="at least 8"):
            stats.newey_west_tstat(series([0.01] * 5))


class TestBlockBootstrap:
    def test_block_resample_preserves_autocorrelation(self):
        # The point of the block version: an i.i.d. resample would destroy
        # the serial dependence and collapse the CI onto the null.
        rng = np.random.default_rng(3)
        n = 400
        eps = rng.normal(0, 0.04, n)
        ar = np.zeros(n)
        for t in range(1, n):
            ar[t] = 0.7 * ar[t - 1] + eps[t]
        x = series(list(ar))

        def lag1_autocorr(s):
            return float(pd.Series(s.to_numpy()).autocorr(lag=1))

        blocked = stats.moving_block_bootstrap(x, lag1_autocorr, n_boot=300, block=30, seed=1)
        iid = stats.moving_block_bootstrap(x, lag1_autocorr, n_boot=300, block=3, seed=1)
        # Longer blocks retain more of the true 0.7 autocorrelation.
        assert blocked["ci_low"] > iid["ci_low"]

    def test_ci_brackets_the_point_estimate(self):
        rng = np.random.default_rng(5)
        x = series(list(rng.normal(0.01, 0.05, 300)))
        out = stats.moving_block_bootstrap(x, metrics.sharpe, n_boot=400, seed=2)
        assert out["ci_low"] < out["point"] < out["ci_high"]

    def test_a_strong_signal_gets_a_ci_clear_of_zero(self):
        x = series([0.02, 0.018, 0.022, 0.019] * 60)
        out = stats.moving_block_bootstrap(x, lambda s: float(s.mean()), n_boot=400, seed=3)
        assert out["ci_low"] > 0
        assert out["p_gt_zero"] > 0.99

    def test_short_sample_raises(self):
        with pytest.raises(StatsError, match="at least 24"):
            stats.moving_block_bootstrap(series([0.01] * 10), metrics.sharpe)


class TestPredictiveRegression:
    def test_recovers_a_planted_positive_beta(self):
        prices = synth.momentum_panel(n_months=600, n_assets=8, persistence=0.5, seed=7)
        rets = prices.pct_change().iloc[1:]
        from momentum import signals as sig

        m = sig.compound_momentum(rets, 12, 1)
        out = stats.predictive_regression(m, rets, overlap=12)
        assert out["beta"] > 0
        assert out["beta_t"] > 2.0

    def test_finds_no_beta_on_a_random_walk(self):
        prices = synth.random_walk_panel(n_months=600, n_assets=8, seed=7)
        rets = prices.pct_change().iloc[1:]
        from momentum import signals as sig

        m = sig.compound_momentum(rets, 12, 1)
        out = stats.predictive_regression(m, rets, overlap=12)
        assert abs(out["beta_t"]) < 2.5

    def test_mismatched_columns_raise(self):
        idx = pd.date_range("2000-01-31", periods=60, freq="ME")
        a = pd.DataFrame(0.0, index=idx, columns=["A"])
        b = pd.DataFrame(0.0, index=idx, columns=["B"])
        with pytest.raises(StatsError, match="columns must match"):
            stats.predictive_regression(a, b)


class TestMultipleTesting:
    def test_threshold_rises_with_the_number_of_trials(self):
        assert stats.deflated_threshold(1) == pytest.approx(1.96, abs=0.01)
        assert stats.deflated_threshold(40) > 3.0
        assert stats.deflated_threshold(400) > stats.deflated_threshold(40)
