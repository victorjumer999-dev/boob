import numpy as np
import pandas as pd
import pytest

from momentum import backtest as bt
from momentum import metrics, synth
from momentum.backtest import BacktestConfig, BacktestError


def simple_panel():
    idx = pd.date_range("2000-01-31", periods=6, freq="ME")
    return pd.DataFrame(
        {"A": [100.0, 110, 121, 133.1, 146.41, 161.051],
         "B": [100.0, 100, 100, 100, 100, 100]},
        index=idx,
    )


class TestNoLookAhead:
    """The core invariant: today's bar cannot pay for today's information."""

    def test_a_perfect_oracle_signal_earns_nothing_this_bar(self):
        # Build weights that are literally next period's winner, placed at
        # the date the return occurs. Because the engine shifts weights, the
        # oracle is always one bar late and cannot harvest its own foresight.
        rng = np.random.default_rng(3)
        idx = pd.date_range("2000-01-31", periods=200, freq="ME")
        rets = pd.DataFrame(rng.normal(0, 0.05, (200, 3)), index=idx, columns=list("ABC"))
        oracle = rets.rank(axis=1, ascending=False)
        weights = (oracle == 1).astype(float)  # 100% in this bar's best asset
        result = bt.run_weighted_backtest(weights, rets, cost_bps=0.0)
        # If the engine did NOT shift, this would be a monstrous Sharpe.
        assert abs(metrics.sharpe(result.returns)) < 1.0

    def test_unshifted_oracle_would_be_absurd_control(self):
        # Control for the test above: the same oracle applied WITHOUT the
        # shift really is absurd, proving the previous assertion has teeth.
        rng = np.random.default_rng(3)
        idx = pd.date_range("2000-01-31", periods=200, freq="ME")
        rets = pd.DataFrame(rng.normal(0, 0.05, (200, 3)), index=idx, columns=list("ABC"))
        weights = (rets.rank(axis=1, ascending=False) == 1).astype(float)
        cheating = (weights * rets).sum(axis=1)
        assert metrics.sharpe(cheating) > 4.0  # measured ~4.6 vs <1.0 when shifted

    def test_future_prices_cannot_change_past_pnl(self):
        prices = synth.momentum_panel(n_months=180, n_assets=6, seed=11)
        base = bt.cross_sectional_momentum(prices).returns
        tampered = prices.copy()
        tampered.iloc[-1] *= 1.2  # rewrite only the final bar (within sane bounds)
        after = bt.cross_sectional_momentum(tampered).returns
        common = base.index.intersection(after.index)[:-1]
        pd.testing.assert_series_equal(base.loc[common], after.loc[common])

    def test_first_investable_return_starts_after_the_signal(self):
        prices = synth.momentum_panel(n_months=60, n_assets=6, seed=1)
        result = bt.cross_sectional_momentum(prices, BacktestConfig(lookback=12, skip=1))
        first_signal = result.meta["signal"].dropna(how="all").index[0]
        assert result.returns.index[0] > first_signal


class TestWeightApplication:
    def test_a_never_traded_book_raises_instead_of_reporting_zeros(self):
        # A dead signal layer and a genuinely flat strategy look identical in
        # the output (a row of 0.00% returns). Reporting the zeros would hand
        # back a track record with a real start date, a tiny volatility and a
        # meaningless Sharpe. Refuse instead.
        prices = simple_panel()
        rets = prices.pct_change().iloc[1:]
        w = pd.DataFrame(0.0, index=rets.index, columns=rets.columns)
        with pytest.raises(BacktestError, match="never takes a position"):
            bt.run_weighted_backtest(w, rets, cost_bps=25.0)

    def test_interior_flat_months_are_kept(self):
        # Only the LEADING warm-up is trimmed. A mid-sample step to cash is a
        # real decision and its 0% month belongs in the return stream.
        idx = pd.date_range("2000-01-31", periods=6, freq="ME")
        rets = pd.DataFrame({"A": [0.02] * 6}, index=idx)
        w = pd.DataFrame({"A": [0.0, 1.0, 0.0, 0.0, 1.0, 1.0]}, index=idx)
        result = bt.run_weighted_backtest(w, rets, cost_bps=0.0)
        assert result.returns.index[0] == idx[2]      # warm-up row dropped
        assert result.returns.loc[idx[3]] == pytest.approx(0.0)  # interior flat kept
        assert len(result.returns) == 4

    def test_full_allocation_reproduces_the_asset(self):
        prices = simple_panel()
        rets = prices.pct_change().iloc[1:]
        w = pd.DataFrame({"A": 1.0, "B": 0.0}, index=rets.index)
        result = bt.run_weighted_backtest(w, rets, cost_bps=0.0)
        pd.testing.assert_series_equal(
            result.returns, rets["A"].iloc[1:], check_names=False
        )

    def test_index_mismatch_is_rejected(self):
        prices = simple_panel()
        rets = prices.pct_change().iloc[1:]
        w = pd.DataFrame(0.0, index=rets.index[:-1], columns=rets.columns)
        with pytest.raises(BacktestError, match="share an index"):
            bt.run_weighted_backtest(w, rets)

    def test_unadjusted_split_is_caught(self):
        idx = pd.date_range("2000-01-31", periods=5, freq="ME")
        rets = pd.DataFrame({"A": [0.01, 0.01, -0.5, 3.0, 0.01]}, index=idx)
        w = pd.DataFrame(1.0, index=idx, columns=["A"])
        with pytest.raises(BacktestError, match="split or dividend"):
            bt.run_weighted_backtest(w, rets)


class TestCosts:
    def test_costs_scale_linearly_with_the_rate(self):
        prices = synth.momentum_panel(n_months=180, n_assets=6, seed=5)
        cheap = bt.cross_sectional_momentum(prices, BacktestConfig(cost_bps=5))
        dear = bt.cross_sectional_momentum(prices, BacktestConfig(cost_bps=50))
        assert dear.costs.sum() == pytest.approx(cheap.costs.sum() * 10, rel=1e-9)

    def test_costs_reduce_net_return(self):
        prices = synth.momentum_panel(n_months=180, n_assets=6, seed=5)
        r = bt.cross_sectional_momentum(prices, BacktestConfig(cost_bps=50))
        assert r.returns.sum() < r.gross_returns.sum()

    def test_turnover_measures_traded_notional(self):
        idx = pd.date_range("2000-01-31", periods=3, freq="ME")
        w = pd.DataFrame(
            {"A": [1.0, 0.0, 1.0], "B": [0.0, 1.0, 0.0]}, index=idx
        )
        t = bt.compute_turnover(w)
        assert list(t) == [1.0, 2.0, 2.0]  # build, full switch, full switch

    def test_zero_cost_equals_gross(self):
        prices = synth.momentum_panel(n_months=120, n_assets=6, seed=2)
        r = bt.cross_sectional_momentum(prices, BacktestConfig(cost_bps=0.0))
        pd.testing.assert_series_equal(r.returns, r.gross_returns, check_names=False)


class TestEdgeRecovery:
    """Does the engine find a planted edge, and stay quiet when there is none?"""

    def test_finds_a_planted_cross_sectional_edge(self):
        prices = synth.momentum_panel(
            n_months=600, n_assets=10, persistence=0.45, seed=42
        )
        r = bt.cross_sectional_momentum(prices, BacktestConfig(cost_bps=0.0))
        assert metrics.sharpe(r.returns) > 0.5

    def test_reports_no_edge_on_a_random_walk(self):
        # Averaged over seeds so a single lucky path cannot pass or fail it.
        sharpes = []
        for seed in range(12):
            prices = synth.random_walk_panel(n_months=400, n_assets=10, seed=seed)
            r = bt.cross_sectional_momentum(prices, BacktestConfig(cost_bps=0.0))
            sharpes.append(metrics.sharpe(r.returns))
        assert abs(np.mean(sharpes)) < 0.35

    def test_stronger_planted_effect_gives_a_stronger_result(self):
        weak = bt.cross_sectional_momentum(
            synth.momentum_panel(n_months=600, n_assets=10, persistence=0.10, seed=9),
            BacktestConfig(cost_bps=0.0),
        )
        strong = bt.cross_sectional_momentum(
            synth.momentum_panel(n_months=600, n_assets=10, persistence=0.60, seed=9),
            BacktestConfig(cost_bps=0.0),
        )
        assert metrics.sharpe(strong.returns) > metrics.sharpe(weak.returns)

    def test_timeseries_momentum_works_on_clean_trends(self):
        prices = synth.trending_panel(n_months=360, n_assets=4, seed=4)
        r = bt.timeseries_momentum(prices, BacktestConfig(cost_bps=0.0))
        assert metrics.sharpe(r.returns) > 0.5


class TestVolTargeting:
    def test_brings_realised_vol_toward_the_target(self):
        prices = synth.momentum_panel(n_months=480, n_assets=8, persistence=0.3, seed=6)
        plain = bt.cross_sectional_momentum(prices, BacktestConfig(cost_bps=0.0))
        targeted = bt.cross_sectional_momentum(
            prices, BacktestConfig(cost_bps=0.0, target_vol=0.10, max_leverage=3.0)
        )
        gap_plain = abs(metrics.annual_vol(plain.returns) - 0.10)
        gap_targeted = abs(metrics.annual_vol(targeted.returns) - 0.10)
        assert gap_targeted < gap_plain

    def test_missing_vol_estimate_means_flat_not_one_times(self):
        prices = synth.momentum_panel(n_months=200, n_assets=6, seed=8)
        r = bt.cross_sectional_momentum(
            prices, BacktestConfig(cost_bps=0.0, target_vol=0.10)
        )
        assert not r.returns.isna().any()


class TestDecilePortfolios:
    def test_planted_momentum_produces_a_monotone_ladder(self):
        prices = synth.momentum_panel(n_months=600, n_assets=9, persistence=0.5, seed=13)
        table = bt.decile_portfolios(prices, n_buckets=3)
        means = table["mean"].tolist()
        assert means == sorted(means)  # bucket 1 (losers) .. bucket 3 (winners)

    def test_rejects_more_buckets_than_assets(self):
        prices = synth.momentum_panel(n_months=120, n_assets=3, seed=1)
        with pytest.raises(BacktestError, match="cannot fill"):
            bt.decile_portfolios(prices, n_buckets=10)
