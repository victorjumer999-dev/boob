import numpy as np
import pandas as pd
import pytest

from momentum import signals as sig
from momentum.signals import SignalError


def const_returns(value, n=40, cols=("A", "B")):
    idx = pd.date_range("2000-01-31", periods=n, freq="ME")
    return pd.DataFrame(value, index=idx, columns=list(cols))


class TestCumulativeMomentum:
    def test_matches_hand_computed_sum(self):
        r = const_returns(0.01, n=20)
        m = sig.cumulative_momentum(r, lookback=12, skip=0)
        # 12 months of 1% summed arithmetically
        assert m["A"].dropna().iloc[0] == pytest.approx(0.12)

    def test_skip_excludes_the_most_recent_month(self):
        idx = pd.date_range("2000-01-31", periods=14, freq="ME")
        r = pd.Series([0.01] * 13 + [10.0], index=idx).to_frame("A")
        # 12-1: the final, huge month must not enter the signal at that date.
        m = sig.cumulative_momentum(r, lookback=12, skip=1)
        assert m["A"].iloc[-1] == pytest.approx(0.11)

    def test_signal_is_nan_during_warmup(self):
        r = const_returns(0.01, n=20)
        m = sig.cumulative_momentum(r, lookback=12, skip=1)
        assert m["A"].iloc[:11].isna().all()
        assert not np.isnan(m["A"].iloc[11])

    def test_rejects_skip_at_or_above_lookback(self):
        r = const_returns(0.01, n=40)
        with pytest.raises(SignalError, match="smaller than lookback"):
            sig.cumulative_momentum(r, lookback=12, skip=12)

    def test_rejects_sample_shorter_than_lookback(self):
        r = const_returns(0.01, n=5)
        with pytest.raises(SignalError, match="at least 12 observations"):
            sig.cumulative_momentum(r, lookback=12)

    def test_does_not_return_a_neutral_default_on_short_sample(self):
        # The failure mode this guards: returning zeros/NaN quietly, leaving
        # a dead signal layer that looks like a working one.
        r = const_returns(0.01, n=5)
        with pytest.raises(SignalError):
            sig.cumulative_momentum(r, lookback=12)


class TestCompoundMomentum:
    def test_geometric_not_arithmetic(self):
        r = const_returns(0.10, n=20)
        m = sig.compound_momentum(r, lookback=12, skip=0)
        assert m["A"].dropna().iloc[0] == pytest.approx(1.10 ** 12 - 1)

    def test_diverges_from_the_sum_at_high_vol(self):
        r = const_returns(0.10, n=20)
        arith = sig.cumulative_momentum(r, lookback=12, skip=0)["A"].dropna().iloc[0]
        geom = sig.compound_momentum(r, lookback=12, skip=0)["A"].dropna().iloc[0]
        assert geom == pytest.approx(1.10 ** 12 - 1)
        assert geom > arith * 1.75  # 2.138 vs 1.200: the gap is ~78%


class TestPriceToSMA:
    def test_flat_price_gives_zero(self):
        idx = pd.date_range("2000-01-31", periods=20, freq="ME")
        p = pd.Series(100.0, index=idx).to_frame("A")
        m = sig.price_to_sma(p, lookback=12)
        assert m["A"].dropna().abs().max() == pytest.approx(0.0, abs=1e-12)

    def test_rising_price_is_above_its_average(self):
        idx = pd.date_range("2000-01-31", periods=30, freq="ME")
        p = pd.Series(np.arange(100, 130, dtype=float), index=idx).to_frame("A")
        assert sig.price_to_sma(p, 12)["A"].dropna().min() > 0


class TestRollingZScore:
    def test_standardises_to_unit_scale(self):
        rng = np.random.default_rng(7)
        idx = pd.date_range("2000-01-31", periods=400, freq="ME")
        s = pd.Series(rng.normal(0.05, 0.2, 400), index=idx)
        z = sig.rolling_zscore(s, window=36).dropna()
        assert abs(z.mean()) < 0.25
        assert 0.7 < z.std() < 1.4

    def test_constant_input_yields_nan_not_infinity(self):
        idx = pd.date_range("2000-01-31", periods=60, freq="ME")
        s = pd.Series(1.0, index=idx)
        z = sig.rolling_zscore(s, window=36)
        assert z.dropna().empty
        assert not np.isinf(z.to_numpy()).any()

    def test_uses_only_trailing_data(self):
        idx = pd.date_range("2000-01-31", periods=60, freq="ME")
        base = pd.Series(np.linspace(0, 1, 60), index=idx)
        spiked = base.copy()
        spiked.iloc[-1] = 99.0
        z_base = sig.rolling_zscore(base, window=24)
        z_spiked = sig.rolling_zscore(spiked, window=24)
        # Changing the LAST point must not change any EARLIER z-score.
        pd.testing.assert_series_equal(z_base.iloc[:-1], z_spiked.iloc[:-1])


class TestQuantileWeights:
    def test_legs_are_dollar_neutral_and_unit_sized(self):
        idx = pd.date_range("2000-01-31", periods=3, freq="ME")
        s = pd.DataFrame(
            [[3, 2, 1, 0, -1, -2]] * 3,
            index=idx,
            columns=list("ABCDEF"),
            dtype=float,
        )
        w = sig.quantile_weights(s, 1 / 3, 1 / 3)
        assert w.sum(axis=1).abs().max() == pytest.approx(0.0, abs=1e-12)
        assert w[w > 0].sum(axis=1).iloc[0] == pytest.approx(1.0)
        assert w[w < 0].sum(axis=1).iloc[0] == pytest.approx(-1.0)

    def test_longs_the_strongest_and_shorts_the_weakest(self):
        idx = pd.date_range("2000-01-31", periods=1, freq="ME")
        s = pd.DataFrame([[3, 2, 1, 0, -1, -2]], index=idx, columns=list("ABCDEF"), dtype=float)
        w = sig.quantile_weights(s, 1 / 3, 1 / 3).iloc[0]
        assert w["A"] > 0 and w["B"] > 0
        assert w["E"] < 0 and w["F"] < 0
        assert w["C"] == 0 and w["D"] == 0

    def test_long_only_holds_no_shorts(self):
        idx = pd.date_range("2000-01-31", periods=1, freq="ME")
        s = pd.DataFrame([[3, 2, 1, 0, -1, -2]], index=idx, columns=list("ABCDEF"), dtype=float)
        w = sig.quantile_weights(s, 1 / 3, 1 / 3, long_only=True).iloc[0]
        assert (w >= 0).all()
        assert w.sum() == pytest.approx(1.0)

    def test_all_nan_row_is_flat_not_concentrated(self):
        idx = pd.date_range("2000-01-31", periods=1, freq="ME")
        s = pd.DataFrame([[np.nan] * 6], index=idx, columns=list("ABCDEF"))
        assert sig.quantile_weights(s).abs().to_numpy().sum() == 0.0

    def test_rejects_overlapping_legs(self):
        idx = pd.date_range("2000-01-31", periods=1, freq="ME")
        s = pd.DataFrame([[1.0, 2.0]], index=idx, columns=list("AB"))
        with pytest.raises(SignalError, match="must not exceed 1"):
            sig.quantile_weights(s, top_q=0.8, bottom_q=0.8)


class TestTimeseriesSignal:
    def test_nan_stays_flat_instead_of_going_maximally_short(self):
        # np.where(m > 0, 1, -1) puts NaN in the -1 bucket. During a 12-month
        # warm-up that is a fully short book on no information.
        idx = pd.date_range("2000-01-31", periods=4, freq="ME")
        s = pd.Series([np.nan, np.nan, 0.05, -0.05], index=idx)
        out = sig.timeseries_signal(s)
        assert out.iloc[:2].isna().all()
        assert out.iloc[2] == 1.0
        assert out.iloc[3] == -1.0

    def test_long_only_mode_goes_flat_not_short(self):
        idx = pd.date_range("2000-01-31", periods=2, freq="ME")
        s = pd.Series([0.05, -0.05], index=idx)
        out = sig.timeseries_signal(s, allow_short=False)
        assert list(out) == [1.0, 0.0]


class TestVolatilityTarget:
    def test_halves_leverage_when_vol_doubles(self):
        idx = pd.date_range("2000-01-31", periods=60, freq="ME")
        calm = pd.Series([0.01, -0.01] * 30, index=idx)
        wild = calm * 2
        lev_calm = sig.volatility_target_scalar(calm, 0.10, 12, max_leverage=100).dropna()
        lev_wild = sig.volatility_target_scalar(wild, 0.10, 12, max_leverage=100).dropna()
        assert (lev_calm / lev_wild).round(6).nunique() == 1
        assert (lev_calm / lev_wild).iloc[0] == pytest.approx(2.0)

    def test_respects_the_leverage_cap(self):
        idx = pd.date_range("2000-01-31", periods=60, freq="ME")
        tiny = pd.Series([0.0001, -0.0001] * 30, index=idx)
        assert sig.volatility_target_scalar(tiny, 0.10, 12, max_leverage=2.0).max() <= 2.0
