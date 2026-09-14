"""Frequency handling: the annualisation constant must never be implicit.

Scaling a Sharpe by the wrong bars-per-year does not raise -- it just
reports a number 4.66x too small (or too large). These tests pin the
scaling law and the weekend handling that a daily spot series needs.
"""

import numpy as np
import pandas as pd
import pytest

from momentum import data, metrics
from momentum import signals as sig
from momentum.data import DataQualityError


def daily_series(n=600, seed=0, weekends=False):
    freq = "D" if weekends else "B"
    idx = pd.date_range("2021-01-04", periods=n, freq=freq)
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0004, 0.011, n), index=idx)


class TestAnnualisationScaling:
    def test_sharpe_scales_as_sqrt_of_frequency(self):
        r = daily_series()
        monthly = metrics.sharpe(r, periods_per_year=12)
        daily = metrics.sharpe(r, periods_per_year=261)
        assert daily / monthly == pytest.approx(np.sqrt(261 / 12), rel=1e-12)

    def test_vol_scales_as_sqrt_of_frequency(self):
        r = daily_series()
        assert metrics.annual_vol(r, 261) / metrics.annual_vol(r, 12) == pytest.approx(
            np.sqrt(261 / 12), rel=1e-12
        )

    def test_cagr_uses_the_frequency_to_count_years(self):
        # 261 bars of +0.1% is one year at daily frequency, 21.75 years monthly.
        r = pd.Series(0.001, index=pd.date_range("2021-01-04", periods=261, freq="B"))
        one_year = metrics.cagr(r, periods_per_year=261)
        assert one_year == pytest.approx(1.001 ** 261 - 1, rel=1e-9)
        assert metrics.cagr(r, periods_per_year=12) < one_year  # spread over 21.75y

    def test_default_is_still_monthly(self):
        # Backward compatibility: the monthly study must not move.
        r = daily_series()
        assert metrics.sharpe(r) == metrics.sharpe(r, periods_per_year=12)

    def test_wrong_frequency_is_a_silent_4x_error(self):
        # Documents the trap: nothing raises, both numbers look plausible,
        # and the reported Sharpe is wrong by a factor of 4.66.
        r = daily_series()
        wrong = metrics.sharpe(r, periods_per_year=12)
        right = metrics.sharpe(r, periods_per_year=261)
        assert np.isfinite(wrong) and np.isfinite(right)
        assert right / wrong == pytest.approx(np.sqrt(261 / 12), rel=1e-12)

        # On a series with a real edge the absolute gap is what bites: a
        # steady drip of gains scores 26.9 at the right frequency and 5.8 at
        # the wrong one -- both finite, both plausible-looking in a table.
        strong = pd.Series(
            [0.004, 0.001] * 300, index=pd.date_range("2021-01-04", periods=600, freq="B")
        )
        assert metrics.sharpe(strong, periods_per_year=261) == pytest.approx(26.90, abs=0.1)
        assert metrics.sharpe(strong, periods_per_year=12) == pytest.approx(5.77, abs=0.1)


class TestInferBarsPerYear:
    def test_detects_weekday_frequency(self):
        idx = pd.date_range("2021-01-04", periods=1305, freq="B")
        assert data.infer_bars_per_year(idx) == pytest.approx(261, abs=2)

    def test_detects_a_calendar_day_series(self):
        # The XAUUSD feed's actual shape before weekends are dropped.
        idx = pd.date_range("2021-01-04", periods=1826, freq="D")
        assert data.infer_bars_per_year(idx) == pytest.approx(365, abs=2)

    def test_detects_monthly(self):
        idx = pd.date_range("2000-01-31", periods=240, freq="ME")
        assert data.infer_bars_per_year(idx) == pytest.approx(12, abs=0.1)

    def test_too_few_bars_raises(self):
        with pytest.raises(DataQualityError, match="at least 3 bars"):
            data.infer_bars_per_year(pd.date_range("2021-01-04", periods=2, freq="B"))


class TestDropWeekends:
    def test_removes_saturday_and_sunday(self):
        frame = pd.DataFrame(
            {"p": range(28)}, index=pd.date_range("2021-01-01", periods=28, freq="D")
        )
        kept = data.drop_weekends(frame)
        assert set(kept.index.dayofweek) <= {0, 1, 2, 3, 4}
        assert len(kept) == 20

    def test_friday_to_monday_return_absorbs_the_weekend(self):
        # Nothing is lost by dropping weekend bars: the Monday return simply
        # spans them, which is what a holder actually experiences.
        idx = pd.to_datetime(["2021-01-08", "2021-01-09", "2021-01-10", "2021-01-11"])
        prices = pd.DataFrame({"p": [100.0, 104.0, 106.0, 110.0]}, index=idx)  # Fri,Sat,Sun,Mon
        weekday = data.drop_weekends(prices)
        assert list(weekday.index.strftime("%a")) == ["Fri", "Mon"]
        monday_return = weekday["p"].pct_change().iloc[-1]
        assert monday_return == pytest.approx(0.10)  # 100 -> 110, weekend included

    def test_rejects_a_non_datetime_index(self):
        with pytest.raises(DataQualityError, match="DatetimeIndex"):
            data.drop_weekends(pd.DataFrame({"p": [1, 2]}))


class TestVolTargetAtDailyFrequency:
    def test_hits_the_target_when_told_the_right_frequency(self):
        r = daily_series(n=1200, seed=3)
        lev = sig.volatility_target_scalar(
            r, target_vol=0.15, window=60, max_leverage=10, periods_per_year=261
        ).dropna()
        realised = metrics.annual_vol(r, 261)
        # Leverage should sit near target/realised.
        assert lev.median() == pytest.approx(0.15 / realised, rel=0.35)

    def test_using_the_monthly_constant_misprices_leverage(self):
        r = daily_series(n=1200, seed=3)
        right = sig.volatility_target_scalar(
            r, 0.15, 60, max_leverage=1e6, periods_per_year=261
        ).dropna()
        wrong = sig.volatility_target_scalar(
            r, 0.15, 60, max_leverage=1e6, periods_per_year=12
        ).dropna()
        # The mistake levers up by sqrt(261/12) -- a 4.7x position error.
        assert (wrong / right).median() == pytest.approx(np.sqrt(261 / 12), rel=1e-9)


class TestSingleAssetIsRefused:
    """A one-column panel has no cross-section; saying so beats a zero book."""

    def test_quantile_weights_raises_on_one_asset(self):
        idx = pd.date_range("2021-01-04", periods=50, freq="B")
        one = pd.DataFrame({"XAUUSD": np.linspace(0.1, 0.5, 50)}, index=idx)
        with pytest.raises(sig.SignalError, match="at least 2 assets"):
            sig.quantile_weights(one)

    def test_the_old_behaviour_would_have_been_an_all_zero_book(self):
        # Regression guard. Before the fix the per-row "fewer than 2 names"
        # branch skipped every date and returned a silent flat book, which
        # reads as "the strategy chose not to trade" rather than "this
        # strategy cannot be formed".
        idx = pd.date_range("2021-01-04", periods=50, freq="B")
        two = pd.DataFrame(
            {"A": np.linspace(0.1, 0.5, 50), "B": np.linspace(0.5, 0.1, 50)}, index=idx
        )
        weights = sig.quantile_weights(two)          # two assets is fine
        assert weights.abs().to_numpy().sum() > 0

    def test_timeseries_momentum_works_on_one_asset(self):
        # The correct construction for a single instrument.
        idx = pd.date_range("2021-01-04", periods=400, freq="B")
        rng = np.random.default_rng(0)
        prices = pd.DataFrame(
            {"XAUUSD": 1800 * np.cumprod(1 + rng.normal(0.0005, 0.01, 400))}, index=idx
        )
        from momentum import backtest as bt

        result = bt.timeseries_momentum(
            prices, bt.BacktestConfig(lookback=126, skip=21, periods_per_year=261)
        )
        assert len(result.returns) > 0


class TestInverseVolWeights:
    """Equal weights are not equal risk on a heterogeneous universe."""

    def _panel(self, n=120, seed=0):
        idx = pd.date_range("2010-01-31", periods=n, freq="ME")
        rng = np.random.default_rng(seed)
        return pd.DataFrame(
            {
                "calm": rng.normal(0, 0.02, n),    # ~7% annualised
                "wild": rng.normal(0, 0.20, n),    # ~69% annualised, like natgas
            },
            index=idx,
        )

    def test_high_vol_asset_gets_a_smaller_weight(self):
        r = self._panel()
        w = pd.DataFrame(0.5, index=r.index, columns=r.columns)
        out = sig.inverse_vol_weights(w, r, window=12).iloc[20:]
        assert (out["calm"] > out["wild"]).all()
        # roughly the inverse-volatility ratio
        assert (out["calm"] / out["wild"]).median() == pytest.approx(10, rel=0.5)

    def test_gross_exposure_is_preserved(self):
        r = self._panel()
        w = pd.DataFrame({"calm": 0.5, "wild": -0.5}, index=r.index)
        out = sig.inverse_vol_weights(w, r, window=12)
        live = out.loc[out.abs().sum(axis=1) > 0]
        assert live.abs().sum(axis=1).round(10).eq(1.0).all()

    def test_no_lookahead_future_returns_cannot_change_todays_weights(self):
        r = self._panel()
        w = pd.DataFrame(0.5, index=r.index, columns=r.columns)
        base = sig.inverse_vol_weights(w, r, window=12)
        tampered = r.copy()
        tampered.iloc[60:] *= 50.0          # violent regime change, later only
        after = sig.inverse_vol_weights(w, tampered, window=12)
        pd.testing.assert_frame_equal(base.iloc[:60], after.iloc[:60])

    def test_rows_without_a_vol_estimate_stay_flat(self):
        r = self._panel()
        w = pd.DataFrame(0.5, index=r.index, columns=r.columns)
        out = sig.inverse_vol_weights(w, r, window=12)
        assert out.iloc[:11].abs().to_numpy().sum() == 0.0

    def test_cap_stops_one_quiet_asset_dominating(self):
        idx = pd.date_range("2010-01-31", periods=60, freq="ME")
        rng = np.random.default_rng(1)
        r = pd.DataFrame(
            {"normal": rng.normal(0, 0.05, 60),
             "frozen": rng.normal(0, 0.0001, 60),   # near-zero vol
             "other": rng.normal(0, 0.05, 60)},
            index=idx,
        )
        w = pd.DataFrame(1 / 3, index=idx, columns=r.columns)
        out = sig.inverse_vol_weights(w, r, window=12, max_scale=5.0).iloc[15:]
        assert (out["frozen"] < 0.95).all()   # capped, not ~100% of the book

    def test_index_and_column_mismatches_raise(self):
        r = self._panel()
        w = pd.DataFrame(0.5, index=r.index, columns=r.columns)
        with pytest.raises(sig.SignalError, match="share an index"):
            sig.inverse_vol_weights(w.iloc[1:], r, window=12)
        with pytest.raises(sig.SignalError, match="column order"):
            sig.inverse_vol_weights(w[["wild", "calm"]], r, window=12)

    def test_bad_parameters_raise(self):
        r = self._panel()
        w = pd.DataFrame(0.5, index=r.index, columns=r.columns)
        with pytest.raises(sig.SignalError, match="window must be"):
            sig.inverse_vol_weights(w, r, window=1)
        with pytest.raises(sig.SignalError, match="max_scale"):
            sig.inverse_vol_weights(w, r, window=12, max_scale=0)


class TestRuinDetection:
    """A book that compounds past zero produces numbers that describe nothing."""

    def test_detects_a_wiped_out_book(self):
        idx = pd.date_range("2010-01-31", periods=5, freq="ME")
        # -120% in one month: a long/short book can do this; equity goes negative.
        r = pd.Series([0.05, 0.02, -1.2, 0.10, 0.03], index=idx)
        assert metrics.is_ruined(r) is True

    def test_a_surviving_book_is_not_ruined(self):
        idx = pd.date_range("2010-01-31", periods=5, freq="ME")
        r = pd.Series([-0.5, -0.5, -0.5, 0.1, 0.1], index=idx)
        assert metrics.is_ruined(r) is False

    def test_exactly_minus_one_hundred_percent_is_ruin(self):
        idx = pd.date_range("2010-01-31", periods=3, freq="ME")
        r = pd.Series([0.05, -1.0, 0.20], index=idx)
        assert metrics.is_ruined(r) is True

    def test_summarise_surfaces_it(self):
        idx = pd.date_range("2010-01-31", periods=40, freq="ME")
        rng = np.random.default_rng(0)
        ok = pd.Series(rng.normal(0.005, 0.03, 40), index=idx)
        assert metrics.summarise(ok)["ruined"] is False
        blown = ok.copy()
        blown.iloc[10] = -1.4
        assert metrics.summarise(blown)["ruined"] is True
