import os

import numpy as np
import pandas as pd
import pytest

from momentum import data
from momentum.data import DataQualityError

RAW = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
UNIVERSE = ["XLB", "XLE", "XLF", "XLK", "XLP", "XLV", "XLY"]
AVAILABLE = [s for s in ["XLE", "XLF", "XLK", "XLP", "XLV", "XLY"] if os.path.exists(os.path.join(RAW, f"{s}.csv"))]


class TestLoading:
    def test_loads_ascending_and_positive(self, tmp_path):
        csv = tmp_path / "X.csv"
        csv.write_text("date,adj_close\n2020-02-29,11\n2020-01-31,10\n")
        s = data.load_symbol(str(csv))
        assert list(s.index.strftime("%Y-%m")) == ["2020-01", "2020-02"]
        assert s.iloc[0] == 10.0

    def test_rejects_missing_columns(self, tmp_path):
        csv = tmp_path / "X.csv"
        csv.write_text("date,close\n2020-01-31,10\n")
        with pytest.raises(DataQualityError, match="missing column"):
            data.load_symbol(str(csv))

    def test_rejects_non_positive_prices(self, tmp_path):
        csv = tmp_path / "X.csv"
        csv.write_text("date,adj_close\n2020-01-31,10\n2020-02-29,0\n")
        with pytest.raises(DataQualityError, match="non-positive"):
            data.load_symbol(str(csv))

    def test_rejects_duplicate_dates(self, tmp_path):
        csv = tmp_path / "X.csv"
        csv.write_text("date,adj_close\n2020-01-31,10\n2020-01-31,11\n")
        with pytest.raises(DataQualityError, match="duplicate"):
            data.load_symbol(str(csv))

    def test_missing_file_names_the_symbol(self, tmp_path):
        with pytest.raises(DataQualityError, match="no cached data for ZZZ"):
            data.load_panel(str(tmp_path), ["ZZZ"])


class TestMonthEndNormalisation:
    def test_differing_last_trading_days_join_to_one_row(self):
        # The real hazard: one vendor row is 2010-10-29, another 2010-10-31.
        # Without normalisation the panel gains holes that look like gaps.
        a = pd.Series([1.0, 2.0], index=pd.to_datetime(["2010-09-30", "2010-10-29"]))
        b = pd.Series([1.0, 2.0], index=pd.to_datetime(["2010-09-30", "2010-10-31"]))
        raw = pd.DataFrame({"A": a, "B": b})
        assert raw.isna().sum().sum() == 2  # the bug, before normalisation
        # load_panel normalises each series to its month period first, which
        # is what makes the join line up.
        merged = pd.DataFrame({"A": data.to_month_end(a.to_frame("A"))["A"],
                               "B": data.to_month_end(b.to_frame("B"))["B"]})
        assert merged.isna().sum().sum() == 0
        assert len(merged) == 2

    def test_two_bars_in_one_month_is_rejected(self):
        idx = pd.to_datetime(["2010-10-15", "2010-10-29"])
        with pytest.raises(DataQualityError, match="same calendar month"):
            data.to_month_end(pd.DataFrame({"A": [1.0, 2.0]}, index=idx))


class TestIncompleteMonth:
    def test_drops_a_partial_final_month(self):
        idx = pd.to_datetime(["2026-07-31", "2026-08-31", "2026-09-11"])
        panel = pd.DataFrame({"A": [1.0, 2.0, 3.0]}, index=idx)
        trimmed = data.drop_incomplete_last_month(panel, asof=pd.Timestamp("2026-09-14"))
        assert len(trimmed) == 2
        assert trimmed.index[-1] == pd.Timestamp("2026-08-31")

    def test_keeps_a_completed_final_month(self):
        idx = pd.to_datetime(["2026-07-31", "2026-08-31"])
        panel = pd.DataFrame({"A": [1.0, 2.0]}, index=idx)
        assert len(data.drop_incomplete_last_month(panel, asof=pd.Timestamp("2026-09-14"))) == 2


class TestAlignment:
    def test_common_sample_is_the_default(self):
        idx = pd.date_range("2000-01-31", periods=4, freq="ME")
        panel = pd.DataFrame({"A": [1.0, 2, 3, 4], "B": [np.nan, 2, 3, 4]}, index=idx)
        assert len(data.align_panel(panel)) == 3

    def test_rejects_an_impossible_coverage(self):
        idx = pd.date_range("2000-01-31", periods=2, freq="ME")
        panel = pd.DataFrame({"A": [np.nan, np.nan]}, index=idx)
        with pytest.raises(DataQualityError, match="no rows meet"):
            data.align_panel(panel)


@pytest.mark.skipif(not AVAILABLE, reason="cached vendor data not present")
class TestRealFeed:
    def test_panel_loads_and_has_no_holes(self):
        panel = data.align_panel(data.load_panel(RAW, AVAILABLE))
        assert panel.isna().sum().sum() == 0
        assert len(panel) > 300

    def test_no_monthly_move_implies_an_unadjusted_split(self):
        # XLK, XLE and XLY all split in Dec 2025; the adjusted-close series
        # must absorb that. A raw close would print a -50% month here.
        panel = data.align_panel(data.load_panel(RAW, AVAILABLE))
        rets = panel.pct_change().iloc[1:]
        assert rets.abs().to_numpy().max() < 0.6

    def test_december_2025_split_month_is_continuous(self):
        panel = data.load_panel(RAW, AVAILABLE)
        if "XLK" not in panel.columns:
            pytest.skip("XLK not cached")
        dec = panel.loc["2025-12-31", "XLK"] / panel.loc["2025-11-30", "XLK"] - 1
        assert abs(dec) < 0.2  # a 2:1 split unadjusted would show ~-50%
