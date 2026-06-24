"""Tests for ABS Excel and RBA CSV parsers."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.parsers import (
    _clean_abs_column_name,
    _excel_serial_to_date,
    parse_rba_csv,
    resample_to_quarter,
)


class TestExcelSerialDate:
    def test_known_date(self):
        # Verify round-trip: compute expected serial dynamically from a known date
        known = pd.Timestamp("2003-09-30")
        serial = (known - pd.Timestamp("1899-12-30")).days
        result = _excel_serial_to_date(serial)
        assert result == known

    def test_epoch(self):
        # Serial 1 = 1899-12-31
        result = _excel_serial_to_date(1)
        assert result == pd.Timestamp("1899-12-31")

    def test_float_input(self):
        result = _excel_serial_to_date(37865.0)
        assert isinstance(result, pd.Timestamp)


class TestCleanAbsColumnName:
    def test_city_extraction(self):
        raw = "Residential Property Price Index ;  Sydney ;"
        assert _clean_abs_column_name(raw) == "Sydney"

    def test_weighted_average(self):
        raw = "Residential Property Price Index ;  Weighted average of eight capital cities ;"
        result = _clean_abs_column_name(raw)
        assert "Weighted" in result or "eight" in result.lower()

    def test_plain_string(self):
        assert _clean_abs_column_name("Melbourne") == "Melbourne"

    def test_strips_whitespace(self):
        result = _clean_abs_column_name("  Index ;  Brisbane  ;  ")
        assert result == "Brisbane"


class TestResampleToQuarter:
    def test_returns_period_index(self):
        dates = pd.date_range("2010-01-01", periods=730, freq="D")
        series = pd.Series(np.random.default_rng(0).uniform(2, 5, 730), index=dates, name="rate")
        result = resample_to_quarter(series)
        assert isinstance(result.index, pd.PeriodIndex)

    def test_last_method_preserves_quarter_end_value(self):
        dates = pd.date_range("2010-01-01", "2010-03-31", freq="D")
        values = list(range(len(dates)))
        series = pd.Series(values, index=dates, name="rate")
        result = resample_to_quarter(series, method="last")
        assert result.iloc[0] == max(values)

    def test_no_nan_output_for_complete_quarters(self):
        dates = pd.date_range("2010-01-01", periods=365, freq="D")
        series = pd.Series(np.ones(365), index=dates, name="rate")
        result = resample_to_quarter(series)
        assert result.notna().all()


class TestParseRbaCsv:
    def _make_rba_csv(self) -> Path:
        content = (
            "﻿F1 Interest Rates and Yields – Money Market\n"
            "Title,Cash Rate Target,Change in Cash Rate Target\n"
            "Description,Cash rate target set by RBA,Change in target\n"
            "Frequency,Daily,Daily\n"
            "Type,Original,Original\n"
            "Units,Per cent per annum,Percentage points\n"
            "Source,RBA,RBA\n"
            "Publication date,2024-01-15,2024-01-15\n"
            "Series ID,FIRMMCRTD,FIRMMCRTC\n"
            "Unique identifier,FIRMMCRTD,FIRMMCRTC\n"
            "\n"
            "\n"
            "04-Jan-2011,4.75,\n"
            "05-Jan-2011,4.75,\n"
            "06-Jan-2011,4.75,\n"
            "01-Nov-2011,4.50,-0.25\n"
        )
        tmp = Path("/tmp/test_rba_f1.csv")
        tmp.write_text(content, encoding="utf-8")
        return tmp

    def test_returns_series(self):
        path = self._make_rba_csv()
        result = parse_rba_csv(path)
        assert isinstance(result, pd.Series)

    def test_series_name_is_cash_rate(self):
        path = self._make_rba_csv()
        result = parse_rba_csv(path)
        assert result.name == "cash_rate"

    def test_values_in_plausible_range(self):
        path = self._make_rba_csv()
        result = parse_rba_csv(path)
        assert result.between(0, 30).all()

    def test_date_index_is_datetime(self):
        path = self._make_rba_csv()
        result = parse_rba_csv(path)
        assert isinstance(result.index, pd.DatetimeIndex)
