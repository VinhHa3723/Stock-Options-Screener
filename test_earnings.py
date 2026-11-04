"""
test_earnings.py
----------------
Unit tests for the earnings feature, with synthetic data and a fixed "today".
Nothing here touches the network.

Run with:  python -m unittest test_earnings -v
"""

from __future__ import annotations
import datetime as dt
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import numpy as np
import pandas as pd

import earnings
from earnings import (
    ALL_FIELDS,
    candidate_tickers,
    fetch_earnings,
    merge_earnings,
    summarize_earnings,
    summarize_revenue,
)
from report import _fmt_earnings_cell, _earnings_tooltip, _fmt_revenue, write_html_dashboard

TODAY = dt.date(2026, 11, 3)
NY = "America/New_York"


def dates_frame(*rows, tz=NY):
    """rows are (timestamp string, estimate, reported, surprise)."""
    index = pd.DatetimeIndex([r[0] for r in rows], name="Earnings Date")
    if tz:
        index = index.tz_localize(tz)
    return pd.DataFrame(
        {
            "EPS Estimate": [r[1] for r in rows],
            "Reported EPS": [r[2] for r in rows],
            "Surprise(%)": [r[3] for r in rows],
        },
        index=index,
    )


def summarize(frame=None, calendar=None, recent=7, soon=14):
    return summarize_earnings(frame, calendar, TODAY, recent, soon)


class JustReported(unittest.TestCase):
    def test_beat(self):
        f = summarize(dates_frame(("2026-10-30 16:00", 1.10, 1.23, 11.82)))
        self.assertEqual(f["earnings_status"], "JUST_REPORTED")
        self.assertEqual(f["last_earnings_date"], "2026-10-30")
        self.assertEqual(f["last_eps_estimate"], 1.10)
        self.assertEqual(f["last_eps_actual"], 1.23)
        self.assertEqual(f["last_surprise_pct"], 11.82)
        self.assertEqual(f["last_result"], "Beat")
        self.assertEqual(f["days_since_last"], 4)

    def test_miss(self):
        f = summarize(dates_frame(("2026-10-30 16:00", 1.10, 1.00, -9.09)))
        self.assertEqual(f["last_result"], "Miss")

    def test_in_line(self):
        f = summarize(dates_frame(("2026-10-30 16:00", 1.10, 1.10, 0.4)))
        self.assertEqual(f["last_result"], "In line")

    def test_exactly_one_percent_is_in_line_both_ways(self):
        for surprise in (1.0, -1.0):
            f = summarize(dates_frame(("2026-10-30 16:00", 1.0, 1.01, surprise)))
            self.assertEqual(f["last_result"], "In line", surprise)

    def test_surprise_computed_when_missing(self):
        f = summarize(dates_frame(("2026-10-30 16:00", 1.00, 1.20, np.nan)))
        self.assertEqual(f["last_surprise_pct"], 20.0)
        self.assertEqual(f["last_result"], "Beat")

    def test_computed_surprise_uses_absolute_estimate(self):
        # Losing less than expected is a beat, even with a negative estimate.
        f = summarize(dates_frame(("2026-10-30 16:00", -0.50, -0.25, np.nan)))
        self.assertEqual(f["last_surprise_pct"], 50.0)
        self.assertEqual(f["last_result"], "Beat")

    def test_zero_estimate_has_result_but_no_percent(self):
        f = summarize(dates_frame(("2026-10-30 16:00", 0.0, 0.05, np.nan)))
        self.assertEqual(f["last_result"], "Beat")
        self.assertIsNone(f["last_surprise_pct"])

    def test_reported_earlier_today_counts_as_zero_days(self):
        f = summarize(dates_frame(("2026-11-03 07:00", 1.0, 1.2, 20.0)))
        self.assertEqual(f["earnings_status"], "JUST_REPORTED")
        self.assertEqual(f["days_since_last"], 0)

    def test_recent_window_edges(self):
        frame = dates_frame(("2026-10-27 16:00", 1.0, 1.2, 20.0))  # 7 days ago
        self.assertEqual(summarize(frame, recent=7)["earnings_status"], "JUST_REPORTED")
        self.assertEqual(summarize(frame, recent=6)["earnings_status"], "NONE")

    def test_most_recent_report_wins(self):
        f = summarize(dates_frame(
            ("2026-07-30 16:00", 1.0, 0.5, -50.0),
            ("2026-10-30 16:00", 1.0, 1.5, 50.0),
            ("2026-04-30 16:00", 1.0, 0.5, -50.0),
        ))
        self.assertEqual(f["last_earnings_date"], "2026-10-30")

    def test_future_row_with_reported_eps_is_neither(self):
        f = summarize(dates_frame(("2026-12-01 16:00", 1.0, 1.2, 20.0)))
        self.assertIsNone(f["last_earnings_date"])
        self.assertIsNone(f["next_earnings_date"])


class Upcoming(unittest.TestCase):
    def test_inside_window(self):
        f = summarize(dates_frame(("2026-11-12 16:00", 1.5, np.nan, np.nan)))
        self.assertEqual(f["earnings_status"], "UPCOMING")
        self.assertEqual(f["next_earnings_date"], "2026-11-12")
        self.assertEqual(f["next_earnings_timing"], "after close")
        self.assertEqual(f["next_eps_estimate"], 1.5)
        self.assertEqual(f["days_until_next"], 9)

    def test_outside_window(self):
        f = summarize(dates_frame(("2027-01-28 16:00", 1.5, np.nan, np.nan)))
        self.assertEqual(f["earnings_status"], "NONE")
        self.assertEqual(f["days_until_next"], 86)

    def test_soon_window_edges(self):
        frame = dates_frame(("2026-11-17 08:00", 1.5, np.nan, np.nan))  # 14 days out
        self.assertEqual(summarize(frame, soon=14)["earnings_status"], "UPCOMING")
        self.assertEqual(summarize(frame, soon=13)["earnings_status"], "NONE")

    def test_earliest_future_row_wins(self):
        f = summarize(dates_frame(
            ("2027-04-28 16:00", 2.0, np.nan, np.nan),
            ("2026-11-12 16:00", 1.0, np.nan, np.nan),
            ("2027-01-28 16:00", 3.0, np.nan, np.nan),
        ))
        self.assertEqual(f["next_earnings_date"], "2026-11-12")

    def test_dated_today_not_yet_reported_is_upcoming_today(self):
        f = summarize(dates_frame(("2026-11-03 16:00", 1.5, np.nan, np.nan)))
        self.assertEqual(f["earnings_status"], "UPCOMING")
        self.assertEqual(f["days_until_next"], 0)
        self.assertIsNone(f["last_earnings_date"])

    def test_past_row_without_eps_is_ignored(self):
        f = summarize(dates_frame(("2026-08-08 16:00", 5.0, np.nan, np.nan)))
        self.assertEqual(f["earnings_status"], "NONE")
        self.assertIsNone(f["last_earnings_date"])
        self.assertIsNone(f["next_earnings_date"])

    def test_blank_duplicate_on_a_reported_day_is_not_upcoming(self):
        # BRK-B really has this shape: a reported row and a blank one, same day.
        f = summarize(dates_frame(
            ("2026-11-03 16:00", 5.13, np.nan, np.nan),
            ("2026-11-03 08:00", 5.13, 6.03, 17.40),
        ))
        self.assertEqual(f["earnings_status"], "JUST_REPORTED")
        self.assertIsNone(f["next_earnings_date"])

    def test_timing_labels(self):
        cases = {
            "07:00": "before open",
            "11:59": "before open",
            "12:00": "time unknown",
            "15:59": "time unknown",
            "16:00": "after close",
            "21:00": "after close",
            "00:00": "time unknown",
            "00:30": "before open",
        }
        for clock, expected in cases.items():
            f = summarize(dates_frame((f"2026-11-12 {clock}", 1.0, np.nan, np.nan)))
            self.assertEqual(f["next_earnings_timing"], expected, clock)


class BothStatuses(unittest.TestCase):
    def test_both_at_once(self):
        f = summarize(dates_frame(
            ("2026-11-12 16:00", 1.5, np.nan, np.nan),
            ("2026-10-30 16:00", 1.0, 1.2, 20.0),
        ))
        self.assertEqual(f["earnings_status"], "JUST_REPORTED+UPCOMING")
        self.assertEqual(f["days_since_last"], 4)
        self.assertEqual(f["days_until_next"], 9)


class MissingValues(unittest.TestCase):
    def test_nan_estimate_is_no_estimate(self):
        f = summarize(dates_frame(("2026-10-30 16:00", np.nan, 1.2, np.nan)))
        self.assertEqual(f["last_result"], "No estimate")
        self.assertIsNone(f["last_eps_estimate"])
        self.assertEqual(f["last_eps_actual"], 1.2)

    def test_dash_strings_are_coerced(self):
        f = summarize(dates_frame(
            ("2026-11-12 16:00", "1.5", "-", "-"),
            ("2026-10-30 16:00", "-", "1.2", "-"),
        ))
        self.assertEqual(f["last_result"], "No estimate")
        self.assertEqual(f["next_eps_estimate"], 1.5)
        self.assertEqual(f["next_earnings_date"], "2026-11-12")

    def test_dash_reported_eps_is_not_a_report(self):
        f = summarize(dates_frame(("2026-10-30 16:00", 1.0, "-", "-")))
        self.assertIsNone(f["last_earnings_date"])

    def test_no_data_at_all(self):
        empty = dates_frame(("2026-10-30 16:00", 1, 1, 1)).iloc[0:0]
        for frame in (None, empty):
            f = summarize(frame)
            self.assertEqual(f["earnings_status"], "NONE")
            self.assertIsNone(f["last_earnings_date"])
            self.assertIsNone(f["next_earnings_date"])

    def test_missing_columns_do_not_raise(self):
        index = pd.DatetimeIndex(["2026-11-12 16:00"]).tz_localize(NY)
        f = summarize(pd.DataFrame({"Other": [1]}, index=index))
        self.assertEqual(f["next_earnings_date"], "2026-11-12")
        self.assertIsNone(f["next_eps_estimate"])


class Timezones(unittest.TestCase):
    def test_new_york_index(self):
        f = summarize(dates_frame(("2026-11-12 16:00", 1.0, np.nan, np.nan)))
        self.assertEqual(f["next_earnings_timing"], "after close")

    def test_utc_index_is_converted_before_taking_the_date(self):
        # 01:00 UTC Nov 4 is 8pm Nov 3 in New York: today, after the close.
        f = summarize(dates_frame(("2026-11-04 01:00", 1.0, np.nan, np.nan), tz="UTC"))
        self.assertEqual(f["next_earnings_date"], "2026-11-03")
        self.assertEqual(f["next_earnings_timing"], "after close")
        self.assertEqual(f["days_until_next"], 0)

    def test_naive_index_is_taken_as_new_york(self):
        f = summarize(dates_frame(("2026-11-12 07:00", 1.0, np.nan, np.nan), tz=None))
        self.assertEqual(f["next_earnings_timing"], "before open")

    def test_mixed_offsets_in_an_object_index(self):
        # Yahoo's rows straddle daylight saving, so offsets can differ.
        index = pd.Index(
            [
                pd.Timestamp("2026-10-30 16:00", tz="UTC-04:00"),
                pd.Timestamp("2027-01-28 16:00", tz="UTC-05:00"),
            ],
            dtype=object,
        )
        frame = pd.DataFrame(
            {"EPS Estimate": [1.0, 1.0], "Reported EPS": [1.2, np.nan], "Surprise(%)": [20.0, np.nan]},
            index=index,
        )
        f = summarize(frame)
        self.assertEqual(f["last_earnings_date"], "2026-10-30")
        self.assertEqual(f["next_earnings_date"], "2027-01-28")
        self.assertEqual(f["next_earnings_timing"], "after close")


class CalendarFallback(unittest.TestCase):
    def test_one_date_is_confirmed(self):
        f = summarize(None, {"Earnings Date": [dt.date(2026, 11, 12)], "Earnings Average": 1.23456})
        self.assertEqual(f["earnings_status"], "UPCOMING")
        self.assertEqual(f["next_earnings_date"], "2026-11-12")
        self.assertEqual(f["next_earnings_timing"], "time unknown")
        self.assertEqual(f["next_eps_estimate"], 1.23)
        self.assertEqual(f["days_until_next"], 9)

    def test_two_dates_are_an_estimated_window(self):
        f = summarize(None, {"Earnings Date": [dt.date(2026, 11, 12), dt.date(2026, 11, 17)]})
        self.assertEqual(f["next_earnings_date"], "2026-11-12")
        self.assertEqual(f["next_earnings_timing"], "Nov 12-17 est.")
        self.assertEqual(f["days_until_next"], 9)

    def test_window_across_a_month(self):
        f = summarize(None, {"Earnings Date": [dt.date(2026, 11, 28), dt.date(2026, 12, 3)]})
        self.assertEqual(f["next_earnings_timing"], "Nov 28-Dec 3 est.")

    def test_used_when_dates_have_no_future_row(self):
        frame = dates_frame(("2026-10-30 16:00", 1.0, 1.2, 20.0))
        f = summarize(frame, {"Earnings Date": [dt.date(2027, 1, 28)], "Earnings Average": 2.0})
        self.assertEqual(f["last_earnings_date"], "2026-10-30")
        self.assertEqual(f["next_earnings_date"], "2027-01-28")
        self.assertEqual(f["next_eps_estimate"], 2.0)

    def test_primary_wins_over_calendar(self):
        frame = dates_frame(("2026-11-12 16:00", 1.0, np.nan, np.nan))
        f = summarize(frame, {"Earnings Date": [dt.date(2026, 12, 1)], "Earnings Average": 9.0})
        self.assertEqual(f["next_earnings_date"], "2026-11-12")
        self.assertEqual(f["next_eps_estimate"], 1.0)

    def test_fills_missing_estimate_only_for_the_same_report(self):
        frame = dates_frame(("2026-11-12 16:00", np.nan, np.nan, np.nan))
        same = {"Earnings Date": [dt.date(2026, 11, 12)], "Earnings Average": 2.0}
        other = {"Earnings Date": [dt.date(2027, 2, 1)], "Earnings Average": 2.0}
        self.assertEqual(summarize(frame, same)["next_eps_estimate"], 2.0)
        self.assertIsNone(summarize(frame, other)["next_eps_estimate"])

    def test_stale_dates_are_ignored(self):
        passed = {"Earnings Date": [dt.date(2026, 10, 1)]}
        self.assertIsNone(summarize(None, passed)["next_earnings_date"])

    def test_calendar_not_rolled_forward_after_a_report(self):
        # The window contains a report that already happened.
        frame = dates_frame(("2026-11-02 16:00", 1.0, 1.2, 20.0))
        cal = {"Earnings Date": [dt.date(2026, 11, 1), dt.date(2026, 11, 5)]}
        self.assertIsNone(summarize(frame, cal)["next_earnings_date"])

    def test_open_window_counts_as_zero_days(self):
        cal = {"Earnings Date": [dt.date(2026, 11, 1), dt.date(2026, 11, 5)]}
        f = summarize(None, cal)
        self.assertEqual(f["days_until_next"], 0)
        self.assertEqual(f["earnings_status"], "UPCOMING")

    def test_bad_calendars_do_not_raise(self):
        for cal in ({}, {"Earnings Date": []}, {"Earnings Date": None},
                    {"Earnings Date": ["nonsense"]}, "not a dict", None):
            f = summarize(None, cal)
            self.assertEqual(f["earnings_status"], "NONE", cal)


def income_stmt(revenues: dict):
    return pd.DataFrame(
        [list(revenues.values())],
        index=["Total Revenue"],
        columns=pd.to_datetime(list(revenues)),
    )


GD_STMT = income_stmt({
    "2026-06-30": 14094e6, "2026-03-31": 13481e6, "2025-12-31": 14379e6,
    "2025-09-30": 12907e6, "2025-06-30": 13041e6, "2025-03-31": np.nan,
})
GD_MODULE = {
    "financialCurrency": "USD",
    "financialsChart": {"quarterly": [
        {"date": "1Q2026", "revenue": 13481000000},
        {"date": "2Q2026", "revenue": 14094000000},
        {"date": "3Q2026", "revenue": 14566000000},
    ]},
}


class Revenue(unittest.TestCase):
    def test_statement_covers_the_last_report(self):
        stmt = income_stmt({"2026-09-30": 4106e6, "2025-09-30": 3407e6})
        cal = {"Earnings Date": [dt.date(2027, 1, 27)], "Revenue Average": 4370528970}
        f = summarize_revenue(stmt, cal, None, dt.date(2026, 10, 28))
        self.assertEqual(f["last_revenue"], 4106000000)
        self.assertEqual(f["last_revenue_period_end"], "2026-09-30")
        self.assertEqual(f["last_revenue_yoy_pct"], 20.52)
        self.assertIs(f["last_revenue_current"], True)
        self.assertEqual(f["next_revenue_estimate"], 4370528970)

    def test_chart_supplies_the_quarter_the_statement_lacks(self):
        # GD really had this on Nov 3: reported Oct 30, statement stops at June.
        f = summarize_revenue(GD_STMT, None, GD_MODULE, dt.date(2026, 10, 30))
        self.assertEqual(f["last_revenue"], 14566000000)
        self.assertEqual(f["last_revenue_period_end"], "2026-09-30")
        self.assertEqual(f["last_revenue_yoy_pct"], 12.85)
        self.assertIs(f["last_revenue_current"], True)
        self.assertEqual(f["revenue_currency"], "USD")

    def test_chart_bridges_two_blank_statement_quarters(self):
        # ABT really has this: June and September blank, chart ties to March.
        stmt = income_stmt({
            "2026-09-30": np.nan, "2026-06-30": np.nan, "2026-03-31": 11164e6,
            "2025-12-31": 11459e6, "2025-09-30": 11369e6, "2025-06-30": 11142e6,
        })
        module = {"financialsChart": {"quarterly": [
            {"revenue": 11459000000}, {"revenue": 11164000000},
            {"revenue": 12593000000}, {"revenue": 13013000000},
        ]}}
        f = summarize_revenue(stmt, None, module, dt.date(2026, 10, 21))
        self.assertEqual(f["last_revenue"], 13013000000)
        self.assertEqual(f["last_revenue_period_end"], "2026-09-30")
        self.assertEqual(f["last_revenue_yoy_pct"], 14.46)
        self.assertIs(f["last_revenue_current"], True)

    def test_chart_no_newer_than_statement_changes_nothing(self):
        stmt = income_stmt({"2026-06-30": 100e6, "2026-03-31": 90e6})
        module = {"financialsChart": {"quarterly": [{"revenue": 90000000}, {"revenue": 100000000}]}}
        f = summarize_revenue(stmt, None, module, dt.date(2026, 10, 30))
        self.assertEqual(f["last_revenue_period_end"], "2026-06-30")
        self.assertIs(f["last_revenue_current"], False)

    def test_revenue_estimate_dropped_when_calendar_not_rolled_forward(self):
        # VRTX: reported Nov 2 and the calendar still says Nov 2.
        stale = {"Earnings Date": [dt.date(2026, 11, 2)], "Revenue Average": 3000000000}
        fresh = {"Earnings Date": [dt.date(2027, 2, 1)], "Revenue Average": 3000000000}
        last = dt.date(2026, 11, 2)
        self.assertIsNone(summarize_revenue(None, stale, None, last)["next_revenue_estimate"])
        self.assertEqual(summarize_revenue(None, fresh, None, last)["next_revenue_estimate"], 3000000000)

    def test_stale_statement_without_chart_is_flagged(self):
        f = summarize_revenue(GD_STMT, None, None, dt.date(2026, 10, 30))
        self.assertEqual(f["last_revenue"], 14094000000)
        self.assertIs(f["last_revenue_current"], False)

    def test_chart_with_another_revenue_definition_is_not_trusted(self):
        # SPG and BRK-B: the chart's figures don't line up with the statement.
        module = {"financialsChart": {"quarterly": [
            {"revenue": 1500000000}, {"revenue": 1659709000},
        ]}}
        f = summarize_revenue(GD_STMT, None, module, dt.date(2026, 10, 30))
        self.assertEqual(f["last_revenue"], 14094000000)
        self.assertIs(f["last_revenue_current"], False)

    def test_chart_ignored_when_statement_is_current(self):
        f = summarize_revenue(GD_STMT, None, GD_MODULE, dt.date(2026, 8, 1))
        self.assertEqual(f["last_revenue"], 14094000000)
        self.assertIs(f["last_revenue_current"], True)

    def test_nan_quarters_are_skipped(self):
        stmt = income_stmt({"2026-09-30": np.nan, "2026-06-30": 100e6})
        f = summarize_revenue(stmt, None, None, None)
        self.assertEqual(f["last_revenue"], 100000000)
        self.assertEqual(f["last_revenue_period_end"], "2026-06-30")
        self.assertIsNone(f["last_revenue_current"])

    def test_no_year_ago_quarter_means_no_growth(self):
        stmt = income_stmt({"2026-06-30": 100e6, "2026-03-31": 90e6})
        self.assertIsNone(summarize_revenue(stmt, None, None, None)["last_revenue_yoy_pct"])

    def test_year_ago_quarter_allows_for_fiscal_drift(self):
        stmt = income_stmt({"2026-09-27": 110e6, "2025-09-28": 100e6})
        self.assertEqual(summarize_revenue(stmt, None, None, None)["last_revenue_yoy_pct"], 10.0)

    def test_non_usd_reporter_keeps_its_currency(self):
        f = summarize_revenue(None, None, {"financialCurrency": "EUR"}, None)
        self.assertEqual(f["revenue_currency"], "EUR")

    def test_missing_data_does_not_raise(self):
        for stmt in (None, pd.DataFrame(), pd.DataFrame({"x": [1]}, index=["Net Income"])):
            f = summarize_revenue(stmt, {}, {}, None)
            self.assertIsNone(f["last_revenue"])

    def test_zero_revenue_estimate_is_treated_as_missing(self):
        cal = {"Earnings Date": [dt.date(2027, 1, 27)], "Revenue Average": 0}
        self.assertIsNone(summarize_revenue(None, cal, None, None)["next_revenue_estimate"])


def row(**fields):
    base = dict.fromkeys(ALL_FIELDS)
    base.update(fields)
    return base


REPORTED = dict(
    last_earnings_date="2026-10-30", last_eps_estimate=1.10, last_eps_actual=1.23,
    last_surprise_pct=11.82, last_result="Beat", days_since_last=4,
)
NEXT = dict(
    next_earnings_date="2026-11-12", next_earnings_timing="after close",
    next_eps_estimate=1.5, days_until_next=9,
)


def plain(cell_html: str) -> str:
    return re.sub(r"<[^>]+>", "", cell_html.replace("<br>", "\n"))


class EarningsCell(unittest.TestCase):
    def test_just_reported_text_and_tag(self):
        cell = _fmt_earnings_cell(row(earnings_status="JUST_REPORTED", **REPORTED))
        self.assertEqual(plain(cell), "Reported Oct 30 (4d ago) · EPS 1.23 vs 1.10 · Beat +11.8%")
        self.assertIn('class="tag earn-beat"', cell)

    def test_result_tag_colours(self):
        for result, css in (("Miss", "earn-miss"), ("In line", "earn-inline")):
            fields = {**REPORTED, "last_result": result}
            cell = _fmt_earnings_cell(row(earnings_status="JUST_REPORTED", **fields))
            self.assertIn(f'class="tag {css}"', cell)

    def test_upcoming_text_and_tag(self):
        cell = _fmt_earnings_cell(row(earnings_status="UPCOMING", **NEXT))
        self.assertEqual(plain(cell), "Earnings Nov 12, after close (in 9d)")
        self.assertIn('class="tag earn-soon"', cell)

    def test_upcoming_today(self):
        fields = {**NEXT, "next_earnings_date": "2026-11-03", "days_until_next": 0}
        cell = _fmt_earnings_cell(row(earnings_status="UPCOMING", **fields))
        self.assertEqual(plain(cell), "Earnings today, after close")

    def test_upcoming_window(self):
        fields = {**NEXT, "next_earnings_timing": "Nov 12-17 est."}
        cell = _fmt_earnings_cell(row(earnings_status="UPCOMING", **fields))
        self.assertEqual(plain(cell), "Earnings Nov 12-17 est. (in 9d)")

    def test_otherwise_a_muted_next_date(self):
        fields = {**NEXT, "next_earnings_date": "2027-01-28", "days_until_next": 86}
        cell = _fmt_earnings_cell(row(earnings_status="NONE", **fields))
        self.assertEqual(plain(cell), "Next Jan 28 (86d)")
        self.assertIn('class="muted"', cell)

    def test_both_on_two_lines(self):
        cell = _fmt_earnings_cell(row(earnings_status="JUST_REPORTED+UPCOMING", **REPORTED, **NEXT))
        self.assertEqual(
            plain(cell).split("\n"),
            ["Reported Oct 30 (4d ago) · EPS 1.23 vs 1.10 · Beat +11.8%",
             "Earnings Nov 12, after close (in 9d)"],
        )

    def test_missing_data_is_a_dash(self):
        self.assertEqual(plain(_fmt_earnings_cell(row())), "—")
        self.assertEqual(plain(_fmt_earnings_cell(row(earnings_status="NONE"))), "—")

    def test_tooltip_has_the_details(self):
        tip = _earnings_tooltip(row(
            earnings_status="JUST_REPORTED+UPCOMING", last_revenue=14566000000,
            revenue_currency="USD", last_revenue_period_end="2026-09-30",
            last_revenue_yoy_pct=12.85, last_revenue_current=True,
            next_revenue_estimate=14700409000, **REPORTED, **NEXT,
        ))
        for expected in ("2026-10-30", "surprise +11.82%", "Revenue $14.57B", "+12.85% YoY",
                         "2026-11-12, after close", "EPS estimate 1.50",
                         "revenue estimate $14.70B", "JUST_REPORTED+UPCOMING"):
            self.assertIn(expected, tip)
        self.assertNotIn("Yahoo hasn't updated", tip)

    def test_tooltip_flags_stale_revenue(self):
        tip = _earnings_tooltip(row(
            earnings_status="NONE", last_revenue=100, last_revenue_period_end="2026-06-30",
            last_revenue_current=False,
        ))
        self.assertIn("Yahoo hasn't updated", tip)

    def test_revenue_format(self):
        self.assertEqual(_fmt_revenue(4106000000, "USD"), "$4.11B")
        self.assertEqual(_fmt_revenue(11662000000, "EUR"), "EUR 11.66B")
        self.assertEqual(_fmt_revenue(1659709000, None), "1.66B")
        self.assertEqual(_fmt_revenue(950000000, "USD"), "$950.00M")
        self.assertEqual(_fmt_revenue(None, "USD"), "—")


def results_frame():
    return pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "CCC", "DDD"],
            "signal": ["STRONG_SHORT", "WATCH_LONG", "NEUTRAL", "WATCH_SHORT"],
            "trigger": ["NONE", "NONE", "NONE", "NONE"],
            "side": ["SHORT", "LONG", "", "SHORT"],
            "score": [4, 3, 0, 1],
            "close": [10.0, 20.0, 30.0, 40.0],
            "pct_change_1d": [1.0, 1.0, 1.0, 1.0],
            "rsi": [75.0, 25.0, 50.0, 70.0],
            "percent_b": [1.1, -0.1, 0.5, 0.9],
            "trend": ["Uptrend"] * 4,
            "pct_from_sma_50": [1.0, 1.0, 1.0, 1.0],
            "volume_ratio": [1.2, 1.2, 1.2, 1.2],
            "vol_signal": ["SELLING", "BUYING", "NONE", "NONE"],
            "vp_node": ["POC", "HVN", "NONE", "NONE"],
            "vp_node_price": [10.0, 20.0, None, None],
            "poc": [10.0, 20.0, 30.0, 40.0],
            "poc_dist_pct": [0.1, 0.1, 0.1, 0.1],
        }
    )


class Merging(unittest.TestCase):
    def test_candidates_need_a_side_and_the_score(self):
        self.assertEqual(candidate_tickers(results_frame(), 3), ["AAA", "BBB"])
        self.assertEqual(candidate_tickers(results_frame(), 4), ["AAA"])
        # Even at --min-score 0, a row with no side isn't a candidate.
        self.assertEqual(candidate_tickers(results_frame(), 0), ["AAA", "BBB", "DDD"])

    def test_existing_columns_and_order_are_untouched(self):
        before = results_frame()
        after = merge_earnings(before, {"AAA": row(earnings_status="UPCOMING", days_until_next=9)})
        pd.testing.assert_frame_equal(after[before.columns.tolist()], before)
        self.assertEqual(after.columns.tolist(), before.columns.tolist() + ALL_FIELDS)

    def test_non_candidates_stay_blank_and_day_counts_stay_whole(self):
        after = merge_earnings(results_frame(), {"AAA": row(earnings_status="UPCOMING", days_until_next=9)})
        self.assertIs(after.loc[0, "days_until_next"], 9)
        self.assertIsNone(after.loc[2, "earnings_status"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.csv"
            after.to_csv(path, index=False)
            lines = path.read_text().splitlines()
        columns = lines[0].split(",")
        first = lines[1].split(",")
        self.assertEqual(first[columns.index("days_until_next")], "9")
        third = lines[3].split(",")
        self.assertEqual(third[columns.index("earnings_status")], "")


class Dashboard(unittest.TestCase):
    def render(self, df, **kwargs):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dash.html"
            write_html_dashboard(
                df, str(path), universe_label="Test", min_price=5.0, rsi_period=14,
                rsi_overbought=70, rsi_oversold=30, bb_period=20, bb_std=2.0, **kwargs,
            )
            return path.read_text(encoding="utf-8")

    def test_column_sits_between_profile_and_trigger(self):
        df = merge_earnings(results_frame(), {"AAA": row(earnings_status="UPCOMING", **NEXT)})
        page = self.render(df)
        self.assertRegex(page, r"<th>Profile</th>\s*<th data-numeric=\"1\">Earnings</th>\s*<th>Trigger</th>")
        self.assertIn('data-sort="9"', page)
        self.assertIn("Earnings Nov 12, after close (in 9d)", page)

    def test_failed_ticker_has_a_dash_and_blank_sort(self):
        df = merge_earnings(results_frame(), {})
        page = self.render(df)
        self.assertIn('class="earnings-cell" data-sort=""', page)
        self.assertNotIn('data-sort="None"', page)

    def test_empty_table_colspan(self):
        df = merge_earnings(results_frame(), {})
        self.assertIn('colspan="13"', self.render(df, min_score=4))

    def test_without_earnings_columns_the_table_is_unchanged(self):
        page = self.render(results_frame(), min_score=4)
        self.assertNotIn(">Earnings</th>", page)
        self.assertIn('colspan="12"', page)
        self.assertNotIn("earnings-cell", page.split("<tbody>")[1])

    def test_tooltip_text_is_escaped(self):
        fields = row(earnings_status="NONE", last_revenue=100, revenue_currency='"><b>',
                     last_revenue_period_end="2026-06-30")
        page = self.render(merge_earnings(results_frame(), {"AAA": fields}))
        self.assertNotIn('"><b>', page)


class FetchBehaviour(unittest.TestCase):
    """fetch_earnings with yfinance swapped out, so still no network."""

    def fake_ticker(self, dates=None, calendar=None, dates_error=None):
        handle = mock.MagicMock()
        if dates_error:
            handle.get_earnings_dates.side_effect = dates_error
        else:
            handle.get_earnings_dates.return_value = dates
        handle.calendar = calendar if calendar is not None else {}
        handle.quarterly_income_stmt = None
        handle._quote._fetch.return_value = None
        return handle

    def setUp(self):
        for target, value in ((earnings.time, "sleep"),):
            patcher = mock.patch.object(target, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_retries_once_after_an_error_then_succeeds(self):
        good = dates_frame(("2026-10-30 16:00", 1.0, 1.2, 20.0))
        bad = self.fake_ticker(dates_error=ConnectionError("boom"))
        with mock.patch.object(earnings.yf, "Ticker", side_effect=[bad, self.fake_ticker(good)]) as ticker:
            out = fetch_earnings(["AAA"])
        self.assertEqual(ticker.call_count, 2)
        self.assertEqual(out["AAA"]["last_earnings_date"], "2026-10-30")

    def test_gives_up_after_one_retry_with_blank_fields(self):
        bad = self.fake_ticker(dates_error=ConnectionError("boom"))
        with mock.patch.object(earnings.yf, "Ticker", return_value=bad) as ticker:
            out = fetch_earnings(["AAA"])
        self.assertEqual(ticker.call_count, 2)
        self.assertEqual(out["AAA"], dict.fromkeys(ALL_FIELDS))

    def test_empty_answer_is_blank_without_a_pointless_retry(self):
        with mock.patch.object(earnings.yf, "Ticker", return_value=self.fake_ticker()) as ticker:
            out = fetch_earnings(["AAA"])
        self.assertEqual(ticker.call_count, 1)
        self.assertIsNone(out["AAA"]["earnings_status"])

    def test_revenue_failure_alone_does_not_retry_or_fail_the_ticker(self):
        good = self.fake_ticker(dates_frame(("2026-10-30 16:00", 1.0, 1.2, 20.0)))
        good._quote._fetch.side_effect = ConnectionError("boom")
        with mock.patch.object(earnings.yf, "Ticker", return_value=good) as ticker:
            out = fetch_earnings(["AAA"])
        self.assertEqual(ticker.call_count, 1)
        self.assertEqual(out["AAA"]["last_earnings_date"], "2026-10-30")

    def test_retry_keeps_what_the_first_pass_got(self):
        day = pd.Timestamp.now(tz=NY).date() + dt.timedelta(days=5)
        first = self.fake_ticker(dates_error=ConnectionError("boom"), calendar={"Earnings Date": [day]})
        second = self.fake_ticker(dates_error=ConnectionError("boom"))
        with mock.patch.object(earnings.yf, "Ticker", side_effect=[first, second]):
            out = fetch_earnings(["AAA"])
        self.assertEqual(out["AAA"]["days_until_next"], 5)

    def test_falls_back_to_calendar_when_dates_keep_failing(self):
        bad = self.fake_ticker(
            dates_error=ConnectionError("boom"),
            calendar={"Earnings Date": [pd.Timestamp.now(tz=NY).date() + dt.timedelta(days=5)]},
        )
        with mock.patch.object(earnings.yf, "Ticker", return_value=bad):
            out = fetch_earnings(["AAA"])
        self.assertEqual(out["AAA"]["days_until_next"], 5)
        self.assertEqual(out["AAA"]["earnings_status"], "UPCOMING")

    def test_one_failure_does_not_affect_the_others(self):
        good = self.fake_ticker(dates_frame(("2026-10-30 16:00", 1.0, 1.2, 20.0)))
        bad = self.fake_ticker(dates_error=ConnectionError("boom"))
        handles = {"AAA": good, "BBB": bad, "CCC": good}
        with mock.patch.object(earnings.yf, "Ticker", side_effect=lambda t: handles[t]):
            out = fetch_earnings(["AAA", "BBB", "CCC"])
        self.assertEqual(out["AAA"]["last_earnings_date"], "2026-10-30")
        self.assertIsNone(out["BBB"]["earnings_status"])
        self.assertEqual(out["CCC"]["last_earnings_date"], "2026-10-30")

    def test_no_tickers_makes_no_requests(self):
        with mock.patch.object(earnings.yf, "Ticker") as ticker:
            self.assertEqual(fetch_earnings([]), {})
        ticker.assert_not_called()

    def test_add_earnings_survives_a_total_failure(self):
        with mock.patch.object(earnings, "fetch_earnings", side_effect=RuntimeError("boom")):
            out = earnings.add_earnings(results_frame(), 3)
        self.assertIn("earnings_status", out.columns)
        self.assertTrue(out["earnings_status"].isna().all())


if __name__ == "__main__":
    unittest.main()
