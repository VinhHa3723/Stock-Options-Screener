"""
earnings.py
-----------
Looks up earnings dates, EPS results and revenue for the candidates that reach
the dashboard. This is information only: nothing here touches the score, the
sort order, or any existing column.

The summarize_* functions are pure (no network calls), so they can be tested
with synthetic data. fetch_earnings() is the only part that talks to Yahoo.
"""

from __future__ import annotations
import datetime as dt
import math
import sys
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeoutError
import pandas as pd
import yfinance as yf

MARKET_TZ = "America/New_York"

MAX_WORKERS = 4
RETRY_DELAY_SECONDS = 2
# Hard ceiling on the whole earnings step. Whatever hasn't finished by then
# gets blank fields, so a stalled Yahoo can't hold up the scan's output.
FETCH_BUDGET_SECONDS = 180

# A surprise beyond this many percent either way is a Beat/Miss, not In line.
RESULT_THRESHOLD_PCT = 1.0

# Companies report within about 90 days of a quarter ending. A newest revenue
# quarter that ended longer than this before the last report must be the
# quarter before the one that report covered.
MAX_DAYS_QUARTER_END_TO_REPORT = 100
# How far a statement date can drift from exactly one year earlier and still
# be the year-ago quarter (fiscal calendars move by a few days).
YEAR_AGO_TOLERANCE_DAYS = 20

EARNINGS_FIELDS = [
    "earnings_status",
    "last_earnings_date",
    "last_eps_estimate",
    "last_eps_actual",
    "last_surprise_pct",
    "last_result",
    "days_since_last",
    "next_earnings_date",
    "next_earnings_timing",
    "next_eps_estimate",
    "days_until_next",
]

REVENUE_FIELDS = [
    "last_revenue",
    "revenue_currency",
    "last_revenue_period_end",
    "last_revenue_yoy_pct",
    "last_revenue_current",
    "next_revenue_estimate",
]

ALL_FIELDS = EARNINGS_FIELDS + REVENUE_FIELDS


def _num(value) -> float | None:
    """A finite float, or None for anything missing or non-numeric."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _rounded(value: float | None) -> float | None:
    return None if value is None else round(value, 2)


def short_date(day: dt.date) -> str:
    """'Nov 3', without the zero padding strftime would add."""
    return f"{day:%b} {day.day}"


def _window_label(start: dt.date, end: dt.date) -> str:
    """'Nov 12-17 est.', or 'Nov 28-Dec 3 est.' across a month boundary."""
    end_text = str(end.day) if (start.year, start.month) == (end.year, end.month) else short_date(end)
    return f"{short_date(start)}-{end_text} est."


def _timing_label(when: pd.Timestamp) -> str:
    if when.hour == 0 and when.minute == 0 and when.second == 0:
        return "time unknown"  # Yahoo uses midnight when it has no time
    if when.hour < 12:
        return "before open"
    if when.hour >= 16:
        return "after close"
    return "time unknown"


def _to_market_time(index) -> pd.DatetimeIndex:
    """Any date index as America/New_York. Timezone-free dates are taken to
    already be New York time."""
    if not isinstance(index, pd.DatetimeIndex):
        # Without utc=True, coercing mixed UTC offsets silently blanks rows.
        has_tz = any(getattr(value, "tzinfo", None) is not None for value in index)
        index = pd.to_datetime(index, errors="coerce", utc=has_tz)
    if index.tz is None:
        return index.tz_localize(MARKET_TZ, ambiguous="NaT", nonexistent="NaT")
    return index.tz_convert(MARKET_TZ)


def _earnings_rows(earnings_dates) -> list[dict]:
    """get_earnings_dates() output as plain dicts, with '-' and NaN as None."""
    if not isinstance(earnings_dates, pd.DataFrame) or earnings_dates.empty:
        return []

    def column(name: str) -> list:
        if name not in earnings_dates.columns:
            return [None] * len(earnings_dates)
        return pd.to_numeric(earnings_dates[name], errors="coerce").tolist()

    rows = []
    for when, estimate, actual, surprise in zip(
        _to_market_time(earnings_dates.index),
        column("EPS Estimate"),
        column("Reported EPS"),
        column("Surprise(%)"),
    ):
        if pd.isna(when):
            continue
        rows.append(
            {
                "when": when,
                "date": when.date(),
                "estimate": _num(estimate),
                "actual": _num(actual),
                "surprise": _num(surprise),
            }
        )
    return rows


def _classify_result(
    estimate: float | None, actual: float, surprise: float | None
) -> tuple[str, float | None]:
    """Returns (Beat / Miss / In line / No estimate, surprise in percent)."""
    if estimate is None:
        return "No estimate", surprise
    if surprise is None and estimate != 0:
        surprise = (actual - estimate) / abs(estimate) * 100
    # A zero estimate has no percent surprise, so compare the two EPS directly.
    gauge = surprise if surprise is not None else actual
    threshold = RESULT_THRESHOLD_PCT if surprise is not None else 0.0
    if gauge > threshold:
        return "Beat", surprise
    if gauge < -threshold:
        return "Miss", surprise
    return "In line", surprise


def _calendar_window(
    calendar, today: dt.date, last_date: dt.date | None
) -> tuple[dt.date, dt.date] | None:
    """The (first, last) next-earnings date from Ticker.calendar, or None if
    it's missing or stale. One date means start == end (confirmed); two dates
    are an estimated window."""
    if not isinstance(calendar, Mapping):
        return None
    raw = calendar.get("Earnings Date")
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)):
        raw = [raw]

    days = set()
    for value in raw:
        try:
            stamp = pd.Timestamp(value)
        except (ValueError, TypeError):
            continue
        if not pd.isna(stamp):
            days.add(stamp.date())
    if not days:
        return None

    start, end = min(days), max(days)
    if end < today:
        return None  # already passed
    if last_date is not None and start <= last_date:
        return None  # that report already happened; Yahoo hasn't rolled forward
    return start, end


def summarize_earnings(
    earnings_dates: pd.DataFrame | None,
    calendar: dict | None,
    today: dt.date,
    recent_days: int = 7,
    soon_days: int = 14,
) -> dict:
    """Turn raw Yahoo earnings data into the EARNINGS_FIELDS dict. `today`
    should be the date in America/New_York. No network calls.

    earnings_status is JUST_REPORTED, UPCOMING, JUST_REPORTED+UPCOMING or NONE.
    For an estimated window, next_earnings_date is its first day and
    next_earnings_timing holds the range, e.g. 'Nov 12-17 est.'.
    """
    fields = dict.fromkeys(EARNINGS_FIELDS)
    rows = _earnings_rows(earnings_dates)

    reported = [r for r in rows if r["actual"] is not None and r["date"] <= today]
    last = max(reported, key=lambda r: r["when"], default=None)
    last_date = last["date"] if last else None

    if last:
        result, surprise = _classify_result(last["estimate"], last["actual"], last["surprise"])
        fields["last_earnings_date"] = last_date.isoformat()
        fields["last_eps_estimate"] = _rounded(last["estimate"])
        fields["last_eps_actual"] = _rounded(last["actual"])
        fields["last_surprise_pct"] = _rounded(surprise)
        fields["last_result"] = result
        fields["days_since_last"] = (today - last_date).days

    # Yahoo sometimes lists a blank row beside the reported one for the same
    # day (BRK-B does), which isn't a second report still to come.
    upcoming = [
        r
        for r in rows
        if r["actual"] is None and r["date"] >= today and r["date"] != last_date
    ]
    next_row = min(upcoming, key=lambda r: r["when"], default=None)
    window = _calendar_window(calendar, today, last_date)
    calendar_estimate = _num(calendar.get("Earnings Average")) if window else None

    if next_row:
        next_date = next_row["date"]
        estimate = next_row["estimate"]
        if estimate is None and window and window[0] <= next_date <= window[1]:
            estimate = calendar_estimate
        fields["next_earnings_date"] = next_date.isoformat()
        fields["next_earnings_timing"] = _timing_label(next_row["when"])
        fields["next_eps_estimate"] = _rounded(estimate)
        fields["days_until_next"] = (next_date - today).days
    elif window:
        start, end = window
        fields["next_earnings_date"] = start.isoformat()
        fields["next_earnings_timing"] = (
            "time unknown" if start == end else _window_label(start, end)
        )
        fields["next_eps_estimate"] = _rounded(calendar_estimate)
        # A window can already be open, in which case it could be any day now.
        fields["days_until_next"] = max((start - today).days, 0)

    statuses = []
    if fields["days_since_last"] is not None and fields["days_since_last"] <= recent_days:
        statuses.append("JUST_REPORTED")
    if fields["days_until_next"] is not None and fields["days_until_next"] <= soon_days:
        statuses.append("UPCOMING")
    fields["earnings_status"] = "+".join(statuses) if statuses else "NONE"
    return fields


def _revenue_by_quarter(income_stmt) -> dict[dt.date, float]:
    """{quarter end: total revenue}, newest first, skipping Yahoo's NaN gaps."""
    if not isinstance(income_stmt, pd.DataFrame) or "Total Revenue" not in income_stmt.index:
        return {}
    quarters = {}
    for period, value in income_stmt.loc["Total Revenue"].items():
        revenue = _num(value)
        stamp = pd.to_datetime(period, errors="coerce")
        if revenue is not None and not pd.isna(stamp):
            quarters[stamp.date()] = revenue
    return dict(sorted(quarters.items(), reverse=True))


def _is_quarter_of_report(period_end: dt.date, report_date: dt.date) -> bool:
    return (report_date - period_end).days <= MAX_DAYS_QUARTER_END_TO_REPORT


def _newer_chart_quarter(
    earnings_module, newest_revenue: float
) -> tuple[float, int] | None:
    """(revenue, quarters ahead of the statement) for the newest quarter in the
    Yahoo 'earnings' chart, which often updates weeks before the statement.

    The chart has no dates and its revenue can be defined differently (SPG's,
    BRK-B's and most banks' are), so it's only trusted when an earlier chart
    quarter exactly matches the statement's newest figure. That can be two
    quarters back, as Yahoo sometimes leaves a statement quarter blank (ABT)."""
    if not isinstance(earnings_module, Mapping):
        return None
    chart = (earnings_module.get("financialsChart") or {}).get("quarterly") or []
    revenues = [_num(quarter.get("revenue")) for quarter in chart]
    if not revenues or revenues[-1] is None or newest_revenue not in revenues[:-1]:
        return None
    # Walk back from the newest quarter to the one the statement ends on.
    ahead = revenues[::-1].index(newest_revenue)
    return revenues[-1], ahead


def summarize_revenue(
    income_stmt: pd.DataFrame | None,
    calendar: dict | None,
    earnings_module: dict | None,
    last_earnings_date: dt.date | None,
) -> dict:
    """Turn raw Yahoo statements into the REVENUE_FIELDS dict. No network calls.

    Yahoo has no revenue estimate for past quarters, so unlike EPS this has no
    beat/miss. last_revenue_current is False when the newest figure Yahoo has
    predates the last report, so it isn't the quarter that report covered.
    """
    fields = dict.fromkeys(REVENUE_FIELDS)

    # When the calendar still shows the report just made, its revenue estimate
    # could be for that quarter, so it can't be called the next one.
    earliest = last_earnings_date or dt.date.min
    if _calendar_window(calendar, earliest, last_earnings_date):
        estimate = _num(calendar.get("Revenue Average"))
        fields["next_revenue_estimate"] = int(estimate) if estimate else None
    if isinstance(earnings_module, Mapping):
        fields["revenue_currency"] = earnings_module.get("financialCurrency")

    quarters = _revenue_by_quarter(income_stmt)
    if not quarters:
        return fields

    period_end, revenue = next(iter(quarters.items()))
    if last_earnings_date and not _is_quarter_of_report(period_end, last_earnings_date):
        newer = _newer_chart_quarter(earnings_module, revenue)
        if newer is not None:
            revenue, quarters_ahead = newer
            # The chart has no dates, so this is the expected quarter end.
            period_end = (
                pd.Timestamp(period_end)
                + pd.DateOffset(months=3 * quarters_ahead)
                + pd.offsets.MonthEnd(0)
            ).date()

    fields["last_revenue"] = int(revenue)
    fields["last_revenue_period_end"] = period_end.isoformat()
    if last_earnings_date:
        fields["last_revenue_current"] = _is_quarter_of_report(period_end, last_earnings_date)

    year_ago = (pd.Timestamp(period_end) - pd.DateOffset(years=1)).date()
    for period, value in quarters.items():
        if abs((period - year_ago).days) <= YEAR_AGO_TOLERANCE_DAYS and value != 0:
            fields["last_revenue_yoy_pct"] = round((revenue - value) / abs(value) * 100, 2)
            break
    return fields


def _market_today() -> dt.date:
    return pd.Timestamp.now(tz=MARKET_TZ).date()


def _earnings_module(ticker: yf.Ticker) -> dict | None:
    """Yahoo's 'earnings' module. yfinance's public get_earnings() is deprecated
    and returns None, so this reaches for its internal fetch. If a later
    yfinance changes that, revenue just falls back to the statement."""
    result = ticker._quote._fetch(modules=["earnings"])
    return result["quoteSummary"]["result"][0]["earnings"] if result else None


def _download_sources(ticker: str, deadline: float) -> tuple[dict, dict]:
    """One pass over Yahoo for a ticker. Returns ({source: data}, {source: error});
    a source that raised is None in the first dict and listed in the second."""
    handle = yf.Ticker(ticker)
    getters = {
        "earnings_dates": lambda: handle.get_earnings_dates(limit=12),
        "calendar": lambda: handle.calendar,
        "income_stmt": lambda: handle.quarterly_income_stmt,
        "earnings_module": lambda: _earnings_module(handle),
    }
    sources = dict.fromkeys(getters)
    errors = {}
    for name, getter in getters.items():
        if time.monotonic() > deadline:
            errors[name] = TimeoutError("earnings time budget used up")
            continue
        try:
            sources[name] = getter()
        except Exception as e:
            errors[name] = e
    return sources, errors


def _has_data(value) -> bool:
    return not value.empty if isinstance(value, pd.DataFrame) else bool(value)


def _has_earnings_data(sources: dict) -> bool:
    return _has_data(sources["earnings_dates"]) or _has_data(sources["calendar"])


def _worth_retrying(sources: dict, errors: dict) -> bool:
    """Only a raised error is: yfinance caches its successful responses, so
    asking again after an empty page returns the same page. Revenue alone
    isn't worth the delay. get_earnings_dates is, as the only source of past
    results."""
    if "earnings_dates" in errors:
        return True
    return "calendar" in errors and not _has_earnings_data(sources)


def _fetch_one(
    ticker: str, today: dt.date, recent_days: int, soon_days: int, deadline: float
) -> dict:
    """All the fields for one ticker. Raises if Yahoo had no earnings data."""
    sources, errors = _download_sources(ticker, deadline)
    if _worth_retrying(sources, errors) and time.monotonic() + RETRY_DELAY_SECONDS < deadline:
        time.sleep(RETRY_DELAY_SECONDS)
        retried, errors = _download_sources(ticker, deadline)
        # Keep what the first pass got if the retry came back with less.
        sources = {k: retried[k] if _has_data(retried[k]) else sources[k] for k in sources}

    if not _has_earnings_data(sources):
        reason = errors.get("earnings_dates") or errors.get("calendar") or "no earnings data"
        raise RuntimeError(str(reason))

    fields = summarize_earnings(
        sources["earnings_dates"], sources["calendar"], today, recent_days, soon_days
    )
    last_date = (
        dt.date.fromisoformat(fields["last_earnings_date"])
        if fields["last_earnings_date"]
        else None
    )
    fields.update(
        summarize_revenue(
            sources["income_stmt"], sources["calendar"], sources["earnings_module"], last_date
        )
    )
    return fields


def fetch_earnings(
    tickers: list[str], recent_days: int = 7, soon_days: int = 14
) -> dict[str, dict]:
    """Fetch earnings fields for each ticker, one request set per ticker, so
    only pass the tickers that need it. Returns {ticker: fields}. Never raises
    over a ticker: one that fails gets blank fields and is reported to stderr.
    """
    results = {t: dict.fromkeys(ALL_FIELDS) for t in tickers}
    if not tickers:
        return results

    today = _market_today()
    started = time.monotonic()
    deadline = started + FETCH_BUDGET_SECONDS
    failed: list[str] = []
    print(f"[earnings] Fetching earnings for {len(tickers)} candidates...", file=sys.stderr)

    executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    futures = {
        executor.submit(_fetch_one, t, today, recent_days, soon_days, deadline): t
        for t in tickers
    }
    done = 0
    try:
        for future in as_completed(futures, timeout=FETCH_BUDGET_SECONDS):
            ticker = futures[future]
            done += 1
            try:
                results[ticker] = future.result()
            except Exception as e:
                failed.append(ticker)
                print(f"[earnings] {ticker} failed: {e}", file=sys.stderr)
            if done % 10 == 0 or done == len(tickers):
                print(f"[earnings] {done}/{len(tickers)} done", file=sys.stderr)
    except FuturesTimeoutError:
        unfinished = [t for f, t in futures.items() if not f.done()]
        failed.extend(unfinished)
        print(f"[earnings] Stopped at the {FETCH_BUDGET_SECONDS}s budget; "
              f"{len(unfinished)} tickers left blank.", file=sys.stderr)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    if failed:
        print(f"[earnings] {len(failed)} tickers had no earnings data: "
              f"{', '.join(failed[:20])}{' ...' if len(failed) > 20 else ''}",
              file=sys.stderr)
    print(f"[earnings] Finished in {time.monotonic() - started:.1f}s.", file=sys.stderr)
    return results


def candidate_tickers(results: pd.DataFrame, min_score: int) -> list[str]:
    """The tickers the dashboard will list: a side set and a high enough score."""
    mask = results["side"].isin(["SHORT", "LONG"]) & (results["score"] >= min_score)
    return results.loc[mask, "ticker"].tolist()


def merge_earnings(results: pd.DataFrame, fields_by_ticker: dict[str, dict]) -> pd.DataFrame:
    """Return a copy of results with the earnings columns added at the end.
    Rows keep their order, and tickers without fields stay blank. Object dtype
    keeps day counts as whole numbers beside the blanks, instead of 4.0."""
    merged = results.copy()
    for field in ALL_FIELDS:
        merged[field] = pd.Series(
            [fields_by_ticker.get(t, {}).get(field) for t in merged["ticker"]],
            index=merged.index,
            dtype=object,
        )
    return merged


def add_earnings(
    results: pd.DataFrame, min_score: int, recent_days: int = 7, soon_days: int = 14
) -> pd.DataFrame:
    """Fetch earnings for the dashboard's candidates and merge them in. If
    anything goes wrong the scan carries on with blank earnings columns."""
    try:
        fields = fetch_earnings(candidate_tickers(results, min_score), recent_days, soon_days)
    except Exception as e:
        print(f"[earnings] Skipped, lookup failed: {e}", file=sys.stderr)
        fields = {}
    return merge_earnings(results, fields)
