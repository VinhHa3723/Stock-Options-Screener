#!/usr/bin/env python3
"""
Stock Screener — RSI / Bollinger Bands / Moving Average scan
==============================================================

Scans the S&P 500 and/or Nasdaq-100 for stocks that are overbought or
oversold, based on the same three indicators you'd check by hand:
RSI, moving averages, and Bollinger Bands.

Usage examples:
    python main.py
    python main.py --universe sp500 --min-price 10
    python main.py --rsi-overbought 75 --rsi-oversold 25
    python main.py --tickers-file my_watchlist.csv

Output:
    output/scan_results_YYYYMMDD_HHMM.csv   <- this run's snapshot
    output/dashboard_YYYYMMDD_HHMM.html     <- this run's snapshot
    output/scan_results_latest.csv          <- always overwritten, bookmark-able
    output/dashboard_latest.html            <- always overwritten, bookmark-able
    The dashboard also opens automatically in your browser when the scan
    finishes (pass --no-open to disable this).
"""

from __future__ import annotations
import argparse
import datetime as dt
import shutil
import sys
import webbrowser
from pathlib import Path

from tickers import get_universe
from screener import run_screen
from report import write_csv, write_html_dashboard


def parse_args():
    p = argparse.ArgumentParser(
        description="Screen for overbought/oversold stocks using RSI, "
        "moving averages, and Bollinger Bands."
    )
    p.add_argument(
        "--universe",
        choices=["sp500", "nasdaq100", "both"],
        default="both",
        help="Which index constituents to scan (default: both)",
    )
    p.add_argument(
        "--tickers-file",
        default=None,
        help="Optional path to a CSV/text file of extra tickers to include "
        "(one per line, or a 'Symbol'/'Ticker' column)",
    )
    p.add_argument(
        "--min-price",
        type=float,
        default=5.0,
        help="Exclude stocks trading below this price, i.e. penny stocks "
        "(default: 5.0)",
    )
    p.add_argument(
        "--min-avg-volume",
        type=float,
        default=0.0,
        help="Optional minimum 20-day average volume filter for liquidity "
        "(default: off)",
    )
    p.add_argument("--rsi-period", type=int, default=14, help="RSI lookback period (default: 14)")
    p.add_argument(
        "--rsi-overbought", type=float, default=70.0,
        help="RSI level considered overbought (default: 70)",
    )
    p.add_argument(
        "--rsi-oversold", type=float, default=30.0,
        help="RSI level considered oversold (default: 30)",
    )
    p.add_argument("--bb-period", type=int, default=20, help="Bollinger Band period (default: 20)")
    p.add_argument(
        "--bb-std", type=float, default=2.0,
        help="Bollinger Band width in standard deviations (default: 2.0)",
    )
    p.add_argument(
        "--volume-avg-period", type=int, default=20,
        help="Period of the simple moving average on volume that a volume "
        "spike has to beat (default: 20)",
    )
    p.add_argument(
        "--volume-ratio-min", type=float, default=1.0,
        help="Today's volume / its SMA must be at least this to count as a "
        "volume confirmation (default: 1.0, i.e. the bar beats the SMA "
        "line). Raise it (e.g. 1.5) to demand a bigger spike.",
    )
    p.add_argument(
        "--vp-lookback", type=int, default=90,
        help="Trading days of history used to build the volume profile "
        "(default: 90). A different lookback can move the POC/HVNs, so "
        "match it to the range you use on your own chart.",
    )
    p.add_argument(
        "--vp-rows", type=int, default=24,
        help="Number of price rows in the volume profile (default: 24, the "
        "TradingView default)",
    )
    p.add_argument(
        "--vp-hvn-min-ratio", type=float, default=1.3,
        help="A profile peak counts as an HVN if its volume is at least this "
        "multiple of the average row (default: 1.3)",
    )
    p.add_argument(
        "--vp-tolerance-pct", type=float, default=1.0,
        help="Close must be within this %% of the POC/HVN price to count as "
        "'at' the node (default: 1.0)",
    )
    p.add_argument(
        "--history",
        default="2y",
        help="How much price history to pull, yfinance period format "
        "(default: 2y; needed for the 200-day moving average)",
    )
    p.add_argument(
        "--output-dir", default="output", help="Directory to write results into"
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only scan the first N tickers in the universe — handy for a "
        "fast test run (e.g. --limit 15) before committing to the full scan",
    )
    p.add_argument(
        "--no-open",
        action="store_true",
        help="Don't automatically open the dashboard in your browser when "
        "the scan finishes",
    )
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[main] Building ticker universe: {args.universe} "
          f"(+ {args.tickers_file})" if args.tickers_file else
          f"[main] Building ticker universe: {args.universe}", file=sys.stderr)
    tickers = get_universe(args.universe, extra_file=args.tickers_file)
    if args.limit:
        tickers = tickers[: args.limit]
        print(f"[main] --limit set: only scanning the first {len(tickers)} "
              f"tickers.", file=sys.stderr)
    print(f"[main] {len(tickers)} unique tickers to scan.", file=sys.stderr)

    results = run_screen(
        tickers,
        min_price=args.min_price,
        min_avg_volume=args.min_avg_volume,
        rsi_period=args.rsi_period,
        rsi_overbought=args.rsi_overbought,
        rsi_oversold=args.rsi_oversold,
        bb_period=args.bb_period,
        bb_std=args.bb_std,
        volume_avg_period=args.volume_avg_period,
        volume_ratio_min=args.volume_ratio_min,
        vp_lookback=args.vp_lookback,
        vp_rows=args.vp_rows,
        vp_hvn_min_ratio=args.vp_hvn_min_ratio,
        vp_tolerance_pct=args.vp_tolerance_pct,
        period=args.history,
    )

    if results.empty:
        print("[main] No results — check your internet connection or "
              "ticker universe.", file=sys.stderr)
        sys.exit(1)

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M")
    csv_path = out_dir / f"scan_results_{timestamp}.csv"
    html_path = out_dir / f"dashboard_{timestamp}.html"

    write_csv(results, str(csv_path))

    universe_label = {
        "sp500": "S&P 500",
        "nasdaq100": "Nasdaq-100",
        "both": "S&P 500 + Nasdaq-100",
    }[args.universe]
    if args.tickers_file:
        universe_label += " + custom list"

    write_html_dashboard(
        results,
        str(html_path),
        universe_label=universe_label,
        min_price=args.min_price,
        rsi_period=args.rsi_period,
        rsi_overbought=args.rsi_overbought,
        rsi_oversold=args.rsi_oversold,
        bb_period=args.bb_period,
        bb_std=args.bb_std,
        volume_avg_period=args.volume_avg_period,
        volume_ratio_min=args.volume_ratio_min,
        vp_lookback=args.vp_lookback,
        vp_rows=args.vp_rows,
        vp_tolerance_pct=args.vp_tolerance_pct,
    )

    counts = results["signal"].value_counts().to_dict()
    trig_counts = results["trigger"].value_counts().to_dict()
    print("\n=== Scan complete ===")
    print(f"Scanned: {len(results)} tickers with usable data")
    print(f"Strong short (overbought) candidates: {counts.get('STRONG_SHORT', 0)}")
    print(f"Strong long (oversold) candidates:    {counts.get('STRONG_LONG', 0)}")
    print(f"Watch list (one condition only):      "
          f"{counts.get('WATCH_SHORT', 0) + counts.get('WATCH_LONG', 0)}")
    print(f"Band re-entry triggers (volume-confirmed): "
          f"{trig_counts.get('SHORT_TRIGGER', 0)} short / {trig_counts.get('LONG_TRIGGER', 0)} long")
    print(f"Re-entries on low volume (not confirmed):  {trig_counts.get('REENTRY_LOW_VOLUME', 0)}")
    full_s = int(((results["side"] == "SHORT") & (results["score"] == 4)).sum())
    full_l = int(((results["side"] == "LONG") & (results["score"] == 4)).sum())
    near_s = int(((results["side"] == "SHORT") & (results["score"] == 3)).sum())
    near_l = int(((results["side"] == "LONG") & (results["score"] == 3)).sum())
    print(f"Full confluence, 4/4 (RSI + band + volume + profile node): {full_s} short / {full_l} long")
    print(f"Near confluence, 3/4:                                      {near_s} short / {near_l} long")

    # Also save fixed-name copies so you can bookmark ONE file/tab and just
    # refresh it after future runs, instead of hunting for the newest
    # timestamped file in the output folder each time.
    latest_csv = out_dir / "scan_results_latest.csv"
    latest_html = out_dir / "dashboard_latest.html"
    shutil.copyfile(csv_path, latest_csv)
    shutil.copyfile(html_path, latest_html)

    print(f"\nThis run:    {csv_path}")
    print(f"             {html_path}")
    print(f"Always-latest: {latest_html}  <- bookmark this one")

    if not args.no_open:
        webbrowser.open(latest_html.resolve().as_uri())
        print("\nOpened the dashboard in your browser.")
    else:
        print("\n--no-open set: not launching a browser.")


if __name__ == "__main__":
    main()