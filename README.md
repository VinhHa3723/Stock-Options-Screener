# Stock Screener — RSI / Bollinger Bands / Moving Averages

Scans the S&P 500 and/or Nasdaq-100 every time you run it and flags stocks
that look **overbought** (candidates to short) or **oversold** (candidates
to go long) — the same read you'd get from manually checking RSI, moving
averages, and Bollinger Bands on each chart, just automated across ~550
tickers in one shot.

## What it checks

For each stock, using daily price data:

- **RSI(14)** — is it above 70 (overbought) or below 30 (oversold)?
- **Bollinger Bands (20-day, 2σ)** — is price touching or beyond the upper
  band (overbought) or the lower band (oversold)? This is measured with
  **%B**, where `%B ≥ 1.0` means at/above the upper band and `%B ≤ 0.0`
  means at/below the lower band — a precise stand-in for "touching the top
  line" or "touching the bottom line."
- **Moving averages (20/50/200-day)** — used for trend context: is the
  stock in an uptrend or downtrend, and how far is it stretched from its
  50-day average?

Each stock gets a signal:

| Signal | Meaning |
|---|---|
| `STRONG_SHORT` | RSI overbought **and** price at/above the upper Bollinger Band |
| `WATCH_SHORT` | Only one of those two conditions is triggered |
| `STRONG_LONG` | RSI oversold **and** price at/below the lower Bollinger Band |
| `WATCH_LONG` | Only one of those two conditions is triggered |
| `NEUTRAL` | Neither condition is close |

On top of that, every row gets three more reads, which together make up a
**confluence score (0-4)**, one point per criterion that lines up in the
trade's direction:

1. **RSI** at the extreme (>= 70 for a short, <= 30 for a long)
2. **Bollinger Band**: price at/through the band on that side
3. **Volume**: the volume bar beats its 20-period SMA *in the trade's direction*
   (`Selling` on a down candle for a short, `Buying` on an up candle for a
   long; direction is read from close vs open, like the colors on the standard
   Volume indicator)
4. **Volume Profile**: today's close sits within 1% of the **POC** (point of
   control, the highest-volume price) or an **HVN** (high volume node)

The score is a count of how many of your criteria agree. It is **not** a
measured win rate. Rows are sorted highest score first.

The volume profile is **approximated from daily bars** (each day's volume is
spread across its high-low range) over the prior 90 trading days in 24 price
rows, and it **excludes today**, so today's own volume can't create the node
it's being tested against. A true profile is built from intraday trades, so
treat nodes as zones, and compare against your own chart's profile settings,
since a different lookback can shift the POC. Match `--vp-lookback` to the
range you use on TradingView.

There's also a separate **Trigger** column (`SHORT_TRIGGER` / `LONG_TRIGGER` /
`REENTRY_LOW_VOLUME` / `NONE`): price closed back *inside* a band it had closed
outside of the day before, on volume at/above its SMA. It's independent of the
score (re-entering the band pulls %B back inside, so a triggered row can score
lower on the band point that day). Triggered rows are pulled into the dashboard
and highlighted.

ADX was removed from the output. The functions are still in `indicators.py`
(`adx`, `adx_regime`, `adx_confidence`) if you ever want it back.

Penny stocks (price below a threshold, $5 by default) are excluded
automatically.

## Quick start

```bash
pip install -r requirements.txt
python main.py
```

That's it — it'll pull the current S&P 500 + Nasdaq-100 ticker lists,
download ~2 years of daily price history for each, compute the indicators,
and write results into an `output/` folder:

- `scan_results_<timestamp>.csv` — every scanned ticker with full indicator
  values, for your own spreadsheet/further filtering
- `dashboard_<timestamp>.html` — **open this in your browser.** It's a
  self-contained, sortable, filterable dashboard split into "Short
  candidates" and "Long candidates" tables.

A run over the full ~550-ticker universe typically takes a couple of
minutes, most of it spent downloading price data from Yahoo Finance.

## Customizing the scan

All the thresholds you'd tweak are CLI flags:

```bash
# Only scan the S&P 500, and require RSI 75/25 instead of 70/30
python main.py --universe sp500 --rsi-overbought 75 --rsi-oversold 25

# Exclude anything under $20, and require decent liquidity for options trading
python main.py --min-price 20 --min-avg-volume 1000000

# Wider Bollinger Bands (2.5 standard deviations) and a faster RSI
python main.py --bb-std 2.5 --rsi-period 9

# Add your own watchlist on top of the two indices
python main.py --tickers-file my_watchlist.csv
```

Full option list:

| Flag | Default | Description |
|---|---|---|
| `--universe` | `both` | `sp500`, `nasdaq100`, or `both` |
| `--tickers-file` | — | extra tickers to include (CSV or one-per-line text) |
| `--min-price` | `5.0` | exclude stocks below this price (penny stock filter) |
| `--min-avg-volume` | off | exclude stocks below this 20-day avg. volume |
| `--rsi-period` | `14` | RSI lookback |
| `--rsi-overbought` | `70` | RSI level considered overbought |
| `--rsi-oversold` | `30` | RSI level considered oversold |
| `--bb-period` | `20` | Bollinger Band moving-average period |
| `--bb-std` | `2.0` | Bollinger Band width, in standard deviations |
| `--volume-avg-period` | `20` | Period of the SMA on volume that a spike has to beat |
| `--volume-ratio-min` | `1.0` | Volume / its SMA must be at least this to confirm (1.0 = beats the SMA; try 1.5 for a bigger spike) |
| `--vp-lookback` | `90` | Trading days used to build the volume profile |
| `--vp-rows` | `24` | Price rows in the volume profile |
| `--vp-hvn-min-ratio` | `1.3` | A profile peak counts as an HVN at this multiple of the average row |
| `--vp-tolerance-pct` | `1.0` | Close must be within this % of the POC/HVN to count as 'at' the node |
| `--history` | `2y` | how much price history to pull (needs enough for the 200-day MA) |
| `--output-dir` | `output` | where to write the CSV/HTML |

## How the ticker lists are fetched

Index membership changes over time (companies get added/removed), so the
tool always pulls a **live** list rather than a hardcoded one:

- **S&P 500**: scrapes Wikipedia's constituent table, falling back to a
  maintained GitHub dataset if that's unreachable.
- **Nasdaq-100**: scrapes Wikipedia, falling back to Slickcharts, falling
  back to a mirrored CSV snapshot as a last resort.

If your machine can't reach any of those (e.g. a locked-down network), pass
your own list with `--tickers-file`.

## Important Note

This is a price-only technical screen. It tells you which stocks are
statistically stretched relative to their *own* recent trading range — it
doesn't know about news, earnings, fundamentals, or overall market
conditions, and an "overbought" stock can stay overbought (or get more so)
for a while before reversing. Treat it as a shortlist to research further,
not a trade signal by itself, and this isn't financial advice.