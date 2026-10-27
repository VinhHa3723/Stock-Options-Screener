"""
tickers.py
----------
Fetches current index constituents for the S&P 500 and Nasdaq-100.

Design notes:
- Index membership changes over time, so we always try to pull a LIVE list
  first rather than shipping a hardcoded ticker list that will go stale.
- Each index has a primary source and a fallback source. If both fail,
  we raise a clear error telling the user what to do (check connection,
  or supply their own list with --tickers-file).
"""

from __future__ import annotations
import sys
import io
import requests
import pandas as pd

USER_AGENT = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

# Well-maintained community dataset mirrored from Wikipedia's S&P 500 table.
SP500_FALLBACK_CSV = (
    "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/"
    "master/data/constituents.csv"
)
SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
NASDAQ100_WIKI_URL = "https://en.wikipedia.org/wiki/Nasdaq-100"
NASDAQ100_SLICKCHARTS_URL = "https://www.slickcharts.com/nasdaq100"
# Community-maintained mirror (scraped from Wikipedia) used as a last-resort
# fallback if both live sources above are unreachable. May occasionally lag
# the live index by a few weeks if that repo hasn't been re-run recently.
NASDAQ100_MIRROR_CSV = (
    "https://raw.githubusercontent.com/Gary-Strauss/NASDAQ100_Constituents/"
    "master/data/nasdaq100_constituents.csv"
)


def _clean_symbol(sym: str) -> str:
    """Normalize ticker formatting for yfinance (e.g. BRK.B -> BRK-B)."""
    return str(sym).strip().upper().replace(".", "-")


def get_sp500_tickers() -> list[str]:
    """Return current S&P 500 ticker symbols. Tries Wikipedia, then a
    mirrored CSV dataset on GitHub as a fallback."""
    # Primary: Wikipedia (most current, community-maintained same-day)
    try:
        resp = requests.get(SP500_WIKI_URL, headers=USER_AGENT, timeout=15)
        resp.raise_for_status()
        tables = pd.read_html(io.StringIO(resp.text))
        df = tables[0]
        symbol_col = "Symbol" if "Symbol" in df.columns else df.columns[0]
        tickers = [_clean_symbol(s) for s in df[symbol_col].tolist()]
        if len(tickers) >= 400:  # sanity check
            return sorted(set(tickers))
    except Exception as e:
        print(f"[tickers] Wikipedia S&P 500 fetch failed ({e}); trying fallback...",
              file=sys.stderr)

    # Fallback: mirrored dataset CSV
    try:
        df = pd.read_csv(SP500_FALLBACK_CSV)
        symbol_col = "Symbol" if "Symbol" in df.columns else df.columns[0]
        tickers = [_clean_symbol(s) for s in df[symbol_col].tolist()]
        if len(tickers) >= 400:
            return sorted(set(tickers))
    except Exception as e:
        print(f"[tickers] Fallback S&P 500 CSV fetch failed ({e})", file=sys.stderr)

    raise RuntimeError(
        "Could not fetch S&P 500 constituents from any source. "
        "Check your internet connection, or supply your own list via --tickers-file."
    )


def get_nasdaq100_tickers() -> list[str]:
    """Return current Nasdaq-100 ticker symbols. Tries Wikipedia, then
    Slickcharts as a fallback."""
    # Primary: Wikipedia
    try:
        resp = requests.get(NASDAQ100_WIKI_URL, headers=USER_AGENT, timeout=15)
        resp.raise_for_status()
        tables = pd.read_html(io.StringIO(resp.text))
        candidate = None
        for t in tables:
            cols = [str(c) for c in t.columns]
            if any("ticker" in c.lower() for c in cols) or any(
                "symbol" in c.lower() for c in cols
            ):
                candidate = t
                break
        if candidate is not None:
            col = next(
                c for c in candidate.columns
                if "ticker" in str(c).lower() or "symbol" in str(c).lower()
            )
            tickers = [_clean_symbol(s) for s in candidate[col].tolist()]
            tickers = [t for t in tickers if t and t.lower() != "nan"]
            if len(tickers) >= 90:
                return sorted(set(tickers))
    except Exception as e:
        print(f"[tickers] Wikipedia Nasdaq-100 fetch failed ({e}); trying fallback...",
              file=sys.stderr)

    # Fallback: Slickcharts
    try:
        resp = requests.get(NASDAQ100_SLICKCHARTS_URL, headers=USER_AGENT, timeout=15)
        resp.raise_for_status()
        tables = pd.read_html(io.StringIO(resp.text))
        df = max(tables, key=lambda t: t.shape[0])  # the constituents table is the largest
        col = next(c for c in df.columns if "symbol" in str(c).lower())
        tickers = [_clean_symbol(s) for s in df[col].tolist()]
        if len(tickers) >= 90:
            return sorted(set(tickers))
    except Exception as e:
        print(f"[tickers] Fallback Nasdaq-100 (Slickcharts) fetch failed ({e}); "
              "trying mirrored CSV...", file=sys.stderr)

    # Last resort: mirrored CSV snapshot on GitHub
    try:
        df = pd.read_csv(NASDAQ100_MIRROR_CSV)
        col = "Ticker" if "Ticker" in df.columns else df.columns[0]
        tickers = [_clean_symbol(s) for s in df[col].tolist()]
        if len(tickers) >= 90:
            print("[tickers] Using mirrored Nasdaq-100 CSV snapshot (may be a few "
                  "weeks out of date).", file=sys.stderr)
            return sorted(set(tickers))
    except Exception as e:
        print(f"[tickers] Mirrored Nasdaq-100 CSV fetch failed ({e})", file=sys.stderr)

    raise RuntimeError(
        "Could not fetch Nasdaq-100 constituents from any source. "
        "Check your internet connection, or supply your own list via --tickers-file."
    )


def get_universe(universe: str = "both", extra_file: str | None = None) -> list[str]:
    """Build the combined, de-duplicated ticker universe.

    universe: "sp500", "nasdaq100", or "both"
    extra_file: optional path to a text/CSV file with one ticker per line
                (or a 'Symbol'/'Ticker' column) to add to the universe.
    """
    tickers: set[str] = set()

    if universe in ("sp500", "both"):
        tickers.update(get_sp500_tickers())
    if universe in ("nasdaq100", "both"):
        tickers.update(get_nasdaq100_tickers())

    if extra_file:
        try:
            df = pd.read_csv(extra_file)
            col = None
            for c in df.columns:
                if str(c).lower() in ("symbol", "ticker", "tickers"):
                    col = c
                    break
            extra = df[col] if col is not None else df.iloc[:, 0]
            tickers.update(_clean_symbol(s) for s in extra.tolist())
        except Exception:
            # Fall back to treating it as a plain text file, one ticker per line
            with open(extra_file) as f:
                tickers.update(_clean_symbol(line) for line in f if line.strip())

    return sorted(tickers)


if __name__ == "__main__":
    # Quick manual check when run directly
    try:
        sp = get_sp500_tickers()
        print(f"S&P 500: {len(sp)} tickers. Sample: {sp[:10]}")
    except Exception as e:
        print("S&P 500 fetch error:", e)
    try:
        nd = get_nasdaq100_tickers()
        print(f"Nasdaq-100: {len(nd)} tickers. Sample: {nd[:10]}")
    except Exception as e:
        print("Nasdaq-100 fetch error:", e)
