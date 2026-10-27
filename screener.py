"""
screener.py
-----------
Fetches historical price data for a universe of tickers and runs the
RSI / Bollinger Band / Moving Average screen over all of them.
"""

from __future__ import annotations
import sys
import time
import pandas as pd
import yfinance as yf

from indicators import (
    compute_all_indicators,
    classify_signal,
    classify_trigger,
    volume_signal,
    compute_volume_profile,
    locate_node,
    confluence_score,
)

BATCH_SIZE = 100  # tickers per yfinance batch request


def chunk(lst: list, size: int):
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


def fetch_price_data(
    tickers: list[str], period: str = "2y", interval: str = "1d", retries: int = 2
) -> dict[str, pd.DataFrame]:
    """Batch-download OHLC data for many tickers at once (much faster and
    friendlier to Yahoo's API than one request per ticker).

    Returns a dict of {ticker: DataFrame with Open/High/Low/Close/Volume}.
    Tickers that fail to download (delisted, renamed, no data, etc.) are
    silently skipped and reported in the summary at the end.
    """
    data: dict[str, pd.DataFrame] = {}
    failed: list[str] = []

    batches = list(chunk(tickers, BATCH_SIZE))
    for i, batch in enumerate(batches, 1):
        print(f"[screener] Downloading batch {i}/{len(batches)} "
              f"({len(batch)} tickers)...", file=sys.stderr)

        attempt = 0
        raw = None
        while attempt <= retries:
            try:
                raw = yf.download(
                    tickers=batch,
                    period=period,
                    interval=interval,
                    group_by="ticker",
                    auto_adjust=True,
                    threads=True,
                    progress=False,
                )
                break
            except Exception as e:
                attempt += 1
                print(f"[screener] Batch {i} attempt {attempt} failed: {e}",
                      file=sys.stderr)
                time.sleep(2)

        if raw is None or raw.empty:
            failed.extend(batch)
            continue

        for t in batch:
            try:
                if len(batch) == 1:
                    df = raw
                else:
                    df = raw[t]
                df = df.dropna(how="all")
                if df.empty or "Close" not in df.columns:
                    failed.append(t)
                    continue
                if df["Close"].dropna().shape[0] < 30:
                    # Not enough history to compute reliable indicators
                    failed.append(t)
                    continue
                data[t] = df
            except (KeyError, Exception):
                failed.append(t)

    if failed:
        print(f"[screener] {len(failed)} tickers had no usable data and were "
              f"skipped: {', '.join(failed[:20])}"
              f"{' ...' if len(failed) > 20 else ''}", file=sys.stderr)

    return data


def run_screen(
    tickers: list[str],
    min_price: float = 5.0,
    min_avg_volume: float = 0.0,
    rsi_period: int = 14,
    rsi_overbought: float = 70.0,
    rsi_oversold: float = 30.0,
    bb_period: int = 20,
    bb_std: float = 2.0,
    volume_avg_period: int = 20,
    volume_ratio_min: float = 1.0,
    vp_lookback: int = 90,
    vp_rows: int = 24,
    vp_hvn_min_ratio: float = 1.3,
    vp_tolerance_pct: float = 1.0,
    period: str = "2y",
) -> pd.DataFrame:
    """Run the full screen and return a results DataFrame, one row per
    ticker, sorted with the strongest signals first."""

    price_data = fetch_price_data(tickers, period=period)

    rows = []
    for ticker, df in price_data.items():
        try:
            enriched = compute_all_indicators(
                df,
                rsi_period=rsi_period,
                bb_period=bb_period,
                bb_std=bb_std,
                volume_avg_period=volume_avg_period,
            )
        except Exception as e:
            print(f"[screener] Indicator calc failed for {ticker}: {e}",
                  file=sys.stderr)
            continue

        last = enriched.iloc[-1]
        prev = enriched.iloc[-2] if len(enriched) > 1 else last

        last_close = last.get("Close")
        if pd.isna(last_close):
            continue
        if last_close < min_price:
            continue  # exclude penny stocks

        avg_vol = df["Volume"].tail(20).mean() if "Volume" in df.columns else float("nan")
        if min_avg_volume and (pd.isna(avg_vol) or avg_vol < min_avg_volume):
            continue

        signal = classify_signal(
            last, rsi_overbought=rsi_overbought, rsi_oversold=rsi_oversold
        )

        trigger = classify_trigger(last, volume_ratio_min=volume_ratio_min)

        # Volume confirmation: volume bar vs its 20-SMA, coloured by candle direction
        vol_sig = volume_signal(last, volume_ratio_min=volume_ratio_min)

        # Volume Profile from the PRIOR vp_lookback bars (today excluded, so
        # today's own volume can't create the very node it's being checked
        # against). Is today's close sitting on the POC or an HVN?
        profile = None
        if {"High", "Low", "Volume"} <= set(df.columns):
            window = df.iloc[-(vp_lookback + 1):-1]
            profile = compute_volume_profile(
                window["High"], window["Low"], window["Volume"],
                rows=vp_rows, hvn_min_ratio=vp_hvn_min_ratio,
            )
        node_label, node_price = locate_node(profile, float(last_close), vp_tolerance_pct)
        poc = profile["poc"] if profile else None
        poc_dist_pct = (last_close - poc) / poc * 100 if poc else None

        # Which side is this ticker a candidate for, and how many of the four
        # criteria (RSI, band, directional volume, profile node) line up?
        if signal in ("STRONG_SHORT", "WATCH_SHORT") or trigger == "SHORT_TRIGGER":
            side = "SHORT"
        elif signal in ("STRONG_LONG", "WATCH_LONG") or trigger == "LONG_TRIGGER":
            side = "LONG"
        else:
            side = ""
        score = confluence_score(
            side, last.get("rsi"), last.get("bb_percent_b"), vol_sig, node_label,
            rsi_overbought=rsi_overbought, rsi_oversold=rsi_oversold,
        )

        pct_change_1d = (
            (last_close - prev["Close"]) / prev["Close"] * 100
            if not pd.isna(prev.get("Close"))
            else float("nan")
        )

        # Simple trend read: is price above/below the longer moving averages?
        if not pd.isna(last.get("sma_50")) and not pd.isna(last.get("sma_200")):
            if last["sma_50"] > last["sma_200"]:
                trend = "Uptrend"
            elif last["sma_50"] < last["sma_200"]:
                trend = "Downtrend"
            else:
                trend = "Flat"
        else:
            trend = "N/A"

        rows.append(
            {
                "ticker": ticker,
                "close": round(float(last_close), 2),
                "pct_change_1d": round(float(pct_change_1d), 2)
                if not pd.isna(pct_change_1d)
                else None,
                "rsi": round(float(last["rsi"]), 1) if not pd.isna(last.get("rsi")) else None,
                "percent_b": round(float(last["bb_percent_b"]), 3)
                if not pd.isna(last.get("bb_percent_b"))
                else None,
                "bb_upper": round(float(last["bb_upper"]), 2)
                if not pd.isna(last.get("bb_upper"))
                else None,
                "bb_lower": round(float(last["bb_lower"]), 2)
                if not pd.isna(last.get("bb_lower"))
                else None,
                "sma_20": round(float(last["sma_20"]), 2)
                if not pd.isna(last.get("sma_20"))
                else None,
                "sma_50": round(float(last["sma_50"]), 2)
                if not pd.isna(last.get("sma_50"))
                else None,
                "sma_200": round(float(last["sma_200"]), 2)
                if not pd.isna(last.get("sma_200"))
                else None,
                "pct_from_sma_50": round(float(last["pct_from_sma_50"]), 2)
                if not pd.isna(last.get("pct_from_sma_50"))
                else None,
                "trend": trend,
                "touched_upper_band_today": bool(last.get("touched_upper_band")),
                "touched_lower_band_today": bool(last.get("touched_lower_band")),
                "signal": signal,
                "trigger": trigger,
                "volume_ratio": round(float(last["volume_ratio"]), 2)
                if not pd.isna(last.get("volume_ratio"))
                else None,
                "vol_signal": vol_sig,
                "vp_node": node_label,
                "vp_node_price": round(node_price, 2) if node_price else None,
                "poc": round(poc, 2) if poc else None,
                "poc_dist_pct": round(poc_dist_pct, 2) if poc_dist_pct is not None else None,
                "side": side,
                "score": score,
            }
        )

    result_df = pd.DataFrame(rows)
    if result_df.empty:
        return result_df

    signal_rank = {
        "STRONG_SHORT": 0,
        "STRONG_LONG": 1,
        "WATCH_SHORT": 2,
        "WATCH_LONG": 3,
        "NEUTRAL": 4,
        "NO_DATA": 5,
    }
    trigger_rank = {
        "SHORT_TRIGGER": 0,
        "LONG_TRIGGER": 0,
        "REENTRY_LOW_VOLUME": 1,
        "NONE": 2,
    }
    result_df["_trig_rank"] = result_df["trigger"].map(trigger_rank)
    result_df["_rank"] = result_df["signal"].map(signal_rank)
    result_df["_rsi_extreme"] = (result_df["rsi"] - 50).abs()  # most stretched first
    # Highest confluence score first, then fresh triggers, then signal
    # strength, then how stretched RSI is.
    result_df = result_df.sort_values(
        ["score", "_trig_rank", "_rank", "_rsi_extreme"],
        ascending=[False, True, True, False],
    )
    result_df = result_df.drop(columns=["_trig_rank", "_rank", "_rsi_extreme"]).reset_index(
        drop=True
    )
    return result_df