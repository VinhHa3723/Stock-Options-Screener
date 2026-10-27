"""
indicators.py
-------------
RSI, moving average, and Bollinger Band calculations, plus the logic that
turns them into overbought/oversold "signals" for a screener.

All functions take/return pandas Series or DataFrames indexed by date, one
ticker at a time. Formulas follow the standard, widely-used definitions
(the same ones behind the indicators on most charting platforms) so results
should line up closely with what you already see when you eyeball a chart.
"""

from __future__ import annotations
import pandas as pd
import numpy as np


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI (the standard/default RSI on most charting platforms).

    RSI = 100 - 100 / (1 + RS), RS = avg gain / avg loss over `period`,
    smoothed with Wilder's method (equivalent to an EWM with alpha=1/period).
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    # When avg_loss is 0 (straight up move), RSI is defined as 100
    out = out.where(avg_loss != 0, 100.0)
    return out


def sma(close: pd.Series, period: int) -> pd.Series:
    return close.rolling(window=period, min_periods=period).mean()


def ema(close: pd.Series, period: int) -> pd.Series:
    return close.ewm(span=period, min_periods=period, adjust=False).mean()


def bollinger_bands(
    close: pd.Series, period: int = 20, num_std: float = 2.0
) -> pd.DataFrame:
    """Returns a DataFrame with columns: bb_mid, bb_upper, bb_lower,
    bb_percent_b, bb_bandwidth.

    bb_percent_b (aka %B) tells you where price sits within the bands:
      %B >= 1.0  -> price is AT or ABOVE the upper band ("touching the top")
      %B <= 0.0  -> price is AT or BELOW the lower band ("touching the bottom")
      %B == 0.5  -> price is exactly at the middle band (the moving average)
    """
    mid = sma(close, period)
    std = close.rolling(window=period, min_periods=period).std(ddof=0)
    upper = mid + num_std * std
    lower = mid - num_std * std
    band_range = (upper - lower).replace(0, np.nan)
    percent_b = (close - lower) / band_range
    bandwidth = band_range / mid

    return pd.DataFrame(
        {
            "bb_mid": mid,
            "bb_upper": upper,
            "bb_lower": lower,
            "bb_percent_b": percent_b,
            "bb_bandwidth": bandwidth,
        }
    )


def adx(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.DataFrame:
    """Wilder's ADX, +DI, and -DI.

    ADX measures trend STRENGTH (0-100), not direction:
      < 20   -> little/no trend, price is range-bound
      20-25  -> a trend may be emerging
      > 25   -> an established trend
      > 40   -> a strong trend
    Whether ADX is RISING or FALLING matters as much as its level - rising
    means the trend is strengthening, falling means it's losing steam
    (regardless of which direction currently dominates).

    +DI / -DI show which side is winning: +DI > -DI means up-moves are
    dominant, -DI > +DI means down-moves are dominant.
    """
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index
    )

    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)

    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    plus_dm_s = plus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    minus_dm_s = minus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    plus_di = 100 * (plus_dm_s / atr)
    minus_di = 100 * (minus_dm_s / atr)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx_val = dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    return pd.DataFrame({"plus_di": plus_di, "minus_di": minus_di, "adx": adx_val})


def adx_regime(adx_now: float, adx_prev: float, trend_threshold: float = 25.0) -> str:
    """Classify the trend regime from the current and previous ADX reading."""
    if pd.isna(adx_now):
        return "unknown"
    if adx_now < trend_threshold:
        return "ranging"  # mean-reversion signals (RSI/BB) are more trustworthy here
    if not pd.isna(adx_prev) and adx_now < adx_prev:
        return "trend_fading"  # was trending, now losing steam - supports a reversal call
    return "trend_strong"  # trending and still strengthening/steady - risky to fade


def adx_confidence(signal: str, regime: str, plus_di: float, minus_di: float) -> str:
    """Read an RSI/Bollinger signal through the lens of the ADX regime.

    Returns one of:
      "confirmed"  -> regime supports the reversal call (ranging, or a
                       fading trend that was running the opposite way)
      "caution"    -> the underlying trend is still strong and running
                       AGAINST your trade - the classic mean-reversion trap
      "neutral"    -> no strong read either way
      "n/a"        -> not an actionable signal to begin with
    """
    if signal not in ("STRONG_SHORT", "WATCH_SHORT", "STRONG_LONG", "WATCH_LONG"):
        return "n/a"
    if pd.isna(plus_di) or pd.isna(minus_di) or regime == "unknown":
        return "unknown"

    is_short = signal in ("STRONG_SHORT", "WATCH_SHORT")
    # For a short: the trend is "against you" if buyers (+DI) are still in control.
    # For a long: the trend is "against you" if sellers (-DI) are still in control.
    trend_against_trade = (plus_di > minus_di) if is_short else (minus_di > plus_di)

    if regime == "trend_strong" and trend_against_trade:
        return "caution"
    if regime == "ranging" or (regime == "trend_fading" and trend_against_trade):
        return "confirmed"
    return "neutral"


def band_reentry(percent_b: pd.Series) -> pd.DataFrame:
    """Detects the day price closes back INSIDE a Bollinger Band after
    having closed outside it the day before - a specific, zero-lag
    confirmation trigger (this is the technique Bollinger himself
    recommends pairing with a band touch, rather than treating the touch
    itself as the signal).

    short_reentry: yesterday closed at/above the upper band (%B >= 1.0),
                   today closed back below it (%B < 1.0)
    long_reentry:  yesterday closed at/below the lower band (%B <= 0.0),
                   today closed back above it (%B > 0.0)
    """
    prev_pb = percent_b.shift(1)
    short_reentry = (prev_pb >= 1.0) & (percent_b < 1.0)
    long_reentry = (prev_pb <= 0.0) & (percent_b > 0.0)
    return pd.DataFrame({"short_reentry": short_reentry, "long_reentry": long_reentry})


def volume_ratio(volume: pd.Series, period: int = 20) -> pd.Series:
    """Today's volume relative to its own recent (period-day) average.
    1.0 = exactly average, 1.5 = 50% above average, etc."""
    avg_volume = volume.rolling(window=period, min_periods=period).mean()
    return volume / avg_volume.replace(0, np.nan)


def classify_trigger(
    row: pd.Series, volume_ratio_min: float = 1.0
) -> str:
    """Combine the band re-entry with a volume check to get an actionable
    entry trigger:

    SHORT_TRIGGER        -> re-entered from above AND volume confirms
    LONG_TRIGGER         -> re-entered from below AND volume confirms
    REENTRY_LOW_VOLUME   -> re-entered, but on below-threshold volume
                            (the price action happened, participation didn't)
    NONE                 -> no re-entry today
    """
    short_reentry = bool(row.get("short_reentry"))
    long_reentry = bool(row.get("long_reentry"))
    vratio = row.get("volume_ratio")
    vol_confirmed = (not pd.isna(vratio)) and (vratio >= volume_ratio_min)

    if short_reentry and vol_confirmed:
        return "SHORT_TRIGGER"
    if long_reentry and vol_confirmed:
        return "LONG_TRIGGER"
    if short_reentry or long_reentry:
        return "REENTRY_LOW_VOLUME"
    return "NONE"


def volume_signal(row: pd.Series, volume_ratio_min: float = 1.0) -> str:
    """Read today's volume bar the way the standard Volume indicator + a
    20-period SMA on volume does on a chart:

    BUYING   -> volume at/above its SMA on an up candle (close > open)
    SELLING  -> volume at/above its SMA on a down candle (close < open)
    NONE     -> volume didn't beat its SMA (or a flat candle)

    volume_ratio is today's volume / its SMA, so 1.0 means "touching the
    SMA line"; raise volume_ratio_min (e.g. 1.5) to demand a bigger spike.
    """
    vr = row.get("volume_ratio")
    if pd.isna(vr) or vr < volume_ratio_min:
        return "NONE"
    o, c = row.get("Open"), row.get("Close")
    if pd.isna(o) or pd.isna(c):
        return "NONE"
    if c > o:
        return "BUYING"
    if c < o:
        return "SELLING"
    return "NONE"


def compute_volume_profile(
    high, low, volume, rows: int = 24, hvn_min_ratio: float = 1.3
) -> dict | None:
    """Approximate a Volume Profile (volume-by-price) from daily bars.

    Each day's volume is spread evenly across the price rows that day's
    High-Low range touches. (A true profile uses intraday trades; this is the
    standard approximation when you only have daily bars, so treat node
    prices as zones rather than exact levels.)

    Returns None if there isn't enough data, else a dict with:
      poc      -> price of the Point of Control (highest-volume row)
      hvn      -> prices of High Volume Nodes (local peaks in the profile
                  whose volume is >= hvn_min_ratio x the average row)
      centers / volume / edges -> the raw profile, for inspection
    """
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    v = np.asarray(volume, dtype=float)
    ok = ~(np.isnan(h) | np.isnan(l) | np.isnan(v))
    h, l, v = h[ok], l[ok], v[ok]
    if len(h) < 10:
        return None

    lo, hi = float(l.min()), float(h.max())
    if hi <= lo:
        return None

    edges = np.linspace(lo, hi, rows + 1)
    bin_lo, bin_hi = edges[:-1], edges[1:]
    centers = (bin_lo + bin_hi) / 2

    # How much of each bar's range falls inside each row: shape (bars, rows)
    overlap = np.clip(
        np.minimum(h[:, None], bin_hi[None, :]) - np.maximum(l[:, None], bin_lo[None, :]),
        0,
        None,
    )
    bar_range = h - l
    share = np.zeros_like(overlap)
    has_range = bar_range > 0
    share[has_range] = overlap[has_range] / bar_range[has_range][:, None]
    if (~has_range).any():  # zero-range bar: all its volume goes in one row
        flat_idx = np.where(~has_range)[0]
        row_idx = np.clip(np.searchsorted(edges, l[flat_idx], side="right") - 1, 0, rows - 1)
        share[flat_idx, row_idx] = 1.0

    profile = (share * v[:, None]).sum(axis=0)
    if profile.sum() <= 0:
        return None

    poc_idx = int(profile.argmax())

    # Light smoothing so one noisy row doesn't create a fake "node"
    padded = np.pad(profile, 1, mode="edge")
    smooth = np.convolve(padded, np.array([1, 2, 1]) / 4, mode="valid")
    mean_row = profile.mean()
    hvn_idx = []
    for i in range(rows):
        left = smooth[i - 1] if i > 0 else -np.inf
        right = smooth[i + 1] if i < rows - 1 else -np.inf
        if smooth[i] > left and smooth[i] >= right and smooth[i] >= hvn_min_ratio * mean_row:
            hvn_idx.append(i)
    if poc_idx not in hvn_idx:
        hvn_idx.append(poc_idx)  # the POC is always the biggest node

    return {
        "poc": float(centers[poc_idx]),
        "hvn": [float(centers[i]) for i in sorted(hvn_idx) if i != poc_idx],
        "centers": centers,
        "volume": profile,
        "edges": edges,
    }


def locate_node(profile: dict | None, close: float, tolerance_pct: float = 1.0):
    """Is the close sitting on the POC or an HVN?

    Returns (label, node_price): label is "POC", "HVN", or "NONE". The close
    counts as 'at' a node if it's within tolerance_pct of the node's price.
    The POC is checked first, since it's the strongest node.
    """
    if profile is None or pd.isna(close) or close <= 0:
        return "NONE", None
    tol = tolerance_pct / 100.0
    if abs(close - profile["poc"]) / close <= tol:
        return "POC", profile["poc"]
    nearby = [p for p in profile["hvn"] if abs(close - p) / close <= tol]
    if nearby:
        return "HVN", min(nearby, key=lambda p: abs(close - p))
    return "NONE", None


def confluence_score(
    side: str,
    rsi_val: float,
    percent_b: float,
    vol_sig: str,
    node_label: str,
    rsi_overbought: float = 70.0,
    rsi_oversold: float = 30.0,
    percent_b_high: float = 1.0,
    percent_b_low: float = 0.0,
) -> int:
    """Count how many of the four criteria line up for the trade direction
    (0-4). One point each:

      1. RSI is at the extreme      (>= overbought for a short / <= oversold for a long)
      2. Price is at/through the Bollinger Band on that side
      3. Volume beat its SMA in the trade's direction (selling for a short,
         buying for a long)
      4. Close is sitting on the Volume Profile's POC or an HVN
    """
    if side not in ("SHORT", "LONG"):
        return 0
    pts = 0
    if not pd.isna(rsi_val):
        if side == "SHORT" and rsi_val >= rsi_overbought:
            pts += 1
        if side == "LONG" and rsi_val <= rsi_oversold:
            pts += 1
    if not pd.isna(percent_b):
        if side == "SHORT" and percent_b >= percent_b_high:
            pts += 1
        if side == "LONG" and percent_b <= percent_b_low:
            pts += 1
    if (side == "SHORT" and vol_sig == "SELLING") or (side == "LONG" and vol_sig == "BUYING"):
        pts += 1
    if node_label in ("POC", "HVN"):
        pts += 1
    return pts


def compute_all_indicators(
    df: pd.DataFrame,
    rsi_period: int = 14,
    bb_period: int = 20,
    bb_std: float = 2.0,
    ma_periods: tuple[int, ...] = (20, 50, 200),
    adx_period: int = 14,
    include_adx: bool = False,
    volume_avg_period: int = 20,
) -> pd.DataFrame:
    """Given a price DataFrame with a 'Close' column (and ideally 'High'/'Low'),
    return a new DataFrame with all indicator columns appended."""
    out = df.copy()
    close = out["Close"]

    out["rsi"] = rsi(close, rsi_period)

    bb = bollinger_bands(close, bb_period, bb_std)
    out = pd.concat([out, bb], axis=1)

    reentry = band_reentry(out["bb_percent_b"])
    out = pd.concat([out, reentry], axis=1)

    if "Volume" in out.columns:
        out["volume_ratio"] = volume_ratio(out["Volume"], volume_avg_period)

    if include_adx and "High" in out.columns and "Low" in out.columns:
        adx_df = adx(out["High"], out["Low"], close, adx_period)
        out = pd.concat([out, adx_df], axis=1)

    for p in ma_periods:
        out[f"sma_{p}"] = sma(close, p)

    # % distance of price from each moving average (extension from trend)
    for p in ma_periods:
        out[f"pct_from_sma_{p}"] = (close - out[f"sma_{p}"]) / out[f"sma_{p}"] * 100

    # Did price actually touch the band intraday? (uses High/Low if available)
    if "High" in out.columns:
        out["touched_upper_band"] = out["High"] >= out["bb_upper"]
    else:
        out["touched_upper_band"] = out["Close"] >= out["bb_upper"]

    if "Low" in out.columns:
        out["touched_lower_band"] = out["Low"] <= out["bb_lower"]
    else:
        out["touched_lower_band"] = out["Close"] <= out["bb_lower"]

    return out


def classify_signal(
    row: pd.Series,
    rsi_overbought: float = 70.0,
    rsi_oversold: float = 30.0,
    percent_b_high: float = 1.0,
    percent_b_low: float = 0.0,
    percent_b_watch_margin: float = 0.05,
) -> str:
    """Classify the most recent row into a signal bucket:

    STRONG_SHORT  -> RSI overbought AND price at/above the upper band
    WATCH_SHORT   -> only one of the two conditions is met
    STRONG_LONG   -> RSI oversold AND price at/below the lower band
    WATCH_LONG    -> only one of the two conditions is met
    NEUTRAL       -> neither condition is close to being met
    """
    rsi_val = row.get("rsi")
    pb = row.get("bb_percent_b")

    if pd.isna(rsi_val) or pd.isna(pb):
        return "NO_DATA"

    rsi_ob = rsi_val >= rsi_overbought
    rsi_os = rsi_val <= rsi_oversold
    bb_ob = pb >= percent_b_high
    bb_os = pb <= percent_b_low
    bb_watch_ob = pb >= (percent_b_high - percent_b_watch_margin)
    bb_watch_os = pb <= (percent_b_low + percent_b_watch_margin)

    if rsi_ob and bb_ob:
        return "STRONG_SHORT"
    if rsi_os and bb_os:
        return "STRONG_LONG"
    if rsi_ob or bb_watch_ob:
        return "WATCH_SHORT"
    if rsi_os or bb_watch_os:
        return "WATCH_LONG"
    return "NEUTRAL"