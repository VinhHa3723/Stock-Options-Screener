"""
report.py
---------
Turns the screener's results DataFrame into:
  1. A CSV export (everything, full precision, for your own spreadsheet work)
  2. A self-contained HTML dashboard (sortable/filterable tables, no
     external dependencies, works by double-clicking the file)
"""

from __future__ import annotations
import datetime as dt
from pathlib import Path
import pandas as pd
from jinja2 import Environment, BaseLoader

SIGNAL_LABELS = {
    "STRONG_SHORT": "Strong short (overbought)",
    "WATCH_SHORT": "Watch — leaning short",
    "STRONG_LONG": "Strong long (oversold)",
    "WATCH_LONG": "Watch — leaning long",
    "NEUTRAL": "Neutral",
    "NO_DATA": "Insufficient data",
}

TRIGGER_LABELS = {
    "SHORT_TRIGGER": "Short trigger",
    "LONG_TRIGGER": "Long trigger",
    "REENTRY_LOW_VOLUME": "Re-entry, low volume",
    "NONE": "—",
}

TRIGGER_CSS_CLASS = {
    "SHORT_TRIGGER": "trig-short",
    "LONG_TRIGGER": "trig-long",
    "REENTRY_LOW_VOLUME": "trig-low-volume",
    "NONE": "trig-none",
}


def write_csv(df: pd.DataFrame, output_path: str) -> str:
    df.to_csv(output_path, index=False)
    return output_path


def _chart_url(ticker: str) -> str:
    """Link to the stockanalysis.com interactive chart for a ticker.
    Our tickers use a dash for share classes (BRK-B, the yfinance style);
    stockanalysis.com uses a dot (BRK.B)."""
    return f"https://stockanalysis.com/chart/{str(ticker).upper().replace('-', '.')}/"


def _is_missing(value) -> bool:
    return value is None or (isinstance(value, float) and pd.isna(value))


def _fmt_ratio(value) -> str:
    return "—" if _is_missing(value) else f"{value:.1f}×"


def _fmt_price(value) -> str:
    return "—" if _is_missing(value) else f"${value:,.2f}"


def _fmt_signed_pct(value) -> str:
    """Format a percent value with an explicit + sign for positive numbers,
    e.g. 6.55 -> '+6.55%', -3.2 -> '-3.20%', None -> '—'."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.2f}%"


def _rows_for(df: pd.DataFrame, signals: list[str], trigger: str | None = None) -> list[dict]:
    mask = df["signal"].isin(signals)
    if trigger:
        # A ticker that just fired a re-entry trigger may no longer read as
        # STRONG on today's bar (that's the point - price moved back inside
        # the band) so it needs to be pulled in on its own merits too.
        mask = mask | (df["trigger"] == trigger)
    subset = df[mask].copy()
    return subset.to_dict(orient="records")


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Market Screen — {{ generated_date }}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;1,6..72,500&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #14171c;
    --panel: #1b1f26;
    --panel-raised: #20242c;
    --border: #2c313b;
    --text: #ece8de;
    --text-muted: #8b909b;
    --gold: #c9a227;
    --rust: #c1584a;
    --teal: #3f9c8a;
    --slate: #6b7280;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font-family: 'Inter', sans-serif;
    line-height: 1.5;
    -webkit-font-smoothing: antialiased;
  }
  .wrap { max-width: 1180px; margin: 0 auto; padding: 48px 24px 80px; }

  header.top {
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    flex-wrap: wrap;
    gap: 16px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 28px;
    margin-bottom: 36px;
  }
  h1.title {
    font-family: 'Newsreader', serif;
    font-weight: 500;
    font-size: 2.6rem;
    margin: 0;
    letter-spacing: -0.01em;
  }
  .meta { color: var(--text-muted); font-size: 0.92rem; margin-top: 6px; }
  .meta b { color: var(--text); font-weight: 500; }

  .stats {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1px;
    background: var(--border);
    border: 1px solid var(--border);
    margin-bottom: 44px;
  }
  .stat {
    background: var(--panel);
    padding: 22px 20px;
  }
  .stat .num {
    font-family: 'Newsreader', serif;
    font-size: 2.4rem;
    line-height: 1;
  }
  .stat .lbl {
    margin-top: 8px;
    font-size: 0.82rem;
    color: var(--text-muted);
  }
  .stat.short .num { color: var(--rust); }
  .stat.long .num { color: var(--teal); }
  .stat.watch .num { color: var(--gold); }
  .stat.trigger { background: rgba(201, 162, 39, 0.08); }
  .stat.trigger .num { color: var(--gold); }
  .stat.confluence { background: rgba(201, 162, 39, 0.08); }
  .stat.confluence .num { color: var(--gold); }

  section.panel { margin-bottom: 40px; }
  .panel-head {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 16px;
    margin-bottom: 14px;
  }
  .panel-head h2 {
    font-family: 'Newsreader', serif;
    font-weight: 500;
    font-size: 1.5rem;
    margin: 0;
  }
  .panel-head h2 .accent-short { color: var(--rust); }
  .panel-head h2 .accent-long { color: var(--teal); }
  .panel-sub { color: var(--text-muted); font-size: 0.88rem; }

  input.filter {
    background: var(--panel-raised);
    border: 1px solid var(--border);
    color: var(--text);
    padding: 8px 12px;
    border-radius: 3px;
    font-family: 'Inter', sans-serif;
    font-size: 0.85rem;
    width: 200px;
  }
  input.filter:focus { outline: 1px solid var(--gold); border-color: var(--gold); }

  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.88rem;
  }
  thead th {
    text-align: left;
    font-weight: 500;
    color: var(--text-muted);
    padding: 8px 14px;
    border-bottom: 1px solid var(--border);
    cursor: pointer;
    white-space: nowrap;
    user-select: none;
  }
  thead th:hover { color: var(--text); }
  thead th.sorted::after { content: " \2193"; color: var(--gold); }
  thead th.sorted-asc::after { content: " \2191"; color: var(--gold); }
  tbody td {
    padding: 10px 14px;
    border-bottom: 1px solid var(--border);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    white-space: nowrap;
  }
  tbody td.ticker-cell {
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    letter-spacing: 0.01em;
  }
  .ticker-cell a {
    color: inherit;
    text-decoration: none;
    border-bottom: 1px dotted var(--text-muted);
  }
  .ticker-cell a:hover {
    color: var(--gold);
    border-bottom-color: var(--gold);
  }
  tbody tr { border-left: 3px solid transparent; }
  tbody tr.strong-short { border-left-color: var(--rust); }
  tbody tr.watch-short { border-left-color: rgba(193, 88, 74, 0.4); }
  tbody tr.strong-long { border-left-color: var(--teal); }
  tbody tr.watch-long { border-left-color: rgba(63, 156, 138, 0.4); }
  tbody tr:hover { background: var(--panel-raised); }
  tbody tr.has-trigger { background: rgba(201, 162, 39, 0.07); }
  tbody tr.has-trigger:hover { background: rgba(201, 162, 39, 0.13); }

  .tag {
    display: inline-block;
    font-family: 'Inter', sans-serif;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 2px;
  }
  .tag.strong-short { background: rgba(193, 88, 74, 0.16); color: var(--rust); }
  .tag.watch-short { background: rgba(193, 88, 74, 0.08); color: var(--rust); }
  .tag.strong-long { background: rgba(63, 156, 138, 0.16); color: var(--teal); }
  .tag.watch-long { background: rgba(63, 156, 138, 0.08); color: var(--teal); }
  .tag.vol-buy { background: rgba(63, 156, 138, 0.16); color: var(--teal); }
  .tag.vol-sell { background: rgba(193, 88, 74, 0.16); color: var(--rust); }
  .tag.node-poc { background: rgba(201, 162, 39, 0.22); color: var(--gold); font-weight: 700; }
  .tag.node-hvn { background: rgba(201, 162, 39, 0.10); color: var(--gold); }
  .muted { color: var(--text-muted); }
  .score {
    display: inline-block;
    min-width: 2.6em;
    text-align: center;
    padding: 2px 6px;
    border-radius: 2px;
    font-weight: 600;
    background: rgba(107, 114, 128, 0.14);
    color: var(--text-muted);
  }
  .score.s3 { background: rgba(201, 162, 39, 0.16); color: var(--gold); }
  .score.s4 { background: var(--gold); color: var(--bg); font-weight: 700; }
  .tag.trig-short { background: var(--rust); color: var(--bg); font-weight: 700; }
  .tag.trig-long { background: var(--teal); color: var(--bg); font-weight: 700; }
  .tag.trig-low-volume { background: rgba(107, 114, 128, 0.16); color: var(--slate); }
  .tag.trig-none { background: none; color: var(--text-muted); font-weight: 400; padding: 0; }

  .empty-note { color: var(--text-muted); padding: 20px 14px; font-style: italic; }

  footer {
    margin-top: 56px;
    padding-top: 24px;
    border-top: 1px solid var(--border);
    color: var(--text-muted);
    font-size: 0.8rem;
    max-width: 760px;
  }
  footer p { margin: 0 0 10px; }

  @media (max-width: 720px) {
    .stats { grid-template-columns: repeat(2, 1fr); }
    h1.title { font-size: 2rem; }
    table { font-size: 0.78rem; }
    thead th, tbody td { padding: 8px 8px; }
  }
</style>
</head>
<body>
<div class="wrap">

  <header class="top">
    <div>
      <h1 class="title">Market Screen</h1>
      <div class="meta">Generated <b>{{ generated_date }}</b> &middot; Universe: <b>{{ universe_label }}</b> &middot; {{ total_scanned }} tickers scanned &middot; penny stocks (&lt; ${{ min_price }}) excluded</div>
    </div>
  </header>

  <div class="stats">
    <div class="stat confluence">
      <div class="num">{{ full_short_count + full_long_count }}</div>
      <div class="lbl">Full-confluence setups, 4/4 ({{ full_short_count }} short / {{ full_long_count }} long)</div>
    </div>
    <div class="stat trigger">
      <div class="num">{{ short_trigger_count + long_trigger_count }}</div>
      <div class="lbl">Confirmed re-entry triggers today ({{ short_trigger_count }} short / {{ long_trigger_count }} long)</div>
    </div>
    <div class="stat short">
      <div class="num">{{ strong_short_count }}</div>
      <div class="lbl">Strong short signals (RSI &ge; {{ rsi_overbought }} and price at/above upper band)</div>
    </div>
    <div class="stat long">
      <div class="num">{{ strong_long_count }}</div>
      <div class="lbl">Strong long signals (RSI &le; {{ rsi_oversold }} and price at/below lower band)</div>
    </div>
    <div class="stat watch">
      <div class="num">{{ watch_count }}</div>
      <div class="lbl">On watch — only one condition triggered</div>
    </div>
    <div class="stat">
      <div class="num">{{ total_scanned }}</div>
      <div class="lbl">Total tickers with usable data</div>
    </div>
  </div>

  <section class="panel">
    <div class="panel-head">
      <div>
        <h2><span class="accent-short">Short candidates</span> — overbought</h2>
        <div class="panel-sub">Highest confluence score first, then fresh triggers, then most stretched RSI</div>
      </div>
      <input type="text" class="filter" data-table="short-table" placeholder="Filter ticker…" />
    </div>
    {{ short_table }}
  </section>

  <section class="panel">
    <div class="panel-head">
      <div>
        <h2><span class="accent-long">Long candidates</span> — oversold</h2>
        <div class="panel-sub">Highest confluence score first, then fresh triggers, then most stretched RSI</div>
      </div>
      <input type="text" class="filter" data-table="long-table" placeholder="Filter ticker…" />
    </div>
    {{ long_table }}
  </section>

  <footer>
    <p>Signals come from price-only technical indicators (RSI-{{ rsi_period }}, {{ bb_period }}-day Bollinger Bands at {{ bb_std }}&sigma;, and {{ ma_periods }}-day moving averages) run against daily closing data. They flag stocks that look statistically stretched relative to their own recent history — not a prediction of what happens next, and not a recommendation to buy, sell, or trade any option. Verify a name's chart, news, earnings date, and options liquidity before acting, and size any position for the risk you're comfortable with.</p>
    <p>The <b>Score</b> counts how many of four criteria line up in the trade's direction, one point each: (1) RSI at the extreme, (2) price at/through the Bollinger Band, (3) <b>Volume</b> beating its {{ volume_avg_period }}-day SMA in the trade's direction (selling volume for a short, buying volume for a long, read from the candle's close vs open, currently &ge;{{ volume_ratio_min }}&times; the SMA), and (4) <b>Profile</b>: the close sitting within {{ vp_tolerance_pct }}% of the Volume Profile's POC or an HVN. A high score means more of your criteria agree. It is not a measured win rate.</p>
    <p>The volume profile is approximated from daily bars (each day's volume spread across its high-low range) over the prior {{ vp_lookback }} trading days in {{ vp_rows }} price rows, excluding today so today's volume can't create the node it's tested against. A true profile uses intraday trades, so treat nodes as zones, and compare against your own chart's profile settings, since a different lookback can move the POC.</p>
    <p>The <b>Trigger</b> column is a same-day entry confirmation, separate from the RSI/Bollinger signal above it: it fires when price closes back <i>inside</i> a band it had closed outside of the day before, on volume at or above its {{ volume_avg_period }}-day average (currently &ge;{{ volume_ratio_min }}&times;). A <b>re-entry, low volume</b> tag means the price action happened but without that volume backing — weaker evidence. Rows with a confirmed trigger get a subtle highlight so they're easy to spot.</p>
  </footer>

</div>

<script>
  function filterTable(inputEl) {
    const tableId = inputEl.getAttribute('data-table');
    const table = document.getElementById(tableId);
    if (!table) return;
    const query = inputEl.value.trim().toUpperCase();
    const rows = table.querySelectorAll('tbody tr');
    rows.forEach(row => {
      const ticker = row.getAttribute('data-ticker') || '';
      row.style.display = ticker.includes(query) ? '' : 'none';
    });
  }
  document.querySelectorAll('input.filter').forEach(inp => {
    inp.addEventListener('input', () => filterTable(inp));
  });

  function sortTable(table, colIndex, numeric) {
    const tbody = table.querySelector('tbody');
    const rows = Array.from(tbody.querySelectorAll('tr'));
    const th = table.querySelectorAll('thead th')[colIndex];
    const asc = !th.classList.contains('sorted');

    table.querySelectorAll('thead th').forEach(h => h.classList.remove('sorted', 'sorted-asc'));
    th.classList.add(asc ? 'sorted' : 'sorted-asc');

    rows.sort((a, b) => {
      let av = a.children[colIndex].getAttribute('data-sort') ?? a.children[colIndex].textContent.trim();
      let bv = b.children[colIndex].getAttribute('data-sort') ?? b.children[colIndex].textContent.trim();
      if (numeric) {
        av = parseFloat(av); bv = parseFloat(bv);
        if (isNaN(av)) av = -Infinity;
        if (isNaN(bv)) bv = -Infinity;
        return asc ? av - bv : bv - av;
      }
      return asc ? String(av).localeCompare(String(bv)) : String(bv).localeCompare(String(av));
    });
    rows.forEach(r => tbody.appendChild(r));
  }

  document.querySelectorAll('table').forEach(table => {
    table.querySelectorAll('thead th').forEach((th, idx) => {
      th.addEventListener('click', () => sortTable(table, idx, th.getAttribute('data-numeric') === '1'));
    });
  });
</script>
</body>
</html>
"""

TABLE_TEMPLATE = r"""
<table id="{{ table_id }}">
  <thead>
    <tr>
      <th>Ticker</th>
      <th data-numeric="1">Score</th>
      <th data-numeric="1">Close</th>
      <th data-numeric="1">1D %</th>
      <th data-numeric="1">RSI</th>
      <th data-numeric="1">%B</th>
      <th>Trend</th>
      <th data-numeric="1">vs 50-MA</th>
      <th>Signal</th>
      <th>Volume</th>
      <th>Profile</th>
      <th>Trigger</th>
    </tr>
  </thead>
  <tbody>
    {% for r in rows %}
    <tr class="{{ r.signal|lower|replace('_', '-') }}{{ ' has-trigger' if r.trigger in ('SHORT_TRIGGER', 'LONG_TRIGGER') else '' }}" data-ticker="{{ r.ticker }}">
      <td class="ticker-cell"><a href="{{ r.ticker|chart_url }}" target="_blank" rel="noopener noreferrer">{{ r.ticker }}</a></td>
      <td data-sort="{{ r.score }}"><span class="score s{{ r.score }}">{{ r.score }}/4</span></td>
      <td data-sort="{{ r.close }}">${{ '%.2f'|format(r.close) }}</td>
      <td data-sort="{{ r.pct_change_1d }}">{{ r.pct_change_1d|pct }}</td>
      <td data-sort="{{ r.rsi }}">{{ '%.1f'|format(r.rsi) if r.rsi is not none else '—' }}</td>
      <td data-sort="{{ r.percent_b }}">{{ '%.2f'|format(r.percent_b) if r.percent_b is not none else '—' }}</td>
      <td>{{ r.trend }}</td>
      <td data-sort="{{ r.pct_from_sma_50 }}">{{ r.pct_from_sma_50|pct }}</td>
      <td><span class="tag {{ r.signal|lower|replace('_', '-') }}">{{ signal_labels[r.signal] }}</span></td>
      <td data-sort="{{ r.volume_ratio }}">{% if r.vol_signal == 'BUYING' %}<span class="tag vol-buy">Buying {{ r.volume_ratio|ratio }}</span>{% elif r.vol_signal == 'SELLING' %}<span class="tag vol-sell">Selling {{ r.volume_ratio|ratio }}</span>{% else %}<span class="muted">{{ r.volume_ratio|ratio }}</span>{% endif %}</td>
      <td title="POC {{ r.poc|price }} ({{ r.poc_dist_pct|pct }} from close)">{% if r.vp_node == 'POC' %}<span class="tag node-poc">At POC {{ r.vp_node_price|price }}</span>{% elif r.vp_node == 'HVN' %}<span class="tag node-hvn">At HVN {{ r.vp_node_price|price }}</span>{% else %}<span class="muted">—</span>{% endif %}</td>
      <td><span class="tag {{ trigger_class.get(r.trigger, 'trig-none') }}">{{ trigger_labels.get(r.trigger, r.trigger) }}</span></td>
    </tr>
    {% else %}
    <tr><td colspan="12" class="empty-note">No tickers currently meet this condition.</td></tr>
    {% endfor %}
  </tbody>
</table>
"""


def write_html_dashboard(
    df: pd.DataFrame,
    output_path: str,
    universe_label: str,
    min_price: float,
    rsi_period: int,
    rsi_overbought: float,
    rsi_oversold: float,
    bb_period: int,
    bb_std: float,
    ma_periods: str = "20/50/200",
    volume_avg_period: int = 20,
    volume_ratio_min: float = 1.0,
    vp_lookback: int = 90,
    vp_rows: int = 24,
    vp_tolerance_pct: float = 1.0,
) -> str:
    env = Environment(loader=BaseLoader())
    env.filters["pct"] = _fmt_signed_pct
    env.filters["chart_url"] = _chart_url
    env.filters["ratio"] = _fmt_ratio
    env.filters["price"] = _fmt_price
    table_tpl = env.from_string(TABLE_TEMPLATE)
    page_tpl = env.from_string(HTML_TEMPLATE)

    short_rows = _rows_for(df, ["STRONG_SHORT", "WATCH_SHORT"], trigger="SHORT_TRIGGER")
    long_rows = _rows_for(df, ["STRONG_LONG", "WATCH_LONG"], trigger="LONG_TRIGGER")

    short_table_html = table_tpl.render(
        table_id="short-table",
        rows=short_rows,
        signal_labels=SIGNAL_LABELS,
        trigger_labels=TRIGGER_LABELS,
        trigger_class=TRIGGER_CSS_CLASS,
    )
    long_table_html = table_tpl.render(
        table_id="long-table",
        rows=long_rows,
        signal_labels=SIGNAL_LABELS,
        trigger_labels=TRIGGER_LABELS,
        trigger_class=TRIGGER_CSS_CLASS,
    )

    counts = df["signal"].value_counts().to_dict() if not df.empty else {}
    trig_counts = df["trigger"].value_counts().to_dict() if not df.empty else {}

    html = page_tpl.render(
        generated_date=dt.datetime.now().strftime("%B %d, %Y at %I:%M %p"),
        universe_label=universe_label,
        total_scanned=len(df),
        min_price=f"{min_price:.2f}",
        rsi_overbought=int(rsi_overbought),
        rsi_oversold=int(rsi_oversold),
        rsi_period=rsi_period,
        bb_period=bb_period,
        bb_std=bb_std,
        ma_periods=ma_periods,
        volume_avg_period=volume_avg_period,
        volume_ratio_min=volume_ratio_min,
        vp_lookback=vp_lookback,
        vp_rows=vp_rows,
        vp_tolerance_pct=vp_tolerance_pct,
        full_short_count=int(((df["side"] == "SHORT") & (df["score"] == 4)).sum()) if not df.empty else 0,
        full_long_count=int(((df["side"] == "LONG") & (df["score"] == 4)).sum()) if not df.empty else 0,
        strong_short_count=counts.get("STRONG_SHORT", 0),
        strong_long_count=counts.get("STRONG_LONG", 0),
        watch_count=counts.get("WATCH_SHORT", 0) + counts.get("WATCH_LONG", 0),
        short_trigger_count=trig_counts.get("SHORT_TRIGGER", 0),
        long_trigger_count=trig_counts.get("LONG_TRIGGER", 0),
        short_table=short_table_html,
        long_table=long_table_html,
    )

    Path(output_path).write_text(html, encoding="utf-8")
    return output_path