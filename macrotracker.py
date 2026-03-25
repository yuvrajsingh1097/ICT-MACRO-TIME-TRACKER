"""
macro_tracker.py
──────────────────────────────────────────────────────────────────────────────
ICT Macro Time Window Tracker

ICT Concept:
  "Macros" are specific 20-minute windows inside ICT Kill Zones where the
  algorithm is expected to deliver price — either to sweep liquidity, fill
  a FVG, or set the high/low of the session.

  Official ICT Macro windows (New York time):
  ┌──────────────────────────────────────────────────────┐
  │  Window            Time (NY)        Kill Zone         │
  │  ─────────────────────────────────────────────────── │
  │  London Open 1     02:33 – 03:00    London Open       │
  │  London Open 2     04:03 – 04:30    London Open       │
  │  NY AM Macro 1     08:50 – 09:10    NY Open           │
  │  NY AM Macro 2     09:50 – 10:10    NY Open           │
  │  NY AM Macro 3     10:50 – 11:10    NY Lunch          │
  │  NY Lunch          11:50 – 12:10    NY Lunch          │
  │  NY PM Macro 1     13:10 – 13:40    NY Afternoon      │
  │  NY PM Macro 2     15:15 – 15:45    NY Close          │
  └──────────────────────────────────────────────────────┘

  This tracker:
    1. Fetches 1-minute intraday data (yfinance)
    2. Tags every bar with the macro window it falls in (if any)
    3. Measures the average range, directional move, and volume inside each window
    4. Computes how often a macro window creates the session high or low
    5. Measures move size in ATR multiples for cross-instrument comparison
    6. Plots a comprehensive dark dashboard

Usage:
  python macro_tracker.py                          # default ticker (NQ=F)
  python macro_tracker.py --ticker EURUSD=X        # forex
  python macro_tracker.py --ticker ES=F --days 30  # 30 days of data
  python macro_tracker.py --ticker AAPL --no-chart
  python macro_tracker.py --help
"""

import argparse
import warnings
import sys
from datetime import time, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.collections import PolyCollection

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# ICT MACRO WINDOWS  (all times in New York / Eastern time)
# ─────────────────────────────────────────────────────────────

NY_TZ = ZoneInfo("America/New_York")

MACRO_WINDOWS = [
    {
        "name":      "London Open 1",
        "short":     "LO-1",
        "kill_zone": "London Open",
        "start":     time(2, 33),
        "end":       time(3, 0),
        "color":     "#a78bfa",   # violet
    },
    {
        "name":      "London Open 2",
        "short":     "LO-2",
        "kill_zone": "London Open",
        "start":     time(4, 3),
        "end":       time(4, 30),
        "color":     "#818cf8",   # indigo
    },
    {
        "name":      "NY AM Macro 1",
        "short":     "NY1",
        "kill_zone": "NY Open",
        "start":     time(8, 50),
        "end":       time(9, 10),
        "color":     "#34d399",   # emerald
    },
    {
        "name":      "NY AM Macro 2",
        "short":     "NY2",
        "kill_zone": "NY Open",
        "start":     time(9, 50),
        "end":       time(10, 10),
        "color":     "#6ee7b7",   # light emerald
    },
    {
        "name":      "NY AM Macro 3",
        "short":     "NY3",
        "kill_zone": "NY Lunch",
        "start":     time(10, 50),
        "end":       time(11, 10),
        "color":     "#fbbf24",   # amber
    },
    {
        "name":      "NY Lunch Macro",
        "short":     "Lunch",
        "kill_zone": "NY Lunch",
        "start":     time(11, 50),
        "end":       time(12, 10),
        "color":     "#f59e0b",   # dark amber
    },
    {
        "name":      "NY PM Macro 1",
        "short":     "PM1",
        "kill_zone": "NY Afternoon",
        "start":     time(13, 10),
        "end":       time(13, 40),
        "color":     "#fb923c",   # orange
    },
    {
        "name":      "NY PM Macro 2",
        "short":     "PM2",
        "kill_zone": "NY Close",
        "start":     time(15, 15),
        "end":       time(15, 45),
        "color":     "#f87171",   # red
    },
]

# Kill zone background shading colours (lighter tint)
KILLZONE_COLORS = {
    "London Open":  "#2d1b6b",
    "NY Open":      "#064e3b",
    "NY Lunch":     "#422006",
    "NY Afternoon": "#431407",
    "NY Close":     "#4c0519",
}

# ─────────────────────────────────────────────────────────────
# CONSTANTS & VISUALS
# ─────────────────────────────────────────────────────────────

BG      = "#0d1117"
PANEL   = "#161b22"
GRID_C  = "#21262d"
TEXT_C  = "#e6edf3"
MUTED_C = "#8b949e"
BULL_C  = "#3fb950"
BEAR_C  = "#f85149"
FLAT_C  = "#58a6ff"

SESSION_START = time(0, 0)    # midnight NY — start of new trading day
SESSION_END   = time(23, 59)


# ─────────────────────────────────────────────────────────────
# DATA FETCHING
# ─────────────────────────────────────────────────────────────

def fetch_1min(ticker: str, days: int = 20) -> pd.DataFrame:
    """
    Fetch 1-minute bars. yfinance allows max 7 days at 1m;
    for longer history we fall back to 5-minute bars.
    """
    if days <= 7:
        interval = "1m"
        period   = f"{days}d"
    else:
        interval = "5m"
        period   = f"{min(days, 60)}d"

    df = yf.download(ticker, interval=interval,
                     period=period, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data for '{ticker}'")

    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index = pd.to_datetime(df.index)

    # Localise to New York time
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert(NY_TZ)
    else:
        df.index = df.index.tz_convert(NY_TZ)

    return df


# ─────────────────────────────────────────────────────────────
# MACRO TAGGING
# ─────────────────────────────────────────────────────────────

def tag_macros(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add columns:
      macro_name  : name of the macro window (or NaN)
      macro_short : abbreviated name
      kill_zone   : kill zone name
      macro_color : hex colour for plotting
    """
    df = df.copy()
    for col in ("macro_name", "macro_short", "kill_zone", "macro_color"):
        df[col] = pd.array([None] * len(df), dtype=object)

    bar_times = df.index.time   # array of time objects, fast

    for mw in MACRO_WINDOWS:
        mask = np.array([(t >= mw["start"]) and (t < mw["end"])
                         for t in bar_times])
        df.loc[mask, "macro_name"]  = mw["name"]
        df.loc[mask, "macro_short"] = mw["short"]
        df.loc[mask, "kill_zone"]   = mw["kill_zone"]
        df.loc[mask, "macro_color"] = mw["color"]

    return df


# ─────────────────────────────────────────────────────────────
# SESSION LABELLING
# ─────────────────────────────────────────────────────────────

def add_session_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'session_date' column — the NY calendar date.
    Bars at/after midnight get the current date's session.
    """
    df = df.copy()
    df["session_date"] = df.index.normalize().date
    return df


# ─────────────────────────────────────────────────────────────
# ATR (daily, used for normalisation)
# ─────────────────────────────────────────────────────────────

def daily_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Compute daily ATR from 1-min data by first resampling to daily bars.
    Returns a Series indexed by date for easy lookup.
    """
    daily = df.resample("1D").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    ).dropna()
    h  = daily["High"]; l = daily["Low"]; pc = daily["Close"].shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr_s = tr.ewm(span=period, adjust=False).mean()
    atr_s.index = atr_s.index.date
    return atr_s


# ─────────────────────────────────────────────────────────────
# MACRO STATISTICS
# ─────────────────────────────────────────────────────────────

def compute_macro_stats(df: pd.DataFrame, atr_by_date: pd.Series) -> pd.DataFrame:
    """
    For each macro window × trading day, compute:
      range_pts     : High - Low of bars inside window
      move_pts      : Close_last - Open_first (signed)
      bull_pct      : % of bullish bars inside window
      vol_total     : total volume inside window
      range_atr     : range_pts / daily ATR (normalised)
      is_session_hi : True if window contains the session high
      is_session_lo : True if window contains the session low
    """
    tagged = df[df["macro_name"].notna()].copy()
    if tagged.empty:
        return pd.DataFrame()

    tagged["date_str"] = tagged["session_date"].astype(str)

    records = []
    for (macro_name, date_str), grp in tagged.groupby(
            ["macro_name", "date_str"], sort=False):
        if grp.empty:
            continue

        grp     = grp.sort_index()
        date    = grp["session_date"].iloc[0]

        # Full session bars for that day
        session = df[df["session_date"] == date]
        sess_hi = float(session["High"].max())
        sess_lo = float(session["Low"].min())

        macro_hi  = float(grp["High"].max())
        macro_lo  = float(grp["Low"].min())
        first_open = float(grp["Open"].iloc[0])
        last_close = float(grp["Close"].iloc[-1])

        range_pts = macro_hi - macro_lo
        move_pts  = last_close - first_open
        bull_bars = (grp["Close"] > grp["Open"]).sum()
        bull_pct  = bull_bars / len(grp) * 100
        vol_total = float(grp["Volume"].sum())

        # ATR normalisation
        atr_val = float(atr_by_date.get(date, np.nan))
        range_atr = range_pts / atr_val if atr_val > 0 else np.nan

        is_hi = macro_hi >= sess_hi * 0.9999
        is_lo = macro_lo <= sess_lo * 1.0001

        # Find the macro metadata
        mw_meta = next((m for m in MACRO_WINDOWS
                        if m["name"] == macro_name), {})

        records.append({
            "macro_name":    macro_name,
            "short":         mw_meta.get("short", macro_name[:5]),
            "kill_zone":     mw_meta.get("kill_zone", ""),
            "color":         mw_meta.get("color", "#888"),
            "date":          date,
            "range_pts":     range_pts,
            "move_pts":      move_pts,
            "bull_pct":      bull_pct,
            "vol_total":     vol_total,
            "range_atr":     range_atr,
            "atr_val":       atr_val,
            "is_session_hi": is_hi,
            "is_session_lo": is_lo,
            "n_bars":        len(grp),
            "macro_hi":      macro_hi,
            "macro_lo":      macro_lo,
            "first_open":    first_open,
            "last_close":    last_close,
        })

    stats = pd.DataFrame(records)
    if stats.empty:
        return stats

    stats["direction"] = stats["move_pts"].apply(
        lambda x: "bull" if x > 0 else ("bear" if x < 0 else "flat"))

    return stats


# ─────────────────────────────────────────────────────────────
# AGGREGATE STATS PER MACRO WINDOW
# ─────────────────────────────────────────────────────────────

def aggregate_per_window(stats: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse per-day stats into one row per macro window name.
    """
    if stats.empty:
        return pd.DataFrame()

    rows = []
    for macro_name, grp in stats.groupby("macro_name"):
        n_days = len(grp)
        rows.append({
            "macro_name":       macro_name,
            "short":            grp["short"].iloc[0],
            "kill_zone":        grp["kill_zone"].iloc[0],
            "color":            grp["color"].iloc[0],
            "n_days":           n_days,
            "avg_range_pts":    grp["range_pts"].mean(),
            "avg_range_atr":    grp["range_atr"].mean(),
            "median_range_pts": grp["range_pts"].median(),
            "avg_move_pts":     grp["move_pts"].mean(),
            "avg_bull_pct":     grp["bull_pct"].mean(),
            "bull_day_pct":     (grp["direction"] == "bull").mean() * 100,
            "bear_day_pct":     (grp["direction"] == "bear").mean() * 100,
            "session_hi_pct":   grp["is_session_hi"].mean() * 100,
            "session_lo_pct":   grp["is_session_lo"].mean() * 100,
            "avg_vol":          grp["vol_total"].mean(),
            "std_range_pts":    grp["range_pts"].std(),
        })

    agg = pd.DataFrame(rows)
    # Sort by macro start time
    order = {mw["name"]: i for i, mw in enumerate(MACRO_WINDOWS)}
    agg["_order"] = agg["macro_name"].map(order).fillna(99)
    agg = agg.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    return agg


# ─────────────────────────────────────────────────────────────
# CONSOLE REPORT
# ─────────────────────────────────────────────────────────────

def print_report(ticker: str, agg: pd.DataFrame, n_days: int):
    W = 82
    print()
    print("╔" + "═" * (W - 2) + "╗")
    print(f"║  ICT Macro Tracker  ·  {ticker}  ·  {n_days} trading days{'':<{W - 46}}║")
    print("╠" + "═" * (W - 2) + "╣")
    print(f"║  {'Window':<16} {'Kill Zone':<14} {'Days':>5}  "
          f"{'AvgRange':>9}  {'RangeATR':>9}  "
          f"{'Bull%':>6}  {'SessHi%':>8}  {'SessLo%':>8}  ║")
    print("║" + "─" * (W - 2) + "║")

    for _, row in agg.iterrows():
        print(
            f"║  {row['macro_name']:<16} {row['kill_zone']:<14} "
            f"{row['n_days']:>5}  "
            f"{row['avg_range_pts']:>9.4f}  "
            f"{row['avg_range_atr']:>9.3f}  "
            f"{row['avg_bull_pct']:>5.1f}%  "
            f"{row['session_hi_pct']:>7.1f}%  "
            f"{row['session_lo_pct']:>7.1f}%  ║"
        )

    print("╠" + "═" * (W - 2) + "╣")

    # Best window by range
    best_range = agg.loc[agg["avg_range_atr"].idxmax()]
    best_hi    = agg.loc[agg["session_hi_pct"].idxmax()]
    best_lo    = agg.loc[agg["session_lo_pct"].idxmax()]

    print(f"║  Most active (ATR range)  : "
          f"{best_range['macro_name']:<16} "
          f"avg {best_range['avg_range_atr']:.3f}× ATR"
          f"{'':>{W - 63}}║")
    print(f"║  Most likely session HIGH : "
          f"{best_hi['macro_name']:<16} "
          f"{best_hi['session_hi_pct']:.1f}% of days"
          f"{'':>{W - 63}}║")
    print(f"║  Most likely session LOW  : "
          f"{best_lo['macro_name']:<16} "
          f"{best_lo['session_lo_pct']:.1f}% of days"
          f"{'':>{W - 63}}║")
    print("╚" + "═" * (W - 2) + "╝")
    print()


# ─────────────────────────────────────────────────────────────
# CHARTING
# ─────────────────────────────────────────────────────────────

def _sax(ax, title=""):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID_C)
    ax.grid(color=GRID_C, lw=0.4, ls="--", alpha=0.6)
    ax.tick_params(colors=MUTED_C, labelsize=7.5)
    ax.xaxis.label.set_color(MUTED_C)
    ax.yaxis.label.set_color(MUTED_C)
    if title:
        ax.set_title(title, color=TEXT_C, fontsize=9, pad=6)


def plot_dashboard(df: pd.DataFrame, stats: pd.DataFrame,
                   agg: pd.DataFrame, ticker: str,
                   output_path: str = "macro_tracker.png"):
    """
    5-panel dashboard:
      Panel A  (top-left,  tall) : intraday price chart of the most recent
                                   trading day with macro windows shaded
      Panel B  (top-right)       : average range per macro window (ATR-normalised)
      Panel C  (mid-right)       : session high/low capture rate per window
      Panel D  (bot-right)       : bull vs bear day split per window
      Panel E  (bottom, wide)    : range distribution violin/box per window
                                   across all sampled days
    """
    if agg.empty:
        print("  No aggregated data — skipping chart.")
        return

    fig = plt.figure(figsize=(18, 13), facecolor=BG)
    gs  = gridspec.GridSpec(
        3, 2, figure=fig,
        width_ratios=[1.45, 1],
        height_ratios=[2.2, 1.3, 1.3],
        hspace=0.48, wspace=0.28
    )

    short_labels = agg["short"].tolist()
    colors       = agg["color"].tolist()
    x_pos        = np.arange(len(agg))

    # ── Panel A: most recent day intraday chart ──────────────────
    ax_price = fig.add_subplot(gs[:, 0])
    _sax(ax_price)

    # Find most recent full trading day with macro data
    days_with_macros = stats["date"].unique()
    if len(days_with_macros) > 0:
        plot_date = sorted(days_with_macros)[-1]
        day_df    = df[df["session_date"] == plot_date].copy()
    else:
        day_df = df.tail(400).copy()
        plot_date = day_df["session_date"].iloc[-1] if "session_date" in day_df.columns else "recent"

    if day_df.empty:
        ax_price.text(0.5, 0.5, "No intraday data",
                      transform=ax_price.transAxes,
                      ha="center", va="center",
                      color=MUTED_C, fontsize=10)
    else:
        # Shade kill zones first (background)
        bar_times_dt = day_df.index
        xs           = np.arange(len(day_df))

        prev_kz    = None
        kz_start_x = 0
        for xi, ts in enumerate(bar_times_dt):
            t  = ts.time()
            kz = None
            for mw in MACRO_WINDOWS:
                if mw["start"] <= t < mw["end"]:
                    kz = mw["kill_zone"]
                    break
            if kz != prev_kz:
                if prev_kz is not None:
                    shade = KILLZONE_COLORS.get(prev_kz, PANEL)
                    ax_price.axvspan(kz_start_x - 0.5, xi - 0.5,
                                     facecolor=shade, alpha=0.25, lw=0)
                kz_start_x = xi
                prev_kz    = kz

        # Draw candles
        opens  = day_df["Open"].values
        highs  = day_df["High"].values
        lows   = day_df["Low"].values
        closes = day_df["Close"].values

        for i, (o, h, l, c) in enumerate(zip(opens, highs, lows, closes)):
            col = BULL_C if c >= o else BEAR_C
            ax_price.plot([i, i], [l, h], color=col, lw=0.55, alpha=0.8)
            blo = min(o, c); bhi = max(o, c)
            ax_price.add_patch(mpatches.FancyBboxPatch(
                (i - 0.3, blo), 0.6, max(bhi - blo, 1e-8),
                boxstyle="square,pad=0",
                fc=col, ec=col, lw=0, alpha=0.85
            ))

        # Shade macro windows and label them
        bar_times_arr = [ts.time() for ts in bar_times_dt]
        n_plot        = len(day_df)
        pr            = float(np.max(highs)) - float(np.min(lows))

        for mw in MACRO_WINDOWS:
            mask_indices = [i for i, t in enumerate(bar_times_arr)
                            if mw["start"] <= t < mw["end"]]
            if not mask_indices:
                continue
            x0 = mask_indices[0]  - 0.5
            x1 = mask_indices[-1] + 0.5
            ax_price.axvspan(x0, x1,
                             facecolor=mw["color"], alpha=0.18, lw=0)
            ax_price.axvline(x0, color=mw["color"],
                             lw=0.6, ls="--", alpha=0.5)

            # Label at the top of the span
            mid_x = (x0 + x1) / 2
            y_lbl = float(np.max(highs)) + pr * 0.025
            ax_price.text(mid_x, y_lbl, mw["short"],
                          color=mw["color"], fontsize=6.5,
                          ha="center", va="bottom",
                          fontweight="bold", alpha=0.9)

        # X-axis: show time labels every 30 minutes
        tick_step = max(1, len(day_df) // 12)
        ax_price.set_xticks(xs[::tick_step])
        ax_price.set_xticklabels(
            [bar_times_arr[i].strftime("%H:%M")
             for i in range(0, len(day_df), tick_step)],
            rotation=30, ha="right", fontsize=7
        )
        ax_price.set_xlim(-1, n_plot + 1)
        ax_price.set_ylim(
            float(np.min(lows))  - pr * 0.04,
            float(np.max(highs)) + pr * 0.10
        )
        ax_price.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x:,.2f}"))
        ax_price.set_title(
            f"{ticker}  ·  {plot_date}  ·  Macro Windows Highlighted  (NY time)",
            color=TEXT_C, fontsize=10, pad=8
        )

    # ── Panel B: avg range (ATR-normalised) ─────────────────────
    ax_range = fig.add_subplot(gs[0, 1])
    _sax(ax_range, "Avg Range per Macro Window (× daily ATR)")

    atr_vals = agg["avg_range_atr"].fillna(0).values
    bars     = ax_range.bar(x_pos, atr_vals, color=colors,
                            alpha=0.85, width=0.65, edgecolor=BG, lw=0.5)

    for bar, val in zip(bars, atr_vals):
        if val > 0.005:
            ax_range.text(
                bar.get_x() + bar.get_width() / 2,
                val + atr_vals.max() * 0.02,
                f"{val:.3f}×",
                ha="center", va="bottom",
                fontsize=7, color=TEXT_C
            )

    ax_range.set_xticks(x_pos)
    ax_range.set_xticklabels(short_labels, fontsize=7.5)
    ax_range.set_ylabel("ATR multiple", fontsize=8)
    ax_range.axhline(atr_vals.mean(), color=MUTED_C,
                     lw=0.8, ls=":", alpha=0.6,
                     label=f"Mean {atr_vals.mean():.3f}×")
    ax_range.legend(fontsize=7, facecolor=PANEL,
                    edgecolor=GRID_C, labelcolor=TEXT_C)

    # ── Panel C: session hi/lo capture rate ──────────────────────
    ax_hl = fig.add_subplot(gs[1, 1])
    _sax(ax_hl, "Session High / Low Capture Rate (%)")

    w  = 0.35
    xp = np.arange(len(agg))
    hi_vals = agg["session_hi_pct"].values
    lo_vals = agg["session_lo_pct"].values

    bars_hi = ax_hl.bar(xp - w/2, hi_vals, width=w,
                        color=BULL_C, alpha=0.80, label="Session High",
                        edgecolor=BG, lw=0.5)
    bars_lo = ax_hl.bar(xp + w/2, lo_vals, width=w,
                        color=BEAR_C, alpha=0.80, label="Session Low",
                        edgecolor=BG, lw=0.5)

    for bars in [bars_hi, bars_lo]:
        for bar in bars:
            h = bar.get_height()
            if h > 3:
                ax_hl.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 1.2, f"{h:.0f}%",
                    ha="center", va="bottom",
                    fontsize=6.5, color=TEXT_C
                )

    ax_hl.set_xticks(xp)
    ax_hl.set_xticklabels(short_labels, fontsize=7.5)
    ax_hl.set_ylim(0, max(np.max(hi_vals), np.max(lo_vals)) * 1.18 + 5)
    ax_hl.set_ylabel("% of trading days", fontsize=8)
    ax_hl.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax_hl.legend(fontsize=7.5, facecolor=PANEL,
                 edgecolor=GRID_C, labelcolor=TEXT_C)

    # ── Panel D: bull vs bear day split ──────────────────────────
    ax_bb = fig.add_subplot(gs[2, 1])
    _sax(ax_bb, "Bull vs Bear Days per Window (%)")

    bull_d = agg["bull_day_pct"].values
    bear_d = agg["bear_day_pct"].values

    ax_bb.bar(xp, bull_d,   color=BULL_C, alpha=0.80,
              width=0.65, label="Bull day", edgecolor=BG, lw=0.5)
    ax_bb.bar(xp, -bear_d,  color=BEAR_C, alpha=0.80,
              width=0.65, label="Bear day", edgecolor=BG, lw=0.5,
              bottom=0)

    ax_bb.axhline(0, color=MUTED_C, lw=0.7, ls="-", alpha=0.5)
    ax_bb.axhline(50,  color=MUTED_C, lw=0.5, ls=":", alpha=0.4)
    ax_bb.axhline(-50, color=MUTED_C, lw=0.5, ls=":", alpha=0.4)

    for i, (bv, be) in enumerate(zip(bull_d, bear_d)):
        if bv > 5:
            ax_bb.text(i, bv + 1.5, f"{bv:.0f}%",
                       ha="center", va="bottom",
                       fontsize=6.5, color=BULL_C)
        if be > 5:
            ax_bb.text(i, -be - 1.5, f"{be:.0f}%",
                       ha="center", va="top",
                       fontsize=6.5, color=BEAR_C)

    ax_bb.set_xticks(xp)
    ax_bb.set_xticklabels(short_labels, fontsize=7.5)
    ax_bb.set_ylabel("% of days", fontsize=8)
    ax_bb.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{abs(x):.0f}%"))
    ax_bb.legend(fontsize=7.5, facecolor=PANEL,
                 edgecolor=GRID_C, labelcolor=TEXT_C)

    # ── Macro window legend strip ────────────────────────────────
    legend_ax = fig.add_axes([0.01, 0.003, 0.98, 0.020],
                             facecolor="#0d1117")
    legend_ax.axis("off")
    legend_patches = [
        mpatches.Patch(fc=mw["color"], ec="none",
                       label=f"{mw['short']}: {mw['start'].strftime('%H:%M')}–"
                             f"{mw['end'].strftime('%H:%M')} NY")
        for mw in MACRO_WINDOWS
    ]
    legend_ax.legend(
        handles=legend_patches,
        ncol=len(MACRO_WINDOWS),
        loc="center",
        fontsize=6.8,
        facecolor=PANEL, edgecolor=GRID_C, labelcolor=TEXT_C,
        framealpha=0.9, handlelength=1.2,
        columnspacing=0.8
    )

    fig.suptitle(
        f"ICT Macro Window Tracker  ·  {ticker}  ·  New York Time",
        color=TEXT_C, fontsize=13, y=1.005, fontweight="bold"
    )

    plt.savefig(output_path, dpi=145, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    print(f"  Chart saved → {output_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────
# CSV EXPORT
# ─────────────────────────────────────────────────────────────

def export_csv(stats: pd.DataFrame, agg: pd.DataFrame,
               ticker: str):
    safe = ticker.replace("=", "").replace("/", "").replace("^", "")

    # Daily per-window stats
    out_stats = stats.drop(columns=["color"], errors="ignore")
    p1 = f"macro_daily_{safe}.csv"
    out_stats.to_csv(p1, index=False)
    print(f"  CSV saved → {p1}  ({len(out_stats)} rows)")

    # Aggregated summary
    out_agg = agg.drop(columns=["color"], errors="ignore")
    p2 = f"macro_summary_{safe}.csv"
    out_agg.to_csv(p2, index=False)
    print(f"  CSV saved → {p2}  ({len(out_agg)} rows)")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ICT Macro Time Window Tracker"
    )
    parser.add_argument("--ticker", type=str, default="NQ=F",
                        help="Ticker symbol (default: NQ=F)")
    parser.add_argument("--days",   type=int, default=20,
                        help="Trading days of history (default: 20; "
                             "max 7 for 1m data, up to 60 for 5m)")
    parser.add_argument("--no-chart", action="store_true",
                        help="Skip chart generation")
    parser.add_argument("--no-csv",   action="store_true",
                        help="Skip CSV export")
    parser.add_argument("--output",   type=str,
                        default=None,
                        help="Chart filename (default: macro_tracker_<TICKER>.png)")
    args = parser.parse_args()

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║             ICT Macro Time Window Tracker               ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Ticker : {args.ticker}")
    print(f"  Days   : {args.days}")
    print(f"  Mode   : {'1-min' if args.days <= 7 else '5-min'} bars")
    print()

    # 1. Fetch
    print(f"  Fetching data ...", end=" ", flush=True)
    try:
        df = fetch_1min(args.ticker, args.days)
    except Exception as e:
        print(f"\n  Error: {e}")
        sys.exit(1)
    print(f"{len(df):,} bars ({df.index[0].date()} → {df.index[-1].date()}) ✓")

    # 2. Tag
    df = tag_macros(df)
    df = add_session_date(df)
    n_macro_bars = df["macro_name"].notna().sum()
    print(f"  Tagged  : {n_macro_bars:,} bars inside macro windows")

    # 3. ATR
    atr_by_date = daily_atr(df)

    # 4. Stats
    stats = compute_macro_stats(df, atr_by_date)
    agg   = aggregate_per_window(stats)
    n_days = df["session_date"].nunique()
    print(f"  Days    : {n_days}  |  Windows analysed: {len(agg)}")

    # 5. Report
    print_report(args.ticker, agg, n_days)

    # 6. Chart
    if not args.no_chart:
        safe   = args.ticker.replace("=", "").replace("/", "").replace("^", "")
        output = args.output or f"macro_tracker_{safe}.png"
        plot_dashboard(df, stats, agg, args.ticker, output_path=output)

    # 7. CSV
    if not args.no_csv and not stats.empty:
        export_csv(stats, agg, args.ticker)

    print("  Done.\n")


if __name__ == "__main__":
    main()