#!/usr/bin/env python3
"""
Signal Performance Analyzer
============================
Reads the bot's signals.db SQLite database and produces detailed
performance analytics with visualizations.

Usage:
    python analyze_signals.py                  # default: signals.db in same dir
    python analyze_signals.py /path/to/signals.db
    python analyze_signals.py signals.db --days 60

Outputs:
    - Console summary of all key metrics
    - signal_performance.png  — multi-panel chart saved to disk
"""

import argparse
import sqlite3
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

# ── Helpers ──────────────────────────────────────────────────────────────────

def connect(db_path: str) -> sqlite3.Connection:
    path = Path(db_path)
    if not path.exists():
        print(f"ERROR: database not found at {path.resolve()}")
        sys.exit(1)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def fetch_resolved(conn: sqlite3.Connection, days: int) -> list[dict]:
    """Fetch all resolved signals with their outcomes within the given window."""
    rows = conn.execute(f"""
        SELECT
            s.id, s.asset, s.timeframe, s.direction,
            s.entry, s.sl, s.tp1, s.tp2, s.tp3,
            s.market_session, s.session_detail, s.created_at, s.source,
            o.tp1_hit, o.tp2_hit, o.tp3_hit, o.sl_hit,
            o.max_favorable, o.max_adverse, o.pnl_percent,
            o.exit_reason, o.candles_to_exit
        FROM signals s
        JOIN outcomes o ON o.signal_id = s.id
        WHERE o.final = 1
          AND s.created_at > datetime('now', '-{days} days')
        ORDER BY s.created_at
    """).fetchall()
    return [dict(r) for r in rows]


# ── Console Report ───────────────────────────────────────────────────────────

def print_report(signals: list[dict], days: int) -> None:
    total = len(signals)
    if total == 0:
        print("No resolved signals found. The bot needs to run and accumulate data first.")
        return

    wins = sum(1 for s in signals if s["tp1_hit"])
    tp2_hits = sum(1 for s in signals if s["tp2_hit"])
    tp3_hits = sum(1 for s in signals if s["tp3_hit"])
    sl_hits = sum(1 for s in signals if s["sl_hit"] and not s["tp1_hit"])
    avg_pnl = np.mean([s["pnl_percent"] for s in signals])
    avg_fav = np.mean([s["max_favorable"] for s in signals if s["max_favorable"] is not None])
    avg_adv = np.mean([s["max_adverse"] for s in signals if s["max_adverse"] is not None])
    avg_candles = np.mean([s["candles_to_exit"] for s in signals if s["candles_to_exit"] is not None])

    print("=" * 65)
    print(f"  SIGNAL PERFORMANCE REPORT  (last {days} days)")
    print("=" * 65)
    print(f"  Total resolved signals : {total}")
    print(f"  Win rate (TP1+)        : {wins/total*100:.1f}%  ({wins}W / {total - wins}L)")
    print(f"  TP1 hits               : {wins}")
    print(f"  TP2 hits               : {tp2_hits}")
    print(f"  TP3 hits               : {tp3_hits}")
    print(f"  SL hits (pure losses)  : {sl_hits}")
    print(f"  Avg P&L                : {avg_pnl:+.3f}%")
    print(f"  Avg max favorable move : {avg_fav:.3f}%")
    print(f"  Avg max adverse move   : {avg_adv:.3f}%")
    print(f"  Avg candles to exit    : {avg_candles:.1f}")
    print()

    # By market session
    sessions: dict[str, list[dict]] = {}
    for s in signals:
        sess = s["market_session"] or "unknown"
        sessions.setdefault(sess, []).append(s)

    print("  BY MARKET SESSION:")
    print(f"  {'Session':<20} {'WR':>6} {'W':>4} {'L':>4} {'Avg P&L':>9} {'Avg Fav':>9} {'Avg Adv':>9}")
    print("  " + "-" * 63)
    for sess in sorted(sessions, key=lambda k: -len(sessions[k])):
        group = sessions[sess]
        w = sum(1 for s in group if s["tp1_hit"])
        l = sum(1 for s in group if s["sl_hit"] and not s["tp1_hit"])
        wr = w / len(group) * 100 if group else 0
        ap = np.mean([s["pnl_percent"] for s in group])
        af = np.mean([s["max_favorable"] for s in group if s["max_favorable"] is not None]) if any(s["max_favorable"] is not None for s in group) else 0
        aa = np.mean([s["max_adverse"] for s in group if s["max_adverse"] is not None]) if any(s["max_adverse"] is not None for s in group) else 0
        print(f"  {sess:<20} {wr:>5.1f}% {w:>4} {l:>4} {ap:>+8.3f}% {af:>8.3f}% {aa:>8.3f}%")
    print()

    # By direction
    for direction in ("LONG", "SHORT"):
        group = [s for s in signals if (s["direction"] or "").upper() == direction]
        if not group:
            continue
        w = sum(1 for s in group if s["tp1_hit"])
        wr = w / len(group) * 100
        ap = np.mean([s["pnl_percent"] for s in group])
        print(f"  {direction}: {wr:.1f}% WR ({w}W/{len(group)-w}L), Avg P&L: {ap:+.3f}%")
    print()

    # By exit reason
    reasons: dict[str, int] = {}
    for s in signals:
        r = s["exit_reason"] or "unknown"
        reasons[r] = reasons.get(r, 0) + 1
    print("  EXIT REASON BREAKDOWN:")
    for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"    {reason:<12}: {count:>4}  ({count/total*100:.1f}%)")
    print()

    # Best/worst conditions
    print("  TOP PERFORMING CONDITIONS (by win rate, min 5 signals):")
    _print_top_conditions(signals, "market_session", "Session")
    _print_top_conditions(signals, "direction", "Direction")
    _print_top_conditions(signals, "source", "Source")


def _print_top_conditions(signals: list[dict], key: str, label: str) -> None:
    groups: dict[str, list[dict]] = {}
    for s in signals:
        val = s.get(key) or "unknown"
        groups.setdefault(val, []).append(s)

    ranked = []
    for val, group in groups.items():
        if len(group) < 3:  # lowered minimum for smaller datasets
            continue
        w = sum(1 for s in group if s["tp1_hit"])
        wr = w / len(group) * 100
        ap = np.mean([s["pnl_percent"] for s in group])
        ranked.append((val, wr, ap, len(group)))

    ranked.sort(key=lambda x: (-x[1], -x[2]))
    for val, wr, ap, n in ranked[:5]:
        print(f"    {label}: {val:<20} WR: {wr:.1f}%  Avg P&L: {ap:+.3f}%  (n={n})")


# ── Chart Generation ─────────────────────────────────────────────────────────

def parse_dates(signals: list[dict]):
    """Parse created_at strings into matplotlib-compatible dates."""
    from datetime import datetime
    dates = []
    for s in signals:
        try:
            dt = datetime.fromisoformat(s["created_at"].replace("Z", "+00:00"))
            dates.append(dt)
        except (ValueError, TypeError):
            dates.append(None)
    return dates


def generate_charts(signals: list[dict], days: int, output_path: str = "signal_performance.png") -> None:
    if not signals:
        print("No data to plot.")
        return

    dates = parse_dates(signals)
    valid = [(d, s) for d, s in zip(dates, signals) if d is not None]
    if not valid:
        print("No valid dates to plot.")
        return

    dates_clean, signals_clean = zip(*valid)
    dates_clean = list(dates_clean)
    signals_clean = list(signals_clean)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor="#1a1a2e")
    fig.suptitle(f"Signal Performance Analysis (last {days} days)",
                 color="white", fontsize=16, fontweight="bold", y=0.98)

    for ax in axes.flat:
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_color("#2a2a4a")

    # ── Panel 1: Cumulative Win Rate Over Time ────────────────────────────
    ax1 = axes[0, 0]
    cumulative_wr = []
    running_wins = 0
    for i, s in enumerate(signals_clean):
        if s["tp1_hit"]:
            running_wins += 1
        cumulative_wr.append(running_wins / (i + 1) * 100)

    ax1.plot(dates_clean, cumulative_wr, color="#26a69a", linewidth=2, label="Cumulative WR")
    ax1.axhline(y=50, color="#ef5350", linestyle="--", alpha=0.5, label="50% baseline")
    if cumulative_wr:
        ax1.axhline(y=cumulative_wr[-1], color="#f0b90b", linestyle=":", alpha=0.7,
                     label=f"Current: {cumulative_wr[-1]:.1f}%")
    ax1.set_title("Cumulative Win Rate Over Time", fontweight="bold")
    ax1.set_ylabel("Win Rate %")
    ax1.legend(facecolor="#1a1a2e", edgecolor="#2a2a4a", labelcolor="white", fontsize=8)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax1.tick_params(axis="x", rotation=45)
    ax1.grid(True, alpha=0.2, color="#2a2a4a")

    # ── Panel 2: P&L Distribution ─────────────────────────────────────────
    ax2 = axes[0, 1]
    pnl_values = [s["pnl_percent"] for s in signals_clean if s["pnl_percent"] is not None]
    if pnl_values:
        colors = ["#26a69a" if p >= 0 else "#ef5350" for p in pnl_values]
        ax2.hist(pnl_values, bins=min(30, max(10, len(pnl_values) // 5)),
                 color="#42a5f5", edgecolor="#1a1a2e", alpha=0.8)
        avg_pnl = np.mean(pnl_values)
        ax2.axvline(x=avg_pnl, color="#f0b90b", linestyle="--", linewidth=2,
                     label=f"Avg: {avg_pnl:+.3f}%")
        ax2.axvline(x=0, color="#ffffff", linestyle="-", linewidth=1, alpha=0.3)
    ax2.set_title("P&L Distribution", fontweight="bold")
    ax2.set_xlabel("P&L %")
    ax2.set_ylabel("Count")
    ax2.legend(facecolor="#1a1a2e", edgecolor="#2a2a4a", labelcolor="white", fontsize=8)
    ax2.grid(True, alpha=0.2, color="#2a2a4a")

    # ── Panel 3: Win Rate by Market Session ───────────────────────────────
    ax3 = axes[1, 0]
    sessions: dict[str, list[dict]] = {}
    for s in signals_clean:
        sess = s["market_session"] or "unknown"
        sessions.setdefault(sess, []).append(s)

    sess_names = []
    sess_wr = []
    sess_counts = []
    sess_pnl = []
    for sess in sorted(sessions, key=lambda k: -len(sessions[k])):
        group = sessions[sess]
        if len(group) < 2:
            continue
        w = sum(1 for s in group if s["tp1_hit"])
        sess_names.append(sess)
        sess_wr.append(w / len(group) * 100)
        sess_counts.append(len(group))
        sess_pnl.append(np.mean([s["pnl_percent"] for s in group]))

    if sess_names:
        bar_colors = ["#26a69a" if wr >= 50 else "#ef5350" for wr in sess_wr]
        bars = ax3.barh(sess_names, sess_wr, color=bar_colors, edgecolor="#1a1a2e", height=0.6)
        ax3.axvline(x=50, color="#ffffff", linestyle="--", alpha=0.3)
        # Add count labels
        for bar, count, pnl in zip(bars, sess_counts, sess_pnl):
            ax3.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                     f" n={count}, P&L:{pnl:+.2f}%",
                     va="center", color="white", fontsize=8)
    ax3.set_title("Win Rate by Market Session", fontweight="bold")
    ax3.set_xlabel("Win Rate %")
    ax3.set_xlim(0, 100)
    ax3.grid(True, alpha=0.2, color="#2a2a4a", axis="x")

    # ── Panel 4: Cumulative P&L Over Time ─────────────────────────────────
    ax4 = axes[1, 1]
    cumulative_pnl = []
    running_pnl = 0
    for s in signals_clean:
        pnl = s["pnl_percent"] or 0
        running_pnl += pnl
        cumulative_pnl.append(running_pnl)

    fill_color_pos = "#26a69a40"
    fill_color_neg = "#ef535040"

    ax4.plot(dates_clean, cumulative_pnl, color="#42a5f5", linewidth=2)
    ax4.fill_between(dates_clean, cumulative_pnl, 0,
                      where=[p >= 0 for p in cumulative_pnl],
                      color="#26a69a", alpha=0.2)
    ax4.fill_between(dates_clean, cumulative_pnl, 0,
                      where=[p < 0 for p in cumulative_pnl],
                      color="#ef5350", alpha=0.2)
    ax4.axhline(y=0, color="#ffffff", linestyle="-", alpha=0.3)
    if cumulative_pnl:
        ax4.annotate(f"{cumulative_pnl[-1]:+.2f}%",
                      xy=(dates_clean[-1], cumulative_pnl[-1]),
                      xytext=(10, 10), textcoords="offset points",
                      color="#f0b90b", fontweight="bold", fontsize=10)
    ax4.set_title("Cumulative P&L Over Time", fontweight="bold")
    ax4.set_ylabel("Cumulative P&L %")
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax4.tick_params(axis="x", rotation=45)
    ax4.grid(True, alpha=0.2, color="#2a2a4a")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor="#1a1a2e", edgecolor="none")
    plt.close(fig)
    print(f"Chart saved to: {output_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyze trading bot signal performance")
    parser.add_argument("db", nargs="?", default="signals.db",
                        help="Path to signals.db (default: signals.db)")
    parser.add_argument("--days", type=int, default=30,
                        help="Look-back window in days (default: 30)")
    parser.add_argument("--output", "-o", default="signal_performance.png",
                        help="Output chart filename (default: signal_performance.png)")
    args = parser.parse_args()

    conn = connect(args.db)
    signals = fetch_resolved(conn, args.days)
    conn.close()

    print_report(signals, args.days)
    generate_charts(signals, args.days, args.output)


if __name__ == "__main__":
    main()
