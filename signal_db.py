"""SQLite database for signal tracking, outcomes, and self-learning."""

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent / "signals.db"

_conn: sqlite3.Connection | None = None


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.execute("PRAGMA foreign_keys=ON")
    return _conn


def init_db() -> None:
    """Create tables if they don't exist."""
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS signals (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id         INTEGER NOT NULL,
            asset           TEXT NOT NULL,
            timeframe       TEXT NOT NULL,
            direction       TEXT,            -- LONG / SHORT
            entry           REAL,
            sl              REAL,
            tp1             REAL,
            tp2             REAL,
            tp3             REAL,
            market_session  TEXT,            -- e.g. "us_market_open", "cme_open", "london", "asia"
            session_detail  TEXT,            -- JSON with session info at signal time
            analysis_text   TEXT,            -- full Claude response (for learning)
            created_at      TEXT NOT NULL,   -- ISO-8601 UTC
            source          TEXT DEFAULT 'autosignal'  -- autosignal / manual
        );

        CREATE TABLE IF NOT EXISTS outcomes (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_id       INTEGER NOT NULL UNIQUE REFERENCES signals(id),
            price_at_check  REAL,
            tp1_hit         INTEGER DEFAULT 0,   -- boolean
            tp2_hit         INTEGER DEFAULT 0,
            tp3_hit         INTEGER DEFAULT 0,
            sl_hit          INTEGER DEFAULT 0,
            max_favorable   REAL,            -- max price move in trade direction
            max_adverse     REAL,            -- max price move against trade direction
            pnl_percent     REAL,            -- realized P&L % (to best TP or SL)
            exit_reason     TEXT,            -- tp1/tp2/tp3/sl/timeout/open
            candles_to_exit INTEGER,         -- how many candles until resolution
            checked_at      TEXT NOT NULL,
            final           INTEGER DEFAULT 0 -- 1 = fully resolved
        );

        CREATE TABLE IF NOT EXISTS autosignal_subs (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id         INTEGER NOT NULL,
            asset           TEXT NOT NULL,
            timeframe       TEXT NOT NULL,
            active          INTEGER DEFAULT 1,
            created_at      TEXT NOT NULL,
            UNIQUE(chat_id, asset, timeframe)
        );

        CREATE INDEX IF NOT EXISTS idx_signals_asset_tf
            ON signals(asset, timeframe);
        CREATE INDEX IF NOT EXISTS idx_signals_session
            ON signals(market_session);
        CREATE INDEX IF NOT EXISTS idx_signals_created
            ON signals(created_at);
        CREATE INDEX IF NOT EXISTS idx_outcomes_final
            ON outcomes(final);
    """)
    conn.commit()
    logger.info("Signal database initialized at %s", DB_PATH)


# ── Signal CRUD ──────────────────────────────────────────────────────────────


def record_signal(
    chat_id: int,
    asset: str,
    timeframe: str,
    direction: str | None,
    entry: float | None,
    sl: float | None,
    tp1: float | None,
    tp2: float | None,
    tp3: float | None,
    market_session: str,
    session_detail: dict,
    analysis_text: str,
    source: str = "autosignal",
) -> int:
    """Store a new signal. Returns the signal ID."""
    conn = _get_conn()
    cur = conn.execute(
        """INSERT INTO signals
           (chat_id, asset, timeframe, direction, entry, sl, tp1, tp2, tp3,
            market_session, session_detail, analysis_text, created_at, source)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            chat_id, asset, timeframe, direction, entry, sl, tp1, tp2, tp3,
            market_session, json.dumps(session_detail), analysis_text,
            datetime.now(timezone.utc).isoformat(), source,
        ),
    )
    conn.commit()
    signal_id = cur.lastrowid
    logger.info("Recorded signal #%d: %s %s %s", signal_id, asset, timeframe, direction)
    return signal_id


def get_pending_outcomes() -> list[dict]:
    """Get signals that haven't been fully resolved yet."""
    conn = _get_conn()
    rows = conn.execute("""
        SELECT s.id, s.asset, s.timeframe, s.direction, s.entry, s.sl,
               s.tp1, s.tp2, s.tp3, s.created_at,
               o.tp1_hit, o.tp2_hit, o.tp3_hit, o.sl_hit
        FROM signals s
        LEFT JOIN outcomes o ON o.signal_id = s.id
        WHERE s.entry IS NOT NULL
          AND s.direction IS NOT NULL
          AND (o.id IS NULL OR o.final = 0)
          AND s.created_at > datetime('now', '-7 days')
        ORDER BY s.created_at
    """).fetchall()
    return [dict(r) for r in rows]


def upsert_outcome(
    signal_id: int,
    price_at_check: float,
    tp1_hit: bool,
    tp2_hit: bool,
    tp3_hit: bool,
    sl_hit: bool,
    max_favorable: float,
    max_adverse: float,
    pnl_percent: float,
    exit_reason: str,
    candles_to_exit: int,
    final: bool,
) -> None:
    """Insert or update an outcome for a signal."""
    conn = _get_conn()
    conn.execute(
        """INSERT INTO outcomes
           (signal_id, price_at_check, tp1_hit, tp2_hit, tp3_hit, sl_hit,
            max_favorable, max_adverse, pnl_percent, exit_reason,
            candles_to_exit, checked_at, final)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(signal_id) DO UPDATE SET
            price_at_check=excluded.price_at_check,
            tp1_hit=excluded.tp1_hit, tp2_hit=excluded.tp2_hit,
            tp3_hit=excluded.tp3_hit, sl_hit=excluded.sl_hit,
            max_favorable=excluded.max_favorable,
            max_adverse=excluded.max_adverse,
            pnl_percent=excluded.pnl_percent,
            exit_reason=excluded.exit_reason,
            candles_to_exit=excluded.candles_to_exit,
            checked_at=excluded.checked_at,
            final=excluded.final
        """,
        (
            signal_id, price_at_check, int(tp1_hit), int(tp2_hit),
            int(tp3_hit), int(sl_hit), max_favorable, max_adverse,
            pnl_percent, exit_reason, candles_to_exit,
            datetime.now(timezone.utc).isoformat(), int(final),
        ),
    )
    conn.commit()


# ── Performance queries ──────────────────────────────────────────────────────


def get_performance_summary(
    asset: str | None = None,
    timeframe: str | None = None,
    days: int = 30,
) -> dict:
    """Aggregate win/loss stats for the learning engine."""
    conn = _get_conn()

    where = ["o.final = 1", f"s.created_at > datetime('now', '-{days} days')"]
    params: list = []
    if asset:
        where.append("s.asset = ?")
        params.append(asset)
    if timeframe:
        where.append("s.timeframe = ?")
        params.append(timeframe)

    where_clause = " AND ".join(where)

    row = conn.execute(f"""
        SELECT
            COUNT(*)                                            AS total_signals,
            SUM(CASE WHEN o.tp1_hit THEN 1 ELSE 0 END)        AS tp1_wins,
            SUM(CASE WHEN o.tp2_hit THEN 1 ELSE 0 END)        AS tp2_wins,
            SUM(CASE WHEN o.tp3_hit THEN 1 ELSE 0 END)        AS tp3_wins,
            SUM(CASE WHEN o.sl_hit THEN 1 ELSE 0 END)         AS sl_losses,
            AVG(o.pnl_percent)                                 AS avg_pnl,
            AVG(o.max_favorable)                               AS avg_max_favorable,
            AVG(o.max_adverse)                                 AS avg_max_adverse,
            AVG(o.candles_to_exit)                             AS avg_candles_to_exit
        FROM signals s
        JOIN outcomes o ON o.signal_id = s.id
        WHERE {where_clause}
    """, params).fetchone()

    return dict(row) if row else {}


def get_session_performance(days: int = 30) -> list[dict]:
    """Performance broken down by market session."""
    conn = _get_conn()
    rows = conn.execute(f"""
        SELECT
            s.market_session,
            COUNT(*)                                            AS total,
            SUM(CASE WHEN o.tp1_hit THEN 1 ELSE 0 END)        AS tp1_wins,
            SUM(CASE WHEN o.sl_hit THEN 1 ELSE 0 END)         AS sl_losses,
            AVG(o.pnl_percent)                                 AS avg_pnl,
            ROUND(100.0 * SUM(CASE WHEN o.tp1_hit THEN 1 ELSE 0 END) / COUNT(*), 1) AS win_rate
        FROM signals s
        JOIN outcomes o ON o.signal_id = s.id
        WHERE o.final = 1
          AND s.created_at > datetime('now', '-{days} days')
          AND s.market_session IS NOT NULL
        GROUP BY s.market_session
        ORDER BY win_rate DESC
    """).fetchall()
    return [dict(r) for r in rows]


def get_asset_performance(days: int = 30) -> list[dict]:
    """Performance broken down by asset."""
    conn = _get_conn()
    rows = conn.execute(f"""
        SELECT
            s.asset,
            s.timeframe,
            COUNT(*)                                            AS total,
            SUM(CASE WHEN o.tp1_hit THEN 1 ELSE 0 END)        AS tp1_wins,
            SUM(CASE WHEN o.sl_hit THEN 1 ELSE 0 END)         AS sl_losses,
            AVG(o.pnl_percent)                                 AS avg_pnl,
            ROUND(100.0 * SUM(CASE WHEN o.tp1_hit THEN 1 ELSE 0 END) / COUNT(*), 1) AS win_rate
        FROM signals s
        JOIN outcomes o ON o.signal_id = s.id
        WHERE o.final = 1
          AND s.created_at > datetime('now', '-{days} days')
        GROUP BY s.asset, s.timeframe
        ORDER BY total DESC
    """).fetchall()
    return [dict(r) for r in rows]


def get_recent_signals_for_learning(
    asset: str, timeframe: str, limit: int = 10
) -> list[dict]:
    """Get the most recent resolved signals for an asset — used to feed into Claude."""
    conn = _get_conn()
    rows = conn.execute("""
        SELECT
            s.direction, s.entry, s.sl, s.tp1, s.tp2, s.tp3,
            s.market_session, s.created_at,
            o.tp1_hit, o.tp2_hit, o.tp3_hit, o.sl_hit,
            o.pnl_percent, o.exit_reason, o.candles_to_exit,
            o.max_favorable, o.max_adverse
        FROM signals s
        JOIN outcomes o ON o.signal_id = s.id
        WHERE s.asset = ? AND s.timeframe = ? AND o.final = 1
        ORDER BY s.created_at DESC
        LIMIT ?
    """, (asset, timeframe, limit)).fetchall()
    return [dict(r) for r in rows]


# ── Autosignal persistence ───────────────────────────────────────────────────


def save_autosignal_sub(chat_id: int, asset: str, timeframe: str) -> None:
    conn = _get_conn()
    conn.execute(
        """INSERT INTO autosignal_subs (chat_id, asset, timeframe, active, created_at)
           VALUES (?, ?, ?, 1, ?)
           ON CONFLICT(chat_id, asset, timeframe) DO UPDATE SET active=1""",
        (chat_id, asset, timeframe, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()


def remove_autosignal_sub(chat_id: int, asset: str, timeframe: str) -> None:
    conn = _get_conn()
    conn.execute(
        "UPDATE autosignal_subs SET active=0 WHERE chat_id=? AND asset=? AND timeframe=?",
        (chat_id, asset, timeframe),
    )
    conn.commit()


def remove_all_autosignal_subs(chat_id: int) -> None:
    conn = _get_conn()
    conn.execute("UPDATE autosignal_subs SET active=0 WHERE chat_id=?", (chat_id,))
    conn.commit()


def get_active_autosignal_subs() -> list[dict]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT chat_id, asset, timeframe FROM autosignal_subs WHERE active=1"
    ).fetchall()
    return [dict(r) for r in rows]
