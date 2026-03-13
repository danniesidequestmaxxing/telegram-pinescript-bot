"""Background outcome tracker — checks real price action against past signals.

Runs as a recurring job via the Telegram bot's JobQueue.
For each unresolved signal, fetches candles since the signal was created and
determines if TP1/TP2/TP3 or SL was hit.
"""

import logging
from datetime import datetime, timezone

import httpx

import config
import signal_db
from chart import TIMEFRAME_MAP, CANDLE_COUNTS

logger = logging.getLogger(__name__)

BINANCE_BASE = config.BINANCE_BASE_URL

# How many candles to wait before marking a signal as "timeout"
# (if neither TP nor SL is hit within this many candles, it's stale)
MAX_CANDLES_TIMEOUT = {
    "15M": 48,   # 12 hours
    "30M": 48,   # 24 hours
    "1H": 48,    # 48 hours
    "2H": 36,    # 72 hours
    "4H": 30,    # 5 days
    "6H": 28,    # 7 days
    "8H": 21,    # 7 days
    "12H": 14,   # 7 days
    "1D": 7,     # 7 days
}


async def _fetch_candles_since(
    asset: str, timeframe: str, since_iso: str, limit: int = 100
) -> list[dict]:
    """Fetch candles from Binance starting at `since_iso` timestamp."""
    interval = TIMEFRAME_MAP.get(timeframe.upper(), timeframe.lower())
    start_dt = datetime.fromisoformat(since_iso)
    start_ms = int(start_dt.timestamp() * 1000)

    try:
        async with httpx.AsyncClient(timeout=10) as http:
            resp = await http.get(
                f"{BINANCE_BASE}/api/v3/klines",
                params={
                    "symbol": asset,
                    "interval": interval,
                    "startTime": start_ms,
                    "limit": limit,
                },
            )
            resp.raise_for_status()
            raw = resp.json()
    except Exception as e:
        logger.warning("Failed to fetch candles for outcome check %s: %s", asset, e)
        return []

    candles = []
    for k in raw:
        candles.append({
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "volume": float(k[5]),
            "timestamp": k[0],
        })
    return candles


def _evaluate_signal(signal: dict, candles: list[dict]) -> dict | None:
    """Check candles against signal levels and return outcome data.

    Returns None if there aren't enough candles yet.
    """
    direction = signal["direction"].upper() if signal["direction"] else None
    entry = signal["entry"]
    sl = signal["sl"]
    tp1 = signal["tp1"]
    tp2 = signal["tp2"]
    tp3 = signal["tp3"]

    if not direction or not entry:
        return None

    is_long = direction == "LONG"

    tp1_hit = bool(signal.get("tp1_hit"))
    tp2_hit = bool(signal.get("tp2_hit"))
    tp3_hit = bool(signal.get("tp3_hit"))
    sl_hit = bool(signal.get("sl_hit"))

    max_favorable = 0.0
    max_adverse = 0.0
    exit_reason = "open"
    final = False
    candles_to_exit = len(candles)

    for i, c in enumerate(candles):
        high = c["high"]
        low = c["low"]

        if is_long:
            favorable = ((high - entry) / entry) * 100
            adverse = ((entry - low) / entry) * 100
        else:
            favorable = ((entry - low) / entry) * 100
            adverse = ((high - entry) / entry) * 100

        max_favorable = max(max_favorable, favorable)
        max_adverse = max(max_adverse, adverse)

        # Check TP hits (in order)
        if tp1 and not tp1_hit:
            if (is_long and high >= tp1) or (not is_long and low <= tp1):
                tp1_hit = True
                if exit_reason == "open":
                    exit_reason = "tp1"
                    candles_to_exit = i + 1

        if tp2 and not tp2_hit:
            if (is_long and high >= tp2) or (not is_long and low <= tp2):
                tp2_hit = True
                exit_reason = "tp2"
                candles_to_exit = i + 1

        if tp3 and not tp3_hit:
            if (is_long and high >= tp3) or (not is_long and low <= tp3):
                tp3_hit = True
                exit_reason = "tp3"
                candles_to_exit = i + 1

        # Check SL hit
        if sl and not sl_hit:
            if (is_long and low <= sl) or (not is_long and high >= sl):
                sl_hit = True
                if not tp1_hit:  # Only mark as SL exit if no TP was hit first
                    exit_reason = "sl"
                    candles_to_exit = i + 1

    # Calculate P&L based on exit reason
    pnl = 0.0
    if exit_reason == "sl" and sl:
        pnl = -abs((sl - entry) / entry) * 100
    elif exit_reason == "tp3" and tp3:
        pnl = abs((tp3 - entry) / entry) * 100
    elif exit_reason == "tp2" and tp2:
        pnl = abs((tp2 - entry) / entry) * 100
    elif exit_reason == "tp1" and tp1:
        pnl = abs((tp1 - entry) / entry) * 100
    elif exit_reason == "open":
        # Still open — use current price for unrealized P&L
        if candles:
            current = candles[-1]["close"]
            if is_long:
                pnl = ((current - entry) / entry) * 100
            else:
                pnl = ((entry - current) / entry) * 100

    # Determine if trade is resolved
    timeout_candles = MAX_CANDLES_TIMEOUT.get(signal["timeframe"], 30)
    if sl_hit or tp3_hit or (tp1_hit and not tp2) or len(candles) >= timeout_candles:
        final = True
        if exit_reason == "open":
            exit_reason = "timeout"

    return {
        "price_at_check": candles[-1]["close"] if candles else entry,
        "tp1_hit": tp1_hit,
        "tp2_hit": tp2_hit,
        "tp3_hit": tp3_hit,
        "sl_hit": sl_hit,
        "max_favorable": round(max_favorable, 4),
        "max_adverse": round(max_adverse, 4),
        "pnl_percent": round(pnl, 4),
        "exit_reason": exit_reason,
        "candles_to_exit": candles_to_exit,
        "final": final,
    }


async def check_all_outcomes() -> int:
    """Check outcomes for all pending signals. Returns number checked."""
    pending = signal_db.get_pending_outcomes()
    checked = 0

    for sig in pending:
        candles = await _fetch_candles_since(
            sig["asset"], sig["timeframe"], sig["created_at"]
        )
        if not candles:
            continue

        result = _evaluate_signal(sig, candles)
        if result is None:
            continue

        signal_db.upsert_outcome(
            signal_id=sig["id"],
            price_at_check=result["price_at_check"],
            tp1_hit=result["tp1_hit"],
            tp2_hit=result["tp2_hit"],
            tp3_hit=result["tp3_hit"],
            sl_hit=result["sl_hit"],
            max_favorable=result["max_favorable"],
            max_adverse=result["max_adverse"],
            pnl_percent=result["pnl_percent"],
            exit_reason=result["exit_reason"],
            candles_to_exit=result["candles_to_exit"],
            final=result["final"],
        )
        checked += 1
        status = "RESOLVED" if result["final"] else "UPDATED"
        logger.info(
            "Outcome %s for signal #%d (%s %s): %s (P&L: %.2f%%)",
            status, sig["id"], sig["asset"], sig["timeframe"],
            result["exit_reason"], result["pnl_percent"],
        )

    return checked
