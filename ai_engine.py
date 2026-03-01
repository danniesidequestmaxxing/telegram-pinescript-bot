"""Claude-powered trading analysis and PineScript generation engine."""

import json
import logging
import re

import anthropic
import httpx

import config
import signal_db
import market_sessions

logger = logging.getLogger(__name__)

client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

BINANCE_BASE = "https://api.binance.us"


async def _fetch_market_data(symbol: str, timeframe: str = "4h") -> str:
    """Fetch live price, 24h stats, and recent klines from Binance."""
    try:
        async with httpx.AsyncClient(timeout=10) as http:
            ticker_resp = await http.get(
                f"{BINANCE_BASE}/api/v3/ticker/24hr", params={"symbol": symbol}
            )
            ticker = ticker_resp.json()

            klines_resp = await http.get(
                f"{BINANCE_BASE}/api/v3/klines",
                params={"symbol": symbol, "interval": timeframe.lower(), "limit": 20},
            )
            klines = klines_resp.json()

        price = ticker.get("lastPrice", "N/A")
        high_24h = ticker.get("highPrice", "N/A")
        low_24h = ticker.get("lowPrice", "N/A")
        change_pct = ticker.get("priceChangePercent", "N/A")
        volume = ticker.get("volume", "N/A")

        candle_lines = []
        for k in klines[-10:]:
            o, h, l, c, v = k[1], k[2], k[3], k[4], k[5]
            candle_lines.append(f"  O:{o} H:{h} L:{l} C:{c} Vol:{v}")

        return (
            f"\n--- LIVE MARKET DATA (Binance) ---\n"
            f"Symbol: {symbol}\n"
            f"Current Price: {price}\n"
            f"24h High: {high_24h}\n"
            f"24h Low: {low_24h}\n"
            f"24h Change: {change_pct}%\n"
            f"24h Volume: {volume}\n"
            f"\nRecent {timeframe.upper()} Candles (last 10):\n"
            + "\n".join(candle_lines)
            + "\n--- END MARKET DATA ---\n"
        )
    except Exception as e:
        logger.warning("Failed to fetch market data for %s: %s", symbol, e)
        return f"\n(Could not fetch live data for {symbol})\n"

SYSTEM_PROMPT = """You are an expert quantitative trading analyst and PineScript v6 developer.

You provide DETAILED, THOROUGH trading analysis with clear structure. Your analysis should
read like a professional trading report — covering market structure, key levels, technical
confirmations, and risk management in depth.

When suggesting trades, use STRUCTURED SECTIONS with clear headers and thorough reasoning.
Always base levels on real price data. Explain WHY each level matters — don't just list numbers.

When generating PineScript code, use v6 syntax with alertcondition() calls.
Format alert messages as JSON: {"action":"LONG","ticker":"{{ticker}}","price":"{{close}}","tp":"<TP>","sl":"<SL>"}
"""


async def _extract_symbol(text: str) -> str | None:
    """Try to extract a trading symbol from user text."""
    import re
    m = re.search(r"\b([A-Z]{2,10}(?:[/-]?USDT?|BUSD))\b", text.upper())
    if m:
        return m.group(1).replace("/", "").replace("-", "")
    return None


async def analyze(prompt: str, context: str = "") -> str:
    """Send a trading/strategy prompt to Claude and return the analysis."""
    symbol = await _extract_symbol(prompt)
    market_info = ""
    if symbol:
        market_info = await _fetch_market_data(symbol)

    enriched_prompt = prompt
    if market_info:
        enriched_prompt = f"{prompt}\n\nReal-time market data:\n{market_info}"

    messages = []
    if context:
        messages.append({"role": "user", "content": context})
        messages.append(
            {
                "role": "assistant",
                "content": "Understood, I have the context. What would you like to analyze?",
            }
        )
    messages.append({"role": "user", "content": enriched_prompt})

    response = client.messages.create(
        model=config.CLAUDE_MODEL,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=messages,
    )
    return response.content[0].text


async def generate_pinescript(description: str) -> str:
    """Generate PineScript v6 code from a natural-language description."""
    prompt = (
        f"Generate a complete, ready-to-use PineScript v6 script for the following:\n\n"
        f"{description}\n\n"
        f"Requirements:\n"
        f"- Use PineScript v6 syntax\n"
        f"- Include alertcondition() calls with JSON-formatted messages for webhook integration\n"
        f"- Add clear comments explaining each section\n"
        f"- Make it production-ready (handle edge cases, use proper na checks)\n"
        f"- Return ONLY the PineScript code block, no extra explanation outside the code"
    )

    response = client.messages.create(
        model=config.CLAUDE_MODEL,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def _parse_levels(text: str) -> dict | None:
    """Extract the TRADE_LEVELS JSON block from Claude's response."""
    m = re.search(r"```json\s*\n(\{[^}]*\"direction\"[^}]*\})\s*\n```", text, re.DOTALL)
    if not m:
        m = re.search(r"TRADE_LEVELS\s*[:=]?\s*(\{[^}]*\"direction\"[^}]*\})", text)
    if not m:
        return None
    try:
        data = json.loads(m.group(1))
        levels = {}
        for key in ("direction", "entry", "sl", "tp1", "tp2", "tp3"):
            val = data.get(key)
            if val is not None:
                levels[key] = float(val) if key != "direction" else val
        return levels if "entry" in levels else None
    except (json.JSONDecodeError, ValueError, TypeError):
        return None


def _strip_levels_block(text: str) -> str:
    """Remove the TRADE_LEVELS JSON block from the display text."""
    text = re.sub(
        r"\n*```json\s*\n\{[^}]*\"direction\"[^}]*\}\s*\n```\n*", "\n", text
    )
    text = re.sub(r"\n*TRADE_LEVELS\s*[:=]?\s*\{[^}]*\"direction\"[^}]*\}\n*", "\n", text)
    return text.strip()


def _build_learning_context(asset: str, timeframe: str) -> str:
    """Build a context block from historical signal performance for Claude."""
    parts = []

    recent = signal_db.get_recent_signals_for_learning(asset, timeframe, limit=10)
    if recent:
        parts.append(f"\n--- SELF-LEARNING: PAST SIGNAL PERFORMANCE ({asset} {timeframe}) ---")
        wins = sum(1 for r in recent if r["tp1_hit"])
        losses = sum(1 for r in recent if r["sl_hit"] and not r["tp1_hit"])
        total = len(recent)
        win_rate = (wins / total * 100) if total > 0 else 0

        parts.append(f"Track record: {wins}W / {losses}L out of {total} signals ({win_rate:.0f}% win rate)")

        for i, r in enumerate(recent[:5], 1):
            result_str = r["exit_reason"] or "open"
            pnl = r["pnl_percent"] or 0
            session = r["market_session"] or "unknown"
            parts.append(
                f"  Signal #{i}: {r['direction']} @ {r['entry']} | "
                f"Result: {result_str} ({pnl:+.2f}%) | "
                f"Session: {session} | "
                f"Max favorable: {r['max_favorable']:.2f}% / adverse: {r['max_adverse']:.2f}% | "
                f"Candles to exit: {r['candles_to_exit']}"
            )

        if wins > 0 and losses > 0:
            win_sessions = [r["market_session"] for r in recent if r["tp1_hit"] and r["market_session"]]
            loss_sessions = [r["market_session"] for r in recent if r["sl_hit"] and not r["tp1_hit"] and r["market_session"]]
            if win_sessions:
                parts.append(f"  Winning sessions: {', '.join(win_sessions)}")
            if loss_sessions:
                parts.append(f"  Losing sessions: {', '.join(loss_sessions)}")

        parts.append("--- END PAST PERFORMANCE ---\n")

    perf = signal_db.get_performance_summary(asset=asset, days=30)
    if perf and perf.get("total_signals") and perf["total_signals"] > 0:
        parts.append(f"\n--- OVERALL 30-DAY STATS ({asset}) ---")
        parts.append(f"Total resolved signals: {perf['total_signals']}")
        parts.append(f"TP1 hit rate: {perf['tp1_wins']}/{perf['total_signals']}")
        parts.append(f"SL hit rate: {perf['sl_losses']}/{perf['total_signals']}")
        parts.append(f"Avg P&L: {perf['avg_pnl']:+.2f}%")
        parts.append(f"Avg max favorable move: {perf['avg_max_favorable']:.2f}%")
        parts.append(f"Avg max adverse move: {perf['avg_max_adverse']:.2f}%")
        parts.append(f"Avg candles to resolution: {perf['avg_candles_to_exit']:.1f}")
        parts.append("--- END STATS ---\n")

    session_perf = signal_db.get_session_performance(days=30)
    if session_perf:
        parts.append("\n--- SESSION PERFORMANCE (all assets, 30 days) ---")
        for sp in session_perf:
            parts.append(
                f"  {sp['market_session']}: {sp['win_rate']}% win rate "
                f"({sp['tp1_wins']}W/{sp['sl_losses']}L, avg P&L: {sp['avg_pnl']:+.2f}%)"
            )
        parts.append("--- END SESSION PERFORMANCE ---\n")

    return "\n".join(parts)


async def suggest_trade(
    asset: str, timeframe: str = "1H", extra: str = ""
) -> tuple[str, dict | None]:
    """Get a trade suggestion. Returns (analysis_text, levels_dict)."""
    from chart import TIMEFRAME_MAP
    interval = TIMEFRAME_MAP.get(timeframe.upper(), timeframe.lower())
    market_data = await _fetch_market_data(asset, interval)

    session_info = market_sessions.get_current_sessions()
    session_context = market_sessions.format_session_context(session_info)
    learning_context = _build_learning_context(asset, timeframe)

    prompt = (
        f"Analyze {asset} on the {timeframe} timeframe and give me a concrete trade setup.\n\n"
        f"USE THE FOLLOWING REAL-TIME MARKET DATA to base your analysis and price levels on:\n"
        f"{market_data}\n"
        f"IMPORTANT: All entry, TP, and SL levels MUST be based on the current price above.\n\n"
    )

    prompt += (
        f"--- CURRENT MARKET SESSION ---\n"
        f"{session_context}\n"
        f"--- END SESSION ---\n\n"
        f"IMPORTANT: Factor the current market session into your analysis. For example:\n"
        f"- At US market open or CME open, expect increased volatility and wider stops may be needed\n"
        f"- During Asian session, momentum may be slower — tighter ranges\n"
        f"- During London/NY overlap, liquidity is highest — best for breakout plays\n"
        f"- CME weekly open (Sunday) has gap fill risk\n\n"
    )

    if learning_context:
        prompt += (
            f"{learning_context}\n"
            f"SELF-LEARNING INSTRUCTIONS: Review the past signal performance above carefully.\n"
            f"- If your past signals had SL hit too often, consider wider stops or more conservative entries\n"
            f"- If TP3 was rarely hit, consider tighter TP targets\n"
            f"- If a particular session had better results, note that in your analysis\n"
            f"- If max adverse excursion was consistently high before winning, consider better entry timing\n"
            f"- Adapt your approach based on what actually worked vs what didn't\n\n"
        )

    prompt += (
        f"Format your analysis using these EXACT sections with headers. Be detailed and thorough:\n\n"
        f"📊 {asset} {timeframe} Analysis\n\n"
        f"📈 Market Structure Assessment\n"
        f"- Current trend direction and strength\n"
        f"- Key support levels (at least 2-3, with why they matter)\n"
        f"- Key resistance levels (at least 2-3, with why they matter)\n"
        f"- Recent price action context (what happened in last few candles)\n\n"
        f"🎯 Trade Recommendation\n"
        f"- Direction: LONG or SHORT\n"
        f"- Confidence: percentage (e.g., 72%)\n"
        f"- Reasoning for this bias in 2-3 sentences\n\n"
        f"💰 Entry Strategy\n"
        f"- Entry zone (specific price or tight range)\n"
        f"- Why this entry level is optimal (confluence factors)\n"
        f"- Ideal entry trigger (e.g., bullish engulfing, break & retest)\n\n"
        f"🎯 Take Profit Levels\n"
        f"- TP1: price (R:R ratio, reasoning)\n"
        f"- TP2: price (R:R ratio, reasoning)\n"
        f"- TP3: price (R:R ratio, reasoning)\n"
        f"- Suggested position sizing across TPs (e.g., 40%/30%/30%)\n\n"
        f"🛑 Stop Loss\n"
        f"- SL: price\n"
        f"- Why this level (what structure it's behind)\n"
        f"- Max risk percentage from entry\n\n"
        f"📊 Risk-to-Reward Analysis\n"
        f"- Overall R:R ratio\n"
        f"- Risk amount vs potential reward breakdown\n\n"
        f"✅ Technical Confirmation\n"
        f"- Which indicators support this trade (RSI, MACD, EMAs, volume, etc.)\n"
        f"- Entry triggers to watch for\n"
        f"- Invalidation criteria — what kills this trade idea\n\n"
        f"⏰ Session & Timing\n"
        f"- How the current market session affects this setup\n"
        f"- Best time window to enter\n\n"
        f"CRITICAL: At the very end of your response, include a JSON block with the exact "
        f"trade levels in this format:\n"
        f"```json\n"
        f'{{"direction": "LONG", "entry": 00000, "sl": 00000, '
        f'"tp1": 00000, "tp2": 00000, "tp3": 00000}}\n'
        f"```\n"
        f"Use actual numbers (no commas, no $ signs). This JSON block is hidden from the user "
        f"and used to draw levels on a chart."
    )
    if extra:
        prompt += f"\nAdditional context: {extra}"

    response = client.messages.create(
        model=config.CLAUDE_MODEL,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text
    levels = _parse_levels(text)
    display_text = _strip_levels_block(text)
    return display_text, levels


async def draw_indicator(indicator_type: str, asset: str = "", params: str = "") -> str:
    """Generate PineScript for a specific indicator type."""
    prompt = (
        f"Generate a PineScript v6 indicator for: {indicator_type}\n"
        f"{'Asset: ' + asset if asset else ''}\n"
        f"{'Parameters: ' + params if params else ''}\n\n"
        f"Include:\n"
        f"- Clean visual styling (proper colors, line widths)\n"
        f"- Alert conditions for key signals\n"
        f"- JSON-formatted alert messages for webhook integration\n"
        f"- Labels or table showing key values on the chart\n"
    )

    response = client.messages.create(
        model=config.CLAUDE_MODEL,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text
