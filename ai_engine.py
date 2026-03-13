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
    """Fetch live price, 24h stats, recent klines, and derived volatility metrics."""
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

        # ── Derived volatility metrics (pre-computed for the LLM) ────────
        volatility_block = ""
        if len(klines) >= 2:
            highs = [float(k[2]) for k in klines]
            lows = [float(k[3]) for k in klines]
            closes = [float(k[4]) for k in klines]
            volumes = [float(k[5]) for k in klines]

            # ATR (Average True Range) over available candles
            true_ranges = []
            for i in range(1, len(klines)):
                h, l, prev_c = highs[i], lows[i], closes[i - 1]
                tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
                true_ranges.append(tr)
            atr_14 = sum(true_ranges[-14:]) / min(14, len(true_ranges)) if true_ranges else 0
            atr_pct = (atr_14 / closes[-1] * 100) if closes[-1] else 0

            # Recent range (last 10 candles high-to-low)
            recent_high = max(highs[-10:])
            recent_low = min(lows[-10:])
            range_pct = ((recent_high - recent_low) / closes[-1] * 100) if closes[-1] else 0

            # Candle body ratio (avg |close-open|/|high-low|) — measures conviction
            body_ratios = []
            for k in klines[-10:]:
                o, h, l, c = float(k[1]), float(k[2]), float(k[3]), float(k[4])
                wick = h - l
                body_ratios.append(abs(c - o) / wick if wick > 0 else 0)
            avg_body_ratio = sum(body_ratios) / len(body_ratios) if body_ratios else 0

            # Volume trend (recent 5 avg vs prior 5 avg)
            vol_recent = sum(volumes[-5:]) / 5 if len(volumes) >= 5 else sum(volumes) / len(volumes)
            vol_prior = sum(volumes[-10:-5]) / 5 if len(volumes) >= 10 else vol_recent
            vol_change = ((vol_recent - vol_prior) / vol_prior * 100) if vol_prior > 0 else 0

            # Consecutive candle direction (streak detection)
            streak = 0
            if len(closes) >= 2:
                direction = 1 if closes[-1] >= closes[-2] else -1
                for i in range(len(closes) - 1, 0, -1):
                    if (closes[i] >= closes[i - 1]) == (direction == 1):
                        streak += 1
                    else:
                        break
                streak *= direction  # positive = bullish streak, negative = bearish

            volatility_block = (
                f"\n  PRE-COMPUTED METRICS (use these — do NOT re-derive from candles):\n"
                f"  ATR ({timeframe.upper()}, 14-period): {atr_14:.2f} ({atr_pct:.3f}% of price)\n"
                f"  Recent 10-candle range: {recent_low:.2f} – {recent_high:.2f} ({range_pct:.3f}%)\n"
                f"  Avg candle body ratio: {avg_body_ratio:.2f} (0=all wick/indecision, 1=full body/conviction)\n"
                f"  Volume trend: {vol_change:+.1f}% (recent 5 candles vs prior 5)\n"
                f"  Candle streak: {streak:+d} ({'bullish' if streak > 0 else 'bearish' if streak < 0 else 'neutral'})\n"
            )

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
            + volatility_block
            + "\n--- END MARKET DATA ---\n"
        )
    except Exception as e:
        logger.warning("Failed to fetch market data for %s: %s", symbol, e)
        return f"\n(Could not fetch live data for {symbol})\n"

SYSTEM_PROMPT = """You are an expert quantitative trading analyst and PineScript v6 developer.
You manage risk first, then seek opportunity. You never force a trade.

CORE METHODOLOGY — follow this on every trade analysis:
1. REGIME FIRST: Before looking at any setup, classify the market regime:
   - Trending (clear HH/HL or LL/LH structure, strong body candles, ATR expanding)
   - Ranging (price oscillating between clear S/R, small bodies, ATR contracting)
   - Volatile/Choppy (wide wicks, no structure, ATR spike with no directional follow-through)
   Your setup type MUST match the regime. Breakouts in trends, mean-reversion in ranges,
   NO TRADE in choppy conditions unless edge is exceptional.

2. VOLATILITY-SCALED RISK: ALWAYS use the provided ATR to size stops.
   - SL distance must be 1.0–2.0x ATR from entry (never arbitrary round numbers)
   - If ATR% > 2.5% on 1H (or proportional), conditions are too volatile for tight setups — widen or sit out
   - If ATR% < 0.3% on 1H, the market is dead — don't chase micro-moves

3. MINIMUM R:R = 1.5:1 to TP1. If the nearest structural TP doesn't give 1.5:1 against
   a proper ATR-based SL, the trade does not exist. Output NO_TRADE.

4. CHAIN OF THOUGHT: You MUST think step-by-step before outputting any levels:
   Step 1 — Regime: trending / ranging / choppy (cite evidence from candles + ATR)
   Step 2 — Bias: bullish / bearish / neutral (cite structure + momentum)
   Step 3 — If regime is choppy OR bias is neutral → output NO_TRADE with reason
   Step 4 — Key levels: identify S/R from recent candle highs/lows
   Step 5 — SL placement: nearest structure + ATR buffer (show the math: level +/- ATR*X)
   Step 6 — TP placement: next structural target. Verify R:R >= 1.5:1 before committing
   Step 7 — Confidence: rate 1-10 based on confluence (regime + structure + session + past performance)

FORMATTING RULES (output is displayed in Telegram):
- NEVER use markdown tables — Telegram cannot render them
- Use bullet points (- ) and bold (**text**) for structure
- Keep trade analysis under 2500 characters (the Chain of Thought section can be brief)
- No PineScript code in trade analysis — only when explicitly asked
- No filler — every sentence must add value

When generating PineScript code (only when asked), use v6 syntax with alertcondition() calls.
Format alert messages as JSON: {"action":"LONG","ticker":"{{ticker}}","price":"{{close}}","tp":"<TP>","sl":"<SL>"}
"""


async def _extract_symbol(text: str) -> str | None:
    """Try to extract a trading symbol from user text."""
    import re
    # Match common patterns like BTCUSDT, BTC/USDT, ETH-USDT
    m = re.search(r"\b([A-Z]{2,10}(?:[/-]?USDT?|BUSD))\b", text.upper())
    if m:
        return m.group(1).replace("/", "").replace("-", "")
    return None


async def analyze(prompt: str, context: str = "") -> str:
    """Send a trading/strategy prompt to Claude and return the analysis."""
    # Try to fetch live data if a symbol is mentioned
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
    """Extract the TRADE_LEVELS JSON block from Claude's response.

    Returns None if no JSON found, or a dict with direction=NO_TRADE if the
    model explicitly chose not to trade.
    """
    m = re.search(r"```json\s*\n(\{[^}]*\"direction\"[^}]*\})\s*\n```", text, re.DOTALL)
    if not m:
        m = re.search(r"TRADE_LEVELS\s*[:=]?\s*(\{[^}]*\"direction\"[^}]*\})", text)
    if not m:
        return None
    try:
        data = json.loads(m.group(1))

        # Handle NO_TRADE responses
        direction = data.get("direction", "")
        if direction and direction.upper() == "NO_TRADE":
            return {"direction": "NO_TRADE", "confidence": data.get("confidence", 0)}

        levels = {}
        for key in ("direction", "entry", "sl", "tp1", "tp2", "tp3"):
            val = data.get(key)
            if val is not None:
                levels[key] = float(val) if key != "direction" else val

        # Capture confidence score if present
        conf = data.get("confidence")
        if conf is not None:
            try:
                levels["confidence"] = int(conf)
            except (ValueError, TypeError):
                pass

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
    """Build a diagnosed context block from historical signal performance.

    Instead of dumping raw numbers for the LLM to interpret, this function
    analyzes the data and produces explicit behavioral directives.
    """
    parts = []

    # 1. Recent signals for this exact asset+timeframe
    recent = signal_db.get_recent_signals_for_learning(asset, timeframe, limit=10)
    if not recent:
        return ""

    wins = sum(1 for r in recent if r["tp1_hit"])
    losses = sum(1 for r in recent if r["sl_hit"] and not r["tp1_hit"])
    timeouts = sum(1 for r in recent if r["exit_reason"] == "timeout")
    total = len(recent)
    win_rate = (wins / total * 100) if total > 0 else 0

    parts.append(f"\n--- SELF-LEARNING CONTEXT ({asset} {timeframe}) ---")
    parts.append(f"Track record (last {total}): {wins}W / {losses}L / {timeouts}T ({win_rate:.0f}% WR)")

    # Show recent signals (compact)
    for i, r in enumerate(recent[:5], 1):
        result_str = r["exit_reason"] or "open"
        pnl = r["pnl_percent"] or 0
        session = r["market_session"] or "unknown"
        parts.append(
            f"  #{i}: {r['direction']} @ {r['entry']} → {result_str} ({pnl:+.2f}%) | "
            f"Fav: {r['max_favorable']:.2f}% Adv: {r['max_adverse']:.2f}% | "
            f"{r['candles_to_exit']} candles | {session}"
        )

    # ── Diagnosed patterns → behavioral directives ───────────────────────
    directives = []

    # A. Stop-loss quality: are we getting stopped out before the move happens?
    sl_signals = [r for r in recent if r["sl_hit"] and not r["tp1_hit"]]
    if sl_signals:
        avg_adverse_on_sl = sum(r["max_adverse"] for r in sl_signals) / len(sl_signals)
        avg_favorable_on_sl = sum(r["max_favorable"] for r in sl_signals) / len(sl_signals)
        if avg_favorable_on_sl > avg_adverse_on_sl * 0.5:
            directives.append(
                f"STOP PLACEMENT ISSUE: On SL losses, price moved {avg_favorable_on_sl:.2f}% "
                f"in your favor before reversing. Your stops are likely too tight — "
                f"use 1.5–2.0x ATR instead of current placement."
            )
        if len(sl_signals) >= 3 and all(
            (r["candles_to_exit"] or 99) <= 3 for r in sl_signals[-3:]
        ):
            directives.append(
                "RAPID STOP-OUTS: Last 3 losses were hit within 3 candles. "
                "This suggests entries at the wrong level (chasing), not bad direction. "
                "Wait for pullbacks to structure before entering."
            )

    # B. Regime detection from recent outcomes
    if total >= 5:
        recent_5_pnl = [r["pnl_percent"] or 0 for r in recent[:5]]
        recent_5_exits = [r["exit_reason"] for r in recent[:5]]
        alternating = sum(
            1 for i in range(len(recent_5_pnl) - 1)
            if (recent_5_pnl[i] >= 0) != (recent_5_pnl[i + 1] >= 0)
        )

        if alternating >= 3 and timeouts >= 2:
            directives.append(
                "CHOPPY MARKET DETECTED: Wins and losses are alternating with timeouts. "
                "The market is likely ranging/choppy. REDUCE SIGNAL FREQUENCY. "
                "Only trade at range extremes (support/resistance) with mean-reversion setups. "
                "If you cannot identify clear range boundaries, output NO_TRADE."
            )
        elif wins >= 4 and sum(1 for r in recent[:5] if r["tp1_hit"]) >= 4:
            directives.append(
                "TRENDING MARKET: Recent signals are hitting targets consistently. "
                "Maintain current approach. Consider holding for TP2/TP3 more aggressively."
            )
        elif losses >= 4:
            losing_dirs = [r["direction"] for r in recent[:5] if r["sl_hit"] and not r["tp1_hit"]]
            if losing_dirs and all(d == losing_dirs[0] for d in losing_dirs):
                directives.append(
                    f"DIRECTIONAL BIAS ERROR: Recent losses are all {losing_dirs[0]}. "
                    f"The market is moving against your bias. Consider the opposite direction, "
                    f"or output NO_TRADE until structure confirms a new trend."
                )
            else:
                directives.append(
                    "LOSING STREAK: 4+ recent losses. DO NOT widen stops blindly. "
                    "Instead, raise your confidence threshold — only take setups with "
                    "3+ confluence factors (structure + session + volume agreement)."
                )

    # C. TP hit rate analysis — are targets realistic?
    tp1_hits = sum(1 for r in recent if r["tp1_hit"])
    tp2_hits = sum(1 for r in recent if r["tp2_hit"])
    tp3_hits = sum(1 for r in recent if r["tp3_hit"])
    if tp1_hits > 0 and tp3_hits / max(tp1_hits, 1) < 0.25:
        avg_fav_on_wins = sum(r["max_favorable"] for r in recent if r["tp1_hit"]) / max(tp1_hits, 1)
        directives.append(
            f"TP CALIBRATION: TP1 hits {tp1_hits}x but TP3 only {tp3_hits}x. "
            f"Avg favorable move on wins: {avg_fav_on_wins:.2f}%. "
            f"Set TP2 closer to avg favorable move, TP3 as stretch target only."
        )

    # D. Session edge
    if wins > 0 and losses > 0:
        win_sessions = [r["market_session"] for r in recent if r["tp1_hit"] and r["market_session"]]
        loss_sessions = [r["market_session"] for r in recent if r["sl_hit"] and not r["tp1_hit"] and r["market_session"]]

        # Find sessions that are clearly bad
        from collections import Counter
        loss_counts = Counter(loss_sessions)
        win_counts = Counter(win_sessions)
        for sess, lcount in loss_counts.items():
            wcount = win_counts.get(sess, 0)
            if lcount >= 3 and wcount <= 1:
                directives.append(
                    f"SESSION AVOID: {sess} has {lcount} losses vs {wcount} wins. "
                    f"If currently in {sess}, output NO_TRADE or require extra confluence."
                )

    if directives:
        parts.append("\n  DIAGNOSED PATTERNS & DIRECTIVES:")
        for d in directives:
            parts.append(f"  >> {d}")

    parts.append("--- END SELF-LEARNING ---\n")

    # 2. Overall 30-day summary (compact)
    perf = signal_db.get_performance_summary(asset=asset, days=30)
    if perf and perf.get("total_signals") and perf["total_signals"] > 0:
        parts.append(
            f"30-DAY STATS ({asset}): {perf['total_signals']} signals, "
            f"TP1: {perf['tp1_wins']}/{perf['total_signals']}, "
            f"SL: {perf['sl_losses']}/{perf['total_signals']}, "
            f"Avg P&L: {perf['avg_pnl']:+.2f}%, "
            f"Avg favorable: {perf['avg_max_favorable']:.2f}%, "
            f"Avg adverse: {perf['avg_max_adverse']:.2f}%"
        )

    # 3. Session performance (compact, only notable sessions)
    session_perf = signal_db.get_session_performance(days=30)
    if session_perf:
        notable = [
            sp for sp in session_perf
            if sp["total"] >= 5 and (sp["win_rate"] >= 70 or sp["win_rate"] <= 40)
        ]
        if notable:
            parts.append("SESSION EDGES:")
            for sp in notable:
                tag = "STRONG" if sp["win_rate"] >= 70 else "WEAK"
                parts.append(
                    f"  {sp['market_session']}: {sp['win_rate']}% WR "
                    f"({sp['tp1_wins']}W/{sp['sl_losses']}L) — {tag}"
                )

    return "\n".join(parts)


async def suggest_trade(
    asset: str, timeframe: str = "1H", extra: str = ""
) -> tuple[str, dict | None]:
    """Get a trade suggestion. Returns (analysis_text, levels_dict)."""
    from chart import TIMEFRAME_MAP
    interval = TIMEFRAME_MAP.get(timeframe.upper(), timeframe.lower())
    market_data = await _fetch_market_data(asset, interval)

    # Build session + learning context
    session_info = market_sessions.get_current_sessions()
    session_context = market_sessions.format_session_context(session_info)
    learning_context = _build_learning_context(asset, timeframe)

    prompt = (
        f"Analyze {asset} on the {timeframe} timeframe.\n\n"
        f"REAL-TIME MARKET DATA:\n{market_data}\n"
        f"CURRENT SESSION:\n{session_context}\n\n"
    )

    # Inject diagnosed learning context
    if learning_context:
        prompt += (
            f"{learning_context}\n"
            f"CRITICAL: The DIRECTIVES above are derived from your own past performance. "
            f"You MUST follow them. If a directive says NO_TRADE, you need strong reasons to override it.\n\n"
        )

    prompt += (
        f"Follow your Chain of Thought methodology. Work through each step IN ORDER, "
        f"showing your reasoning briefly before committing to any levels.\n\n"
        f"RESPONSE FORMAT:\n\n"
        f"🧠 **Thesis** (REQUIRED — do this FIRST, before any levels)\n"
        f"- Regime: trending / ranging / choppy — cite ATR trend + candle structure\n"
        f"- Bias: bullish / bearish / neutral — cite price action evidence\n"
        f"- Session factor: how does current session affect this setup?\n"
        f"- If regime is choppy or bias is neutral with no clear structure → output NO_TRADE\n\n"
        f"📊 **{asset} {timeframe} Analysis**\n"
        f"1-2 sentences on trend + market structure.\n\n"
        f"🎯 **Trade Setup** (or **NO TRADE** if conditions are unclear)\n"
        f"- Direction: LONG/SHORT (confidence 1-10)\n"
        f"- Entry: price — brief reason\n"
        f"- SL: price — MUST be structure level +/- ATR buffer (show: level ± ATR*X = SL)\n"
        f"- TP1: price — nearest structure target (verify R:R ≥ 1.5:1)\n"
        f"- TP2: price — secondary target\n"
        f"- TP3: price — stretch target\n"
        f"- R:R to TP1: X.X:1 (if < 1.5:1, this trade does NOT qualify — output NO_TRADE)\n\n"
        f"📈 **Key Levels**\n"
        f"- Support: price, price\n"
        f"- Resistance: price, price\n\n"
        f"✅ **Confirmation**: What must happen before entry (e.g. candle close, volume spike)\n\n"
        f"⚠️ **Invalidation**: One sentence — what kills the thesis\n\n"
        f"RULES:\n"
        f"- NO markdown tables. Bullet points only.\n"
        f"- NO PineScript code.\n"
        f"- SL MUST be volatility-scaled (ATR-based), not arbitrary round numbers.\n"
        f"- If R:R < 1.5:1 to TP1, output NO_TRADE. Do not force a bad trade.\n"
        f"- Keep total response under 2500 characters.\n\n"
        f"CRITICAL: At the very end, include a JSON block:\n"
        f"For a valid trade:\n"
        f"```json\n"
        f'{{"direction": "LONG", "entry": 00000, "sl": 00000, '
        f'"tp1": 00000, "tp2": 00000, "tp3": 00000, "confidence": 7}}\n'
        f"```\n"
        f"For no trade:\n"
        f"```json\n"
        f'{{"direction": "NO_TRADE", "entry": null, "sl": null, '
        f'"tp1": null, "tp2": null, "tp3": null, "confidence": 0}}\n'
        f"```\n"
        f"Use actual numbers, no commas or $ signs."
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
