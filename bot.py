"""Telegram bot handlers for the PineScript trading assistant."""

import html
import io
import logging
import re

from telegram import InputFile, Update
from telegram.constants import ParseMode, ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

import ai_engine
import chart
import config
import signal_db
import market_sessions
import outcome_tracker

logger = logging.getLogger(__name__)

# Per-chat conversation context (last assistant reply kept for follow-ups)
_chat_context: dict[int, str] = {}

# Registered chat IDs that receive webhook alerts
alert_subscribers: set[int] = set()

# Auto-signal subscriptions: chat_id -> list of {"asset": str, "timeframe": str}
_auto_signals: dict[int, list[dict]] = {}

# Valid intervals for auto-signals (in hours)
INTERVAL_HOURS = {
    "15M": 0.25, "30M": 0.5,
    "1H": 1, "2H": 2, "4H": 4,
    "6H": 6, "8H": 8, "12H": 12, "1D": 24,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

MAX_TG_MSG = 4096  # Telegram message length limit


def _md_to_tg_html(text: str) -> str:
    """Convert common Markdown to Telegram-supported HTML tags."""
    # 1. Extract fenced code blocks so they aren't mangled
    code_blocks: list[str] = []

    def _save_block(m: re.Match) -> str:
        code_blocks.append(html.escape(m.group(2)))
        return f"\x00CB{len(code_blocks) - 1}\x00"

    text = re.sub(r"```(\w*)\n?(.*?)```", _save_block, text, flags=re.DOTALL)

    # 2. Extract inline code
    inline_codes: list[str] = []

    def _save_inline(m: re.Match) -> str:
        inline_codes.append(html.escape(m.group(1)))
        return f"\x00IC{len(inline_codes) - 1}\x00"

    text = re.sub(r"`([^`]+)`", _save_inline, text)

    # 3. HTML-escape the rest
    text = html.escape(text)

    # 4. Headings → bold
    text = re.sub(r"^#{1,6}\s+(.+)$", r"<b>\1</b>", text, flags=re.MULTILINE)

    # 5. Bold **text** / __text__
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"__(.+?)__", r"<b>\1</b>", text)

    # 6. Italic *text* / _text_ (not inside words)
    text = re.sub(r"(?<!\w)\*([^*]+?)\*(?!\w)", r"<i>\1</i>", text)
    text = re.sub(r"(?<!\w)_([^_]+?)_(?!\w)", r"<i>\1</i>", text)

    # 7. Strikethrough
    text = re.sub(r"~~(.+?)~~", r"<s>\1</s>", text)

    # 8. Restore code blocks / inline code
    for i, code in enumerate(code_blocks):
        text = text.replace(f"\x00CB{i}\x00", f"<pre>{code}</pre>")
    for i, code in enumerate(inline_codes):
        text = text.replace(f"\x00IC{i}\x00", f"<code>{code}</code>")

    return text


def _split_html_chunks(text: str) -> list[str]:
    """Split HTML text into <=4096-char chunks without breaking <pre> tags."""
    chunks: list[str] = []
    while text:
        if len(text) <= MAX_TG_MSG:
            chunks.append(text)
            break
        split_at = text.rfind("\n", 0, MAX_TG_MSG)
        if split_at == -1:
            split_at = MAX_TG_MSG

        candidate = text[:split_at]

        # Check if we're splitting inside an unclosed <pre> block
        open_count = candidate.count("<pre>")
        close_count = candidate.count("</pre>")
        if open_count > close_count:
            # Close the <pre> in this chunk, re-open in the next
            candidate += "\n</pre>"
            text = "<pre>" + text[split_at:].lstrip("\n")
        else:
            text = text[split_at:].lstrip("\n")

        chunks.append(candidate)
    return chunks


async def _send_long(update: Update, text: str) -> None:
    """Convert Markdown to Telegram HTML and split into <=4096-char chunks."""
    converted = _md_to_tg_html(text)

    for chunk in _split_html_chunks(converted):
        try:
            await update.message.reply_text(chunk, parse_mode=ParseMode.HTML)
        except Exception:
            # Fallback: strip tags and send as plain text
            plain = re.sub(r"<[^>]+>", "", chunk)
            await update.message.reply_text(plain)


async def _send_long_to_chat(bot, chat_id: int, text: str) -> None:
    """Send formatted text directly to a chat_id (for scheduled jobs)."""
    converted = _md_to_tg_html(text)

    for chunk in _split_html_chunks(converted):
        try:
            await bot.send_message(chat_id, chunk, parse_mode=ParseMode.HTML)
        except Exception:
            plain = re.sub(r"<[^>]+>", "", chunk)
            await bot.send_message(chat_id, plain)


async def _thinking(update: Update) -> None:
    await update.message.chat.send_action(ChatAction.TYPING)


# ── Command handlers ─────────────────────────────────────────────────────────


async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "<b>PineScript Trading Assistant</b>\n\n"
        "I use Claude AI to help you with trading strategies, "
        "PineScript code, and live TradingView alerts.\n\n"
        "<b>Commands:</b>\n"
        "/analyze &lt;asset&gt; [timeframe] — Full trade analysis + chart\n"
        "  Timeframes: 15M, 30M, 1H, 4H, 1D\n"
        "/script &lt;description&gt; — Generate PineScript v6 code\n"
        "/indicator &lt;type&gt; — Generate a specific indicator\n"
        "/autosignal &lt;asset&gt; [timeframe] — Auto signals on schedule\n"
        "  e.g. /autosignal BTCUSDT 4H\n"
        "/stopsignal [asset] [timeframe] — Stop auto signals\n"
        "/signals — List active auto signals\n"
        "/performance [asset] [days] — View signal track record\n"
        "/alerts — Subscribe to TradingView webhook alerts\n"
        "/stopalerts — Unsubscribe from alerts\n"
        "/help — Show this message\n\n"
        "The bot <b>self-learns</b> from past signals — it tracks outcomes and "
        "adapts future analysis based on what worked.\n\n"
        "Or just <b>send any message</b> and I'll treat it as a trading question!",
        parse_mode=ParseMode.HTML,
    )


async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await cmd_start(update, ctx)


async def cmd_analyze(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Usage: /analyze BTCUSDT 4H [extra notes]"""
    args = ctx.args or []
    if not args:
        await update.message.reply_text(
            "Usage: /analyze &lt;asset&gt; [timeframe] [extra notes]\n"
            "Example: /analyze BTCUSDT 4H looking for swing long\n"
            "Timeframes: 15M, 30M, 1H, 4H, 1D",
            parse_mode=ParseMode.HTML,
        )
        return

    asset = args[0].upper()
    timeframe = args[1].upper() if len(args) > 1 else "1H"
    extra = " ".join(args[2:]) if len(args) > 2 else ""

    await _thinking(update)

    # Fetch Claude analysis + chart data in parallel-ish fashion
    analysis_text, levels = await ai_engine.suggest_trade(asset, timeframe, extra)
    _chat_context[update.effective_chat.id] = analysis_text

    # Record signal in database for self-learning
    session_info = market_sessions.get_current_sessions()
    signal_db.record_signal(
        chat_id=update.effective_chat.id,
        asset=asset,
        timeframe=timeframe,
        direction=levels.get("direction") if levels else None,
        entry=levels.get("entry") if levels else None,
        sl=levels.get("sl") if levels else None,
        tp1=levels.get("tp1") if levels else None,
        tp2=levels.get("tp2") if levels else None,
        tp3=levels.get("tp3") if levels else None,
        market_session=session_info["primary_session"],
        session_detail=session_info,
        analysis_text=analysis_text,
        source="manual",
    )

    # Generate and send chart
    try:
        df = await chart.fetch_klines(asset, timeframe)
        img_bytes = chart.generate_chart(df, asset, timeframe, levels)
        await update.message.reply_photo(
            photo=InputFile(io.BytesIO(img_bytes), filename=f"{asset}_{timeframe}.png"),
        )
    except Exception as e:
        logger.warning("Chart generation failed: %s", e)

    # Send text analysis
    await _send_long(update, analysis_text)


async def cmd_script(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Usage: /script RSI divergence strategy with alerts"""
    description = " ".join(ctx.args or [])
    if not description:
        await update.message.reply_text(
            "Usage: /script <description>\n"
            "Example: /script EMA crossover strategy 9/21 with volume filter and alerts"
        )
        return

    await _thinking(update)
    result = await ai_engine.generate_pinescript(description)
    _chat_context[update.effective_chat.id] = result
    await _send_long(update, result)


async def cmd_indicator(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Usage: /indicator volume profile"""
    args = ctx.args or []
    if not args:
        await update.message.reply_text(
            "Usage: /indicator <type> [asset] [params]\n"
            "Example: /indicator volume_profile BTCUSDT\n"
            "Example: /indicator RSI divergence\n"
            "Example: /indicator support_resistance ETHUSDT"
        )
        return

    indicator_type = args[0]
    asset = args[1] if len(args) > 1 else ""
    params = " ".join(args[2:]) if len(args) > 2 else ""

    await _thinking(update)
    result = await ai_engine.draw_indicator(indicator_type, asset, params)
    _chat_context[update.effective_chat.id] = result
    await _send_long(update, result)


async def cmd_alerts(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Subscribe this chat to TradingView webhook alerts."""
    chat_id = update.effective_chat.id
    alert_subscribers.add(chat_id)
    webhook_url = config.WEBHOOK_URL or f"http://YOUR_SERVER:{config.WEBHOOK_PORT}/webhook"
    await update.message.reply_text(
        f"Alerts enabled for this chat.\n\n"
        f"<b>TradingView webhook URL:</b>\n"
        f"<code>{webhook_url}</code>\n\n"
        f"<b>Secret header:</b>\n"
        f"<code>X-Webhook-Secret: {config.WEBHOOK_SECRET}</code>\n\n"
        f"Set this as your alert webhook URL in TradingView. "
        f"The alert message should be the JSON from the PineScript alertcondition().",
        parse_mode=ParseMode.HTML,
    )


async def cmd_stopalerts(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    alert_subscribers.discard(chat_id)
    await update.message.reply_text("Alerts disabled for this chat.")


# ── Auto-signal scheduler ─────────────────────────────────────────────────────


async def _auto_signal_job(ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Scheduled job: run analysis and send chart + signal to a chat."""
    chat_id: int = ctx.job.data["chat_id"]
    asset: str = ctx.job.data["asset"]
    timeframe: str = ctx.job.data["timeframe"]

    logger.info("Auto-signal firing for %s %s -> chat %s", asset, timeframe, chat_id)

    try:
        await ctx.bot.send_chat_action(chat_id, ChatAction.TYPING)

        analysis_text, levels = await ai_engine.suggest_trade(asset, timeframe)

        # Record signal in database for self-learning
        session_info = market_sessions.get_current_sessions()
        signal_db.record_signal(
            chat_id=chat_id,
            asset=asset,
            timeframe=timeframe,
            direction=levels.get("direction") if levels else None,
            entry=levels.get("entry") if levels else None,
            sl=levels.get("sl") if levels else None,
            tp1=levels.get("tp1") if levels else None,
            tp2=levels.get("tp2") if levels else None,
            tp3=levels.get("tp3") if levels else None,
            market_session=session_info["primary_session"],
            session_detail=session_info,
            analysis_text=analysis_text,
            source="autosignal",
        )

        # Generate and send chart
        try:
            df = await chart.fetch_klines(asset, timeframe)
            img_bytes = chart.generate_chart(df, asset, timeframe, levels)
            await ctx.bot.send_photo(
                chat_id,
                photo=InputFile(io.BytesIO(img_bytes), filename=f"{asset}_{timeframe}.png"),
            )
        except Exception as e:
            logger.warning("Auto-signal chart failed: %s", e)

        # Send text analysis
        await _send_long_to_chat(ctx.bot, chat_id, analysis_text)

    except Exception as e:
        logger.error("Auto-signal job failed for %s %s: %s", asset, timeframe, e)


def _job_name(chat_id: int, asset: str, timeframe: str) -> str:
    return f"autosignal_{chat_id}_{asset}_{timeframe}"


async def cmd_autosignal(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Usage: /autosignal BTCUSDT 4H"""
    args = ctx.args or []
    if not args:
        await update.message.reply_text(
            "<b>Auto Signal — Scheduled alerts</b>\n\n"
            "Usage: /autosignal &lt;asset&gt; [timeframe]\n\n"
            "Examples:\n"
            "  /autosignal BTCUSDT 4H\n"
            "  /autosignal ETHUSDT 1H\n"
            "  /autosignal SOLUSDT 30M\n\n"
            "Timeframes: 15M, 30M, 1H, 2H, 4H, 6H, 8H, 12H, 1D\n\n"
            "Sends a full chart + analysis at every interval.\n"
            "Use /stopsignal to stop.",
            parse_mode=ParseMode.HTML,
        )
        return

    asset = args[0].upper()
    timeframe = args[1].upper() if len(args) > 1 else "4H"
    chat_id = update.effective_chat.id

    if timeframe not in INTERVAL_HOURS:
        await update.message.reply_text(
            f"Invalid timeframe: {timeframe}\n"
            f"Valid: {', '.join(INTERVAL_HOURS.keys())}"
        )
        return

    # Check if this exact signal already exists
    job_name = _job_name(chat_id, asset, timeframe)
    existing = ctx.job_queue.get_jobs_by_name(job_name)
    if existing:
        await update.message.reply_text(
            f"Already tracking <b>{asset} {timeframe}</b>. "
            f"Use /stopsignal to remove it first.",
            parse_mode=ParseMode.HTML,
        )
        return

    # Store subscription (memory + database)
    if chat_id not in _auto_signals:
        _auto_signals[chat_id] = []
    _auto_signals[chat_id].append({"asset": asset, "timeframe": timeframe})
    signal_db.save_autosignal_sub(chat_id, asset, timeframe)

    # Schedule the recurring job
    interval_secs = INTERVAL_HOURS[timeframe] * 3600
    ctx.job_queue.run_repeating(
        _auto_signal_job,
        interval=interval_secs,
        first=10,  # fire first signal in 10 seconds
        name=job_name,
        data={"chat_id": chat_id, "asset": asset, "timeframe": timeframe},
    )

    await update.message.reply_text(
        f"Auto-signal enabled: <b>{asset} {timeframe}</b>\n"
        f"You'll receive a chart + analysis every <b>{timeframe}</b>.\n"
        f"First signal coming in a few seconds...\n\n"
        f"Use /stopsignal to manage signals.",
        parse_mode=ParseMode.HTML,
    )


async def cmd_stopsignal(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Usage: /stopsignal [asset] [timeframe] — stop one or all auto-signals."""
    args = ctx.args or []
    chat_id = update.effective_chat.id

    if not _auto_signals.get(chat_id):
        await update.message.reply_text("No active auto-signals for this chat.")
        return

    if not args:
        # Stop all signals for this chat
        subs = _auto_signals.pop(chat_id, [])
        removed = []
        for sub in subs:
            name = _job_name(chat_id, sub["asset"], sub["timeframe"])
            for job in ctx.job_queue.get_jobs_by_name(name):
                job.schedule_removal()
            removed.append(f"{sub['asset']} {sub['timeframe']}")
        signal_db.remove_all_autosignal_subs(chat_id)
        await update.message.reply_text(
            f"Stopped all auto-signals:\n" + "\n".join(f"  - {r}" for r in removed)
        )
        return

    # Stop a specific signal
    asset = args[0].upper()
    timeframe = args[1].upper() if len(args) > 1 else "4H"
    name = _job_name(chat_id, asset, timeframe)

    jobs = ctx.job_queue.get_jobs_by_name(name)
    if not jobs:
        await update.message.reply_text(
            f"No active auto-signal for <b>{asset} {timeframe}</b>.",
            parse_mode=ParseMode.HTML,
        )
        return

    for job in jobs:
        job.schedule_removal()

    # Remove from our tracking dict + database
    subs = _auto_signals.get(chat_id, [])
    _auto_signals[chat_id] = [
        s for s in subs if not (s["asset"] == asset and s["timeframe"] == timeframe)
    ]
    if not _auto_signals[chat_id]:
        del _auto_signals[chat_id]
    signal_db.remove_autosignal_sub(chat_id, asset, timeframe)

    await update.message.reply_text(
        f"Stopped auto-signal: <b>{asset} {timeframe}</b>",
        parse_mode=ParseMode.HTML,
    )


async def cmd_signals(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Show active auto-signals for this chat."""
    chat_id = update.effective_chat.id
    subs = _auto_signals.get(chat_id, [])

    if not subs:
        await update.message.reply_text("No active auto-signals. Use /autosignal to add one.")
        return

    lines = [f"  - <b>{s['asset']} {s['timeframe']}</b>" for s in subs]
    await update.message.reply_text(
        "<b>Active auto-signals:</b>\n" + "\n".join(lines) + "\n\n"
        "Use /stopsignal to stop one or all.",
        parse_mode=ParseMode.HTML,
    )


# ── Outcome checker (background job) ─────────────────────────────────────────


async def _outcome_check_job(ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Periodic job that checks real price outcomes for all pending signals."""
    try:
        checked = await outcome_tracker.check_all_outcomes()
        if checked > 0:
            logger.info("Outcome checker: evaluated %d signals", checked)
    except Exception as e:
        logger.error("Outcome checker failed: %s", e)


# ── Performance command ──────────────────────────────────────────────────────


async def cmd_performance(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Show bot's signal track record — /performance [asset] [days]"""
    args = ctx.args or []
    asset = args[0].upper() if args else None
    days = 30
    if len(args) > 1:
        try:
            days = int(args[1])
        except ValueError:
            pass

    lines = [f"<b>Signal Performance Report</b> (last {days} days)\n"]

    # Overall stats
    perf = signal_db.get_performance_summary(asset=asset, days=days)
    total = perf.get("total_signals", 0)
    if total == 0:
        await update.message.reply_text(
            "No resolved signals yet. The bot needs time to track outcomes.\n"
            "Signals are checked automatically — results typically appear after "
            "a few candle closes.",
            parse_mode=ParseMode.HTML,
        )
        return

    tp1_wins = perf.get("tp1_wins", 0)
    tp2_wins = perf.get("tp2_wins", 0)
    tp3_wins = perf.get("tp3_wins", 0)
    sl_losses = perf.get("sl_losses", 0)
    win_rate = (tp1_wins / total * 100) if total > 0 else 0

    lines.append(f"<b>Overall:</b>")
    lines.append(f"  Signals resolved: {total}")
    lines.append(f"  Win rate (TP1+): <b>{win_rate:.1f}%</b>")
    lines.append(f"  TP1 hits: {tp1_wins} | TP2: {tp2_wins} | TP3: {tp3_wins}")
    lines.append(f"  SL hits: {sl_losses}")
    lines.append(f"  Avg P&L: <b>{perf.get('avg_pnl', 0):+.2f}%</b>")
    lines.append(f"  Avg max favorable: {perf.get('avg_max_favorable', 0):.2f}%")
    lines.append(f"  Avg max adverse: {perf.get('avg_max_adverse', 0):.2f}%")
    lines.append(f"  Avg candles to exit: {perf.get('avg_candles_to_exit', 0):.1f}")
    lines.append("")

    # By session
    session_perf = signal_db.get_session_performance(days=days)
    if session_perf:
        lines.append("<b>By Market Session:</b>")
        for sp in session_perf:
            emoji = "+" if (sp["avg_pnl"] or 0) >= 0 else "-"
            lines.append(
                f"  {sp['market_session']}: {sp['win_rate']}% WR "
                f"({sp['tp1_wins']}W/{sp['sl_losses']}L) "
                f"avg {sp['avg_pnl']:+.2f}%"
            )
        lines.append("")

    # By asset
    asset_perf = signal_db.get_asset_performance(days=days)
    if asset_perf:
        lines.append("<b>By Asset:</b>")
        for ap in asset_perf:
            lines.append(
                f"  {ap['asset']} {ap['timeframe']}: {ap['win_rate']}% WR "
                f"({ap['tp1_wins']}W/{ap['sl_losses']}L) "
                f"avg {ap['avg_pnl']:+.2f}%"
            )

    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)


# ── Free-form message handler ────────────────────────────────────────────────


async def handle_message(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Treat any plain text message as a trading question sent to Claude."""
    text = update.message.text
    if not text:
        return

    await _thinking(update)
    context = _chat_context.get(update.effective_chat.id, "")
    result = await ai_engine.analyze(text, context)
    _chat_context[update.effective_chat.id] = result
    await _send_long(update, result)


# ── Bot builder ───────────────────────────────────────────────────────────────


def _restore_autosignal_subs(app: Application) -> None:
    """Restore persisted autosignal subscriptions from the database on startup."""
    subs = signal_db.get_active_autosignal_subs()
    if not subs:
        return

    for sub in subs:
        chat_id = sub["chat_id"]
        asset = sub["asset"]
        timeframe = sub["timeframe"]

        if timeframe not in INTERVAL_HOURS:
            continue

        job_name = _job_name(chat_id, asset, timeframe)

        # Restore in-memory tracking
        if chat_id not in _auto_signals:
            _auto_signals[chat_id] = []
        _auto_signals[chat_id].append({"asset": asset, "timeframe": timeframe})

        # Schedule the job
        interval_secs = INTERVAL_HOURS[timeframe] * 3600
        app.job_queue.run_repeating(
            _auto_signal_job,
            interval=interval_secs,
            first=30,  # give 30s for bot to fully initialize
            name=job_name,
            data={"chat_id": chat_id, "asset": asset, "timeframe": timeframe},
        )
        logger.info("Restored autosignal: %s %s for chat %d", asset, timeframe, chat_id)

    logger.info("Restored %d autosignal subscriptions from database", len(subs))


def build_app() -> Application:
    """Build and return the Telegram Application (does not start polling)."""
    # Initialize the signal database
    signal_db.init_db()

    app = Application.builder().token(config.TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("analyze", cmd_analyze))
    app.add_handler(CommandHandler("script", cmd_script))
    app.add_handler(CommandHandler("indicator", cmd_indicator))
    app.add_handler(CommandHandler("alerts", cmd_alerts))
    app.add_handler(CommandHandler("stopalerts", cmd_stopalerts))
    app.add_handler(CommandHandler("autosignal", cmd_autosignal))
    app.add_handler(CommandHandler("stopsignal", cmd_stopsignal))
    app.add_handler(CommandHandler("signals", cmd_signals))
    app.add_handler(CommandHandler("performance", cmd_performance))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Schedule the outcome checker — runs every 15 minutes
    app.job_queue.run_repeating(
        _outcome_check_job,
        interval=900,  # 15 minutes
        first=60,      # first check 60s after startup
        name="outcome_checker",
    )
    logger.info("Outcome checker scheduled (every 15 min)")

    # Restore persisted autosignal subscriptions
    _restore_autosignal_subs(app)

    return app
