"""Lightweight aiohttp server that receives TradingView webhook alerts
and forwards them to subscribed Telegram chats."""

import json
import logging

from aiohttp import web

import config

logger = logging.getLogger(__name__)

# These are set by main.py at startup
_bot_app = None  # telegram Application
_get_subscribers = None  # callable -> set[int]


def init(bot_app, get_subscribers):
    """Wire up the bot application and subscriber accessor."""
    global _bot_app, _get_subscribers
    _bot_app = bot_app
    _get_subscribers = get_subscribers


def _format_alert(data: dict) -> str:
    """Turn a webhook JSON payload into a human-readable Telegram message."""
    action = data.get("action", "SIGNAL").upper()
    ticker = data.get("ticker", "???")
    price = data.get("price", "—")
    tp = data.get("tp", "—")
    sl = data.get("sl", "—")
    extra = {k: v for k, v in data.items() if k not in ("action", "ticker", "price", "tp", "sl")}

    arrow = "\u2B06" if action == "LONG" else "\u2B07" if action == "SHORT" else "\u26A1"
    lines = [
        f"{arrow} <b>{action} {ticker}</b>",
        f"Price: <code>{price}</code>",
        f"TP: <code>{tp}</code>",
        f"SL: <code>{sl}</code>",
    ]
    if extra:
        lines.append(f"Details: <code>{json.dumps(extra)}</code>")
    return "\n".join(lines)


async def handle_webhook(request: web.Request) -> web.Response:
    """POST /webhook — receives TradingView alert JSON."""
    # Verify secret
    secret = request.headers.get("X-Webhook-Secret", "")
    if secret != config.WEBHOOK_SECRET:
        return web.Response(status=403, text="Forbidden")

    try:
        body = await request.text()
        # TradingView sends the alertcondition message as the body
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            # Fallback: plain text alert
            data = {"action": "ALERT", "message": body}
    except Exception:
        return web.Response(status=400, text="Bad request")

    msg = _format_alert(data)
    logger.info("Webhook alert received: %s", msg)

    subscribers = _get_subscribers() if _get_subscribers else set()
    bot = _bot_app.bot if _bot_app else None

    if bot and subscribers:
        for chat_id in subscribers:
            try:
                await bot.send_message(
                    chat_id=chat_id,
                    text=msg,
                    parse_mode="HTML",
                )
            except Exception as e:
                logger.error("Failed to send alert to chat %s: %s", chat_id, e)

    return web.Response(text="OK")


def create_webhook_app() -> web.Application:
    """Create the aiohttp web app for the webhook server."""
    app = web.Application()
    app.router.add_post("/webhook", handle_webhook)
    # Health check
    app.router.add_get("/health", lambda _: web.Response(text="OK"))
    return app
