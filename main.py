#!/usr/bin/env python3
"""Entry point — runs the Telegram bot and TradingView webhook server together."""

import asyncio
import logging

from aiohttp import web

import config
import bot
import webhook_server

logging.basicConfig(
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


async def main() -> None:
    # ── Validate config ──────────────────────────────────────────────────
    if not config.TELEGRAM_BOT_TOKEN:
        raise SystemExit("TELEGRAM_BOT_TOKEN is not set. Copy .env.example to .env and fill it in.")
    if not config.ANTHROPIC_API_KEY:
        raise SystemExit("ANTHROPIC_API_KEY is not set. Copy .env.example to .env and fill it in.")

    # ── Build Telegram bot ───────────────────────────────────────────────
    app = bot.build_app()

    # ── Wire up webhook server ───────────────────────────────────────────
    webhook_server.init(app, lambda: bot.alert_subscribers)
    web_app = webhook_server.create_webhook_app()

    runner = web.AppRunner(web_app)
    await runner.setup()
    site = web.TCPSite(runner, config.WEBHOOK_HOST, config.WEBHOOK_PORT)

    # ── Start both ───────────────────────────────────────────────────────
    logger.info(
        "Starting webhook server on %s:%s", config.WEBHOOK_HOST, config.WEBHOOK_PORT
    )
    await site.start()

    logger.info("Starting Telegram bot polling...")
    await app.initialize()
    await app.start()
    await app.updater.start_polling()

    logger.info("Bot is running. Press Ctrl+C to stop.")

    # Keep alive until interrupted
    stop_event = asyncio.Event()
    try:
        await stop_event.wait()
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        logger.info("Shutting down...")
        await app.updater.stop()
        await app.stop()
        await app.shutdown()
        await runner.cleanup()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
