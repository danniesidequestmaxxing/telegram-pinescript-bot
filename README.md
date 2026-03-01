# PineScript Trading Assistant — Telegram Bot

A Telegram bot powered by Claude AI that provides real-time trading strategy analysis, generates PineScript v6 code, and forwards TradingView webhook alerts to your Telegram.

## Features

| Command | What it does |
|---------|-------------|
| `/analyze BTCUSDT 4H` | Full trade analysis — direction, entry, TP/SL, key levels, indicator confirmation, plus a PineScript indicator |
| `/script EMA crossover with volume filter` | Generates production-ready PineScript v6 code from plain English |
| `/indicator volume_profile ETHUSDT` | Generates a specific indicator (RSI, MACD, volume profile, S/R, etc.) |
| `/alerts` | Subscribe this chat to receive live TradingView webhook alerts |
| `/stopalerts` | Unsubscribe from alerts |
| *Any free text* | Treated as a trading question — ask anything |

## Quick Start

### 1. Prerequisites

- Python 3.11+
- A **Telegram Bot Token** — talk to [@BotFather](https://t.me/BotFather) on Telegram
- An **Anthropic API key** — from [console.anthropic.com](https://console.anthropic.com/)

### 2. Install

```bash
cd telegram-pinescript-bot
pip install -r requirements.txt
cp .env.example .env
```

### 3. Configure

Edit `.env` and fill in your keys:

```
TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
ANTHROPIC_API_KEY=sk-ant-...
WEBHOOK_SECRET=some_random_string
```

### 4. Run

```bash
python main.py
```

The bot starts polling Telegram and listening for webhooks on port 8443.

## TradingView Webhook Setup (for live alerts)

1. In Telegram, send `/alerts` to the bot — it will show you the webhook URL and secret.
2. In TradingView, create an alert on your chart.
3. Under **Notifications → Webhook URL**, paste the URL the bot gave you.
4. Add the header `X-Webhook-Secret: <your secret>` (TradingView Pro+ required for webhooks).
5. Set the **Alert message** to the JSON from the PineScript `alertcondition()`.

Example alert message:
```json
{"action": "LONG", "ticker": "BTCUSDT", "price": "{{close}}", "tp": "45000", "sl": "41000"}
```

When the alert fires, TradingView sends this JSON to your bot, which formats it and pushes it to your Telegram chat instantly.

## Architecture

```
┌─────────────┐     prompt      ┌─────────────┐    API     ┌──────────┐
│  Telegram    │ ──────────────► │   Bot        │ ────────► │  Claude  │
│  (you)       │ ◄────────────── │  (bot.py)    │ ◄──────── │  AI      │
└─────────────┘     response     └─────────────┘           └──────────┘
                                        ▲
                                        │ alerts
                                        │
                                 ┌──────┴──────┐
                                 │  Webhook     │ ◄──── TradingView
                                 │  Server      │       alert fires
                                 │  (port 8443) │
                                 └─────────────┘
```

## VPS Deployment Tips

- Open port 8443 (or your configured port) in your firewall for TradingView webhooks.
- Use a reverse proxy (nginx/caddy) with HTTPS for production — TradingView expects HTTPS webhook URLs.
- Run with `systemd` or `tmux`/`screen` to keep it alive.

Example systemd unit:
```ini
[Unit]
Description=PineScript Telegram Bot
After=network.target

[Service]
WorkingDirectory=/path/to/telegram-pinescript-bot
ExecStart=/usr/bin/python3 main.py
Restart=always
EnvironmentFile=/path/to/telegram-pinescript-bot/.env

[Install]
WantedBy=multi-user.target
```
