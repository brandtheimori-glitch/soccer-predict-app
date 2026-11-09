"""Telegram -> MT5 signal copier.

Logs into YOUR Telegram account (not a bot account, so it can read any group
you're a member of), listens to one group/channel, parses each new message and
hands it to the MT5 trader.

    python bot.py                 # uses config.yaml
    python bot.py my-config.yaml
"""

from __future__ import annotations

import asyncio
import logging
import sys
from concurrent.futures import ThreadPoolExecutor

import yaml
from telethon import TelegramClient, events

from signal_parser import ManageCommand, TradeSignal, parse_message
from trader import Trader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("bot.log", encoding="utf-8")],
)
log = logging.getLogger("bot")


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_chat(value):
    # Numeric ids (e.g. -1001234567890) must be ints for Telethon.
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


async def main(cfg_path: str) -> None:
    cfg = load_config(cfg_path)
    tg = cfg["telegram"]
    parser_cfg = cfg.get("parser") or {}

    trader = Trader(cfg)
    # The MT5 API isn't thread-safe: run every call on one dedicated thread.
    mt5_thread = ThreadPoolExecutor(max_workers=1)
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(mt5_thread, trader.connect)
    if trader.dry_run:
        log.warning("DRY RUN is ON - no real orders will be sent. Set dry_run: false to trade.")

    client = TelegramClient(tg.get("session_name", "signal_session"), int(tg["api_id"]), tg["api_hash"])
    await client.start(phone=tg.get("phone"))  # first run asks for the login code Telegram sends you
    chat = await client.get_entity(resolve_chat(tg["chat"]))
    log.info("Listening to: %s", getattr(chat, "title", chat))

    seen: set[int] = set()

    @client.on(events.NewMessage(chats=chat))
    async def on_message(event):
        msg = event.message
        if msg.id in seen:
            return
        seen.add(msg.id)
        text = msg.message or ""
        result = parse_message(text, parser_cfg.get("aliases"), parser_cfg.get("extra_symbols"))
        if result is None:
            log.debug("Ignored message %s: %r", msg.id, text[:80])
            return
        log.info("Message %s -> %s", msg.id, result)
        try:
            if isinstance(result, TradeSignal):
                await loop.run_in_executor(mt5_thread, trader.open_signal, result, f"tg{msg.id}")
            elif isinstance(result, ManageCommand):
                reply_tag = f"tg{msg.reply_to_msg_id}#" if msg.reply_to_msg_id else None
                await loop.run_in_executor(mt5_thread, trader.manage, result, reply_tag)
        except Exception:
            log.exception("Failed to handle message %s", msg.id)

    try:
        await client.run_until_disconnected()
    finally:
        await loop.run_in_executor(mt5_thread, trader.shutdown)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "config.yaml"))
