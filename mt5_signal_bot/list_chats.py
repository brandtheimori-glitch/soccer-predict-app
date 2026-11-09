"""Print the id and name of every chat/group/channel on your Telegram account.

Run once to find the id of the signal group, then put it in config.yaml.

    python list_chats.py
"""

import asyncio

import yaml
from telethon import TelegramClient


async def main() -> None:
    with open("config.yaml", encoding="utf-8") as f:
        tg = yaml.safe_load(f)["telegram"]
    async with TelegramClient(tg.get("session_name", "signal_session"), int(tg["api_id"]), tg["api_hash"]) as client:
        async for d in client.iter_dialogs():
            kind = "channel" if d.is_channel else "group" if d.is_group else "user"
            print(f"{d.id:>16}  {kind:<8} {d.name}")


if __name__ == "__main__":
    asyncio.run(main())
