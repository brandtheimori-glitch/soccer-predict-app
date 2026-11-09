# Telegram → MT5 Signal Copier

Reads new messages from **one Telegram group you follow**, works out the trade
(symbol, buy/sell, entry, SL, TPs) and places it in your **MetaTrader 5**
account automatically. It also follows up messages like *"close gold"* or
*"move SL to breakeven"*.

> ⚠️ **Risk warning.** Copying signals automatically can lose money fast:
> a misread message, a typo from the provider or a price spike all become real
> trades. Run it on a **demo account** with `dry_run: true` first, watch
> `bot.log` for a few days, and only then go live with small size.

## How it works

```
Telegram group ──(your account, via Telethon)──► signal_parser.py ──► trader.py ──► MT5 terminal
```

- **bot.py** logs into *your* Telegram account (not a Telegram bot), so it can
  read any group you're in, even if you're not an admin.
- **signal_parser.py** turns the text into a trade or a command.
- **trader.py** sizes the position, splits it over the TPs, and sends it to
  MT5 through the official `MetaTrader5` Python package.

Each trade is tagged with a magic number and the Telegram message id, so the bot
never touches your manual trades, and a *"close"* sent as a **reply** to a
signal closes only that signal's trades.

## Requirements

- **Windows** PC or VPS (the MetaTrader5 Python package is Windows-only).
- MetaTrader 5 terminal installed and logged in to your broker.
- Python 3.10+ from python.org (tick *"Add to PATH"* when installing).
- The PC/VPS must stay on while you want it to trade.

## Setup (once)

1. **Telegram API keys:** go to <https://my.telegram.org> → *API development
   tools* → create an app. Copy the `api_id` and `api_hash`.
2. **Install:** open a terminal in this folder and run
   ```
   pip install -r requirements.txt
   ```
3. **Config:** copy `config.example.yaml` to `config.yaml` and fill in your
   Telegram keys, phone number and MT5 login.
4. **Find the group id:**
   ```
   python list_chats.py
   ```
   The first run asks for the login code Telegram sends you. Find the signal
   group in the list and put its id (e.g. `-1001234567890`) under
   `telegram.chat` in `config.yaml`.
5. **MT5 terminal:** click the **Algo Trading** button so it's green, and check
   your broker's symbol names (Market Watch). If gold is `XAUUSDm`, set
   `symbol_suffix: "m"`; for odd names use `symbol_map`.

## Run

```
python bot.py
```

Leave it running. Everything it does is printed and written to `bot.log`.
With `dry_run: true` it only logs what it *would* send. When you're happy,
set `dry_run: false` and restart.

## Supported signal formats

The parser is flexible. All of these work:

```
XAUUSD BUY @ 2350
SL 2340
TP1 2360
TP2 2370
```
```
🔥 GOLD SELL NOW 2351-2355  SL: 2362  TP: 2345 / 2338 / 2330
```
```
EURUSD buy limit 1.0820 sl 1.0790 tp 1.0870
```
```
📉 GBP/JPY SELL
Entry: 191.50
❌ Stop Loss: 192.10
✅ Take Profit 1: 190.90
```

| Message | What the bot does |
|---|---|
| BUY / SELL (with or without a price) | Market order. Skipped if price has moved more than `max_entry_distance` from the quoted entry/zone. |
| BUY LIMIT / SELL STOP etc. | Pending order at the given price. |
| Several TPs | Lot is split evenly, one trade per TP (up to `max_tps`). |
| "close", "close gold", "exit" | Closes the bot's trades (and cancels its pending orders). |
| "move SL to entry", "breakeven" | Moves SL to the open price. |

Management messages act on the signal they **reply to**, or on the symbol they
name. A bare "close all" with neither is ignored unless you set
`allow_global_commands: true`.

**Every group writes differently.** If yours uses a format that isn't picked
up, add an example to `tests/test_parser.py` and adjust `signal_parser.py`.
Run the tests with `python -m pytest tests`.

## Risk settings (in `config.yaml`)

- `mode: fixed` — always trade `fixed_lot` lots per signal.
- `mode: risk_percent` — size so hitting SL loses `risk_percent` % of balance.
- `max_lot`, `max_open_trades`, `require_sl`, `allowed_symbols` — safety caps.

## Limits to know about

- Edited or deleted Telegram messages are ignored; only new messages count.
- Signals sent as **images** can't be read.
- "Close half" / partial closes are treated as a full close.
- Keep `config.yaml` and the `*.session` file private: the session file is a
  login to your Telegram account. Both are git-ignored.
