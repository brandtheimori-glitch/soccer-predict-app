import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from signal_parser import ManageCommand, TradeSignal, parse_message, validate  # noqa: E402


def test_multiline_gold_signal():
    s = parse_message("XAUUSD BUY @ 2350\nSL 2340\nTP1 2360\nTP2 2370")
    assert isinstance(s, TradeSignal)
    assert (s.symbol, s.side, s.order_kind) == ("XAUUSD", "BUY", "MARKET")
    assert s.entry == 2350 and s.sl == 2340 and s.tps == [2360, 2370]
    assert validate(s) is None


def test_alias_zone_and_slash_tps():
    s = parse_message("🔥 GOLD SELL NOW 2351-2355  SL: 2362  TP: 2345 / 2338 / 2330")
    assert s.symbol == "XAUUSD" and s.side == "SELL"
    assert (s.entry_low, s.entry_high, s.entry) == (2351, 2355, 2353)
    assert s.sl == 2362 and s.tps == [2345, 2338, 2330]
    assert validate(s) is None


def test_limit_order_lowercase():
    s = parse_message("EURUSD buy limit 1.0820 sl 1.0790 tp 1.0870")
    assert s.order_kind == "LIMIT" and s.entry == 1.082 and s.sl == 1.079 and s.tps == [1.087]


def test_slash_pair_and_emoji_labels():
    s = parse_message("📉 GBP/JPY SELL\nEntry: 191.50\n❌ Stop Loss: 192.10\n✅ Take Profit 1: 190.90\n✅ Take Profit 2: 190.20")
    assert s.symbol == "GBPJPY" and s.entry == 191.5 and s.sl == 192.1 and s.tps == [190.9, 190.2]


def test_market_without_entry():
    s = parse_message("BUY GOLD NOW\nSL 2330\nTP 2360")
    assert s.symbol == "XAUUSD" and s.entry is None and s.sl == 2330


def test_custom_symbol_and_alias():
    s = parse_message("BOOM1000 sell sl 10500 tp 10300", extra_symbols=["BOOM1000"])
    assert s.symbol == "BOOM1000"
    s = parse_message("JAPAN buy sl 38000 tp 39000", aliases={"JAPAN": "JP225"})
    assert s.symbol == "JP225"


def test_bad_sl_is_flagged():
    s = parse_message("XAUUSD BUY 2350 SL 2360 TP 2370")
    assert validate(s) is not None


def test_management_commands():
    c = parse_message("Close all trades now")
    assert isinstance(c, ManageCommand) and c.action == "CLOSE" and c.symbol is None
    c = parse_message("close gold")
    assert c.action == "CLOSE" and c.symbol == "XAUUSD"
    c = parse_message("Move SL to entry")
    assert c.action == "BREAKEVEN"
    c = parse_message("TP1 hit ✅ set breakeven")
    assert c.action == "BREAKEVEN"


def test_chatter_is_ignored():
    assert parse_message("Good morning traders! Market opens in 1 hour") is None
    assert parse_message("") is None


def test_buy_stop_is_not_stop_loss():
    s = parse_message("XAUUSD BUY STOP 2360 SL 2350 TP 2380")
    assert s.order_kind == "STOP" and s.entry == 2360 and s.sl == 2350 and s.tps == [2380]
