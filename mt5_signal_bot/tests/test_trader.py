"""Exercise Trader against a fake MetaTrader5 module (the real one is Windows-only)."""

import os
import sys
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import trader as trader_mod  # noqa: E402
from signal_parser import parse_message  # noqa: E402


class FakeMT5:
    TRADE_ACTION_DEAL, TRADE_ACTION_PENDING, TRADE_ACTION_SLTP, TRADE_ACTION_REMOVE = 1, 5, 6, 8
    ORDER_TYPE_BUY, ORDER_TYPE_SELL = 0, 1
    ORDER_TYPE_BUY_LIMIT, ORDER_TYPE_SELL_LIMIT, ORDER_TYPE_BUY_STOP, ORDER_TYPE_SELL_STOP = 2, 3, 4, 5
    ORDER_FILLING_FOK, ORDER_FILLING_IOC, ORDER_FILLING_RETURN = 0, 1, 2
    ORDER_TIME_GTC = 0
    POSITION_TYPE_BUY = 0
    TRADE_RETCODE_DONE, TRADE_RETCODE_PLACED = 10009, 10008

    def __init__(self, bid, ask):
        self.tick = NS(bid=bid, ask=ask)
        self.sent = []
        self.positions = []

    def symbol_select(self, s, _):
        return s == "XAUUSDm"

    def symbol_info(self, s):
        return NS(volume_step=0.01, volume_min=0.01, volume_max=100, digits=2, filling_mode=2)

    def symbol_info_tick(self, s):
        return self.tick

    def account_info(self):
        return NS(balance=10_000)

    def order_calc_profit(self, t, s, vol, price, sl):
        return -abs(price - sl) * 100 * vol  # gold: $100 per $1 move per lot

    def positions_get(self, symbol=None):
        return self.positions

    def orders_get(self, symbol=None):
        return []

    def order_send(self, req):
        self.sent.append(req)
        return NS(retcode=self.TRADE_RETCODE_DONE, order=1, volume=req.get("volume"), price=req.get("price"))

    def last_error(self):
        return (0, "")


CFG = {
    "dry_run": False,
    "mt5": {"magic": 7, "symbol_suffix": "m"},
    "risk": {"mode": "risk_percent", "risk_percent": 1.0, "max_lot": 5, "max_tps": 3,
             "require_sl": True, "max_entry_distance": 3.0},
}


def make(bid, ask):
    fake = FakeMT5(bid, ask)
    trader_mod.mt5 = fake
    return trader_mod.Trader(CFG), fake


def test_market_buy_split_with_risk_sizing():
    t, fake = make(2350.0, 2350.3)
    t.open_signal(parse_message("XAUUSD BUY 2350 SL 2340 TP1 2360 TP2 2370"), "tg42")
    assert len(fake.sent) == 2
    r = fake.sent[0]
    assert r["symbol"] == "XAUUSDm" and r["type"] == FakeMT5.ORDER_TYPE_BUY
    assert r["comment"] == "tg42#1" and r["sl"] == 2340 and r["tp"] == 2360
    # 1% of 10k = $100 risk; ~10.3 SL distance * $100/lot -> ~0.097 lots total -> 0.04 each
    assert r["volume"] == 0.04 and r["type_filling"] == FakeMT5.ORDER_FILLING_IOC


def test_skips_when_price_ran_away():
    t, fake = make(2360.0, 2360.3)
    t.open_signal(parse_message("XAUUSD BUY 2350 SL 2340 TP 2380"), "tg1")
    assert fake.sent == []


def test_skips_without_sl_and_unknown_symbol():
    t, fake = make(2350.0, 2350.3)
    t.open_signal(parse_message("XAUUSD BUY 2350 TP 2380"), "tg1")
    t.open_signal(parse_message("EURUSD BUY SL 1.07 TP 1.09"), "tg2")  # EURUSDm not at broker
    assert fake.sent == []


def test_sell_limit_is_pending():
    t, fake = make(2340.0, 2340.3)
    t.open_signal(parse_message("GOLD SELL LIMIT 2355 SL 2365 TP 2335"), "tg3")
    assert fake.sent[0]["action"] == FakeMT5.TRADE_ACTION_PENDING
    assert fake.sent[0]["type"] == FakeMT5.ORDER_TYPE_SELL_LIMIT and fake.sent[0]["price"] == 2355


def test_reply_close_only_hits_that_signal():
    t, fake = make(2350.0, 2350.3)
    pos = lambda tk, c: NS(ticket=tk, symbol="XAUUSDm", magic=7, comment=c, type=0,
                           volume=0.02, price_open=2349.0, tp=2360.0)
    fake.positions = [pos(1, "tg42#1"), pos(2, "tg42#2"), pos(3, "tg99#1"), NS(**{**vars(pos(4, "manual")), "magic": 0})]
    t.manage(parse_message("close now"), "tg42#")
    assert sorted(r["position"] for r in fake.sent) == [1, 2]
    fake.sent.clear()
    t.manage(parse_message("move SL to breakeven"), None)  # no reply, no symbol -> ignored
    assert fake.sent == []
