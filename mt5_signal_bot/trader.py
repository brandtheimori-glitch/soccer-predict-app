"""Place and manage MT5 orders from parsed signals.

Uses the official `MetaTrader5` Python package, which only runs on Windows
(or Wine) next to an installed, logged-in MT5 terminal. Every order the bot
places carries its magic number, so it never touches your manual trades.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional

from signal_parser import ManageCommand, TradeSignal, validate

log = logging.getLogger("trader")

try:
    import MetaTrader5 as mt5  # type: ignore
except ImportError:  # lets the parser / dry-run work on any OS
    mt5 = None


class Trader:
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.mt = cfg["mt5"]
        self.risk = cfg["risk"]
        self.dry_run = bool(cfg.get("dry_run", True))
        self.magic = int(self.mt.get("magic", 20261109))

    # ------------------------------------------------------------------ setup
    def connect(self) -> None:
        if mt5 is None:
            if self.dry_run:
                log.warning("MetaTrader5 package not available - running parse-only dry run.")
                return
            raise RuntimeError("pip install MetaTrader5 (Windows only) to trade live.")
        kwargs = {}
        if self.mt.get("terminal_path"):
            kwargs["path"] = self.mt["terminal_path"]
        if self.mt.get("login"):
            kwargs.update(
                login=int(self.mt["login"]),
                password=str(self.mt["password"]),
                server=str(self.mt["server"]),
            )
        if not mt5.initialize(**kwargs):
            raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
        acc = mt5.account_info()
        log.info("Connected to MT5 account %s (%s), balance %.2f %s",
                 acc.login, acc.server, acc.balance, acc.currency)
        if acc.trade_mode != mt5.ACCOUNT_TRADE_MODE_DEMO:
            log.warning("*** THIS IS A REAL-MONEY ACCOUNT ***")
        if not mt5.terminal_info().trade_allowed:
            log.warning("'Algo Trading' is OFF in the MT5 terminal - orders will be rejected.")

    def shutdown(self) -> None:
        if mt5 is not None:
            mt5.shutdown()

    # ---------------------------------------------------------------- helpers
    def broker_symbol(self, symbol: str) -> str:
        mapping = {k.upper(): v for k, v in (self.mt.get("symbol_map") or {}).items()}
        if symbol in mapping:
            return mapping[symbol]
        return f"{self.mt.get('symbol_prefix', '')}{symbol}{self.mt.get('symbol_suffix', '')}"

    def _bot_positions(self, symbol: Optional[str] = None, comment: Optional[str] = None):
        positions = (mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()) or []
        out = [p for p in positions if p.magic == self.magic]
        if comment:
            out = [p for p in out if p.comment.startswith(comment)]
        return out

    def _bot_orders(self, symbol: Optional[str] = None, comment: Optional[str] = None):
        orders = (mt5.orders_get(symbol=symbol) if symbol else mt5.orders_get()) or []
        out = [o for o in orders if o.magic == self.magic]
        if comment:
            out = [o for o in out if o.comment.startswith(comment)]
        return out

    @staticmethod
    def _filling(info) -> int:
        # symbol_info.filling_mode is a bitmask: 1 = FOK allowed, 2 = IOC allowed.
        if info.filling_mode & 1:
            return mt5.ORDER_FILLING_FOK
        if info.filling_mode & 2:
            return mt5.ORDER_FILLING_IOC
        return mt5.ORDER_FILLING_RETURN

    @staticmethod
    def _round_volume(vol: float, info) -> float:
        step = info.volume_step
        vol = math.floor(vol / step + 1e-9) * step
        vol = min(vol, info.volume_max)
        return round(vol, 8) if vol >= info.volume_min else 0.0

    def _total_volume(self, symbol: str, side: str, price: float, sl: Optional[float], info) -> float:
        mode = self.risk.get("mode", "fixed")
        if mode == "risk_percent" and sl:
            acc = mt5.account_info()
            risk_money = acc.balance * float(self.risk["risk_percent"]) / 100.0
            order_type = mt5.ORDER_TYPE_BUY if side == "BUY" else mt5.ORDER_TYPE_SELL
            loss_per_lot = mt5.order_calc_profit(order_type, symbol, 1.0, price, sl)
            if not loss_per_lot:
                log.error("Could not compute risk for %s, falling back to fixed lot", symbol)
                return float(self.risk.get("fixed_lot", 0.01))
            return risk_money / abs(loss_per_lot)
        return float(self.risk.get("fixed_lot", 0.01))

    def _send(self, request: dict) -> bool:
        if self.dry_run:
            log.info("[DRY RUN] would send: %s", request)
            return True
        result = mt5.order_send(request)
        if result is None or result.retcode not in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED):
            log.error("Order rejected: %s | %s", result, mt5.last_error())
            return False
        log.info("Order OK: ticket=%s volume=%s price=%s", result.order, result.volume, result.price)
        return True

    # ------------------------------------------------------------- new trades
    def open_signal(self, sig: TradeSignal, tag: str) -> None:
        problem = validate(sig)
        if problem:
            log.warning("Skipping signal (%s): %r", problem, sig.raw)
            return
        allowed = [s.upper() for s in self.risk.get("allowed_symbols") or []]
        if allowed and sig.symbol not in allowed:
            log.info("Skipping %s - not in allowed_symbols", sig.symbol)
            return
        if self.risk.get("require_sl", True) and sig.sl is None:
            log.warning("Skipping signal without SL (require_sl is on): %r", sig.raw)
            return

        if mt5 is None:  # parse-only dry run on non-Windows
            log.info("[DRY RUN] parsed signal: %s", sig)
            return

        symbol = self.broker_symbol(sig.symbol)
        if not mt5.symbol_select(symbol, True):
            log.error("Symbol %s not found at broker - set symbol_suffix / symbol_map", symbol)
            return
        info = mt5.symbol_info(symbol)
        tick = mt5.symbol_info_tick(symbol)

        max_open = int(self.risk.get("max_open_trades", 0))
        if max_open and len(self._bot_positions()) >= max_open:
            log.warning("max_open_trades (%d) reached - skipping", max_open)
            return

        is_buy = sig.side == "BUY"
        market_price = tick.ask if is_buy else tick.bid
        tps = sig.tps[: int(self.risk.get("max_tps", 3)) or None] or [None]

        # Decide market vs pending.
        if sig.order_kind == "MARKET":
            action = mt5.TRADE_ACTION_DEAL
            order_type = mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL
            price = market_price
            if sig.entry is not None:
                max_dist = float(self.risk.get("max_entry_distance", 0))
                in_zone = sig.entry_low <= market_price <= sig.entry_high
                dist = abs(market_price - sig.entry)
                if max_dist and not in_zone and dist > max_dist:
                    log.warning("Price %.5f is %.5f away from entry %.5f (limit %s) - skipping",
                                market_price, dist, sig.entry, max_dist)
                    return
        else:
            action = mt5.TRADE_ACTION_PENDING
            price = sig.entry
            order_type = {
                ("BUY", "LIMIT"): mt5.ORDER_TYPE_BUY_LIMIT,
                ("SELL", "LIMIT"): mt5.ORDER_TYPE_SELL_LIMIT,
                ("BUY", "STOP"): mt5.ORDER_TYPE_BUY_STOP,
                ("SELL", "STOP"): mt5.ORDER_TYPE_SELL_STOP,
            }[(sig.side, sig.order_kind)]

        # Live-price sanity checks (e.g. market BUY whose SL is already above price).
        if sig.sl is not None and ((is_buy and sig.sl >= price) or (not is_buy and sig.sl <= price)):
            log.warning("SL %.5f is on the wrong side of price %.5f - skipping", sig.sl, price)
            return
        tps = [tp for tp in tps if tp is None or (tp > price if is_buy else tp < price)] or [None]

        total = self._total_volume(symbol, sig.side, price, sig.sl, info)
        total = min(total, float(self.risk.get("max_lot", total)))
        per_tp = self._round_volume(total / len(tps), info)
        if per_tp == 0.0:  # too small to split; put it all on the first TP
            tps = tps[:1]
            per_tp = self._round_volume(total, info) or info.volume_min
        log.info("Opening %s %s x%d @ %.5f lots=%s SL=%s TPs=%s",
                 sig.side, symbol, len(tps), price, per_tp, sig.sl, tps)

        for i, tp in enumerate(tps, 1):
            request = {
                "action": action,
                "symbol": symbol,
                "volume": per_tp,
                "type": order_type,
                "price": round(price, info.digits),
                "deviation": int(self.mt.get("deviation_points", 20)),
                "magic": self.magic,
                "comment": f"{tag}#{i}"[:31],
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": self._filling(info),
            }
            if sig.sl is not None:
                request["sl"] = round(sig.sl, info.digits)
            if tp is not None:
                request["tp"] = round(tp, info.digits)
            self._send(request)

    # ------------------------------------------------------------- management
    def manage(self, cmd: ManageCommand, reply_tag: Optional[str]) -> None:
        if mt5 is None:
            log.info("[DRY RUN] parsed command: %s (reply to %s)", cmd, reply_tag)
            return
        symbol = self.broker_symbol(cmd.symbol) if cmd.symbol else None
        if not reply_tag and not symbol and not self.risk.get("allow_global_commands", False):
            log.warning("Ignoring '%s' - not a reply to a signal and no symbol named "
                        "(set allow_global_commands: true to act on all bot trades)", cmd.raw)
            return
        positions = self._bot_positions(symbol, reply_tag)

        if cmd.action == "BREAKEVEN":
            for p in positions:
                self._send({"action": mt5.TRADE_ACTION_SLTP, "symbol": p.symbol,
                            "position": p.ticket, "sl": p.price_open, "tp": p.tp,
                            "magic": self.magic})
            log.info("Breakeven applied to %d position(s)", len(positions))
            return

        if cmd.action == "CLOSE":
            for p in positions:
                tick = mt5.symbol_info_tick(p.symbol)
                is_buy = p.type == mt5.POSITION_TYPE_BUY
                self._send({
                    "action": mt5.TRADE_ACTION_DEAL, "symbol": p.symbol, "position": p.ticket,
                    "volume": p.volume,
                    "type": mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY,
                    "price": tick.bid if is_buy else tick.ask,
                    "deviation": int(self.mt.get("deviation_points", 20)),
                    "magic": self.magic, "comment": "tg close",
                    "type_filling": self._filling(mt5.symbol_info(p.symbol)),
                })
            orders = self._bot_orders(symbol, reply_tag)
            for o in orders:
                self._send({"action": mt5.TRADE_ACTION_REMOVE, "order": o.ticket})
            log.info("Closed %d position(s), cancelled %d pending order(s)", len(positions), len(orders))
