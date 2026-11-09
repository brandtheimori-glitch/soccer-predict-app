"""Turn Telegram signal messages into structured trade instructions.

Signal groups word things differently, so this parser is deliberately
forgiving. It understands messages such as:

    XAUUSD BUY @ 2350
    SL 2340
    TP1 2360
    TP2 2370

    GOLD SELL NOW 2351-2355  SL: 2362  TP: 2345 / 2338 / 2330

    EURUSD buy limit 1.0820 sl 1.0790 tp 1.0870

and management messages such as "close all", "close gold",
"move SL to entry" / "set breakeven".

It has no dependency on MetaTrader 5, so it can be tested anywhere.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

NUMBER = r"(\d+(?:[.,]\d+)?)"

# Common nicknames used by signal providers -> canonical symbol.
DEFAULT_ALIASES = {
    "GOLD": "XAUUSD",
    "XAU": "XAUUSD",
    "SILVER": "XAGUSD",
    "XAG": "XAGUSD",
    "OIL": "USOIL",
    "WTI": "USOIL",
    "BRENT": "UKOIL",
    "BTC": "BTCUSD",
    "BITCOIN": "BTCUSD",
    "ETH": "ETHUSD",
    "US30": "US30",
    "DOW": "US30",
    "NAS100": "NAS100",
    "NASDAQ": "NAS100",
    "US100": "NAS100",
    "SPX500": "SPX500",
    "US500": "SPX500",
    "GER40": "GER40",
    "DAX": "GER40",
}

CURRENCIES = "USD|EUR|GBP|JPY|CHF|AUD|NZD|CAD|XAU|XAG|SGD|HKD|NOK|SEK|ZAR|MXN|TRY|PLN"
FX_PAIR = re.compile(rf"\b((?:{CURRENCIES})(?:{CURRENCIES}))\b")


@dataclass
class TradeSignal:
    symbol: str
    side: str  # "BUY" or "SELL"
    order_kind: str = "MARKET"  # MARKET, LIMIT or STOP
    entry: Optional[float] = None  # single price (midpoint if a zone was given)
    entry_low: Optional[float] = None
    entry_high: Optional[float] = None
    sl: Optional[float] = None
    tps: list[float] = field(default_factory=list)
    raw: str = ""


@dataclass
class ManageCommand:
    action: str  # "CLOSE" or "BREAKEVEN"
    symbol: Optional[str] = None  # None = applies to the replied-to signal, or all
    raw: str = ""


def _num(text: str) -> float:
    return float(text.replace(",", "."))


def _normalise(text: str) -> str:
    text = text.upper()
    # Strip emoji / decoration but keep characters that carry meaning.
    text = re.sub(r"[^\w\s.,:@/\-]", " ", text)
    # "XAU/USD" -> "XAUUSD"
    text = re.sub(r"\b([A-Z]{3})\s*/\s*([A-Z]{3})\b", r"\1\2", text)
    return text


def _find_symbol(text: str, aliases: dict[str, str], extra_symbols: list[str]) -> Optional[str]:
    for sym in sorted(extra_symbols, key=len, reverse=True):
        if re.search(rf"\b{re.escape(sym.upper())}\b", text):
            return sym.upper()
    m = FX_PAIR.search(text)
    if m:
        return m.group(1)
    for alias in sorted(aliases, key=len, reverse=True):
        if re.search(rf"\b{re.escape(alias)}\b", text):
            return aliases[alias]
    return None


def _find_side(text: str) -> tuple[Optional[str], str]:
    m = re.search(r"\b(BUY|SELL|LONG|SHORT)\b(?:\s+(LIMIT|STOP))?", text)
    if not m:
        return None, "MARKET"
    side = {"LONG": "BUY", "SHORT": "SELL"}.get(m.group(1), m.group(1))
    return side, (m.group(2) or "MARKET")


def _find_sl(text: str) -> Optional[float]:
    m = re.search(rf"\b(?:SL|S/L|STOP\s*LOSS|STOPLOSS)\b\s*[:@=\-]?\s*{NUMBER}", text)
    return _num(m.group(1)) if m else None


def _find_tps(text: str) -> list[float]:
    tps: list[float] = []
    label = r"\b(?:TP|T/P|TAKE\s*PROFIT|TARGET)\s*\d?\b\s*[:@=\-]?\s*"
    for m in re.finditer(label + NUMBER + r"((?:\s*[/|,]\s*\d+(?:\.\d+)?)*)", text):
        tps.append(_num(m.group(1)))
        # "TP: 2345 / 2338 / 2330"
        for extra in re.findall(r"\d+(?:\.\d+)?", m.group(2) or ""):
            tps.append(_num(extra))
    # De-duplicate while keeping order.
    seen, out = set(), []
    for tp in tps:
        if tp not in seen:
            seen.add(tp)
            out.append(tp)
    return out


def _find_entry(text: str, side_pos: int) -> tuple[Optional[float], Optional[float]]:
    # Explicit "ENTRY 2350" / "@ 2350" / "PRICE 2350"; zones "2350-2355" or "2350/2355".
    zone = rf"{NUMBER}(?:\s*(?:-|/|TO)\s*{NUMBER})?"
    m = re.search(rf"(?:\bENTRY\b|\bENTER\b|\bPRICE\b|\bAT\b|@)\s*[:=\-]?\s*{zone}", text)
    if not m:
        # Fall back to the first number after BUY/SELL (and optional NOW/LIMIT/STOP).
        tail = text[side_pos:]
        m = re.match(rf"\s*(?:BUY|SELL|LONG|SHORT)(?:\s+(?:LIMIT|STOP|NOW))*\s*[:@=\-]?\s*{zone}", tail)
    if not m:
        return None, None
    low = _num(m.group(1))
    high = _num(m.group(2)) if m.group(2) else low
    return min(low, high), max(low, high)


def parse_message(
    text: str,
    aliases: Optional[dict[str, str]] = None,
    extra_symbols: Optional[list[str]] = None,
) -> Optional[TradeSignal | ManageCommand]:
    """Return a TradeSignal, a ManageCommand, or None if the message isn't actionable."""
    if not text or not text.strip():
        return None
    aliases = {**DEFAULT_ALIASES, **{k.upper(): v.upper() for k, v in (aliases or {}).items()}}
    extra_symbols = extra_symbols or []
    t = _normalise(text)

    symbol = _find_symbol(t, aliases, extra_symbols)

    # --- management messages -------------------------------------------------
    if re.search(r"\b(BREAK\s*EVEN|BREAKEVEN|BE)\b", t) and re.search(
        r"\b(MOVE|SET|PUT|SL|STOP|SECURE|BREAK\s*EVEN|BREAKEVEN)\b", t
    ) or re.search(r"\b(?:MOVE|SET)\s+(?:SL|STOP(?:\s*LOSS)?)\s+TO\s+(?:ENTRY|BE)\b", t):
        if not re.search(r"\b(BUY|SELL)\b", t):
            return ManageCommand("BREAKEVEN", symbol, text)
    if re.search(r"\b(CLOSE|EXIT|CANCEL)\b", t) and not re.search(r"\b(BUY|SELL)\b", t):
        return ManageCommand("CLOSE", symbol, text)

    # --- new trade ------------------------------------------------------------
    side, kind = _find_side(t)
    if not side or not symbol:
        return None
    side_pos = re.search(r"\b(BUY|SELL|LONG|SHORT)\b", t).start()

    sl = _find_sl(t)
    tps = _find_tps(t)
    low, high = _find_entry(t, side_pos)
    # Don't mistake the SL/TP value for the entry.
    if low is not None and (low == sl or low in tps) and low == high:
        low = high = None

    entry = None if low is None else round((low + high) / 2, 10)
    return TradeSignal(
        symbol=symbol,
        side=side,
        order_kind=kind,
        entry=entry,
        entry_low=low,
        entry_high=high,
        sl=sl,
        tps=tps,
        raw=text,
    )


def validate(sig: TradeSignal) -> Optional[str]:
    """Return a reason string if the signal's numbers don't make sense, else None."""
    ref = sig.entry
    if ref is None:
        return None  # can only sanity-check against live price later
    if sig.sl is not None:
        if sig.side == "BUY" and sig.sl >= ref:
            return f"BUY with SL {sig.sl} not below entry {ref}"
        if sig.side == "SELL" and sig.sl <= ref:
            return f"SELL with SL {sig.sl} not above entry {ref}"
    for tp in sig.tps:
        if sig.side == "BUY" and tp <= ref:
            return f"BUY with TP {tp} not above entry {ref}"
        if sig.side == "SELL" and tp >= ref:
            return f"SELL with TP {tp} not below entry {ref}"
    return None
