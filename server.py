from __future__ import annotations

import os
import re
import sys
import time
import logging
from typing import Any

import httpx
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("hyperliquid-futures-indicator-mcp")

mcp = FastMCP(
    "hyperliquid-futures-indicator-mcp",
    host="0.0.0.0",
    port=int(os.getenv("PORT", "8000")),
)

HYPERLIQUID_KLINES_URL = os.getenv(
    "HYPERLIQUID_KLINES_URL",
    "https://api.hyperliquid.xyz/info",
).strip()
HTTP_TIMEOUT_SECONDS = float(os.getenv("HTTP_TIMEOUT_SECONDS", "30"))

HYPERLIQUID_INFO_URL = os.getenv(
    "HYPERLIQUID_INFO_URL",
    "https://api.hyperliquid.xyz/info",
).strip()

DEFAULT_COLUMNS = [
    "open", "high", "low", "close", "volume",
    "VWAP", "RSI", "ADX", "ATR", "AO", "Mom", "ROC",
    "Stoch.K", "Stoch.D",
    "Ichimoku.BLine", "Ichimoku.CLine", "Ichimoku.Lead1", "Ichimoku.Lead2",
    "EMA5", "EMA10", "EMA20", "EMA30", "EMA50", "EMA100", "EMA200",
    "Pivot.Classic.P", "Pivot.Classic.R1", "Pivot.Classic.R2", "Pivot.Classic.R3",
    "Pivot.Classic.S1", "Pivot.Classic.S2", "Pivot.Classic.S3",
    "MACD.macd", "MACD.signal", "MACD.hist",
    "BB.upper", "BB.lower", "BB.basis",
]

ALIASES = {
    "btc": "BTC", "bitcoin": "BTC", "btcusdt": "BTC", "btcusd": "BTC",
    "eth": "ETH", "ethereum": "ETH", "ethusdt": "ETH", "ethusd": "ETH",
    "sol": "SOL", "solana": "SOL", "solusdt": "SOL", "solusd": "SOL",
    "xrp": "XRP", "xrpusdt": "XRP",
    "doge": "DOGE", "dogeusdt": "DOGE",
    "ada": "ADA", "adausdt": "ADA",
    "link": "LINK", "linkusdt": "LINK",
    "avax": "AVAX", "avaxusdt": "AVAX",
    "bnb": "BNB", "bnbusdt": "BNB",
}

# Hyperliquid supported intervals
INTERVAL_MAP = {
    "1": "1m",
    "5": "5m",
    "15": "15m",
    "60": "1h",
    "120": "2h",
    "240": "4h",
    "1D": "1d",
    "1W": "1w",
    "1M": "1d",   # Hyperliquid has no monthly; fall back to daily
}

# Milliseconds per interval (used to compute startTime from limit)
INTERVAL_MS = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
    "1w": 604_800_000,
}

def normalize_symbol(raw: str) -> str:
    original = raw.strip()
    lowered = re.sub(r"[^a-zA-Z0-9]", "", original).lower()

    if original.upper().startswith("HYPERLIQUID:"):
        return original.upper()

    if lowered in ALIASES:
        return f"HYPERLIQUID:{ALIASES[lowered]}"

    # Strip trailing usdt/usd if present
    if lowered.endswith("usdt") or lowered.endswith("usd"):
        base = lowered.replace("usdt", "").replace("usd", "").upper()
        return f"HYPERLIQUID:{base}"

    if lowered.isalnum():
        return f"HYPERLIQUID:{lowered.upper()}"

    raise ValueError(f"Could not normalize symbol: {raw}")

def provider_symbol(hl_symbol: str) -> str:
    return hl_symbol.replace("HYPERLIQUID:", "")

def map_interval(tf: str) -> str:
    tf = tf.strip().upper()
    return INTERVAL_MAP.get(tf, "1h")

async def fetch_klines(symbol: str, timeframe: str, limit: int = 300) -> pd.DataFrame:
    interval = map_interval(timeframe)
    interval_ms = INTERVAL_MS.get(interval, 3_600_000)

    end_time = int(time.time() * 1000)
    start_time = end_time - (limit * interval_ms)

    body = {
        "type": "candleSnapshot",
        "req": {
            "coin": provider_symbol(symbol),
            "interval": interval,
            "startTime": start_time,
            "endTime": end_time,
        },
    }

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
        resp = await client.post(HYPERLIQUID_KLINES_URL, json=body)
        resp.raise_for_status()
        payload = resp.json()

    return parse_candles(payload)

def parse_candles(payload: Any) -> pd.DataFrame:
    # Hyperliquid returns a list of candle objects:
    # {"t": openTimeMs, "T": closeTimeMs, "s": coin, "i": interval,
    #  "o": open, "h": high, "l": low, "c": close, "v": volume, "n": trades}
    if not isinstance(payload, list) or not payload:
        raise RuntimeError("Candle payload is empty or invalid")

    normalized = []
    for row in payload:
        if isinstance(row, dict):
            normalized.append({
                "timestamp": int(row["t"]),
                "open": float(row["o"]),
                "high": float(row["h"]),
                "low": float(row["l"]),
                "close": float(row["c"]),
                "volume": float(row["v"]),
            })
        else:
            raise RuntimeError("Unsupported candle row format")

    df = pd.DataFrame(normalized)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/length, adjust=False).mean()

def adx(df: pd.DataFrame, length: int = 14) -> pd.Series:
    up_move = df["high"].diff()
    down_move = -df["low"].diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr_smoothed = tr.ewm(alpha=1/length, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/length, adjust=False).mean() / atr_smoothed
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/length, adjust=False).mean() / atr_smoothed
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1/length, adjust=False).mean()

def ao(df: pd.DataFrame) -> pd.Series:
    median = (df["high"] + df["low"]) / 2
    return median.rolling(5).mean() - median.rolling(34).mean()

def mom(close: pd.Series, length: int = 10) -> pd.Series:
    return close - close.shift(length)

def roc(close: pd.Series, length: int = 10) -> pd.Series:
    return (close / close.shift(length) - 1) * 100

def stochastic(df: pd.DataFrame, k_len: int = 14, d_len: int = 3):
    lowest_low = df["low"].rolling(k_len).min()
    highest_high = df["high"].rolling(k_len).max()
    k = 100 * (df["close"] - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
    d = k.rolling(d_len).mean()
    return k, d

def ichimoku(df: pd.DataFrame):
    conversion = (df["high"].rolling(9).max() + df["low"].rolling(9).min()) / 2
    base = (df["high"].rolling(26).max() + df["low"].rolling(26).min()) / 2
    lead1 = ((conversion + base) / 2).shift(26)
    lead2 = ((df["high"].rolling(52).max() + df["low"].rolling(52).min()) / 2).shift(26)
    return {
        "Ichimoku.CLine": conversion,
        "Ichimoku.BLine": base,
        "Ichimoku.Lead1": lead1,
        "Ichimoku.Lead2": lead2,
    }

def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal_len: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_len, adjust=False).mean()
    histogram = macd_line - signal_line
    return {"MACD.macd": macd_line, "MACD.signal": signal_line, "MACD.hist": histogram}

def bollinger_bands(close: pd.Series, length: int = 20, mult: float = 2.0):
    basis = close.rolling(length).mean()
    std = close.rolling(length).std()
    upper = basis + mult * std
    lower = basis - mult * std
    return {"BB.basis": basis, "BB.upper": upper, "BB.lower": lower}

def vwap(df: pd.DataFrame) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    return (tp * df["volume"]).cumsum() / df["volume"].cumsum().replace(0, np.nan)

def classic_pivots(df: pd.DataFrame):
    prev_high = df["high"].shift(1)
    prev_low = df["low"].shift(1)
    prev_close = df["close"].shift(1)
    p = (prev_high + prev_low + prev_close) / 3
    r1 = 2 * p - prev_low
    s1 = 2 * p - prev_high
    r2 = p + (prev_high - prev_low)
    s2 = p - (prev_high - prev_low)
    r3 = prev_high + 2 * (p - prev_low)
    s3 = prev_low - 2 * (prev_high - p)
    return {
        "Pivot.Classic.P": p,
        "Pivot.Classic.R1": r1,
        "Pivot.Classic.R2": r2,
        "Pivot.Classic.R3": r3,
        "Pivot.Classic.S1": s1,
        "Pivot.Classic.S2": s2,
        "Pivot.Classic.S3": s3,
    }

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["VWAP"] = vwap(out)
    out["RSI"] = rsi(out["close"])
    out["ATR"] = atr(out)
    out["ADX"] = adx(out)
    out["AO"] = ao(out)
    out["Mom"] = mom(out["close"])
    out["ROC"] = roc(out["close"])
    k, d = stochastic(out)
    out["Stoch.K"] = k
    out["Stoch.D"] = d
    out["EMA5"] = ema(out["close"], 5)
    out["EMA10"] = ema(out["close"], 10)
    out["EMA20"] = ema(out["close"], 20)
    out["EMA30"] = ema(out["close"], 30)
    out["EMA50"] = ema(out["close"], 50)
    out["EMA100"] = ema(out["close"], 100)
    out["EMA200"] = ema(out["close"], 200)
    for key, value in ichimoku(out).items():
        out[key] = value
    for key, value in classic_pivots(out).items():
        out[key] = value
    for key, value in macd(out["close"]).items():
        out[key] = value
    for key, value in bollinger_bands(out["close"]).items():
        out[key] = value
    return out

def latest_values(df: pd.DataFrame, columns: list[str]) -> dict[str, Any]:
    row = df.iloc[-1]
    result = {}
    for col in columns:
        value = row.get(col)
        if pd.isna(value):
            result[col] = None
        elif isinstance(value, (np.floating, float)):
            result[col] = float(value)
        elif isinstance(value, (np.integer, int)):
            result[col] = int(value)
        else:
            result[col] = value
    return result

@mcp.tool()
def normalize_hyperliquid_futures_symbols(symbols: list[str]) -> dict[str, Any]:
    normalized = []
    errors = []
    for symbol in symbols:
        try:
            normalized.append(normalize_symbol(symbol))
        except Exception as exc:
            errors.append({"input": symbol, "error": str(exc)})
    return {
        "exchange": "HYPERLIQUID",
        "market_type": "futures",
        "normalized": normalized,
        "errors": errors,
    }

@mcp.tool()
async def get_hyperliquid_market_data(symbols: list[str]) -> dict[str, Any]:
    """Fetch live market data from the Hyperliquid API: mark price, oracle price,
    funding rate, open interest, premium, and 24h volume for one or more symbols."""
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
        resp = await client.post(
            HYPERLIQUID_INFO_URL,
            json={"type": "metaAndAssetCtxs"},
        )
        resp.raise_for_status()
        payload = resp.json()

    # payload is [meta, assetCtxs] where meta.universe[i] corresponds to assetCtxs[i]
    meta = payload[0]
    asset_ctxs = payload[1]
    universe = meta.get("universe", [])

    # Build lookup: coin name -> asset context
    coin_lookup: dict[str, tuple[dict, dict]] = {}
    for i, asset_info in enumerate(universe):
        coin_lookup[asset_info["name"].upper()] = (asset_info, asset_ctxs[i])

    results = []
    errors = []
    for raw_symbol in symbols:
        try:
            canonical = normalize_symbol(raw_symbol)
            coin = provider_symbol(canonical)
        except Exception as exc:
            errors.append({"input": raw_symbol, "error": str(exc)})
            continue

        entry = coin_lookup.get(coin.upper())
        if entry is None:
            errors.append({"input": raw_symbol, "error": f"{coin} not found on Hyperliquid"})
            continue

        asset_info, ctx = entry
        mark_price = float(ctx.get("markPx", 0))
        oracle_price = float(ctx.get("oraclePx", 0))
        funding_rate = float(ctx.get("funding", 0))
        open_interest = float(ctx.get("openInterest", 0))
        day_volume_usd = float(ctx.get("dayNtlVlm", 0))
        premium = float(ctx.get("premium", 0)) if "premium" in ctx else (
            (mark_price - oracle_price) / oracle_price if oracle_price else 0.0
        )

        results.append({
            "symbol": f"HYPERLIQUID:{coin.upper()}",
            "mark_price": mark_price,
            "oracle_price": oracle_price,
            "funding_rate": funding_rate,
            "funding_rate_annualized": round(funding_rate * 3 * 365, 6),
            "open_interest": open_interest,
            "open_interest_usd": round(open_interest * mark_price, 2),
            "premium": round(premium, 8),
            "day_volume_usd": round(day_volume_usd, 2),
        })

    return {
        "exchange": "HYPERLIQUID",
        "market_type": "futures",
        "results": results,
        "errors": errors,
    }

@mcp.tool()
async def retrieve_crypto_indicators_mtf(
    ticker: str,
    timeframes: list[str],
    columns: list[str] | None = None,
    limit: int = 300,
) -> dict[str, Any]:
    """Multi-timeframe indicator retrieval. Fetches indicators for a single ticker
    across multiple timeframes in one call."""
    requested = columns or DEFAULT_COLUMNS
    canonical = normalize_symbol(ticker)

    tf_results: dict[str, dict[str, Any]] = {}
    for tf in timeframes:
        candles = await fetch_klines(canonical, tf, limit=limit)
        enriched = compute_indicators(candles)
        tf_results[tf] = latest_values(enriched, requested)

    return {
        "exchange": "HYPERLIQUID",
        "market_type": "futures",
        "results": {
            canonical: tf_results,
        },
    }

@mcp.tool()
async def retrieve_crypto_indicators(
    tickers: list[str],
    columns: list[str] | None = None,
    timeframe: str = "60",
    limit: int = 300,
) -> dict[str, Any]:
    requested = columns or DEFAULT_COLUMNS
    results = []

    for ticker in tickers:
        canonical = normalize_symbol(ticker)
        candles = await fetch_klines(canonical, timeframe, limit=limit)
        enriched = compute_indicators(candles)
        results.append({
            "ticker": canonical,
            "timeframe": timeframe,
            "values": latest_values(enriched, requested),
        })

    return {
        "exchange": "HYPERLIQUID",
        "market_type": "futures",
        "timeframe": timeframe,
        "results": results,
    }

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
