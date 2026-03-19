"""
Data fetching from Binance REST API (no auth required for public endpoints).
Falls back to CoinGecko for daily data when Binance unavailable.
"""

import requests
import pandas as pd
import numpy as np
import time
import logging
from typing import Optional
from config.settings import (
    BINANCE_BASE_URL, BINANCE_FUTURES_URL,
    COINGECKO_BASE_URL, TIMEFRAME_BARS, CACHE_TTL_SECONDS
)
from utils.cache import cache_get, cache_set

import yfinance as yf

logger = logging.getLogger(__name__)

# Map our timeframe labels to Binance interval strings
_BINANCE_INTERVAL_MAP = {
    "1m": "1m", "5m": "5m", "15m": "15m",
    "1h": "1h", "4h": "4h", "1d": "1d",
}


class YahooFetcher:
    """Fetch OHLCV klines from Yahoo Finance."""

    def fetch(
        self,
        symbol: str,
        interval: str,
        limit: int = 1000,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch from Yahoo. Note: only supports interval '1d' for long history easily.
        """
        # Map BTCUSDT -> BTC-USD
        yahoo_symbol = symbol.replace("USDT", "-USD")
        
        # Yahoo intervals: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
        yahoo_interval = "1d" if interval == "1d" else "1h"
        
        cache_key = f"yahoo_{yahoo_symbol}_{yahoo_interval}_{limit}"
        ttl = 86400 # 24h
        
        if use_cache:
            cached = cache_get(cache_key, ttl)
            if cached is not None:
                return cached

        logger.info(f"Fetching {yahoo_symbol} from Yahoo Finance...")
        
        # We fetch a bit more than limit to be safe
        ticker = yf.Ticker(yahoo_symbol)
        # For '1d', we can just use period='max' or a large window
        df = ticker.history(period="max", interval=yahoo_interval)
        
        if df.empty:
            raise ValueError(f"No data found for {yahoo_symbol} on Yahoo Finance")

        # Clean columns to match our format
        df = df.rename(columns={
            "Open": "open", "High": "high", "Low": "low", 
            "Close": "close", "Volume": "volume"
        })
        
        # Add dummy columns for compatibility
        df["quote_asset_volume"] = 0.0
        df["num_trades"] = 0
        df["taker_buy_base"] = 0.0
        
        df = df[["open", "high", "low", "close", "volume",
                 "quote_asset_volume", "num_trades", "taker_buy_base"]]
        
        # Keep only the last 'limit' bars
        df = df.tail(limit)
        df.index.name = "timestamp"

        if use_cache:
            cache_set(cache_key, df)
            
        return df


class BinanceFetcher:
    """Fetch OHLCV klines from Binance Spot REST API."""

    def fetch(
        self,
        symbol: str,
        interval: str,
        limit: Optional[int] = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        limit = limit or TIMEFRAME_BARS.get(interval, 500)
        ttl = CACHE_TTL_SECONDS.get(interval, 60)
        cache_key = f"binance_{symbol}_{interval}_{limit}"

        if use_cache:
            cached = cache_get(cache_key, ttl)
            if cached is not None:
                return cached

        binance_interval = _BINANCE_INTERVAL_MAP[interval]
        url = f"{BINANCE_BASE_URL}/api/v3/klines"
        
        all_data = []
        remaining = limit
        last_end_time = None

        while remaining > 0:
            fetch_count = min(remaining, 1000)
            params = {
                "symbol": symbol, 
                "interval": binance_interval, 
                "limit": fetch_count
            }
            if last_end_time:
                # endTime is inclusive, we subtract 1ms to avoid overlap
                params["endTime"] = last_end_time - 1

            try:
                resp = requests.get(url, params=params, timeout=10)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                logger.error(f"Binance fetch error: {e}")
                if not all_data:
                    raise
                break

            if not data:
                break
            
            all_data = data + all_data
            remaining -= len(data)
            last_end_time = data[0][0] # first candle's open time

            # If we got fewer than requested, we hit the start of history
            if len(data) < fetch_count:
                break

        df = pd.DataFrame(all_data, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "num_trades",
            "taker_buy_base", "taker_buy_quote", "ignore"
        ])
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df.set_index("open_time", inplace=True)

        for col in ["open", "high", "low", "close", "volume",
                    "quote_asset_volume", "taker_buy_base", "taker_buy_quote"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df[["open", "high", "low", "close", "volume",
                 "quote_asset_volume", "num_trades", "taker_buy_base"]]
        df.index.name = "timestamp"

        if use_cache:
            cache_set(cache_key, df)

        return df


class BinanceFuturesFetcher:
    """Fetch funding rates and open interest from Binance Futures API."""

    def fetch_funding_rate(
        self, symbol: str, limit: int = 200, use_cache: bool = True
    ) -> pd.Series:
        """Returns a Series of funding rates indexed by time."""
        cache_key = f"funding_{symbol}_{limit}"
        ttl = 3600  # 1 hr TTL for funding data
        if use_cache:
            cached = cache_get(cache_key, ttl)
            if cached is not None:
                return cached

        url = f"{BINANCE_FUTURES_URL}/fapi/v1/fundingRate"
        params = {"symbol": symbol, "limit": limit}
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, list) or len(data) == 0:
                return pd.Series(dtype=float)
            df = pd.DataFrame(data)
            df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms")
            df.set_index("fundingTime", inplace=True)
            s = pd.to_numeric(df["fundingRate"], errors="coerce")
            if use_cache:
                cache_set(cache_key, s)
            return s
        except Exception as e:
            logger.warning(f"Funding rate fetch failed for {symbol}: {e}")
            return pd.Series(dtype=float)

    def fetch_open_interest(
        self, symbol: str, period: str = "1h", limit: int = 200, use_cache: bool = True
    ) -> pd.Series:
        """Returns a Series of open interest values indexed by time."""
        cache_key = f"oi_{symbol}_{period}_{limit}"
        ttl = 3600
        if use_cache:
            cached = cache_get(cache_key, ttl)
            if cached is not None:
                return cached

        # Map period to Binance accepted values
        period_map = {"1m": "5m", "5m": "5m", "15m": "15m",
                      "1h": "1h", "4h": "4h", "1d": "1d"}
        binance_period = period_map.get(period, "1h")
        url = f"{BINANCE_FUTURES_URL}/futures/data/openInterestHist"
        params = {"symbol": symbol, "period": binance_period, "limit": limit}
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, list) or len(data) == 0:
                return pd.Series(dtype=float)
            df = pd.DataFrame(data)
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            s = pd.to_numeric(df["sumOpenInterest"], errors="coerce")
            if use_cache:
                cache_set(cache_key, s)
            return s
        except Exception as e:
            logger.warning(f"Open interest fetch failed for {symbol}: {e}")
            return pd.Series(dtype=float)


class DataManager:
    """
    Orchestrates data fetching, enrichment with futures data, and retry logic.
    Returns a clean OHLCV+ DataFrame ready for feature engineering.
    """

    def __init__(self):
        self.spot = BinanceFetcher()
        self.futures = BinanceFuturesFetcher()
        self.yahoo = YahooFetcher()

    def get_data(
        self,
        symbol: str,
        interval: str,
        limit: Optional[int] = None,
        use_cache: bool = True,
        source: str = "binance",
    ) -> pd.DataFrame:
        """
        Fetches OHLCV data and enriches with futures signals if available.
        """
        fetch_limit = (limit or 500) + 200
        
        if source == "yahoo":
            df = self.yahoo.fetch(symbol, interval, fetch_limit, use_cache)
        else:
            df = self.spot.fetch(symbol, interval, fetch_limit, use_cache)

        # Attempt to add funding rate (only for Binance source usually, but we try as enrichment)
        funding = self.futures.fetch_funding_rate(symbol, limit=fetch_limit, use_cache=use_cache)
        if len(funding) > 0:
            funding = funding.reindex(df.index, method="ffill").fillna(0.0)
            df["funding_rate"] = funding
        else:
            df["funding_rate"] = 0.0

        # Attempt to add open interest change
        oi = self.futures.fetch_open_interest(symbol, period=interval, limit=fetch_limit, use_cache=use_cache)
        if len(oi) > 0:
            oi = oi.reindex(df.index, method="ffill").bfill()
            df["open_interest"] = oi
            df["oi_change"] = df["open_interest"].pct_change().fillna(0.0)
        else:
            df["open_interest"] = np.nan
            df["oi_change"] = 0.0

        df.dropna(subset=["open", "high", "low", "close", "volume"], inplace=True)
        return df
