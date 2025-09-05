# src/data_loader.py
import os
import json
import requests
import time
import random
from datetime import datetime, timedelta, timezone
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf

try:
    from yahooquery import Ticker as YQ_Ticker
except ImportError:
    YQ_Ticker = None

MODEL_DIR = "models"

# ----------------------------
# Supported intervals (request-level)
# ----------------------------
SUPPORTED_INTERVALS = [
    "1m", "2m", "5m", "15m", "30m",
    "60m", "90m", "1h", "1d"
]

# ----------------------------
# Provider interval mapping
# ----------------------------
INTERVAL_MAPPING = {
    "twelvedata": {
        "1m": "1min", "2m": "2min", "5m": "5min", "15m": "15min", "30m": "30min",
        "60m": "60min", "90m": "60min", "1h": "1h", "1d": "1day"
    },
    "finnhub": {
        "1m": "1", "2m": "1", "5m": "5", "15m": "15", "30m": "30",
        "60m": "60", "90m": "60", "1h": "60", "1d": "D"
    },
    "alphavantage": {
        "1m": "1min", "2m": "1min", "5m": "5min", "15m": "15min", "30m": "30min",
        "60m": "60min", "90m": "60min", "1h": "60min", "1d": "daily"
    },
    "coingecko": {"1d": "daily"}
}

# ----------------------------
# Base ticker mapping
# ----------------------------
TICKER_MAPPING = {
    # Stocks / ETFs
    "AAPL": {"yahoo": "AAPL"},
    "TSLA": {"yahoo": "TSLA"},
    "SPY": {"yahoo": "SPY"},
    "QQQ": {"yahoo": "QQQ"},
    "MSFT": {"yahoo": "MSFT"},
    "GOOGL": {"yahoo": "GOOGL"},
    "AMZN": {"yahoo": "AMZN"},

    # Indices (Yahoo)
    "^GSPC": {"yahoo": "^GSPC"},   # S&P 500
    "^IXIC": {"yahoo": "^IXIC"},   # Nasdaq Composite
    "^DJI": {"yahoo": "^DJI"},     # Dow Jones
    "^VIX": {"yahoo": "^VIX"},

    # Commodities / Metals (Yahoo)
    "XAUUSD=X": {"yahoo": "GC=F"},  # Gold futures (more reliable than XAUUSD=X)
    "XAGUSD=X": {"yahoo": "SI=F"},  # Silver futures
    "CL=F": {"yahoo": "CL=F"},      # Crude oil futures
    "GC=F": {"yahoo": "GC=F"},      # Gold futures
    "SI=F": {"yahoo": "SI=F"},      # Silver futures
    "NG=F": {"yahoo": "NG=F"},      # Natural gas futures

    # Forex
    "EURUSD=X": {"yahoo": "EURUSD=X"},
    "GBPUSD=X": {"yahoo": "GBPUSD=X"},
    "USDJPY=X": {"yahoo": "USDJPY=X"},
    "XAUUSD": {"yahoo": "GC=F"},  # Gold futures (more reliable than XAUUSD)
    "XAGUSD": {"yahoo": "SI=F"},

    # Alternative crypto ticker
    "BTCUSD": {"yahoo": "BTC-USD"},
    "ETHUSD": {"yahoo": "ETH-USD"},
    "DOGEUSD": {"yahoo": "DOGE-USD"},
    "LTCUSD": {"yahoo": "LTC-USD"},
    "TRXUSD": {"yahoo": "TRX-USD"},
    #Others
    
    
    "NZD/CAD": {"yahoo": "NZDCAD=X"},
    "NZD/JPY": {"yahoo": "NZDJPY=X"},
    "AUD/CAD": {"yahoo": "AUDCAD=X"},
    "AUD/USD": {"yahoo": "AUDUSD=X"},
    "GBP/JPY": {"yahoo": "GBPJPY=X"},
    "GBP/USD": {"yahoo": "GBPUSD=X"},

    "NZDCAD": {"yahoo": "NZDCAD=X"},
    "NZDJPY": {"yahoo": "NZDJPY=X"},
    "AUDCAD": {"yahoo": "AUDCAD=X"},
    "AUDUSD": {"yahoo": "AUDUSD=X"},
    "GBPJPY": {"yahoo": "GBPJPY=X"},
    "GBPUSD": {"yahoo": "GBPUSD=X"},
    "USDJPY": {"yahoo": "USDJPY=X"},

    "NZDUSD": {"yahoo": "NZDUSD=X"},
    "EURUSD": {"yahoo": "EURUSD=X"},
    "EURJPY": {"yahoo": "EURJPY=X"},
    "JPYUSD": {"yahoo": "JPYUSD=X"},
    "CADUSD": {"yahoo": "CADUSD=X"},
    "CHFUSD": {"yahoo": "CHFUSD=X"},
    "CNHUSD": {"yahoo": "CNHUSD=X"},
    "HKDUSD": {"yahoo": "HKDUSD=X"},

    "XAUUSD": {"yahoo": "GC=F"},  # Gold futures (more reliable than XAUUSD=X)
    "XAGUSD": {"yahoo": "SI=F"},

    "COPPER": {"yahoo": "HG=F"},

    # Crypto
    "BTC-USD": {
        "yahoo": "BTC-USD",
        "twelvedata": "BTC/USD",
        "finnhub": "BINANCE:BTCUSDT",
        "alphavantage": "BTCUSD",
        "coingecko": "bitcoin"
    },
    "ETH-USD": {
        "yahoo": "ETH-USD",
        "twelvedata": "ETH/USD",
        "finnhub": "BINANCE:ETHUSDT",
        "alphavantage": "ETHUSD",
        "coingecko": "ethereum"
    },
}

# ----------------------------
# Helpers: asset detection
# ----------------------------
def is_cross_pair(ticker: str) -> bool:
    return "/" in ticker and not ticker.endswith("=X")

def is_forex(ticker: str) -> bool:
    return ticker.endswith("=X")

def is_crypto_usd(ticker: str) -> bool:
    return "-USD" in ticker

def is_index(ticker: str) -> bool:
    return ticker.startswith("^")

# ----------------------------
# Normalization rules
# ----------------------------
def normalize_ticker(ticker: str, provider: Optional[str] = "yahoo") -> str:
    """
    Convert broker-style tickers (MT4, TradingView, Exness, OctaFX, etc.)
    into unified symbols compatible with Yahoo, TwelveData, Finnhub, AlphaVantage, CoinGecko.

    provider defaults to "yahoo" for backward compatibility with calls that pass only ticker.
    """
    if not isinstance(ticker, str) or ticker.strip() == "":
        return ticker

    ticker = ticker.strip()

    # If provider is None, default to yahoo
    provider = provider or "yahoo"

    # Handle uppercase broker tickers like "BTCUSD", "EURUSD", "XAUUSD"
    if ticker.isupper() and len(ticker) in (6, 7):
        # Forex like EURUSD, GBPJPY
        if ticker.endswith("USD") or ticker.endswith("JPY") or ticker.endswith("EUR") or ticker.endswith("GBP") or ticker.endswith("CAD") or ticker.endswith("CHF") or ticker.endswith("NZD") or ticker.endswith("MXN"):
            base, quote = ticker[:3], ticker[3:]
            yahoo_symbol = f"{base}{quote}=X"
            if provider == "yahoo":
                return yahoo_symbol
            if provider == "twelvedata":
                return f"{base}/{quote}"
            if provider == "finnhub":
                return f"OANDA:{base}_{quote}"
            if provider == "alphavantage":
                return f"{base}{quote}"
            return ticker
           
        if ticker.endswith("USDT") and len(ticker) > 4:
            base = ticker[:-4]
            if provider == "yahoo":
                return f"{base}-USD"
            if provider == "twelvedata":
                return f"{base}/USD"
            if provider == "finnhub":
                return f"BINANCE:{base}USDT"
            if provider == "alphavantage":
                return f"{base}USD"
            if provider == "coingecko":
                return base.lower()
            return ticker
         
        if ticker.endswith("USD") and len(ticker) == 6:  # e.g. EURUSD, GBPUSD
            base = ticker[:3]
            quote = ticker[3:]
            if provider == "yahoo":
                return f"{base}{quote}=X"   # EURUSD → EURUSD=X
            if provider == "twelvedata":
                return f"{base}/{quote}"   # EURUSD → EUR/USD
            if provider == "finnhub":
                return f"OANDA:{base}_{quote}"  # EURUSD → OANDA:EUR_USD
            if provider == "alphavantage":
                return f"{base}{quote}"    # EURUSD → EURUSD
            return ticker
        
        # Metals/commodities like XAUUSD, XAGUSD
        if ticker.startswith("XAU") and ticker.endswith("USD"):
            if provider == "yahoo":
                return "GC=F"
            if provider == "twelvedata":
                return "XAU/USD"
            return ticker
        if ticker.startswith("XAG") and ticker.endswith("USD"):
            if provider == "yahoo":
                return "SI=F"
            if provider == "twelvedata":
                return "XAG/USD"
            return ticker

        # Crypto like BTCUSD, ETHUSD etc.
        if ticker.endswith("USD") and ticker[:3] in ("BTC", "ETH", "SOL", "ADA", "XRP", "LTC", "DOGE"):
            symbol = ticker[:-3]
            if provider == "yahoo":
                return f"{symbol}-USD"
            if provider == "twelvedata":
                return f"{symbol}/USD"
            if provider == "finnhub":
                return f"BINANCE:{symbol}USDT"
            if provider == "alphavantage":
                return f"{symbol}USD"
            if provider == "coingecko":
                return symbol.lower()
            return ticker

    # TradingView format: "BINANCE:BTCUSDT", "OANDA:EURUSD"
    if ":" in ticker:
        prefix, symbol = ticker.split(":", 1)
        symbol = symbol.strip()
        # Crypto like BINANCE:BTCUSDT
        if symbol.endswith("USDT"):
            base = symbol.replace("USDT", "")
            if provider == "yahoo":
                return f"{base}-USD"
            if provider == "twelvedata":
                return f"{base}/USD"
            if provider == "finnhub":
                return f"BINANCE:{base}USDT"
            if provider == "alphavantage":
                return f"{base}USD"
            if provider == "coingecko":
                return base.lower()
        # Forex like OANDA:EURUSD
        if len(symbol) == 6:
            base, quote = symbol[:3], symbol[3:]
            if provider == "yahoo":
                return f"{base}{quote}=X"
            if provider == "twelvedata":
                return f"{base}/{quote}"
            if provider == "finnhub":
                return f"OANDA:{base}_{quote}"
            if provider == "alphavantage":
                return f"{base}{quote}"
        return symbol

    # Fallback to static mapping if present
    mapped = TICKER_MAPPING.get(ticker, {})
    if provider in mapped:
        return mapped[provider]

    # Cross crypto pairs (e.g., ETH/BTC)
    if is_cross_pair(ticker):
        base, quote = ticker.split("/")
        if provider == "twelvedata":
            return f"{base}/{quote}"
        if provider == "finnhub":
            return f"BINANCE:{base}{quote}"
        if provider == "alphavantage":
            return f"{base}{quote}"
        if provider == "coingecko":
            return base.lower()
        return ticker

    # Forex already in Yahoo style: EURUSD=X
    if is_forex(ticker):
        if len(ticker.replace("=X", "")) >= 6:
            base, quote = ticker.replace("=X", "")[:3], ticker.replace("=X", "")[3:]
            if provider == "twelvedata":
                return f"{base}/{quote}"
            if provider == "finnhub":
                return f"OANDA:{base}_{quote}"
            if provider == "alphavantage":
                return f"{base}{quote}"
        return ticker

    # Crypto USD pairs: BTC-USD
    if is_crypto_usd(ticker):
        symbol = ticker.split("-")[0]
        if provider == "twelvedata":
            return f"{symbol}/USD"
        if provider == "finnhub":
            return f"BINANCE:{symbol}USDT"
        if provider == "alphavantage":
            return f"{symbol}USD"
        if provider == "coingecko":
            return symbol.lower()
        return ticker

    # Default: return unchanged
    return ticker

# ----------------------------
# Interval fallback logic
# ----------------------------
def fallback_interval(interval: str, period: str) -> str:
    if period.endswith("y") or period.endswith("mo") or period.endswith("w"):
        if interval in {"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"}:
            print(f"[INFO] Interval fallback: {interval} is too granular for period={period}. Using 1d.")
            return "1d"
    return interval

def map_interval(interval: str, provider: str) -> str:
    provider_map = INTERVAL_MAPPING.get(provider, {})
    mapped = provider_map.get(interval)
    if mapped:
        return mapped

    if provider == "coingecko":
        return "daily"

    if provider in ("finnhub", "alphavantage", "twelvedata"):
        if interval in ["2m", "90m"]:
            return provider_map.get("60m", "daily")
        return provider_map.get("1d", "daily")

    return interval

# ----------------------------
# Metadata / storage
# ----------------------------
def get_metadata_path(ticker: str, interval: str):
    return os.path.join(MODEL_DIR, ticker, f"metadata_{interval}.json")

def save_last_train_date(ticker: str, interval: str, last_date: pd.Timestamp):
    os.makedirs(os.path.join(MODEL_DIR, ticker), exist_ok=True)
    metadata = {"last_train_date": pd.to_datetime(last_date).strftime("%Y-%m-%d")}
    with open(get_metadata_path(ticker, interval), "w") as f:
        json.dump(metadata, f)

def load_last_train_date(ticker: str, interval: str) -> Optional[pd.Timestamp]:
    path = get_metadata_path(ticker, interval)
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                metadata = json.load(f)
            val = metadata.get("last_train_date")
            if val:
                return pd.to_datetime(val)
        except Exception:
            pass
    return None

# ----------------------------
# Period parsing
# ----------------------------
def period_to_timedelta(period: str) -> timedelta:
    period = period.strip().lower()
    num = "".join([c for c in period if c.isdigit()])
    suf = "".join([c for c in period if not c.isdigit()])
    if not num:
        return timedelta(days=365)
    n = int(num)
    if suf in ("d", "day", "days"):
        return timedelta(days=n)
    if suf in ("w", "wk", "week", "weeks"):
        return timedelta(days=7 * n)
    if suf in ("mo", "mon", "month", "months"):
        return timedelta(days=30 * n)
    if suf in ("y", "yr", "year", "years"):
        return timedelta(days=365 * n)
    return timedelta(days=365)

def now_utc() -> datetime:
    return datetime.now(timezone.utc)

# ----------------------------
# Rate limiting helper
# ----------------------------
def rate_limited_request(func, *args, max_retries=3, base_delay=1.0, **kwargs):
    """Execute a request with exponential backoff retry logic"""
    for attempt in range(max_retries):
        try:
            # Add random delay to avoid rate limiting
            if attempt > 0:
                delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0.1, 0.5)
                print(f"[INFO] Waiting {delay:.1f}s before retry {attempt + 1}...")
                time.sleep(delay)
            else:
                time.sleep(random.uniform(0.2, 0.8))  # Small initial delay
            
            return func(*args, **kwargs)
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            print(f"[WARN] Attempt {attempt + 1} failed: {str(e)[:100]}...")
    return None

# ----------------------------
# CoinGecko helper
# ----------------------------
def fetch_coingecko_usd(symbol: str) -> pd.DataFrame:
    def _fetch():
        url = f"https://api.coingecko.com/api/v3/coins/{symbol}/market_chart"
        res = requests.get(url, params={"vs_currency": "usd", "days": "max", "interval": "daily"}, timeout=30)
        res.raise_for_status()
        return res.json()
    
    data = rate_limited_request(_fetch)
    if data and "prices" in data:
        df = pd.DataFrame(data["prices"], columns=["ds", "y"])
        df["ds"] = pd.to_datetime(df["ds"], unit="ms", utc=True)
        df = df.sort_values("ds").reset_index(drop=True)
        return df
    raise ValueError(f"CoinGecko returned no 'prices' for {symbol}")

# ----------------------------
# ATR utility (added)
# ----------------------------
def compute_atr(df: pd.DataFrame, period: int = 14, price_col: str = "y") -> pd.Series:
    """
    Compute ATR (Average True Range) over 'period' rows.
    If High/Low/Close columns exist, use them; otherwise fallback to close-based TR.
    Returns a pandas Series aligned with df.index (same length), filled forward/backward where needed.
    """
    try:
        # Prefer using 'High', 'Low', 'Close' if present
        if all(col in df.columns for col in ["High", "Low", "Close"]):
            high = pd.to_numeric(df["High"], errors="coerce")
            low = pd.to_numeric(df["Low"], errors="coerce")
            close = pd.to_numeric(df["Close"], errors="coerce")
            prev_close = close.shift(1)
            tr1 = (high - low).abs()
            tr2 = (high - prev_close).abs()
            tr3 = (low - prev_close).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        else:
            # Fallback: use absolute close change as TR
            close = pd.to_numeric(df.get(price_col, df.columns[0]), errors="coerce")
            prev_close = close.shift(1)
            tr = (close - prev_close).abs().fillna(0.0)

        atr = tr.rolling(window=period, min_periods=max(1, period // 2)).mean()
        atr = atr.fillna(method="bfill").fillna(method="ffill")
        # Ensure small positive values to avoid division by zero
        atr = atr.fillna(0.0)
        return atr
    except Exception as e:
        print(f"[WARN] ATR calculation failed: {e}")
        # Return a series of small positive values
        return pd.Series(np.full(len(df), 1e-8), index=df.index)

# ----------------------------
# Data validation
# ----------------------------
def validate_data(df: pd.DataFrame, ticker: str) -> bool:
    if df is None or df.empty:
        raise ValueError(f"No data available for {ticker}")
    
    if len(df) < 50:
        raise ValueError(f"Insufficient data for {ticker}: only {len(df)} rows (need at least 50)")
    
    if 'y' not in df.columns:
        raise ValueError(f"Missing price column 'y' for {ticker}")
    
    if df['y'].isna().sum() > len(df) * 0.5:
        raise ValueError(f"Too many missing values in {ticker} data: {df['y'].isna().sum()}/{len(df)}")
    
    return True

# ----------------------------
# Main data download function
# ----------------------------
def download_ticker(ticker: str, period: str = "5y", interval: str = "1d", incremental: bool = True) -> pd.DataFrame:
    # Validate inputs
    if interval not in SUPPORTED_INTERVALS:
        raise ValueError(f"Interval '{interval}' not supported. Choose from {SUPPORTED_INTERVALS}")
    interval = fallback_interval(interval, period)

    print(f"\n[INFO] Fetching {ticker} (period={period}, interval={interval})")
    ticker_dir = os.path.join(MODEL_DIR, ticker)
    os.makedirs(ticker_dir, exist_ok=True)
    csv_path = os.path.join(ticker_dir, f"full_data_{interval}.csv")

    # Handle cross-pair construction (FIXED: no recursion)
    if is_cross_pair(ticker):
        base, quote = ticker.split("/")
        print(f"[INFO] Constructing cross-pair {ticker} from USD legs...")
        try:
            def fetch_leg(symbol):
                yf_ticker = f"{symbol}-USD"
                return yf.Ticker(yf_ticker).history(period=period, interval=interval)
            
            base_hist = rate_limited_request(fetch_leg, base)
            quote_hist = rate_limited_request(fetch_leg, quote)
            
            if base_hist.empty or quote_hist.empty:
                raise ValueError("Failed to get USD legs data")
            
            base_df = normalize_df_columns(base_hist.reset_index())
            quote_df = normalize_df_columns(quote_hist.reset_index())
            
            merged = pd.merge(base_df[["ds", "y"]], quote_df[["ds", "y"]],
                            on="ds", how="inner", suffixes=(f"_{base}", f"_{quote}"))
            if merged.empty:
                raise ValueError(f"No overlapping dates between {base} and {quote}")
            
            merged["y"] = merged[f"y_{base}"] / merged[f"y_{quote}"]
            result = merged[["ds", "y"]].sort_values("ds").reset_index(drop=True)
            
            validate_data(result, ticker)
            result.to_csv(csv_path, index=False)
            save_last_train_date(ticker, interval, result["ds"].max())
            print(f"[SUCCESS] Constructed cross rate {ticker}")
            return result
            
        except Exception as e:
            print(f"[ERROR] Cross-pair construction failed: {e}")
            # Fall through to try other providers

    # Provider cascade with improved error handling
    providers = [
        ("yahoo", try_yahoo_finance),
        ("yahooquery", try_yahooquery),
        ("twelvedata", try_twelvedata),
        ("finnhub", try_finnhub),
        ("alphavantage", try_alphavantage),
        ("coingecko", try_coingecko)
    ]
    
    last_error = None
    for provider_name, provider_func in providers:
        try:
            print(f"[INFO] Trying {provider_name}...")
            df = provider_func(ticker, period, interval)
            if df is not None and not df.empty:
                validate_data(df, ticker)
                df.to_csv(csv_path, index=False)
                save_last_train_date(ticker, interval, df["ds"].max())
                print(f"[SUCCESS] {provider_name} returned {len(df)} rows")
                return df
            else:
                print(f"[WARN] {provider_name} returned empty data")
        except Exception as e:
            print(f"[ERROR] {provider_name} failed: {str(e)[:150]}")
            last_error = e
            continue

    # If all providers failed
    error_msg = f"[FAIL] No data available for {ticker} with period={period} interval={interval} across all APIs."
    if last_error:
        error_msg += f" Last error: {str(last_error)[:200]}"
    raise ValueError(error_msg)

# ----------------------------
# Provider functions
# ----------------------------
def try_yahoo_finance(ticker: str, period: str, interval: str) -> pd.DataFrame:
    def fetch():
        yf_ticker = normalize_ticker(ticker, "yahoo")
        return yf.Ticker(yf_ticker).history(period=period, interval=interval)
    
    df = rate_limited_request(fetch)
    if df is not None and not df.empty:
        return normalize_df_columns(df.reset_index())
    return pd.DataFrame()

def try_yahooquery(ticker: str, period: str, interval: str) -> pd.DataFrame:
    if not YQ_Ticker:
        return pd.DataFrame()
    
    def fetch():
        yq_ticker = normalize_ticker(ticker, "yahoo")
        yq = YQ_Ticker(yq_ticker)
        return yq.history(period=period, interval=interval)
    
    df = rate_limited_request(fetch)
    if isinstance(df, pd.DataFrame) and not df.empty:
        df = df.reset_index()
        if "date" in df.columns:
            df = df.rename(columns={"date": "ds", "close": "y"})
        df["ds"] = pd.to_datetime(df["ds"], utc=True)
        return df.sort_values("ds").reset_index(drop=True)
    return pd.DataFrame()

def try_twelvedata(ticker: str, period: str, interval: str) -> pd.DataFrame:
    td_key = os.getenv("TWELVEDATA_API_KEY", "").strip()
    if not td_key:
        print("[INFO] TWELVEDATA_API_KEY not set")
        return pd.DataFrame()
    
    def fetch():
        td_symbol = normalize_ticker(ticker, "twelvedata")
        td_interval = map_interval(interval, "twelvedata")
        url = "https://api.twelvedata.com/time_series"
        r = requests.get(url, params={
            "symbol": td_symbol,
            "interval": td_interval,
            "apikey": td_key,
            "outputsize": 5000
        }, timeout=30)
        r.raise_for_status()
        return r.json()
    
    data = rate_limited_request(fetch)
    if data and "values" in data:
        df = pd.DataFrame(data["values"])
        df = df.rename(columns={
            "datetime": "ds", "close": "y",
            "open": "Open", "high": "High",
            "low": "Low", "volume": "Volume"
        })
        df["ds"] = pd.to_datetime(df["ds"], utc=True)
        return df.sort_values("ds").reset_index(drop=True)
    return pd.DataFrame()

def try_finnhub(ticker: str, period: str, interval: str) -> pd.DataFrame:
    fh_key = os.getenv("FINNHUB_API_KEY", "").strip()
    if not fh_key:
        print("[INFO] FINNHUB_API_KEY not set")
        return pd.DataFrame()
    
    def fetch():
        fh_symbol = normalize_ticker(ticker, "finnhub")
        fh_interval = map_interval(interval, "finnhub")
        end = int(now_utc().timestamp())
        start = int((now_utc() - period_to_timedelta(period)).timestamp())

        if fh_symbol.startswith("BINANCE:"):
            endpoint = "https://finnhub.io/api/v1/crypto/candle"
        elif fh_symbol.startswith("OANDA:"):
            endpoint = "https://finnhub.io/api/v1/forex/candle"
        else:
            endpoint = "https://finnhub.io/api/v1/stock/candle"

        params = {"symbol": fh_symbol, "resolution": fh_interval,
                  "from": start, "to": end, "token": fh_key}
        res = requests.get(endpoint, params=params, timeout=30)
        res.raise_for_status()
        return res.json()
    
    data = rate_limited_request(fetch)
    if data and "t" in data and data.get("s") == "ok":
        df = pd.DataFrame({
            "ds": pd.to_datetime(data["t"], unit="s", utc=True),
            "y": data["c"], "Open": data["o"],
            "High": data["h"], "Low": data["l"], "Volume": data["v"]
        })
        return df.sort_values("ds").reset_index(drop=True)
    return pd.DataFrame()

def try_alphavantage(ticker: str, period: str, interval: str) -> pd.DataFrame:
    av_key = os.getenv("ALPHAVANTAGE_API_KEY", "").strip()
    if not av_key:
        print("[INFO] ALPHAVANTAGE_API_KEY not set")
        return pd.DataFrame()
    
    def fetch():
        av_symbol = normalize_ticker(ticker, "alphavantage")
        av_interval = map_interval(interval, "alphavantage")
        function = "TIME_SERIES_DAILY_ADJUSTED" if av_interval == "daily" else "TIME_SERIES_INTRADAY"

        params = {"symbol": av_symbol, "apikey": av_key, "outputsize": "full"}
        if function == "TIME_SERIES_INTRADAY":
            params["function"] = "TIME_SERIES_INTRADAY"
            params["interval"] = av_interval
        else:
            params["function"] = "TIME_SERIES_DAILY_ADJUSTED"

        res = requests.get("https://www.alphavantage.co/query", params=params, timeout=30)
        res.raise_for_status()
        return res.json()
    
    data = rate_limited_request(fetch)
    if data:
        key = "Time Series (Daily)" if "TIME_SERIES_DAILY_ADJUSTED" in str(data) else f"Time Series ({map_interval(interval, 'alphavantage')})"
        series = data.get(key, {})
        if series:
            df = pd.DataFrame.from_dict(series, orient="index")
            df = df.rename(columns={
                "5. adjusted close": "y",
                "1. open": "Open",
                "2. high": "High", 
                "3. low": "Low",
                "6. volume": "Volume",
                "4. close": "Close"
            })
            df["ds"] = pd.to_datetime(df.index, utc=True)
            df = df.sort_values("ds").reset_index(drop=True)
            
            if "y" not in df.columns and "Close" in df.columns:
                df["y"] = pd.to_numeric(df["Close"], errors="coerce")
            elif "y" not in df.columns:
                for c in ["4. close", "close", "Adj Close"]:
                    if c in df.columns:
                        df["y"] = pd.to_numeric(df[c], errors="coerce")
                        break
            return df
    return pd.DataFrame()

def try_coingecko(ticker: str, period: str, interval: str) -> pd.DataFrame:
    try:
        cg_symbol = normalize_ticker(ticker, "coingecko")
        if cg_symbol and (is_crypto_usd(ticker) or ticker.lower() in ['bitcoin', 'ethereum']):
            return fetch_coingecko_usd(cg_symbol)
    except Exception:
        pass
    return pd.DataFrame()

# ----------------------------
# DataFrame utilities
# ----------------------------
def normalize_df_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    
    # Handle different date column names
    date_cols = ["Date", "Datetime", "date", "datetime", "timestamp", "ds"]
    for col in date_cols:
        if col in df.columns:
            df = df.rename(columns={col: "ds"})
            break
    else:
        # If no date column found, use index
        if df.index.name:
            df = df.reset_index().rename(columns={df.index.name: "ds"})
        else:
            df["ds"] = pd.to_datetime(df.index, utc=True)

    # Handle price column
    price_cols = ["Close", "close", "CLOSE", "price", "Price"]
    for col in price_cols:
        if col in df.columns:
            df = df.rename(columns={col: "y"})
            break

    # Ensure ds is datetime
    df["ds"] = pd.to_datetime(df["ds"], utc=True)
    df = df.sort_values("ds").reset_index(drop=True)

    # Ensure standard columns exist
    standard_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in standard_cols:
        if col not in df.columns:
            df[col] = np.nan

    # Ensure y is numeric
    if "y" in df.columns:
        df["y"] = pd.to_numeric(df["y"], errors="coerce")

    return df

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    
    df = df.copy()
    price_col = "Close" if "Close" in df.columns else "y"
    df["y"] = pd.to_numeric(df[price_col], errors="coerce").astype(float)

    # Technical indicators with error handling
    try:
        df["return"] = df["y"].pct_change()
        df["ma_5"] = df["y"].rolling(window=5, min_periods=1).mean()
        df["ma_20"] = df["y"].rolling(window=20, min_periods=1).mean()
        df["vol_10"] = df["y"].rolling(window=10, min_periods=1).std()

        # RSI calculation
        delta = df["y"].diff()
        up = delta.clip(lower=0).rolling(window=14, min_periods=1).mean()
        down = (-delta.clip(upper=0)).rolling(window=14, min_periods=1).mean()
        rs = up / (down.replace(0, np.nan))
        df["rsi"] = 100 - (100 / (1 + rs))

        # --- NEW: compute ATR and ATR percent for volatility-adjusted SL/TP ---
        # Add atr_14 (absolute ATR) and atr_pct_14 (ATR expressed as percentage of price)
        try:
            atr_series = compute_atr(df, period=14, price_col="y")
            df["atr_14"] = atr_series
            # Avoid division by zero and fill small positive number if needed
            df["atr_pct_14"] = (atr_series / df["y"].replace(0, np.nan)) * 100.0
            df["atr_pct_14"] = df["atr_pct_14"].replace([np.inf, -np.inf], np.nan).fillna(method="bfill").fillna(method="ffill").fillna(0.0)
        except Exception as _e:
            # If ATR calculation fails, ensure the columns exist with fallback values
            df["atr_14"] = np.nan
            df["atr_pct_14"] = 0.0

        # FIXED: Use new pandas methods instead of deprecated ones
        df = df.bfill().ffill()
        
        # Final validation
        for col in ["return", "ma_5", "ma_20", "vol_10", "rsi", "atr_14", "atr_pct_14"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        
    except Exception as e:
        print(f"[WARN] Technical indicators calculation failed: {e}")
        # Provide fallback values
        for col in ["return", "ma_5", "ma_20", "vol_10", "rsi", "atr_14", "atr_pct_14"]:
            if col not in df.columns:
                df[col] = 0.0

    return df

def prepare_features_for_model(df: pd.DataFrame, feature_cols=None) -> Tuple[np.ndarray, pd.DataFrame]:
    if feature_cols is None:
        feature_cols = ["y", "ma_5", "ma_20", "vol_10", "rsi"]
    
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    
    # Validate data quality
    for col in feature_cols:
        if df[col].isna().all():
            raise ValueError(f"All values are NaN in column: {col}")
    
    values = df[feature_cols].values.astype(float)
    
    # Check for any remaining NaN values
    if np.isnan(values).any():
        print(f"[WARN] Found NaN values in features, filling with forward fill")
        values = pd.DataFrame(values).ffill().bfill().values
    
    return values, df
