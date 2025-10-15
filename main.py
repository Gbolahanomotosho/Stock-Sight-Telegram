# main.py — Enhanced Telegram Service with Google Sheets Persistence
# NEW FEATURES: 
# - Google Sheets persistent subscription storage
# - Automatic 3-day free trial for new users
# - Admin notifications for new users and payments
# - Fallback to local JSON if Google Sheets unavailable

import os
import io
import math
import asyncio
import json
import datetime
import numpy as np
import pandas as pd
import joblib
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import traceback
import signal
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

from src.data_loader import (
    download_ticker,
    normalize_df_columns,
    add_technical_indicators,
    prepare_features_for_model,
    validate_data
)
from src.prophet_model import train_prophet, prophet_predict
from src.lstm_model import train_lstm, predict_lstm
from src.transformer_model import train_transformer, predict_transformer
from src.timesnet_model import train_timesnet, predict_timesnet
from src.ensemble import fit_meta, predict_meta
from src.utils import create_sliding_windows, scale_train_val_test, rmse, mape, clean_predictions

try:
    from src.data_loader import normalize_ticker
except Exception:
    def normalize_ticker(t: str) -> str:
        return (t or "").strip().upper()

from telegram import Update, InputFile
from telegram.constants import ParseMode, ChatAction
from telegram.ext import Application, CommandHandler, ContextTypes

# Google Sheets Integration
try:
    import gspread
    from google.oauth2.service_account import Credentials
    GSPREAD_AVAILABLE = True
except ImportError:
    GSPREAD_AVAILABLE = False
    print("[WARN] gspread not installed. Run: pip install gspread google-auth")

VERSION = "Stock Sight Telegram Service v2.1 (Google Sheets Edition)"

# ----------------------------
# Configuration
# ----------------------------
SUBSCRIBE_URL = os.getenv("SUBSCRIBE_URL", "").strip()
ADMIN_IDS = [int(x) for x in os.getenv("ADMIN_IDS", "").split(",") if x.strip().isdigit()]
SUBS_FILE = os.getenv("SUBS_FILE", "subscriptions.json")
FREE_TRIAL_DAYS = 3

# Google Sheets Configuration
GOOGLE_SHEETS_ENABLED = os.getenv("GOOGLE_SHEETS_ENABLED", "false").lower() == "true"
GOOGLE_CREDENTIALS_JSON = os.getenv("GOOGLE_CREDENTIALS_JSON", "")
GOOGLE_SHEET_NAME = os.getenv("GOOGLE_SHEET_NAME", "StockSight-Subscriptions")
GOOGLE_WORKSHEET_NAME = os.getenv("GOOGLE_WORKSHEET_NAME", "Users")

# ----------------------------
# Google Sheets Manager
# ----------------------------
class GoogleSheetsManager:
    def __init__(self):
        self.client = None
        self.sheet = None
        self.worksheet = None
        self.enabled = False
        self.last_error = None
        
        if GOOGLE_SHEETS_ENABLED and GSPREAD_AVAILABLE:
            self._initialize()
    
    def _initialize(self):
        """Initialize Google Sheets connection"""
        try:
            if not GOOGLE_CREDENTIALS_JSON:
                print("[WARN] GOOGLE_CREDENTIALS_JSON not configured")
                return
            
            # Parse credentials
            try:
                creds_dict = json.loads(GOOGLE_CREDENTIALS_JSON)
            except json.JSONDecodeError:
                # Try loading from file path
                if os.path.exists(GOOGLE_CREDENTIALS_JSON):
                    with open(GOOGLE_CREDENTIALS_JSON, 'r') as f:
                        creds_dict = json.load(f)
                else:
                    print("[ERROR] Invalid Google credentials format")
                    return
            
            # Set up credentials with required scopes
            scopes = [
                'https://www.googleapis.com/auth/spreadsheets',
                'https://www.googleapis.com/auth/drive'
            ]
            
            credentials = Credentials.from_service_account_info(
                creds_dict,
                scopes=scopes
            )
            
            # Authorize and get client
            self.client = gspread.authorize(credentials)
            
            # Open or create spreadsheet
            try:
                self.sheet = self.client.open(GOOGLE_SHEET_NAME)
                print(f"[INFO] Opened existing Google Sheet: {GOOGLE_SHEET_NAME}")
            except gspread.SpreadsheetNotFound:
                self.sheet = self.client.create(GOOGLE_SHEET_NAME)
                print(f"[INFO] Created new Google Sheet: {GOOGLE_SHEET_NAME}")
            
            # Get or create worksheet
            try:
                self.worksheet = self.sheet.worksheet(GOOGLE_WORKSHEET_NAME)
            except gspread.WorksheetNotFound:
                self.worksheet = self.sheet.add_worksheet(
                    title=GOOGLE_WORKSHEET_NAME,
                    rows=1000,
                    cols=10
                )
                # Set up headers
                headers = [
                    "user_id", "username", "full_name", "expires", 
                    "activated_at", "is_trial", "days", "last_updated"
                ]
                self.worksheet.append_row(headers)
                print(f"[INFO] Created worksheet with headers: {GOOGLE_WORKSHEET_NAME}")
            
            self.enabled = True
            print("[SUCCESS] Google Sheets integration enabled")
            
        except Exception as e:
            self.last_error = str(e)
            print(f"[ERROR] Google Sheets initialization failed: {e}")
            print("[INFO] Falling back to local JSON storage")
    
    def load_subscriptions(self) -> Dict[str, Any]:
        """Load all subscriptions from Google Sheets"""
        if not self.enabled or not self.worksheet:
            return {}
        
        try:
            # Get all records (skip header row)
            records = self.worksheet.get_all_records()
            
            subscriptions = {}
            for record in records:
                user_id = str(record.get("user_id", ""))
                if user_id and user_id != "user_id":  # Skip header if present
                    subscriptions[user_id] = {
                        "username": record.get("username", ""),
                        "full_name": record.get("full_name", ""),
                        "expires": record.get("expires", ""),
                        "activated_at": record.get("activated_at", ""),
                        "is_trial": str(record.get("is_trial", "")).lower() == "true",
                        "days": int(record.get("days", 0)) if record.get("days") else 0,
                        "last_updated": record.get("last_updated", "")
                    }
            
            print(f"[INFO] Loaded {len(subscriptions)} subscriptions from Google Sheets")
            return subscriptions
            
        except Exception as e:
            print(f"[ERROR] Failed to load from Google Sheets: {e}")
            self.last_error = str(e)
            return {}
    
    def save_subscription(self, user_id: int, data: Dict[str, Any], 
                         username: str = "", full_name: str = "") -> bool:
        """Save or update a single subscription"""
        if not self.enabled or not self.worksheet:
            return False
        
        try:
            user_id_str = str(user_id)
            
            # Check if user exists
            cell = self.worksheet.find(user_id_str, in_column=1)
            
            row_data = [
                user_id_str,
                username,
                full_name,
                data.get("expires", ""),
                data.get("activated_at", ""),
                str(data.get("is_trial", False)),
                str(data.get("days", 0)),
                datetime.datetime.utcnow().isoformat()
            ]
            
            if cell:
                # Update existing row
                row_num = cell.row
                self.worksheet.update(f'A{row_num}:H{row_num}', [row_data])
                print(f"[INFO] Updated user {user_id} in Google Sheets")
            else:
                # Append new row
                self.worksheet.append_row(row_data)
                print(f"[INFO] Added user {user_id} to Google Sheets")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to save to Google Sheets: {e}")
            self.last_error = str(e)
            return False
    
    def delete_subscription(self, user_id: int) -> bool:
        """Delete a subscription"""
        if not self.enabled or not self.worksheet:
            return False
        
        try:
            user_id_str = str(user_id)
            cell = self.worksheet.find(user_id_str, in_column=1)
            
            if cell:
                self.worksheet.delete_rows(cell.row)
                print(f"[INFO] Deleted user {user_id} from Google Sheets")
                return True
            
            return False
            
        except Exception as e:
            print(f"[ERROR] Failed to delete from Google Sheets: {e}")
            self.last_error = str(e)
            return False

# Initialize Google Sheets Manager
sheets_manager = GoogleSheetsManager()

# ----------------------------
# Subscription Management (with Google Sheets + JSON fallback)
# ----------------------------
def _load_subs() -> Dict[str, Any]:
    """Load subscriptions from Google Sheets with JSON fallback"""
    # Try Google Sheets first
    if sheets_manager.enabled:
        subs = sheets_manager.load_subscriptions()
        if subs:
            return subs
        print("[WARN] Google Sheets returned empty, trying local JSON")
    
    # Fallback to local JSON
    try:
        if os.path.exists(SUBS_FILE):
            with open(SUBS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to load from JSON: {e}")
    
    return {}

def _save_subs(subs: Dict[str, Any]) -> None:
    """Save subscriptions to both Google Sheets and JSON"""
    # Save to Google Sheets
    sheets_saved = False
    if sheets_manager.enabled:
        for user_id, data in subs.items():
            if sheets_manager.save_subscription(
                int(user_id), 
                data,
                data.get("username", ""),
                data.get("full_name", "")
            ):
                sheets_saved = True
    
    # Always save to local JSON as backup
    try:
        os.makedirs(os.path.dirname(SUBS_FILE) if os.path.dirname(SUBS_FILE) else ".", exist_ok=True)
        with open(SUBS_FILE, "w", encoding="utf-8") as f:
            json.dump(subs, f, indent=2, default=str)
        print(f"[INFO] Subscriptions saved to local JSON: {SUBS_FILE}")
    except Exception as e:
        print(f"[WARN] Failed to save to JSON: {e}")
    
    if sheets_saved:
        print("[INFO] Subscriptions synced to Google Sheets")

def is_subscribed(user_id: int) -> Tuple[bool, Any]:
    """Check if user has active subscription"""
    subs = _load_subs()
    rec = subs.get(str(user_id))
    if not rec:
        return False, None
    exp_iso = rec.get("expires")
    if not exp_iso:
        return False, None
    try:
        exp_dt = datetime.datetime.fromisoformat(exp_iso)
    except Exception:
        try:
            exp_dt = datetime.datetime.utcfromtimestamp(int(exp_iso))
        except Exception:
            return False, None
    now = datetime.datetime.utcnow()
    return now < exp_dt, exp_dt

def activate_subscription_for(user_id: int, days: int = 30, trial: bool = False,
                              username: str = "", full_name: str = "") -> datetime.datetime:
    """Activate subscription for a user"""
    subs = _load_subs()
    new_exp = datetime.datetime.utcnow() + datetime.timedelta(days=days)
    subs[str(user_id)] = {
        "username": username,
        "full_name": full_name,
        "expires": new_exp.isoformat(),
        "activated_at": datetime.datetime.utcnow().isoformat(),
        "is_trial": trial,
        "days": days
    }
    _save_subs(subs)
    print(f"[INFO] Activated subscription for user {user_id} until {new_exp.isoformat()} (trial={trial})")
    return new_exp

def deactivate_subscription_for(user_id: int) -> bool:
    """Deactivate subscription for a user"""
    subs = _load_subs()
    if str(user_id) in subs:
        del subs[str(user_id)]
        _save_subs(subs)
        
        # Also delete from Google Sheets
        if sheets_manager.enabled:
            sheets_manager.delete_subscription(user_id)
        
        print(f"[INFO] Deactivated subscription for user {user_id}")
        return True
    return False

def is_new_user(user_id: int) -> bool:
    """Check if this is a new user (never had subscription)"""
    subs = _load_subs()
    return str(user_id) not in subs

async def notify_admins(context: ContextTypes.DEFAULT_TYPE, message: str):
    """Send notification to all admins"""
    if not ADMIN_IDS:
        print(f"[WARN] No admins configured to receive notification: {message}")
        return
    
    for admin_id in ADMIN_IDS:
        try:
            await context.bot.send_message(chat_id=admin_id, text=message, parse_mode=ParseMode.MARKDOWN)
            print(f"[INFO] Notification sent to admin {admin_id}")
        except Exception as e:
            print(f"[WARN] Failed to notify admin {admin_id}: {e}")

# ----------------------------
# Model storage
# ----------------------------
def get_model_dir(ticker: str, interval: str) -> str:
    return os.path.join("models", ticker, interval)

# ----------------------------
# Request model
# ----------------------------
@dataclass
class ForecastRequest:
    ticker: str
    period: str = "5y"
    interval: str = "1d"
    steps: int = 30
    context: int = 60
    horizon: int = 1
    window_size: int = 250
    device: str = "cpu"
    buy_threshold_pct: float = 0.3
    sell_threshold_pct: float = -0.3
    stop_loss_pct: float = 0.5
    take_profit_rr: float = 2.0

    def __post_init__(self):
        if not self.ticker or not isinstance(self.ticker, str):
            raise ValueError("Ticker must be a non-empty string")
        if self.steps <= 0 or self.steps > 100:
            raise ValueError("Steps must be between 1 and 100")
        if self.context <= 0 or self.context > 500:
            raise ValueError("Context must be between 1 and 500")
        if self.window_size < 50 or self.window_size > 2000:
            raise ValueError("Window size must be between 50 and 2000")

# ----------------------------
# Volatility helpers
# ----------------------------
def compute_atr_pct(df: pd.DataFrame, period: int = 14) -> Optional[pd.Series]:
    try:
        cols = {c.lower(): c for c in df.columns}
        has_hlc = all(k in cols for k in ["high", "low", "close"])
        if not has_hlc:
            return None
        
        H, L, C = cols["high"], cols["low"], cols["close"]
        high = pd.to_numeric(df[H], errors='coerce')
        low = pd.to_numeric(df[L], errors='coerce')
        close = pd.to_numeric(df[C], errors='coerce')

        prev_close = close.shift(1)
        tr1 = (high - low).abs()
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period, min_periods=max(1, period//2)).mean()
        atr_pct = (atr / close) * 100.0
        
        return atr_pct.fillna(1.0)
    except Exception as e:
        print(f"[WARN] ATR calculation failed: {e}")
        return None

def compute_returns_vol_pct(df: pd.DataFrame, period: int = 14) -> Optional[pd.Series]:
    try:
        if "y" not in df.columns:
            return None
        y = pd.to_numeric(df["y"], errors='coerce')
        ret = y.pct_change() * 100.0
        vol = ret.rolling(window=period, min_periods=max(1, period//2)).std()
        return vol.fillna(1.0)
    except Exception as e:
        print(f"[WARN] Returns volatility calculation failed: {e}")
        return None

def get_current_and_median_vol(df: pd.DataFrame, window_size: int) -> Tuple[float, float]:
    try:
        atr_pct = compute_atr_pct(df)
        vol_series = atr_pct if atr_pct is not None else compute_returns_vol_pct(df)
        
        if vol_series is None:
            return 1.0, 1.0
        
        vol_recent = vol_series.dropna().iloc[-min(window_size, len(vol_series)):]
        if len(vol_recent) == 0:
            return 1.0, 1.0
        
        current_vol = float(vol_recent.iloc[-1]) if len(vol_recent) > 0 else 1.0
        median_vol = float(vol_recent.median()) if len(vol_recent) > 0 else 1.0
        
        current_vol = max(current_vol, 0.1)
        median_vol = max(median_vol, 0.1)
        
        return current_vol, median_vol
    except Exception as e:
        print(f"[WARN] Volatility calculation failed: {e}")
        return 1.0, 1.0

def adapt_thresholds(buy_th: float, sell_th: float, sl_pct: float,
                     current_vol: float, median_vol: float) -> Tuple[float, float, float]:
    try:
        ratio = current_vol / (median_vol + 1e-8)
        ratio = max(0.6, min(ratio, 1.8))
        
        buy_adapted = buy_th * ratio
        sell_adapted = sell_th * ratio
        sl_adapted = sl_pct * ratio
        
        buy_adapted = float(max(0.05, min(buy_adapted, 2.0)))
        sell_adapted = float(max(-2.0, min(sell_adapted, -0.05)))
        sl_adapted = float(max(0.1, min(sl_adapted, 5.0)))
        
        return buy_adapted, sell_adapted, sl_adapted
    except Exception as e:
        print(f"[WARN] Threshold adaptation failed: {e}")
        return buy_th, sell_th, sl_pct

# ----------------------------
# Signal logic
# ----------------------------
def make_signal(current_price: float, forecast_price: float,
                buy_threshold_pct: float, sell_threshold_pct: float,
                stop_loss_pct: float, take_profit_rr: float):
    try:
        if not all(np.isfinite([current_price, forecast_price])):
            return "HOLD", current_price, None, None, None
        
        pct_diff = ((forecast_price - current_price) / current_price) * 100.0
        signal = "HOLD"
        entry = current_price
        stop_loss = None
        take_profit = None
        rr = None
        
        if pct_diff >= buy_threshold_pct:
            signal = "BUY"
            sl_dist = entry * (stop_loss_pct / 100.0)
            stop_loss = entry - sl_dist
            take_profit = entry + take_profit_rr * sl_dist
            rr = take_profit_rr
        elif pct_diff <= sell_threshold_pct:
            signal = "SELL"
            sl_dist = entry * (stop_loss_pct / 100.0)
            stop_loss = entry + sl_dist
            take_profit = entry - take_profit_rr * sl_dist
            rr = take_profit_rr
            
        return (signal, float(entry), 
                float(stop_loss) if stop_loss is not None else None,
                float(take_profit) if take_profit is not None else None, 
                rr)
    except Exception as e:
        print(f"[WARN] Signal generation failed: {e}")
        return "HOLD", current_price, None, None, None

# ----------------------------
# Market Regime detection
# ----------------------------
def detect_market_regime(df: pd.DataFrame, lookback: int = 50) -> str:
    try:
        if len(df) < lookback + 1:
            return "Unknown"
        
        y = pd.to_numeric(df["y"], errors='coerce')
        recent = y.iloc[-lookback:]
        
        if len(recent) < 2:
            return "Unknown"
        
        ret = recent.pct_change().dropna()
        if len(ret) == 0:
            return "Unknown"
        
        trend = recent.iloc[-1] - recent.iloc[0]
        vol = ret.std()
        
        if pd.isna(trend) or pd.isna(vol):
            return "Unknown"
        
        if trend > 0 and vol < 2:
            return "Bullish"
        elif trend < 0 and vol < 2:
            return "Bearish"
        else:
            return "Sideways"
    except Exception as e:
        print(f"[WARN] Regime detection failed: {e}")
        return "Unknown"

# ----------------------------
# Timestamp helper
# ----------------------------
def _compute_forecast_timestamp(last_ts: pd.Timestamp, interval: str, step_index: int) -> str:
    try:
        interval = (interval or "1d").lower()
        
        if interval.endswith("m"):
            minutes = int(interval[:-1])
            result = last_ts + pd.to_timedelta(minutes * step_index, unit="m")
        elif interval.endswith("h"):
            hours = int(interval[:-1])
            result = last_ts + pd.to_timedelta(hours * step_index, unit="h")
        elif interval.endswith("d"):
            days = int(interval[:-1])
            result = last_ts + pd.to_timedelta(days * step_index, unit="D")
            if isinstance(result, pd.Timestamp):
                result = result.normalize()
        elif interval.endswith("wk"):
            weeks = int(interval[:-2])
            result = last_ts + pd.to_timedelta(7 * weeks * step_index, unit="D")
        elif interval.endswith("mo"):
            months = int(interval[:-2])
            result = last_ts + pd.DateOffset(months=months * step_index)
        else:
            result = last_ts + pd.to_timedelta(step_index, unit="D")
        
        return str(result)
    except Exception as e:
        print(f"[WARN] Timestamp computation failed: {e}")
        return str(last_ts + pd.to_timedelta(step_index, unit="D"))

# ----------------------------
# Core forecast function (PRESERVED)
# ----------------------------
def forecast_core(req: ForecastRequest) -> Dict[str, Any]:
    """Core forecasting function - UNCHANGED from original"""
    try:
        print(f"[INFO] Starting forecast for {req.ticker}")
        
        df = download_ticker(req.ticker, period=req.period, interval=req.interval)
        validate_data(df, req.ticker)
        
        max_rows = 5000
        if len(df) > max_rows:
            print(f"[INFO] Large dataset ({len(df)} rows), using last {max_rows} rows")
            df = df.tail(max_rows).reset_index(drop=True)
        
        df = add_technical_indicators(df)
        values, _ = prepare_features_for_model(df)
        
        save_dir = get_model_dir(req.ticker, req.interval)
        os.makedirs(save_dir, exist_ok=True)

        train_df_prophet = df[["ds", "y"]].iloc[-min(req.window_size, len(df)):]
        
        try:
            m_prophet = train_prophet(train_df_prophet)
            p_preds_future = prophet_predict(m_prophet, periods=req.steps)["yhat"].values[-req.steps:]
            joblib.dump(m_prophet, os.path.join(save_dir, "prophet.pkl"))
            print(f"[INFO] Prophet training successful")
        except Exception as e:
            print(f"[ERROR] Prophet training failed: {e}")
            recent_prices = df["y"].tail(10)
            trend = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / len(recent_prices)
            last_price = df["y"].iloc[-1]
            p_preds_future = np.array([last_price + trend * i for i in range(1, req.steps + 1)])
            m_prophet = None

        train_slice = values[-min(req.window_size, len(values)):]
        
        if len(train_slice) < req.context + 1:
            raise ValueError(f"Insufficient data: need at least {req.context + 1} samples, got {len(train_slice)}")
        
        X_all, y_all = create_sliding_windows(train_slice, req.context, req.horizon)
        
        if len(X_all) < 10:
            raise ValueError(f"Too few training windows: {len(X_all)} (need at least 10)")
        
        split = max(1, int(len(X_all) * 0.7))
        X_train, y_train = X_all[:split], y_all[:split, 0] if y_all.ndim > 1 else y_all[:split]
        X_val, y_val = X_all[split:], y_all[split:, 0] if y_all.ndim > 1 else y_all[split:]
        
        if len(X_val) == 0:
            val_split = max(1, int(len(X_train) * 0.8))
            X_val, y_val = X_train[val_split:], y_train[val_split:]
            X_train, y_train = X_train[:val_split], y_train[:val_split]
        
        try:
            X_train_s, X_val_s, scalers = scale_train_val_test(X_train, X_val)
        except Exception as e:
            print(f"[ERROR] Data scaling failed: {e}")
            raise ValueError(f"Data preprocessing failed: {e}")

        models = {}
        
        try:
            lstm = train_lstm(X_train_s, y_train, X_val_s, y_val, device=req.device, epochs=30)
            models['lstm'] = lstm
            if lstm is not None:
                torch.save(lstm.state_dict(), os.path.join(save_dir, "lstm.pt"))
            print(f"[INFO] LSTM training {'successful' if lstm is not None else 'failed'}")
        except Exception as e:
            print(f"[ERROR] LSTM training failed: {e}")
            models['lstm'] = None

        try:
            transformer = train_transformer(X_train_s, y_train, X_val_s, y_val, device=req.device, epochs=30)
            models['transformer'] = transformer
            if transformer is not None:
                torch.save(transformer.state_dict(), os.path.join(save_dir, "transformer.pt"))
            print(f"[INFO] Transformer training {'successful' if transformer is not None else 'failed'}")
        except Exception as e:
            print(f"[ERROR] Transformer training failed: {e}")
            models['transformer'] = None

        try:
            timesnet = train_timesnet(X_train_s, y_train, X_val_s, y_val, device=req.device, epochs=30)
            models['timesnet'] = timesnet
            if timesnet is not None:
                torch.save(timesnet.state_dict(), os.path.join(save_dir, "timesnet.pt"))
            print(f"[INFO] TimesNet training {'successful' if timesnet is not None else 'failed'}")
        except Exception as e:
            print(f"[ERROR] TimesNet training failed: {e}")
            models['timesnet'] = None

        successful_models = [k for k, v in models.items() if v is not None]
        if not successful_models:
            print("[WARN] All deep learning models failed to train, using Prophet only")
        
        try:
            joblib.dump(scalers, os.path.join(save_dir, "scalers.pkl"))
        except Exception as e:
            print(f"[WARN] Failed to save scalers: {e}")

        try:
            n_available = len(values)
            test_indices = []
            for i in range(req.steps):
                start_idx = n_available - req.context + i
                if start_idx >= 0 and start_idx + req.context <= n_available:
                    test_indices.append((start_idx, start_idx + req.context))
                else:
                    test_indices.append((n_available - req.context, n_available))
            
            X_test_windows = []
            for start_idx, end_idx in test_indices:
                X_test_windows.append(values[start_idx:end_idx])
            
            X_test = np.array(X_test_windows)
            
            for f in range(X_test.shape[-1]):
                if f < len(scalers):
                    X_test[:, :, f] = scalers[f].transform(X_test[:, :, f])
        
        except Exception as e:
            print(f"[ERROR] Test data preparation failed: {e}")
            last_window = values[-req.context:]
            X_test = np.array([last_window] * req.steps)
            for f in range(X_test.shape[-1]):
                if f < len(scalers):
                    X_test[:, :, f] = scalers[f].transform(X_test[:, :, f])

        l_preds = predict_lstm(models.get('lstm'), X_test, device=req.device)
        t_preds = predict_transformer(models.get('transformer'), X_test, device=req.device)
        tn_preds = predict_timesnet(models.get('timesnet'), X_test, device=req.device)

        try:
            if m_prophet is not None:
                val_start_date = df["ds"].iloc[-len(y_val) - req.steps]
                prophet_val_df = pd.DataFrame({
                    'ds': pd.date_range(start=val_start_date, periods=len(y_val), freq='D')
                })
                p_val_forecast = m_prophet.predict(prophet_val_df)
                p_val = p_val_forecast["yhat"].values
            else:
                p_val = np.full(len(y_val), df["y"].iloc[-1])
        except Exception as e:
            print(f"[WARN] Prophet validation predictions failed: {e}, using constant values")
            p_val = np.full(len(y_val), df["y"].iloc[-1])

        l_val = predict_lstm(models.get('lstm'), X_val_s, device=req.device)
        t_val = predict_transformer(models.get('transformer'), X_val_s, device=req.device)
        tn_val = predict_timesnet(models.get('timesnet'), X_val_s, device=req.device)

        try:
            min_len = min(len(p_val), len(l_val), len(t_val), len(tn_val), len(y_val))
            if min_len > 0:
                val_preds_matrix = np.column_stack([
                    p_val[:min_len], l_val[:min_len], 
                    t_val[:min_len], tn_val[:min_len]
                ])
                meta_model = fit_meta(val_preds_matrix, y_val[:min_len])
            else:
                meta_model = None
        except Exception as e:
            print(f"[ERROR] Meta-learner training failed: {e}")
            meta_model = None

        try:
            future_preds_matrix = np.column_stack([p_preds_future, l_preds, t_preds, tn_preds])
            
            if meta_model is not None:
                final_preds = predict_meta(meta_model, future_preds_matrix)
            else:
                print("[INFO] Using simple average for ensemble predictions")
                final_preds = np.mean(future_preds_matrix, axis=1)
            
            final_preds = clean_predictions(final_preds, method="clip")
            
        except Exception as e:
            print(f"[ERROR] Ensemble prediction failed: {e}")
            final_preds = clean_predictions(p_preds_future, method="clip")

        try:
            recent_actuals = values[-req.steps:, 0] if len(values) >= req.steps else values[:, 0]
            if len(recent_actuals) == len(final_preds):
                rm = rmse(recent_actuals, final_preds)
                mp = mape(recent_actuals, final_preds)
            else:
                rm, mp = 0.0, 0.0
        except Exception as e:
            print(f"[WARN] Metrics calculation failed: {e}")
            rm, mp = 0.0, 0.0

        current_vol, median_vol = get_current_and_median_vol(df, req.window_size)
        buy_th, sell_th, sl_pct = adapt_thresholds(
            req.buy_threshold_pct, req.sell_threshold_pct, req.stop_loss_pct,
            current_vol=current_vol, median_vol=median_vol
        )

        regime = detect_market_regime(df)
        last_actual_price = float(df["y"].iloc[-1])
        last_ts = pd.to_datetime(df["ds"].iloc[-1])
        
        results = []
        for i in range(len(final_preds)):
            try:
                price = float(final_preds[i])
                pct_change = ((price - last_actual_price) / last_actual_price) * 100.0
                trend = "Uptrend" if pct_change > 0.1 else ("Downtrend" if pct_change < -0.1 else "Sideways")

                signal, entry, stop_loss, take_profit, rr = make_signal(
                    current_price=last_actual_price,
                    forecast_price=price,
                    buy_threshold_pct=buy_th,
                    sell_threshold_pct=sell_th,
                    stop_loss_pct=sl_pct,
                    take_profit_rr=req.take_profit_rr
                )
                
                confidence = max(0.1, min(1.0, (abs(pct_change) / max(current_vol, 1e-6)) * (current_vol / median_vol)))
                future_time = _compute_forecast_timestamp(last_ts, req.interval, i + 1)

                results.append({
                    "date": future_time,
                    "prophet": float(p_preds_future[i]) if i < len(p_preds_future) else price,
                    "lstm": float(l_preds[i]) if i < len(l_preds) else price,
                    "transformer": float(t_preds[i]) if i < len(t_preds) else price,
                    "timesnet": float(tn_preds[i]) if i < len(tn_preds) else price,
                    "blended": price,
                    "trend": trend,
                    "change_pct": round(pct_change, 2),
                    "signal": signal,
                    "entry": entry,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "risk_reward": rr,
                    "confidence": round(float(confidence), 3),
                    "market_regime": regime
                })
            except Exception as e:
                print(f"[WARN] Failed to generate result for step {i}: {e}")
                continue

        hist_dates = df["ds"].dt.strftime("%Y-%m-%d %H:%M").tolist()
        hist_prices = df["y"].astype(float).tolist()
        future_dates = [r["date"] for r in results]

        response_data = {
            "ticker": req.ticker,
            "rmse": float(rm),
            "mape": float(mp),
            "market_regime": regime,
            "predictions": results,
            "history": {"dates": hist_dates, "prices": hist_prices},
            "future": {
                "dates": future_dates,
                "prophet": [float(x) for x in p_preds_future.tolist()],
                "lstm": [float(x) for x in l_preds.tolist()],
                "transformer": [float(x) for x in t_preds.tolist()],
                "timesnet": [float(x) for x in tn_preds.tolist()],
                "blended": [float(x) for x in final_preds.tolist()],
            }
        }
        
        print(f"[SUCCESS] Forecast completed for {req.ticker}")
        return response_data
        
    except Exception as e:
        print(f"[ERROR] Forecast core failed: {e}")
        print(traceback.format_exc())
        raise

# ----------------------------
# Chart generation (PRESERVED)
# ----------------------------
def make_watermarked_chart(data: Dict[str, Any], title: str = "") -> bytes:
    try:
        hist = data.get("history", {})
        fut = data.get("future", {})
        hist_dates = hist.get("dates", [])
        hist_prices = hist.get("prices", [])
        fut_dates = fut.get("dates", [])
        p_line = fut.get("prophet", [])
        l_line = fut.get("lstm", [])
        t_line = fut.get("transformer", [])
        tn_line = fut.get("timesnet", [])
        b_line = fut.get("blended", [])

        if not hist_prices:
            raise ValueError("No historical price data for chart")

        fig, ax = plt.subplots(figsize=(12, 6))
        
        hist_x = list(range(len(hist_dates)))
        ax.plot(hist_x, hist_prices, label="History", linewidth=2, color='blue')
        
        start_x = len(hist_dates)
        fut_x = list(range(start_x, start_x + len(fut_dates)))
        
        if p_line and len(p_line) == len(fut_x):
            ax.plot(fut_x, p_line, label="Prophet", linewidth=1.8, linestyle="--", alpha=0.8)
        if l_line and len(l_line) == len(fut_x):
            ax.plot(fut_x, l_line, label="LSTM", linewidth=1.8, linestyle="--", alpha=0.8)
        if t_line and len(t_line) == len(fut_x):
            ax.plot(fut_x, t_line, label="Transformer", linewidth=1.8, linestyle="--", alpha=0.8)
        if tn_line and len(tn_line) == len(fut_x):
            ax.plot(fut_x, tn_line, label="TimesNet", linewidth=1.8, linestyle="--", alpha=0.8)
        if b_line and len(b_line) == len(fut_x):
            ax.plot(fut_x, b_line, label="Blended", linewidth=2.4, color='red')

        ax.set_title(title or "Stock Sight Forecast")
        ax.set_xlabel("Time")
        ax.set_ylabel("Price")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.25)

        wm_text = "Stock Sight AI Forex Trading Forecasting Tool\n\nPowered By Pluto Technology"
        fig.text(0.5, 0.5, wm_text, fontsize=22, color="gray", ha="center", va="center", alpha=0.15, rotation=30)

        fig.text(0.995, 0.01, "Stock Sight • Telegram Service",
                 ha="right", va="bottom", fontsize=9, alpha=0.6)

        buf = io.BytesIO()
        plt.tight_layout()
        fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf.read()
        
    except Exception as e:
        print(f"[ERROR] Chart generation failed: {e}")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, f"Chart generation failed\n{title}", ha='center', va='center')
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        plt.close(fig)
        buf.seek(0)
        return buf.read()

# ----------------------------
# Formatting helpers (PRESERVED)
# ----------------------------
def accuracy_badge(mape_value: float) -> str:
    try:
        if not np.isfinite(mape_value) or mape_value <= 0:
            return ""
        if mape_value <= 2:
            return "✅ (Excellent)"
        elif mape_value <= 5:
            return "✅ (Good)"
        elif mape_value <= 10:
            return "⚠️ (Fair)"
        else:
            return "❌ (Weak)"
    except Exception:
        return ""

def fmt(x, nd=4):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    if isinstance(x, (int, np.integer)):
        return str(x)
    try:
        return f"{float(x):,.{nd}f}".rstrip("0").rstrip(".")
    except Exception:
        return str(x)

def chunk_text(s: str, limit: int = 4000) -> List[str]:
    out, cur = [], []
    cur_len = 0
    for line in s.splitlines(True):
        if cur_len + len(line) > limit:
            out.append("".join(cur))
            cur, cur_len = [line], len(line)
        else:
            cur.append(line)
            cur_len += len(line)
    if cur:
        out.append("".join(cur))
    return out

def build_forecast_table(preds: List[Dict[str, Any]]) -> List[str]:
    if not preds:
        return ["No predictions available."]
    
    header = (
        "Forecast Detail\n"
        "```text\n"
        f"{'Date':<16} {'Prophet':>8} {'LSTM':>8} {'Trans':>8} {'Times':>8} {'Blend':>8} {'Trend':>8} {'%':>5}\n"
        f"{'-'*16} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*5}\n"
    )
    lines = []
    for r in preds[:20]:
        date = r.get("date", "")[:16]
        lines.append(
            f"{date:<16} {fmt(r.get('prophet', 0), 2):>8} {fmt(r.get('lstm', 0), 2):>8} "
            f"{fmt(r.get('transformer', 0), 2):>8} {fmt(r.get('timesnet', 0), 2):>8} "
            f"{fmt(r.get('blended', 0), 2):>8} {r.get('trend', '')[:7]:>8} "
            f"{fmt(r.get('change_pct', 0), 1):>5}%\n"
        )
    footer = "```\n"
    chunks = chunk_text(header + "".join(lines) + footer)
    return chunks

def build_signal_table(preds: List[Dict[str, Any]]) -> List[str]:
    if not preds:
        return ["No signals available."]
        
    header = (
        "Trading Signals\n"
        "```text\n"
        f"{'Date':<16} {'Sig':>4} {'Entry':>9} {'SL':>9} {'TP':>9} {'RR':>4} {'Conf':>5}\n"
        f"{'-'*16} {'-'*4} {'-'*9} {'-'*9} {'-'*9} {'-'*4} {'-'*5}\n"
    )
    lines = []
    for r in preds[:20]:
        date = r.get("date", "")[:16]
        lines.append(
            f"{date:<16} {r.get('signal', '')[:4]:>4} {fmt(r.get('entry', 0)):>9} "
            f"{fmt(r.get('stop_loss', 0)):>9} {fmt(r.get('take_profit', 0)):>9} "
            f"{fmt(r.get('risk_reward', 0), 1):>4} {fmt(r.get('confidence', 0), 2):>5}\n"
        )
    footer = "```\n"
    chunks = chunk_text(header + "".join(lines) + footer)
    return chunks

# ----------------------------
# TELEGRAM COMMAND HANDLERS
# ----------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Enhanced /start command with automatic free trial and admin notification"""
    user = update.effective_user
    user_id = user.id if user else 0
    username = user.username if user else "Unknown"
    full_name = user.full_name if user else "Unknown"
    
    print(f"[INFO] User {user_id} (@{username}) started bot")
    
    if is_new_user(user_id):
        expiry = activate_subscription_for(user_id, days=FREE_TRIAL_DAYS, trial=True, 
                                          username=username, full_name=full_name)
        
        admin_message = (
            f"🆕 *NEW USER STARTED BOT*\n\n"
            f"👤 User: {full_name}\n"
            f"🆔 Telegram ID: `{user_id}`\n"
            f"📱 Username: @{username}\n"
            f"🎁 Status: {FREE_TRIAL_DAYS}-Day Free Trial Activated\n"
            f"⏰ Expires: {expiry.strftime('%Y-%m-%d %H:%M UTC')}\n\n"
            f"Use `/activate {user_id} 30` to extend after payment."
        )
        await notify_admins(context, admin_message)
        
        welcome_msg = (
            f"🎉 *Welcome to {VERSION}!*\n\n"
            f"You have been automatically activated with a *{FREE_TRIAL_DAYS}-day free trial*!\n\n"
            f"⏰ Trial expires: {expiry.strftime('%Y-%m-%d %H:%M UTC')}\n\n"
            f"*Commands:*\n"
            f"📊 /forecast TICKER [PERIOD] [INTERVAL] [STEPS]\n"
            f"Example: `/forecast AAPL 1y 1d 30`\n\n"
            f"💳 /subscribe - Get subscription info\n"
            f"📱 /status - Check your subscription status\n\n"
            f"Enjoy your free trial! 🚀"
        )
    else:
        subscribed, expiry = is_subscribed(user_id)
        
        if subscribed:
            welcome_msg = (
                f"👋 *Welcome back to {VERSION}!*\n\n"
                f"✅ Your subscription is active until: {expiry.strftime('%Y-%m-%d %H:%M UTC')}\n\n"
                f"*Commands:*\n"
                f"📊 /forecast TICKER [PERIOD] [INTERVAL] [STEPS]\n"
                f"Example: `/forecast AAPL 1y 1d 30`\n\n"
                f"📱 /status - Check subscription status\n"
            )
        else:
            welcome_msg = (
                f"👋 *Welcome back to {VERSION}!*\n\n"
                f"⚠️ Your subscription has expired.\n\n"
                f"💳 Use /subscribe to renew your access.\n"
                f"📱 Use /status to check details."
            )
    
    await update.message.reply_text(welcome_msg, parse_mode=ParseMode.MARKDOWN)

async def subscribe_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Enhanced subscribe command"""
    user_id = update.effective_user.id if update.effective_user else 0
    
    if SUBSCRIBE_URL:
        text = (
            f"💳 *SUBSCRIPTION INFORMATION*\n\n"
            f"To subscribe or renew, complete payment at:\n\n"
            f"{SUBSCRIBE_URL}\n\n"
            f"📝 After payment, use this command:\n"
            f"`/paid <transaction_id>`\n\n"
            f"This will notify our admins who will activate your account immediately.\n\n"
            f"🆔 Your Telegram ID: `{user_id}`\n"
            f"(Please include this in your payment reference)"
        )
    else:
        text = (
            f"💳 *SUBSCRIPTION INFORMATION*\n\n"
            f"Subscription system not configured.\n\n"
            f"🆔 Your Telegram ID: `{user_id}`\n\n"
            f"Please contact admin for activation."
        )
    
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)

async def paid_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Enhanced paid command with admin notification including user Telegram ID"""
    user = update.effective_user
    uid = user.id if user else None
    username = user.username if user else "Unknown"
    full_name = user.full_name if user else "Unknown"
    
    details = " ".join(context.args) if context.args else "(no transaction details provided)"
    
    msg = (
        f"💰 *PAYMENT NOTIFICATION*\n\n"
        f"👤 User: {full_name}\n"
        f"🆔 Telegram ID: `{uid}`\n"
        f"📱 Username: @{username}\n"
        f"💳 Transaction Details: {details}\n\n"
        f"⚡ *Quick Activation:*\n"
        f"`/activate {uid} 30`\n\n"
        f"(Tap to copy command, then send to activate for 30 days)"
    )
    
    await notify_admins(context, msg)
    
    user_msg = (
        f"✅ *Payment notification sent to admins!*\n\n"
        f"🆔 Your Telegram ID: `{uid}`\n"
        f"💳 Transaction: {details}\n\n"
        f"Our team will activate your account shortly.\n"
        f"You'll receive a confirmation message once activated."
    )
    
    await update.message.reply_text(user_msg, parse_mode=ParseMode.MARKDOWN)

async def activate_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Enhanced activate command (admin only)"""
    caller = update.effective_user.id if update.effective_user else None
    
    if caller not in ADMIN_IDS:
        await update.message.reply_text("❌ Unauthorized. Admin access required.")
        return
    
    if not context.args:
        await update.message.reply_text(
            "📝 *Usage:* `/activate <telegram_id> [days]`\n\n"
            "Example: `/activate 123456789 30`\n"
            "(Default: 30 days if not specified)",
            parse_mode=ParseMode.MARKDOWN
        )
        return
    
    try:
        target = int(context.args[0])
        days = int(context.args[1]) if len(context.args) >= 2 else 30
    except ValueError:
        await update.message.reply_text("❌ Invalid arguments. Use numeric values.")
        return
    
    # Try to get user info
    try:
        user_obj = await context.bot.get_chat(target)
        username = user_obj.username if user_obj.username else ""
        full_name = user_obj.full_name if user_obj.full_name else ""
    except:
        username = ""
        full_name = ""
    
    exp = activate_subscription_for(target, days=days, trial=False, 
                                   username=username, full_name=full_name)
    
    await update.message.reply_text(
        f"✅ *User Activated Successfully*\n\n"
        f"🆔 Telegram ID: `{target}`\n"
        f"⏰ Active until: {exp.strftime('%Y-%m-%d %H:%M UTC')}\n"
        f"📅 Duration: {days} days",
        parse_mode=ParseMode.MARKDOWN
    )
    
    try:
        await context.bot.send_message(
            chat_id=target,
            text=(
                f"🎉 *Your subscription has been activated!*\n\n"
                f"✅ Active until: {exp.strftime('%Y-%m-%d %H:%M UTC')}\n"
                f"📅 Duration: {days} days\n\n"
                f"You can now use /forecast to get trading signals!"
            ),
            parse_mode=ParseMode.MARKDOWN
        )
    except Exception as e:
        print(f"[WARN] Could not notify user {target}: {e}")
        await update.message.reply_text(
            f"⚠️ User activated but couldn't send notification (user may have blocked bot)"
        )

async def deactivate_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Enhanced deactivate command (admin only)"""
    caller = update.effective_user.id if update.effective_user else None
    
    if caller not in ADMIN_IDS:
        await update.message.reply_text("❌ Unauthorized. Admin access required.")
        return
    
    if not context.args:
        await update.message.reply_text(
            "📝 *Usage:* `/deactivate <telegram_id>`\n\n"
            "Example: `/deactivate 123456789`",
            parse_mode=ParseMode.MARKDOWN
        )
        return
    
    try:
        target = int(context.args[0])
    except ValueError:
        await update.message.reply_text("❌ Invalid telegram_id. Use numeric value.")
        return
    
    success = deactivate_subscription_for(target)
    
    if success:
        await update.message.reply_text(
            f"✅ *Subscription Deactivated*\n\n"
            f"🆔 Telegram ID: `{target}`",
            parse_mode=ParseMode.MARKDOWN
        )
        
        try:
            await context.bot.send_message(
                chat_id=target,
                text="⚠️ Your subscription has been deactivated.\n\nUse /subscribe to renew."
            )
        except Exception as e:
            print(f"[WARN] Could not notify user {target}: {e}")
    else:
        await update.message.reply_text("❌ No active subscription found for this user.")

async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Enhanced status command"""
    user = update.effective_user
    user_id = user.id if user else None
    
    if not user_id:
        await update.message.reply_text("❌ Could not determine your user ID.")
        return
    
    subscribed, expiry = is_subscribed(user_id)
    subs = _load_subs()
    user_data = subs.get(str(user_id), {})
    is_trial = user_data.get("is_trial", False)
    activated_at = user_data.get("activated_at", "Unknown")
    
    # Add Google Sheets sync status
    sheets_status = "✅ Connected" if sheets_manager.enabled else "⚠️ Using Local Storage"
    
    if subscribed and expiry:
        now = datetime.datetime.utcnow()
        days_left = (expiry - now).days
        hours_left = ((expiry - now).seconds // 3600)
        
        status_icon = "🎁" if is_trial else "✅"
        status_text = "Free Trial" if is_trial else "Active Subscription"
        
        text = (
            f"{status_icon} *{status_text}*\n\n"
            f"🆔 Your Telegram ID: `{user_id}`\n"
            f"📅 Activated: {activated_at[:10] if activated_at != 'Unknown' else 'Unknown'}\n"
            f"⏰ Expires: {expiry.strftime('%Y-%m-%d %H:%M UTC')}\n"
            f"⏳ Time remaining: {days_left} days, {hours_left} hours\n"
            f"☁️ Storage: {sheets_status}\n\n"
        )
        
        if is_trial and days_left <= 1:
            text += (
                f"⚠️ *Trial expiring soon!*\n"
                f"Use /subscribe to continue access."
            )
        elif days_left <= 3:
            text += (
                f"⚠️ *Subscription expiring soon!*\n"
                f"Use /subscribe to renew."
            )
        else:
            text += "✨ Enjoy your access!"
            
    else:
        if expiry:
            text = (
                f"⚠️ *Subscription Expired*\n\n"
                f"🆔 Your Telegram ID: `{user_id}`\n"
                f"❌ Expired on: {expiry.strftime('%Y-%m-%d %H:%M UTC')}\n"
                f"☁️ Storage: {sheets_status}\n\n"
                f"💳 Use /subscribe to renew access."
            )
        else:
            text = (
                f"❌ *No Active Subscription*\n\n"
                f"🆔 Your Telegram ID: `{user_id}`\n"
                f"☁️ Storage: {sheets_status}\n\n"
                f"💳 Use /subscribe for payment instructions."
            )
    
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)

def parse_forecast_args(args: List[str]) -> ForecastRequest:
    """Parse forecast command arguments with validation"""
    try:
        ticker = args[0].upper() if len(args) >= 1 else ""
        ticker = normalize_ticker(ticker)
        period = args[1] if len(args) >= 2 else "1y"
        interval = args[2] if len(args) >= 3 else "1d"
        
        try:
            steps = int(args[3]) if len(args) >= 4 else 30
        except (ValueError, IndexError):
            steps = 30
        
        steps = max(1, min(steps, 50))
        
        return ForecastRequest(
            ticker=ticker, period=period, interval=interval, steps=steps,
            context=60, horizon=1, window_size=300, device="cpu",
            buy_threshold_pct=0.3, sell_threshold_pct=-0.3, 
            stop_loss_pct=0.5, take_profit_rr=2.0
        )
    except Exception as e:
        raise ValueError(f"Invalid forecast arguments: {e}")

async def forecast_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Enhanced forecast command with subscription check"""
    if not update.message:
        return
        
    args = context.args or []
    if not args:
        await update.message.reply_text(
            "📝 *Usage:* `/forecast <ticker> [period] [interval] [steps]`\n\n"
            "*Example:* `/forecast AAPL 1y 1d 30`\n\n"
            "*Parameters:*\n"
            "• ticker: Stock/crypto symbol (required)\n"
            "• period: Data period (default: 1y)\n"
            "• interval: Timeframe (default: 1d)\n"
            "• steps: Forecast steps (default: 30)",
            parse_mode=ParseMode.MARKDOWN
        )
        return

    user_id = update.effective_user.id if update.effective_user else None
    
    subscribed, expiry = is_subscribed(user_id)
    if not subscribed:
        subs = _load_subs()
        user_data = subs.get(str(user_id), {})
        was_trial = user_data.get("is_trial", False)
        
        if expiry:
            if was_trial:
                msg = (
                    f"⚠️ *Free Trial Expired*\n\n"
                    f"Your {FREE_TRIAL_DAYS}-day trial ended on {expiry.strftime('%Y-%m-%d %H:%M UTC')}\n\n"
                    f"💳 Use /subscribe to continue access."
                )
            else:
                msg = (
                    f"⚠️ *Subscription Expired*\n\n"
                    f"Expired on {expiry.strftime('%Y-%m-%d %H:%M UTC')}\n\n"
                    f"💳 Use /subscribe to renew."
                )
        else:
            msg = (
                f"❌ *No Active Subscription*\n\n"
                f"💳 Use /subscribe for payment instructions."
            )
        await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)
        return

    try:
        req = parse_forecast_args(args)
    except Exception as e:
        await update.message.reply_text(f"❌ Invalid arguments: {e}")
        return

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    
    async def timeout_handler():
        await asyncio.sleep(600)
        return None
    
    try:
        status_msg = await update.message.reply_text(f"🔄 Processing forecast for {req.ticker}...")
        
        forecast_task = asyncio.create_task(asyncio.to_thread(forecast_core, req))
        timeout_task = asyncio.create_task(timeout_handler())
        
        done, pending = await asyncio.wait(
            [forecast_task, timeout_task], 
            return_when=asyncio.FIRST_COMPLETED
        )
        
        for task in pending:
            task.cancel()
        
        if forecast_task in done:
            data = await forecast_task
        else:
            await status_msg.edit_text(f"⏰ Forecast for {req.ticker} timed out (10min limit)")
            return
        
        if not data or not data.get("predictions"):
            await status_msg.edit_text(f"❌ No forecast data generated for {req.ticker}")
            return
        
        predictions = data.get("predictions", [])
        if len(predictions) == 0:
            await status_msg.edit_text(f"❌ No predictions generated for {req.ticker}")
            return
        
        await status_msg.edit_text(f"📊 Generating results for {req.ticker}...")
        
        mape_val = data.get('mape', 0)
        badge = accuracy_badge(mape_val)
        summary = (
            f"📈 **{req.ticker}** ({req.period}, {req.interval}, {req.steps} steps)\n"
            f"**RMSE:** {fmt(data.get('rmse', 0))}\n"
            f"**MAPE:** {fmt(mape_val)}% {badge}\n"
            f"**Market Regime:** {data.get('market_regime', 'Unknown')}\n"
        )
        await update.message.reply_text(summary, parse_mode=ParseMode.MARKDOWN)
        
        forecast_chunks = build_forecast_table(predictions)
        for chunk in forecast_chunks[:3]:
            await update.message.reply_text(chunk, parse_mode=ParseMode.MARKDOWN)
        
        signal_chunks = build_signal_table(predictions)
        for chunk in signal_chunks[:3]:
            await update.message.reply_text(chunk, parse_mode=ParseMode.MARKDOWN)
        
        try:
            title = f"{req.ticker} Forecast ({req.period}, {req.interval})"
            chart_bytes = make_watermarked_chart(data, title=title)
            chart_bio = io.BytesIO(chart_bytes)
            chart_bio.name = f"{req.ticker}_forecast.png"
            await update.message.reply_photo(photo=InputFile(chart_bio))
        except Exception as chart_error:
            print(f"[ERROR] Chart generation failed: {chart_error}")
            await update.message.reply_text("⚠️ Chart generation failed")
        
        try:
            await status_msg.delete()
        except Exception:
            pass
        
        print(f"[SUCCESS] Forecast completed for {req.ticker} (user: {user_id})")
        
    except Exception as e:
        error_msg = str(e)[:200]
        await update.message.reply_text(f"❌ Forecast failed: {error_msg}")
        print(f"[ERROR] Forecast command failed: {e}")
        print(traceback.format_exc())

# ----------------------------
# Main application
# ----------------------------
def main():
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN environment variable required")
    
    print(f"[INFO] Starting {VERSION}")
    print(f"[INFO] Admin IDs: {ADMIN_IDS}")
    print(f"[INFO] Subscription URL configured: {bool(SUBSCRIBE_URL)}")
    print(f"[INFO] Free trial period: {FREE_TRIAL_DAYS} days")
    print(f"[INFO] Local subscriptions file: {SUBS_FILE}")
    print(f"[INFO] Google Sheets enabled: {GOOGLE_SHEETS_ENABLED}")
    print(f"[INFO] Google Sheets manager status: {sheets_manager.enabled}")
    
    if sheets_manager.enabled:
        print(f"[INFO] ✅ Google Sheets integration active")
        print(f"[INFO] Sheet name: {GOOGLE_SHEET_NAME}")
        print(f"[INFO] Worksheet: {GOOGLE_WORKSHEET_NAME}")
    else:
        print(f"[INFO] ⚠️ Using local JSON storage only")
        if GOOGLE_SHEETS_ENABLED and not GSPREAD_AVAILABLE:
            print(f"[WARN] gspread library not installed")
        elif GOOGLE_SHEETS_ENABLED:
            print(f"[WARN] Google Sheets initialization failed: {sheets_manager.last_error}")
    
    os.makedirs(os.path.dirname(SUBS_FILE) if os.path.dirname(SUBS_FILE) else ".", exist_ok=True)
    
    app = Application.builder().token(token).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("forecast", forecast_cmd))
    app.add_handler(CommandHandler("subscribe", subscribe_cmd))
    app.add_handler(CommandHandler("paid", paid_cmd))
    app.add_handler(CommandHandler("activate", activate_cmd))
    app.add_handler(CommandHandler("deactivate", deactivate_cmd))
    app.add_handler(CommandHandler("status", status_cmd))

    print("[INFO] Bot starting...")
    app.run_polling(close_loop=False)

if __name__ == "__main__":
    main()
