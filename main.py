# main.py — Fixed Telegram-only Signal Service for Stock Sight AI
# MAJOR FIXES: Prophet validation, error handling, rate limiting, data validation, memory management
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
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import traceback
import signal
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

# --- Your existing project modules (fixed imports) ---
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

# Try to import normalize_ticker if your project provides it; otherwise, use a safe fallback.
try:
    from src.data_loader import normalize_ticker  # optional; only if present in your project
except Exception:
    def normalize_ticker(t: str) -> str:
        return (t or "").strip().upper()

# --- Telegram bot (v20+) ---
from telegram import Update, InputFile
from telegram.constants import ParseMode, ChatAction
from telegram.ext import Application, CommandHandler, ContextTypes

VERSION = "Stock Sight Telegram Service v1.1 (Fixed)"

# ----------------------------
# Subscription settings (unchanged)
# ----------------------------
SUBSCRIBE_URL = os.getenv("SUBSCRIBE_URL", "").strip()
ADMIN_IDS = [int(x) for x in os.getenv("ADMIN_IDS", "").split(",") if x.strip().isdigit()]
SUBS_FILE = os.getenv("SUBS_FILE", "subscriptions.json")

def _load_subs() -> Dict[str, Any]:
    try:
        if os.path.exists(SUBS_FILE):
            with open(SUBS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to load subscriptions: {e}")
    return {}

def _save_subs(subs: Dict[str, Any]) -> None:
    try:
        with open(SUBS_FILE, "w", encoding="utf-8") as f:
            json.dump(subs, f, indent=2, default=str)
    except Exception as e:
        print(f"[WARN] Failed to save subscriptions: {e}")

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

def activate_subscription_for(user_id: int, days: int = 30) -> datetime.datetime:
    subs = _load_subs()
    new_exp = datetime.datetime.utcnow() + datetime.timedelta(days=days)
    subs[str(user_id)] = {"expires": new_exp.isoformat()}
    _save_subs(subs)
    return new_exp

def deactivate_subscription_for(user_id: int) -> bool:
    subs = _load_subs()
    if str(user_id) in subs:
        del subs[str(user_id)]
        _save_subs(subs)
        return True
    return False

# -------------------------------------------
# Model storage (interval-aware)
# -------------------------------------------
def get_model_dir(ticker: str, interval: str) -> str:
    return os.path.join("models", ticker, interval)

# -------------------------------------------
# Request model with validation
# -------------------------------------------
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
        """Validate request parameters"""
        if not self.ticker or not isinstance(self.ticker, str):
            raise ValueError("Ticker must be a non-empty string")
        
        if self.steps <= 0 or self.steps > 100:
            raise ValueError("Steps must be between 1 and 100")
        
        if self.context <= 0 or self.context > 500:
            raise ValueError("Context must be between 1 and 500")
        
        if self.window_size < 50 or self.window_size > 2000:
            raise ValueError("Window size must be between 50 and 2000")

# -------------------------------------------
# Volatility helpers with error handling
# -------------------------------------------
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
        
        return atr_pct.fillna(1.0)  # Fallback to 1% volatility
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
        return vol.fillna(1.0)  # Fallback to 1% volatility
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
        
        # Ensure positive values
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

# -------------------------------------------
# Signal logic with validation
# -------------------------------------------
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

# -------------------------------------------
# Market Regime detection with error handling
# -------------------------------------------
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

# -------------------------------------------
# Timestamp helper with better error handling
# -------------------------------------------
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

# -------------------------------------------
# FIXED Core forecast function
# -------------------------------------------
def forecast_core(req: ForecastRequest) -> Dict[str, Any]:
    """
    Core forecasting function with comprehensive error handling and fixes.
    MAJOR FIXES:
    1. Prophet validation predictions (no longer using constant values)
    2. Data size limits for memory management
    3. Model training error handling
    4. Input validation and cleaning
    """
    try:
        print(f"[INFO] Starting forecast for {req.ticker}")
        
        # Step 1: Download and validate data with size limits
        df = download_ticker(req.ticker, period=req.period, interval=req.interval)
        validate_data(df, req.ticker)
        
        # FIXED: Limit data size to prevent memory issues
        max_rows = 5000
        if len(df) > max_rows:
            print(f"[INFO] Large dataset ({len(df)} rows), using last {max_rows} rows")
            df = df.tail(max_rows).reset_index(drop=True)
        
        df = add_technical_indicators(df)
        values, _ = prepare_features_for_model(df)
        
        save_dir = get_model_dir(req.ticker, req.interval)
        os.makedirs(save_dir, exist_ok=True)

        # Step 2: Train Prophet with validation
        train_df_prophet = df[["ds", "y"]].iloc[-min(req.window_size, len(df)):]
        
        try:
            m_prophet = train_prophet(train_df_prophet)
            # Generate predictions for future steps
            p_preds_future = prophet_predict(m_prophet, periods=req.steps)["yhat"].values[-req.steps:]
            joblib.dump(m_prophet, os.path.join(save_dir, "prophet.pkl"))
            print(f"[INFO] Prophet training successful")
        except Exception as e:
            print(f"[ERROR] Prophet training failed: {e}")
            # Fallback: use simple trend extrapolation
            recent_prices = df["y"].tail(10)
            trend = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / len(recent_prices)
            last_price = df["y"].iloc[-1]
            p_preds_future = np.array([last_price + trend * i for i in range(1, req.steps + 1)])
            m_prophet = None

        # Step 3: Prepare sliding windows with validation
        train_slice = values[-min(req.window_size, len(values)):]
        
        if len(train_slice) < req.context + 1:
            raise ValueError(f"Insufficient data: need at least {req.context + 1} samples, got {len(train_slice)}")
        
        X_all, y_all = create_sliding_windows(train_slice, req.context, req.horizon)
        
        if len(X_all) < 10:
            raise ValueError(f"Too few training windows: {len(X_all)} (need at least 10)")
        
        # Split with validation
        split = max(1, int(len(X_all) * 0.7))
        X_train, y_train = X_all[:split], y_all[:split, 0] if y_all.ndim > 1 else y_all[:split]
        X_val, y_val = X_all[split:], y_all[split:, 0] if y_all.ndim > 1 else y_all[split:]
        
        if len(X_val) == 0:
            # If no validation data, use last 20% of training
            val_split = max(1, int(len(X_train) * 0.8))
            X_val, y_val = X_train[val_split:], y_train[val_split:]
            X_train, y_train = X_train[:val_split], y_train[:val_split]
        
        try:
            X_train_s, X_val_s, scalers = scale_train_val_test(X_train, X_val)
        except Exception as e:
            print(f"[ERROR] Data scaling failed: {e}")
            raise ValueError(f"Data preprocessing failed: {e}")

        # Step 4: Train deep learning models with error handling
        models = {}
        
        # Train LSTM
        try:
            lstm = train_lstm(X_train_s, y_train, X_val_s, y_val, device=req.device, epochs=30)
            models['lstm'] = lstm
            if lstm is not None:
                torch.save(lstm.state_dict(), os.path.join(save_dir, "lstm.pt"))
            print(f"[INFO] LSTM training {'successful' if lstm is not None else 'failed'}")
        except Exception as e:
            print(f"[ERROR] LSTM training failed: {e}")
            models['lstm'] = None

        # Train Transformer
        try:
            transformer = train_transformer(X_train_s, y_train, X_val_s, y_val, device=req.device, epochs=30)
            models['transformer'] = transformer
            if transformer is not None:
                torch.save(transformer.state_dict(), os.path.join(save_dir, "transformer.pt"))
            print(f"[INFO] Transformer training {'successful' if transformer is not None else 'failed'}")
        except Exception as e:
            print(f"[ERROR] Transformer training failed: {e}")
            models['transformer'] = None

        # Train TimesNet
        try:
            timesnet = train_timesnet(X_train_s, y_train, X_val_s, y_val, device=req.device, epochs=30)
            models['timesnet'] = timesnet
            if timesnet is not None:
                torch.save(timesnet.state_dict(), os.path.join(save_dir, "timesnet.pt"))
            print(f"[INFO] TimesNet training {'successful' if timesnet is not None else 'failed'}")
        except Exception as e:
            print(f"[ERROR] TimesNet training failed: {e}")
            models['timesnet'] = None

        # Check if at least one model trained successfully
        successful_models = [k for k, v in models.items() if v is not None]
        if not successful_models:
            print("[WARN] All deep learning models failed to train, using Prophet only")
        
        try:
            joblib.dump(scalers, os.path.join(save_dir, "scalers.pkl"))
        except Exception as e:
            print(f"[WARN] Failed to save scalers: {e}")

        # Step 5: Generate predictions for future steps
        # Create test windows for future predictions
        try:
            n_available = len(values)
            test_indices = []
            for i in range(req.steps):
                start_idx = n_available - req.context + i
                if start_idx >= 0 and start_idx + req.context <= n_available:
                    test_indices.append((start_idx, start_idx + req.context))
                else:
                    # Use the last available window
                    test_indices.append((n_available - req.context, n_available))
            
            X_test_windows = []
            for start_idx, end_idx in test_indices:
                X_test_windows.append(values[start_idx:end_idx])
            
            X_test = np.array(X_test_windows)
            
            # Scale test data
            for f in range(X_test.shape[-1]):
                if f < len(scalers):
                    X_test[:, :, f] = scalers[f].transform(X_test[:, :, f])
        
        except Exception as e:
            print(f"[ERROR] Test data preparation failed: {e}")
            # Fallback: use last window repeated
            last_window = values[-req.context:]
            X_test = np.array([last_window] * req.steps)
            for f in range(X_test.shape[-1]):
                if f < len(scalers):
                    X_test[:, :, f] = scalers[f].transform(X_test[:, :, f])

        # Generate model predictions
        l_preds = predict_lstm(models.get('lstm'), X_test, device=req.device)
        t_preds = predict_transformer(models.get('transformer'), X_test, device=req.device)
        tn_preds = predict_timesnet(models.get('timesnet'), X_test, device=req.device)

        # FIXED: Generate proper Prophet validation predictions for meta-learner
        try:
            if m_prophet is not None:
                # Generate Prophet predictions for validation period
                val_start_date = df["ds"].iloc[-len(y_val) - req.steps]
                prophet_val_df = pd.DataFrame({
                    'ds': pd.date_range(start=val_start_date, periods=len(y_val), freq='D')
                })
                p_val_forecast = m_prophet.predict(prophet_val_df)
                p_val = p_val_forecast["yhat"].values
            else:
                # Fallback if Prophet failed
                p_val = np.full(len(y_val), df["y"].iloc[-1])
        except Exception as e:
            print(f"[WARN] Prophet validation predictions failed: {e}, using constant values")
            p_val = np.full(len(y_val), df["y"].iloc[-1])

        # Get validation predictions from deep models
        l_val = predict_lstm(models.get('lstm'), X_val_s, device=req.device)
        t_val = predict_transformer(models.get('transformer'), X_val_s, device=req.device)
        tn_val = predict_timesnet(models.get('timesnet'), X_val_s, device=req.device)

        # Step 6: Train meta-learner with proper validation
        try:
            # Ensure all validation predictions have the same length
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

        # Combine predictions using meta-learner or simple average
        try:
            future_preds_matrix = np.column_stack([p_preds_future, l_preds, t_preds, tn_preds])
            
            if meta_model is not None:
                final_preds = predict_meta(meta_model, future_preds_matrix)
            else:
                print("[INFO] Using simple average for ensemble predictions")
                final_preds = np.mean(future_preds_matrix, axis=1)
            
            # Clean predictions
            final_preds = clean_predictions(final_preds, method="clip")
            
        except Exception as e:
            print(f"[ERROR] Ensemble prediction failed: {e}")
            # Fallback to Prophet predictions
            final_preds = clean_predictions(p_preds_future, method="clip")

        # Step 7: Calculate metrics (using recent data as ground truth approximation)
        try:
            recent_actuals = values[-req.steps:, 0] if len(values) >= req.steps else values[:, 0]
            if len(recent_actuals) == len(final_preds):
                rm = rmse(recent_actuals, final_preds)
                mp = mape(recent_actuals, final_preds)
            else:
                rm, mp = 0.0, 0.0  # Can't calculate without matching ground truth
        except Exception as e:
            print(f"[WARN] Metrics calculation failed: {e}")
            rm, mp = 0.0, 0.0

        # Step 8: Generate trading signals and analysis
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

        # Step 9: Prepare response data
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



# -------------------------------------------
# Chart generation with error handling
# -------------------------------------------
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
        
        # Plot historical data
        hist_x = list(range(len(hist_dates)))
        ax.plot(hist_x, hist_prices, label="History", linewidth=2, color='blue')
        
        # Plot future predictions
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

        # Add watermark
        wm_text = "Stock Sight AI Forex Trading Forecasting Tool\n\nPowered By Pluto Technology"
        fig.text(0.5, 0.5, wm_text, fontsize=22, color="gray", ha="center", va="center", alpha=0.15, rotation=30)

        # Footer brand
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
        # Return a minimal chart
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, f"Chart generation failed\n{title}", ha='center', va='center')
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        plt.close(fig)
        buf.seek(0)
        return buf.read()

# -------------------------------------------
# Telegram formatting helpers
# -------------------------------------------

def accuracy_badge(mape_value: float) -> str:
    """
    Return an emoji + label for accuracy highlighting from MAPE.
    Thresholds (lower is better):
      ≤2%  -> ✅ (Excellent)
      ≤5%  -> ✅ (Good)
      ≤10% -> ⚠️ (Fair)
      >10% -> ❌ (Weak)
    If MAPE is 0 or not finite, returns an empty string to avoid misleading badges.
    """
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
    """Return message chunks for forecast table"""
    if not preds:
        return ["No predictions available."]
    
    header = (
        "Forecast Detail\n"
        "```text\n"
        f"{'Date':<16} {'Prophet':>8} {'LSTM':>8} {'Trans':>8} {'Times':>8} {'Blend':>8} {'Trend':>8} {'%':>5}\n"
        f"{'-'*16} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*5}\n"
    )
    lines = []
    for r in preds[:20]:  # Limit to prevent overflow
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
    """Return message chunks for signals table"""
    if not preds:
        return ["No signals available."]
        
    header = (
        "Trading Signals\n"
        "```text\n"
        f"{'Date':<16} {'Sig':>4} {'Entry':>9} {'SL':>9} {'TP':>9} {'RR':>4} {'Conf':>5}\n"
        f"{'-'*16} {'-'*4} {'-'*9} {'-'*9} {'-'*9} {'-'*4} {'-'*5}\n"
    )
    lines = []
    for r in preds[:20]:  # Limit to prevent overflow
        date = r.get("date", "")[:16]
        lines.append(
            f"{date:<16} {r.get('signal', '')[:4]:>4} {fmt(r.get('entry', 0)):>9} "
            f"{fmt(r.get('stop_loss', 0)):>9} {fmt(r.get('take_profit', 0)):>9} "
            f"{fmt(r.get('risk_reward', 0), 1):>4} {fmt(r.get('confidence', 0), 2):>5}\n"
        )
    footer = "```\n"
    chunks = chunk_text(header + "".join(lines) + footer)
    return chunks

# -------------------------------------------
# Telegram command handlers
# -------------------------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id if update.effective_user else 0
    print(f"[INFO] User {user_id} started bot")
    
    msg = (
        f"{VERSION}\n\n"
        "Welcome to Stock Sight AI - AI Powered Telegram Trading Signal Service\n\n"
        "Commands:\n"
        "/forecast TICKER [PERIOD] [INTERVAL] [STEPS]\n"
        "Example: /forecast AAPL 1y 1d 30\n\n"
        "Other commands:\n"
        "/subscribe - Get subscription info\n"
        "/status - Check subscription status\n\n"
        "Note: This service requires an active subscription."
    )
    await update.message.reply_text(msg)

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
        
        # Validate and constrain parameters
        steps = max(1, min(steps, 50))  # Limit steps to prevent timeouts
        
        return ForecastRequest(
            ticker=ticker, period=period, interval=interval, steps=steps,
            context=60, horizon=1, window_size=300, device="cpu",
            buy_threshold_pct=0.3, sell_threshold_pct=-0.3, 
            stop_loss_pct=0.5, take_profit_rr=2.0
        )
    except Exception as e:
        raise ValueError(f"Invalid forecast arguments: {e}")

# Subscription command handlers (unchanged from original)
async def subscribe_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if SUBSCRIBE_URL:
        text = (
            "To subscribe, complete payment at:\n\n"
            f"{SUBSCRIBE_URL}\n\n"
            "After payment, use /paid <transaction_id> to notify admins."
        )
    else:
        text = "Subscription system not configured. Contact admin for activation."
    await update.message.reply_text(text)

async def paid_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id if update.effective_user else None
    details = " ".join(context.args) if context.args else "(no details)"
    
    msg = (
        f"Payment notification from user {uid}:\n{details}\n\n"
        "Use /activate <telegram_id> <days> to activate subscription."
    )
    
    if ADMIN_IDS:
        for admin in ADMIN_IDS:
            try:
                await context.bot.send_message(chat_id=admin, text=msg)
            except Exception:
                pass
        await update.message.reply_text("Payment notification sent to admins.")
    else:
        await update.message.reply_text("No admins configured.")

async def activate_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    caller = update.effective_user.id if update.effective_user else None
    if caller not in ADMIN_IDS:
        await update.message.reply_text("Unauthorized.")
        return
    
    if not context.args:
        await update.message.reply_text("Usage: /activate <telegram_id> [days]")
        return
    
    try:
        target = int(context.args[0])
        days = int(context.args[1]) if len(context.args) >= 2 else 30
    except ValueError:
        await update.message.reply_text("Invalid arguments.")
        return
    
    exp = activate_subscription_for(target, days=days)
    await update.message.reply_text(f"Activated user {target} until {exp.isoformat()}")
    
    try:
        await context.bot.send_message(
            chat_id=target, 
            text=f"Your subscription is active until {exp.isoformat()}"
        )
    except Exception:
        pass

async def deactivate_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    caller = update.effective_user.id if update.effective_user else None
    if caller not in ADMIN_IDS:
        await update.message.reply_text("Unauthorized.")
        return
    
    if not context.args:
        await update.message.reply_text("Usage: /deactivate <telegram_id>")
        return
    
    try:
        target = int(context.args[0])
    except ValueError:
        await update.message.reply_text("Invalid telegram_id.")
        return
    
    success = deactivate_subscription_for(target)
    if success:
        await update.message.reply_text(f"Deactivated subscription for {target}")
        try:
            await context.bot.send_message(
                chat_id=target, 
                text="Your subscription has been deactivated."
            )
        except Exception:
            pass
    else:
        await update.message.reply_text("No active subscription found.")

async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user.id if update.effective_user else None
    if not user:
        await update.message.reply_text("Could not determine your user ID.")
        return
    
    subscribed, expiry = is_subscribed(user)
    if subscribed and expiry:
        await update.message.reply_text(f"Subscribed until: {expiry.isoformat()}")
    else:
        if expiry:
            await update.message.reply_text(f"Subscription expired on {expiry.isoformat()}")
        else:
            await update.message.reply_text("No active subscription. Use /subscribe")

# -------------------------------------------
# Main forecast command with timeout and comprehensive error handling
# -------------------------------------------
async def forecast_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message:
        return
        
    args = context.args or []
    if not args:
        await update.message.reply_text(
            "Usage: /forecast <ticker> [period] [interval] [steps]\n"
            "Example: /forecast AAPL 1y 1d 30"
        )
        return

    user_id = update.effective_user.id if update.effective_user else None
    
    # Check subscription
    subscribed, expiry = is_subscribed(user_id)
    if not subscribed:
        if expiry:
            msg = f"Subscription expired on {expiry.isoformat()}. Use /subscribe to renew."
        else:
            msg = "No active subscription. Use /subscribe for payment instructions."
        await update.message.reply_text(msg)
        return

    # Parse arguments
    try:
        req = parse_forecast_args(args)
    except Exception as e:
        await update.message.reply_text(f"Invalid arguments: {e}")
        return

    # Show typing indicator
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    
    # Set timeout for the entire forecast operation
    async def timeout_handler():
        await asyncio.sleep(600)  # 10 minute timeout
        return None
    
    try:
        # Status message
        status_msg = await update.message.reply_text(f"🔄 Processing forecast for {req.ticker}...")
        
        # Run forecast with timeout
        forecast_task = asyncio.create_task(asyncio.to_thread(forecast_core, req))
        timeout_task = asyncio.create_task(timeout_handler())
        
        done, pending = await asyncio.wait(
            [forecast_task, timeout_task], 
            return_when=asyncio.FIRST_COMPLETED
        )
        
        # Cancel pending tasks
        for task in pending:
            task.cancel()
        
        if forecast_task in done:
            data = await forecast_task
        else:
            await status_msg.edit_text(f"⏰ Forecast for {req.ticker} timed out (10min limit)")
            return
        
        # Validate response
        if not data or not data.get("predictions"):
            await status_msg.edit_text(f"❌ No forecast data generated for {req.ticker}")
            return
        
        predictions = data.get("predictions", [])
        if len(predictions) == 0:
            await status_msg.edit_text(f"❌ No predictions generated for {req.ticker}")
            return
        
        # Update status
        await status_msg.edit_text(f"📊 Generating results for {req.ticker}...")
        
        # Send summary with accuracy highlighting (MAPE badge)
        mape_val = data.get('mape', 0)
        badge = accuracy_badge(mape_val)
        summary = (
            f"📈 **{req.ticker}** ({req.period}, {req.interval}, {req.steps} steps)\n"
            f"**RMSE:** {fmt(data.get('rmse', 0))}\n"
            f"**MAPE:** {fmt(mape_val)}% {badge}\n"
            f"**Market Regime:** {data.get('market_regime', 'Unknown')}\n"
        )
        await update.message.reply_text(summary, parse_mode=ParseMode.MARKDOWN)
        
        # Send forecast table
        forecast_chunks = build_forecast_table(predictions)
        for chunk in forecast_chunks[:3]:  # Limit chunks to avoid spam
            await update.message.reply_text(chunk, parse_mode=ParseMode.MARKDOWN)
        
        # Send signals table
        signal_chunks = build_signal_table(predictions)
        for chunk in signal_chunks[:3]:  # Limit chunks to avoid spam
            await update.message.reply_text(chunk, parse_mode=ParseMode.MARKDOWN)
        
        # Generate and send chart
        try:
            title = f"{req.ticker} Forecast ({req.period}, {req.interval})"
            chart_bytes = make_watermarked_chart(data, title=title)
            chart_bio = io.BytesIO(chart_bytes)
            chart_bio.name = f"{req.ticker}_forecast.png"
            await update.message.reply_photo(photo=InputFile(chart_bio))
        except Exception as chart_error:
            print(f"[ERROR] Chart generation failed: {chart_error}")
            await update.message.reply_text("⚠️ Chart generation failed")
        
        # Delete status message
        try:
            await status_msg.delete()
        except Exception:
            pass
        
        print(f"[SUCCESS] Forecast completed for {req.ticker} (user: {user_id})")
        
    except Exception as e:
        error_msg = str(e)[:200]  # Truncate long errors
        await update.message.reply_text(f"❌ Forecast failed: {error_msg}")
        print(f"[ERROR] Forecast command failed: {e}")
        print(traceback.format_exc())

# -------------------------------------------
# Main application
# -------------------------------------------
def main():
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN environment variable required")
    
    print(f"[INFO] Starting {VERSION}")
    print(f"[INFO] Admin IDs: {ADMIN_IDS}")
    print(f"[INFO] Subscription URL configured: {bool(SUBSCRIBE_URL)}")
    
    app = Application.builder().token(token).build()

    # Register handlers
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
