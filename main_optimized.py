# main_optimized.py - LIGHTWEIGHT version for better performance
import os
import io
import asyncio
import json
import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests
from typing import Dict, Any, Tuple, Optional

# Minimal imports - only Prophet for speed
from src.data_loader import download_ticker, validate_data, normalize_ticker
from src.prophet_model import train_prophet, prophet_predict

from telegram import Update, InputFile
from telegram.constants import ParseMode, ChatAction
from telegram.ext import Application, CommandHandler, ContextTypes

VERSION = "Stock Sight Telegram Service v2.0 (OPTIMIZED)"

# Environment variables
SUBSCRIBE_URL = os.getenv("SUBSCRIBE_URL", "").strip()
ADMIN_IDS = [int(x) for x in os.getenv("ADMIN_IDS", "").split(",") if x.strip().isdigit()]
BACKUP_URL = os.getenv("BACKUP_URL", "").strip()  # For external subscription storage

# PERFORMANCE OPTIMIZATIONS
MAX_STEPS = 7  # Reduced from 30
MAX_DATA_ROWS = 500  # Reduced from 5000
TIMEOUT_SECONDS = 90  # Reduced from 600
operation_count = 0  # For memory cleanup

# In-memory subscription storage with backup
subscription_cache = {}

def backup_subscriptions():
    """Backup subscriptions to external storage"""
    try:
        if BACKUP_URL:
            requests.post(f"{BACKUP_URL}/backup_subscriptions", 
                         json=subscription_cache, timeout=10)
        print(f"[INFO] Subscriptions backed up: {len(subscription_cache)} users")
    except Exception as e:
        print(f"[WARN] Subscription backup failed: {e}")

def restore_subscriptions():
    """Restore subscriptions from external storage"""
    global subscription_cache
    try:
        if BACKUP_URL:
            response = requests.get(f"{BACKUP_URL}/restore_subscriptions", timeout=10)
            if response.status_code == 200:
                data = response.json().get('data', {})
                subscription_cache.update(data)
                print(f"[INFO] Subscriptions restored: {len(subscription_cache)} users")
        
        # Also try to load from local file if it exists
        if os.path.exists("subscriptions.json"):
            with open("subscriptions.json", "r") as f:
                local_data = json.load(f)
                subscription_cache.update(local_data)
                print(f"[INFO] Local subscriptions loaded: {len(local_data)} users")
    except Exception as e:
        print(f"[WARN] Subscription restore failed: {e}")

def save_subscription(user_id: int, expires: str):
    """Save single subscription with backup"""
    subscription_cache[str(user_id)] = {"expires": expires}
    
    # Backup every 5 operations or immediately for new subscriptions
    global operation_count
    operation_count += 1
    if operation_count % 5 == 0:
        backup_subscriptions()

def is_subscribed(user_id: int) -> Tuple[bool, Any]:
    """Check subscription with memory cleanup"""
    rec = subscription_cache.get(str(user_id))
    if not rec:
        return False, None
    
    try:
        exp_dt = datetime.datetime.fromisoformat(rec.get("expires"))
        now = datetime.datetime.utcnow()
        return now < exp_dt, exp_dt
    except Exception:
        return False, None

def activate_subscription_for(user_id: int, days: int = 30) -> datetime.datetime:
    """Activate subscription with backup"""
    new_exp = datetime.datetime.utcnow() + datetime.timedelta(days=days)
    save_subscription(user_id, new_exp.isoformat())
    return new_exp

def simple_forecast(ticker: str, steps: int = 5) -> Dict[str, Any]:
    """Ultra-lightweight forecasting - Prophet only"""
    try:
        print(f"[INFO] Lightweight forecast: {ticker}, {steps} steps")
        
        # Force memory cleanup
        global operation_count
        operation_count += 1
        if operation_count % 3 == 0:
            import gc
            gc.collect()
        
        # Download minimal data
        df = download_ticker(ticker, period="3mo", interval="1d")  # Much smaller
        validate_data(df, ticker)
        
        # Limit data size aggressively
        if len(df) > MAX_DATA_ROWS:
            df = df.tail(MAX_DATA_ROWS).reset_index(drop=True)
        
        prophet_df = df[["ds", "y"]].tail(100)  # Only last 100 points
        
        # Ultra-fast Prophet training
        try:
            m = train_prophet(
                prophet_df, 
                daily_seasonality=False,
                weekly_seasonality=False, 
                yearly_seasonality=False
            )
            
            if m is None:
                raise ValueError("Prophet training failed")
            
            forecast = prophet_predict(m, periods=steps)
            predictions = forecast["yhat"].tail(steps).values
            
        except Exception:
            # Fallback: simple moving average
            recent = df["y"].tail(20)
            avg_change = recent.pct_change().mean()
            last_price = df["y"].iloc[-1]
            predictions = [last_price * (1 + avg_change) ** i for i in range(1, steps + 1)]
        
        # Simple signals
        current_price = float(df["y"].iloc[-1])
        signals = []
        
        for i, pred in enumerate(predictions):
            pct_change = ((pred - current_price) / current_price) * 100
            signal = "BUY" if pct_change > 2 else ("SELL" if pct_change < -2 else "HOLD")
            
            signals.append({
                "step": i + 1,
                "price": round(float(pred), 4),
                "change_pct": round(pct_change, 2),
                "signal": signal
            })
        
        # Force cleanup
        del df, prophet_df
        import gc
        gc.collect()
        
        return {
            "ticker": ticker,
            "current_price": current_price,
            "predictions": predictions,
            "signals": signals,
            "method": "prophet_optimized"
        }
        
    except Exception as e:
        print(f"[ERROR] Lightweight forecast failed: {e}")
        return {"ticker": ticker, "error": str(e)}

def create_simple_chart(data: Dict[str, Any]) -> bytes:
    """Create minimal chart quickly"""
    try:
        if "error" in data:
            raise ValueError("No data for chart")
        
        predictions = data.get("predictions", [])
        current_price = data.get("current_price", 0)
        
        if not predictions:
            raise ValueError("No predictions")
        
        # Simple chart
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Plot current price and predictions
        x_vals = list(range(len(predictions) + 1))
        prices = [current_price] + list(predictions)
        
        ax.plot(x_vals, prices, marker='o', linewidth=2, markersize=4)
        ax.set_title(f"{data['ticker']} - Quick Forecast")
        ax.set_xlabel("Days")
        ax.set_ylabel("Price")
        ax.grid(True, alpha=0.3)
        
        # Add simple watermark
        fig.text(0.99, 0.01, "Stock Sight AI", ha="right", va="bottom", 
                fontsize=8, alpha=0.5)
        
        buf = io.BytesIO()
        plt.tight_layout()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        
        return buf.read()
        
    except Exception as e:
        print(f"[ERROR] Chart creation failed: {e}")
        # Return minimal error chart
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, f"Chart Error\n{data.get('ticker', 'Unknown')}", 
                ha='center', va='center')
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=80)
        plt.close(fig)
        buf.seek(0)
        return buf.read()

# Telegram Commands
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id if update.effective_user else 0
    print(f"[INFO] User {user_id} started optimized bot")
    
    msg = (
        f"{VERSION}\n\n"
        "⚡ OPTIMIZED Telegram Trading Service\n\n"
        "Commands:\n"
        "/forecast TICKER [STEPS]\n"
        "Example: /forecast AAPL 5\n"
        "Max steps: 7 (optimized for speed)\n\n"
        "/subscribe - Get subscription\n"
        "/status - Check status\n\n"
        "🚀 Features:\n"
        "• Ultra-fast Prophet forecasting\n"
        "• Persistent subscriptions\n"
        "• Optimized for reliability"
    )
    await update.message.reply_text(msg)

async def forecast_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message:
        return
    
    args = context.args or []
    if not args:
        await update.message.reply_text(
            "Usage: /forecast <ticker> [steps]\n"
            "Example: /forecast AAPL 5\n"
            "Max steps: 7"
        )
        return
    
    user_id = update.effective_user.id if update.effective_user else None
    
    # Check subscription
    subscribed, expiry = is_subscribed(user_id)
    if not subscribed:
        msg = "⚠️ No active subscription. Use /subscribe for access."
        await update.message.reply_text(msg)
        return
    
    # Parse arguments
    ticker = normalize_ticker(args[0].upper())
    try:
        steps = min(int(args[1]) if len(args) > 1 else 5, MAX_STEPS)
    except (ValueError, IndexError):
        steps = 5
    
    # Show typing
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    
    # Quick timeout
    try:
        status_msg = await update.message.reply_text(f"⚡ Quick forecast for {ticker}...")
        
        # Run forecast with short timeout
        forecast_task = asyncio.create_task(asyncio.to_thread(simple_forecast, ticker, steps))
        
        try:
            data = await asyncio.wait_for(forecast_task, timeout=TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            await status_msg.edit_text(f"⏰ Forecast timed out - try fewer steps")
            return
        
        if "error" in data:
            await status_msg.edit_text(f"❌ Error: {data['error']}")
            return
        
        # Send results
        signals = data.get("signals", [])
        current = data.get("current_price", 0)
        
        # Summary
        summary = f"📊 **{ticker}** Quick Forecast\n"
        summary += f"Current: ${current:.2f}\n\n"
        
        for sig in signals[:5]:  # Limit output
            arrow = "🟢" if sig["signal"] == "BUY" else ("🔴" if sig["signal"] == "SELL" else "🔵")
            summary += f"Day {sig['step']}: ${sig['price']} ({sig['change_pct']:+.1f}%) {arrow}\n"
        
        await update.message.reply_text(summary, parse_mode=ParseMode.MARKDOWN)
        
        # Create and send chart
        try:
            chart_bytes = create_simple_chart(data)
            chart_bio = io.BytesIO(chart_bytes)
            chart_bio.name = f"{ticker}_forecast.png"
            await update.message.reply_photo(photo=InputFile(chart_bio))
        except Exception:
            await update.message.reply_text("⚠️ Chart generation skipped")
        
        # Clean up status message
        try:
            await status_msg.delete()
        except:
            pass
        
        print(f"[SUCCESS] Fast forecast completed: {ticker} ({user_id})")
        
    except Exception as e:
        await update.message.reply_text(f"❌ Forecast failed: {str(e)[:100]}")
        print(f"[ERROR] Forecast command failed: {e}")

async def subscribe_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if SUBSCRIBE_URL:
        text = f"💳 Subscribe for access:\n{SUBSCRIBE_URL}\n\nAfter payment: /paid <transaction_id>"
    else:
        text = "💳 Subscription system not configured. Contact admin."
    await update.message.reply_text(text)

async def paid_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id if update.effective_user else None
    details = " ".join(context.args) if context.args else "(no details)"
    
    msg = f"💳 Payment notification from user {uid}:\n{details}\n\nUse /activate <id> <days>"
    
    for admin in ADMIN_IDS:
        try:
            await context.bot.send_message(chat_id=admin, text=msg)
        except:
            pass
    
    await update.message.reply_text("✅ Payment notification sent to admins.")

async def activate_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    caller = update.effective_user.id if update.effective_user else None
    if caller not in ADMIN_IDS:
        await update.message.reply_text("❌ Unauthorized.")
        return
    
    if not context.args:
        await update.message.reply_text("Usage: /activate <telegram_id> [days]")
        return
    
    try:
        target = int(context.args[0])
        days = int(context.args[1]) if len(context.args) >= 2 else 30
        exp = activate_subscription_for(target, days=days)
        
        await update.message.reply_text(f"✅ Activated user {target} until {exp.strftime('%Y-%m-%d')}")
        
        # Notify user
        try:
            await context.bot.send_message(
                chat_id=target, 
                text=f"🎉 Subscription activated until {exp.strftime('%Y-%m-%d')}"
            )
        except:
            pass
            
    except ValueError:
        await update.message.reply_text("❌ Invalid arguments.")

async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user.id if update.effective_user else None
    subscribed, expiry = is_subscribed(user)
    
    if subscribed and expiry:
        await update.message.reply_text(f"✅ Subscribed until: {expiry.strftime('%Y-%m-%d')}")
    else:
        await update.message.reply_text("❌ No active subscription. Use /subscribe")

def main_lightweight():
    """Optimized main function"""
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN required")
    
    print(f"[INFO] Starting {VERSION}")
    print(f"[INFO] Max steps: {MAX_STEPS}, Max data rows: {MAX_DATA_ROWS}")
    
    # Restore subscriptions on startup
    restore_subscriptions()
    
    app = Application.builder().token(token).build()
    
    # Register handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("forecast", forecast_cmd))
    app.add_handler(CommandHandler("subscribe", subscribe_cmd))
    app.add_handler(CommandHandler("paid", paid_cmd))
    app.add_handler(CommandHandler("activate", activate_cmd))
    app.add_handler(CommandHandler("status", status_cmd))
    
    print("[INFO] Lightweight bot starting...")
    
    # Optimized polling settings
    app.run_polling(
        close_loop=False,
        stop_signals=None,
        drop_pending_updates=True,
        allowed_updates=['message']  # Only process messages
    )

if __name__ == "__main__":
    main_lightweight()
