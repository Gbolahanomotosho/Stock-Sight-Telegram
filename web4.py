# web.py - SIMPLE FIX: Just disable signal handlers in the bot
import os
import threading
import time
import sys
from flask import Flask, jsonify

app = Flask(__name__)

# Simple bot status tracking
bot_thread = None
bot_status = {"alive": False, "start_time": None, "error": None}

@app.route("/")
def home():
    return "Stock Sight AI Telegram Bot is running successfully!"

@app.route("/health")
def health():
    import main
    uptime = int(time.time() - bot_status["start_time"]) if bot_status["start_time"] else 0
    
    return jsonify({
        "status": "healthy" if bot_status["alive"] else "degraded",
        "service": "stock-sight-telegram-bot",
        "bot_alive": bot_status["alive"],
        "bot_uptime_seconds": uptime,
        "last_error": bot_status.get("error"),
        "version": getattr(main, 'VERSION', 'unknown'),
        "telegram_token_configured": bool(os.getenv("TELEGRAM_BOT_TOKEN")),
        "admin_ids_configured": bool(getattr(main, 'ADMIN_IDS', [])),
        "subscription_url_configured": bool(getattr(main, 'SUBSCRIBE_URL', ''))
    })

def run_bot_no_signals():
    """Run bot with signal handlers disabled"""
    import main
    import asyncio
    
    try:
        bot_status["start_time"] = time.time()
        bot_status["alive"] = True
        bot_status["error"] = None
        
        print("[INFO] Starting Telegram bot with disabled signal handlers...")
        
        # Create new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Get the token
        token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        if not token:
            raise RuntimeError("TELEGRAM_BOT_TOKEN environment variable required")
        
        # Import telegram components
        from telegram.ext import Application, CommandHandler
        
        # Create application
        print(f"[INFO] Starting {main.VERSION}")
        print(f"[INFO] Admin IDs: {main.ADMIN_IDS}")
        print(f"[INFO] Subscription URL configured: {bool(main.SUBSCRIBE_URL)}")
        
        application = Application.builder().token(token).build()

        # Register handlers (same as main.py)
        application.add_handler(CommandHandler("start", main.start))
        application.add_handler(CommandHandler("forecast", main.forecast_cmd))
        application.add_handler(CommandHandler("subscribe", main.subscribe_cmd))
        application.add_handler(CommandHandler("paid", main.paid_cmd))
        application.add_handler(CommandHandler("activate", main.activate_cmd))
        application.add_handler(CommandHandler("deactivate", main.deactivate_cmd))
        application.add_handler(CommandHandler("status", main.status_cmd))

        print("[INFO] Bot starting with polling...")
        
        # THIS IS THE KEY FIX: disable signal handlers
        application.run_polling(
            close_loop=False,
            stop_signals=None,  # This disables signal handlers!
            drop_pending_updates=True
        )
        
    except Exception as e:
        print(f"[ERROR] Bot crashed: {e}")
        bot_status["alive"] = False
        bot_status["error"] = str(e)
        import traceback
        traceback.print_exc()

def start_bot():
    """Start bot in background thread"""
    global bot_thread
    
    bot_thread = threading.Thread(target=run_bot_no_signals, daemon=True)
    bot_thread.start()
    print(f"[INFO] Bot thread started")
    return bot_thread

if __name__ == "__main__":
    # Validate environment
    required_env = ["TELEGRAM_BOT_TOKEN"]
    missing_env = [var for var in required_env if not os.getenv(var)]
    
    if missing_env:
        print(f"[ERROR] Missing required environment variables: {missing_env}")
        sys.exit(1)
    
    print("[INFO] Starting Stock Sight Telegram Bot Service...")
    
    # Start bot
    start_bot()
    
    # Give bot time to start
    time.sleep(3)
    
    # Start Flask
    port = int(os.environ.get("PORT", 8080))
    host = "0.0.0.0"
    
    print(f"[INFO] Starting Flask web server on {host}:{port}")
    
    try:
        app.run(
            host=host, 
            port=port, 
            debug=False,
            threaded=True,
            use_reloader=False
        )
    except Exception as e:
        print(f"[ERROR] Flask server failed: {e}")
        sys.exit(1)
