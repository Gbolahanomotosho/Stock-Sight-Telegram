# web.py - DEBUG VERSION with enhanced logging
import os
import threading
import time
import sys
import traceback
from flask import Flask, jsonify

app = Flask(__name__)

# Enhanced bot status tracking
bot_thread = None
bot_status = {
    "alive": False, 
    "start_time": None, 
    "error": None,
    "last_update": None,
    "commands_received": 0,
    "startup_logs": []
}

def log_startup(message):
    """Log startup messages for debugging"""
    print(message)
    bot_status["startup_logs"].append(f"{time.strftime('%H:%M:%S')}: {message}")
    if len(bot_status["startup_logs"]) > 20:  # Keep only last 20 logs
        bot_status["startup_logs"] = bot_status["startup_logs"][-20:]

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
        "last_update": bot_status.get("last_update"),
        "commands_received": bot_status.get("commands_received", 0),
        "startup_logs": bot_status.get("startup_logs", []),
        "version": getattr(main, 'VERSION', 'unknown'),
        "telegram_token_configured": bool(os.getenv("TELEGRAM_BOT_TOKEN")),
        "admin_ids_configured": bool(getattr(main, 'ADMIN_IDS', [])),
        "subscription_url_configured": bool(getattr(main, 'SUBSCRIBE_URL', ''))
    })

@app.route("/debug")
def debug_info():
    """Debug endpoint with detailed information"""
    try:
        import main
        token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        
        return jsonify({
            "bot_thread_alive": bot_thread.is_alive() if bot_thread else False,
            "bot_status": bot_status,
            "environment": {
                "token_length": len(token) if token else 0,
                "token_format_ok": token.count(':') == 1 if token else False,
                "admin_ids": getattr(main, 'ADMIN_IDS', []),
                "subscribe_url": bool(getattr(main, 'SUBSCRIBE_URL', ''))
            },
            "main_module_attributes": [attr for attr in dir(main) if not attr.startswith('_')]
        })
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()})

def run_bot_no_signals():
    """Run bot with signal handlers disabled and enhanced debugging"""
    try:
        log_startup("Starting bot thread...")
        bot_status["start_time"] = time.time()
        bot_status["alive"] = True
        bot_status["error"] = None
        bot_status["last_update"] = time.strftime('%H:%M:%S')
        
        # Import main module
        log_startup("Importing main module...")
        import main
        import asyncio
        
        # Create new event loop for this thread
        log_startup("Setting up asyncio event loop...")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Get and validate token
        token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        if not token:
            raise RuntimeError("TELEGRAM_BOT_TOKEN environment variable required")
        
        log_startup(f"Token configured: {len(token)} characters")
        log_startup(f"Token format check: {token.count(':') == 1}")
        
        # Import telegram components
        log_startup("Importing telegram components...")
        from telegram.ext import Application, CommandHandler
        from telegram import Update
        
        # Test token by creating application
        log_startup("Creating Telegram application...")
        application = Application.builder().token(token).build()
        
        # Log main module info
        log_startup(f"Main VERSION: {getattr(main, 'VERSION', 'NOT_FOUND')}")
        log_startup(f"Main ADMIN_IDS: {getattr(main, 'ADMIN_IDS', 'NOT_FOUND')}")
        log_startup(f"Main SUBSCRIBE_URL configured: {bool(getattr(main, 'SUBSCRIBE_URL', ''))}")
        
        # Test if command handlers exist
        handler_tests = ['start', 'forecast_cmd', 'subscribe_cmd', 'paid_cmd', 'activate_cmd', 'deactivate_cmd', 'status_cmd']
        for handler_name in handler_tests:
            if hasattr(main, handler_name):
                log_startup(f"Handler {handler_name}: EXISTS")
            else:
                log_startup(f"Handler {handler_name}: MISSING")
        
        # Wrap command handlers to track usage
        def wrap_handler(original_handler, name):
            async def wrapped_handler(update: Update, context):
                try:
                    bot_status["commands_received"] += 1
                    bot_status["last_update"] = time.strftime('%H:%M:%S')
                    log_startup(f"Received command: {name}")
                    return await original_handler(update, context)
                except Exception as e:
                    log_startup(f"Handler {name} failed: {e}")
                    raise
            return wrapped_handler
        
        # Register handlers with error tracking
        log_startup("Registering command handlers...")
        try:
            application.add_handler(CommandHandler("start", wrap_handler(main.start, "start")))
            log_startup("Registered: start")
        except Exception as e:
            log_startup(f"Failed to register start: {e}")
        
        try:
            application.add_handler(CommandHandler("forecast", wrap_handler(main.forecast_cmd, "forecast")))
            log_startup("Registered: forecast")
        except Exception as e:
            log_startup(f"Failed to register forecast: {e}")
            
        try:
            application.add_handler(CommandHandler("subscribe", wrap_handler(main.subscribe_cmd, "subscribe")))
            log_startup("Registered: subscribe")
        except Exception as e:
            log_startup(f"Failed to register subscribe: {e}")
            
        try:
            application.add_handler(CommandHandler("paid", wrap_handler(main.paid_cmd, "paid")))
            log_startup("Registered: paid")
        except Exception as e:
            log_startup(f"Failed to register paid: {e}")
            
        try:
            application.add_handler(CommandHandler("activate", wrap_handler(main.activate_cmd, "activate")))
            log_startup("Registered: activate")
        except Exception as e:
            log_startup(f"Failed to register activate: {e}")
            
        try:
            application.add_handler(CommandHandler("deactivate", wrap_handler(main.deactivate_cmd, "deactivate")))
            log_startup("Registered: deactivate")
        except Exception as e:
            log_startup(f"Failed to register deactivate: {e}")
            
        try:
            application.add_handler(CommandHandler("status", wrap_handler(main.status_cmd, "status")))
            log_startup("Registered: status")
        except Exception as e:
            log_startup(f"Failed to register status: {e}")

        log_startup("Starting polling...")
        
        # Start polling with signal handlers disabled
        application.run_polling(
            close_loop=False,
            stop_signals=None,  # This disables signal handlers
            drop_pending_updates=True
        )
        
    except Exception as e:
        error_msg = f"Bot crashed: {e}"
        log_startup(error_msg)
        log_startup(f"Traceback: {traceback.format_exc()}")
        bot_status["alive"] = False
        bot_status["error"] = error_msg
        raise

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
    time.sleep(5)  # Increased startup time
    
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
