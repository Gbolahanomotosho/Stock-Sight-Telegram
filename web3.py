# web.py - Fixed version with auto-restart and better monitoring
import os
import threading
import asyncio
import traceback
import time
import signal
import sys
from flask import Flask, jsonify
import main  # import your fixed main.py

app = Flask(__name__)

# Global variables for bot monitoring
bot_thread = None
bot_start_time = None
bot_restart_count = 0
MAX_RESTARTS = 10  # Maximum restarts before giving up

# Health check route for Render
@app.route("/")
def home():
    return "Stock Sight AI Telegram Bot is running successfully!"

@app.route("/health")
def health():
    global bot_thread, bot_start_time, bot_restart_count
    
    bot_alive = bot_thread is not None and bot_thread.is_alive()
    uptime = int(time.time() - bot_start_time) if bot_start_time else 0
    
    return jsonify({
        "status": "healthy" if bot_alive else "degraded",
        "service": "stock-sight-telegram-bot",
        "bot_alive": bot_alive,
        "bot_uptime_seconds": uptime,
        "bot_restart_count": bot_restart_count,
        "version": getattr(main, 'VERSION', 'unknown'),
        "telegram_token_configured": bool(os.getenv("TELEGRAM_BOT_TOKEN")),
        "admin_ids_configured": bool(main.ADMIN_IDS),
        "subscription_url_configured": bool(main.SUBSCRIBE_URL)
    })

@app.route("/status")
def status():
    return health()  # Redirect to health endpoint

@app.route("/restart_bot")
def restart_bot_endpoint():
    """Emergency endpoint to manually restart the bot"""
    if restart_bot_thread():
        return jsonify({"status": "success", "message": "Bot restart initiated"})
    else:
        return jsonify({"status": "error", "message": "Bot restart failed"}), 500

def run_bot():
    """Run the Telegram bot with improved error handling and logging"""
    global bot_start_time, bot_restart_count
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        bot_start_time = time.time()
        print(f"[INFO] Starting Telegram bot thread... (Restart #{bot_restart_count})")
        
        # Validate token before starting
        token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        if not token:
            raise RuntimeError("TELEGRAM_BOT_TOKEN environment variable is missing or empty")
        
        # Check token format
        if not token.count(':') == 1 or len(token.split(':')[0]) < 8:
            raise RuntimeError("TELEGRAM_BOT_TOKEN appears to be invalid format")
        
        # Start the main bot
        main.main()
        
    except KeyboardInterrupt:
        print("[INFO] Telegram bot stopped by user")
    except Exception as e:
        print(f"[ERROR] Telegram bot crashed: {e}")
        print(f"[ERROR] Full traceback:\n{traceback.format_exc()}")
        
        # Log specific error types
        error_str = str(e).lower()
        if "token" in error_str or "unauthorized" in error_str:
            print("[ERROR] Bot token issue detected - check TELEGRAM_BOT_TOKEN")
        elif "network" in error_str or "timeout" in error_str:
            print("[ERROR] Network connectivity issue detected")
        elif "rate limit" in error_str:
            print("[ERROR] Rate limiting detected - will retry after delay")
            time.sleep(60)  # Wait 1 minute for rate limits
        
        # Don't restart immediately on certain errors
        if "token" in error_str or bot_restart_count >= MAX_RESTARTS:
            print(f"[ERROR] Critical error or max restarts reached ({MAX_RESTARTS}), not restarting")
            return
    finally:
        try:
            loop.close()
        except:
            pass

def start_bot_thread():
    """Start the bot in a monitored thread"""
    global bot_thread
    
    try:
        bot_thread = threading.Thread(target=run_bot, daemon=False)  # Changed from daemon=True
        bot_thread.start()
        print(f"[INFO] Bot thread started with ID: {bot_thread.ident}")
        return bot_thread
    except Exception as e:
        print(f"[ERROR] Failed to start bot thread: {e}")
        return None

def restart_bot_thread():
    """Restart the bot thread"""
    global bot_thread, bot_restart_count
    
    try:
        # Stop existing thread if alive
        if bot_thread and bot_thread.is_alive():
            print("[INFO] Stopping existing bot thread...")
            # Note: There's no clean way to stop a thread in Python
            # The bot should handle shutdown gracefully via signals
        
        bot_restart_count += 1
        if bot_restart_count > MAX_RESTARTS:
            print(f"[ERROR] Maximum restarts ({MAX_RESTARTS}) exceeded")
            return False
        
        print(f"[INFO] Restarting bot thread (restart #{bot_restart_count})...")
        time.sleep(5)  # Wait a bit before restart
        
        bot_thread = start_bot_thread()
        return bot_thread is not None
        
    except Exception as e:
        print(f"[ERROR] Failed to restart bot thread: {e}")
        return False

def monitor_bot():
    """Monitor bot health and restart if needed"""
    global bot_thread
    
    while True:
        try:
            time.sleep(30)  # Check every 30 seconds
            
            if bot_thread is None or not bot_thread.is_alive():
                print("[WARN] Bot thread is dead, attempting restart...")
                if not restart_bot_thread():
                    print("[ERROR] Bot restart failed, will retry in 60 seconds")
                    time.sleep(60)
            
        except Exception as e:
            print(f"[ERROR] Bot monitor error: {e}")
            time.sleep(60)

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    print(f"\n[INFO] Received signal {signum}, shutting down gracefully...")
    
    # Try to stop the bot gracefully
    if bot_thread and bot_thread.is_alive():
        print("[INFO] Waiting for bot thread to stop...")
        bot_thread.join(timeout=10)
    
    print("[INFO] Shutdown complete")
    sys.exit(0)

if __name__ == "__main__":
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Validate required environment variables
    required_env = ["TELEGRAM_BOT_TOKEN"]
    missing_env = [var for var in required_env if not os.getenv(var)]
    
    if missing_env:
        print(f"[ERROR] Missing required environment variables: {missing_env}")
        sys.exit(1)
    
    # Start services
    print("[INFO] Starting Stock Sight Telegram Bot Service...")
    
    # Start bot thread
    if not start_bot_thread():
        print("[ERROR] Failed to start bot thread")
        sys.exit(1)
    
    # Start bot monitor in background
    monitor_thread = threading.Thread(target=monitor_bot, daemon=True)
    monitor_thread.start()
    
    # Give services time to initialize
    time.sleep(3)
    
    # Start Flask web service (this keeps Render happy)
    port = int(os.environ.get("PORT", 8080))
    host = "0.0.0.0"
    
    print(f"[INFO] Starting Flask web server on {host}:{port}")
    print(f"[INFO] Bot monitoring active with auto-restart (max: {MAX_RESTARTS} restarts)")
    
    try:
        app.run(
            host=host, 
            port=port, 
            debug=False,  # Disable debug in production
            threaded=True,
            use_reloader=False  # Disable reloader to prevent duplicate instances
        )
    except Exception as e:
        print(f"[ERROR] Flask server failed: {e}")
        sys.exit(1)
