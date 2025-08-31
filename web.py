# web.py - Fixed version with better error handling
import os
import threading
import asyncio
import traceback
from flask import Flask
import main  # import your fixed main.py

app = Flask(__name__)

# Health check route for Render
@app.route("/")
def home():
    return "Stock Sight AI Telegram Bot is running successfully!"

@app.route("/health")
def health():
    return {"status": "healthy", "service": "stock-sight-telegram-bot"}

# Status route to check bot status
@app.route("/status")
def status():
    return {
        "version": main.VERSION,
        "telegram_token_configured": bool(os.getenv("TELEGRAM_BOT_TOKEN")),
        "admin_ids_configured": bool(main.ADMIN_IDS),
        "subscription_url_configured": bool(main.SUBSCRIBE_URL)
    }

def run_bot():
    """Run the Telegram bot in a separate thread with error handling"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        print("[INFO] Starting Telegram bot thread...")
        main.main()
    except KeyboardInterrupt:
        print("[INFO] Telegram bot stopped by user")
    except Exception as e:
        print(f"[ERROR] Telegram bot crashed: {e}")
        print(traceback.format_exc())
        # In production, you might want to restart the bot here

def start_bot_thread():
    """Start the bot in a daemon thread"""
    bot_thread = threading.Thread(target=run_bot, daemon=True)
    bot_thread.start()
    return bot_thread

if __name__ == "__main__":
    # Validate required environment variables
    required_env = ["TELEGRAM_BOT_TOKEN"]
    missing_env = [var for var in required_env if not os.getenv(var)]
    
    if missing_env:
        print(f"[ERROR] Missing required environment variables: {missing_env}")
        exit(1)
    
    # Start Telegram bot in background
    print("[INFO] Starting services...")
    bot_thread = start_bot_thread()
    
    # Give the bot thread a moment to start
    import time
    time.sleep(2)
    
    # Start Flask web service (this keeps Render happy)
    port = int(os.environ.get("PORT", 8080))
    host = "0.0.0.0"
    
    print(f"[INFO] Starting Flask web server on {host}:{port}")
    
    try:
        app.run(
            host=host, 
            port=port, 
            debug=False,  # Disable debug in production
            threaded=True,
            use_reloader=False  # Disable reloader to prevent duplicate bot instances
        )
    except Exception as e:
        print(f"[ERROR] Flask server failed: {e}")
        exit(1)
