# web.py - Telegram bot + Flask with auto-restart safety
import os
import time
import threading
import traceback
from flask import Flask
import main  # your main.py

app = Flask(__name__)

@app.route("/")
def home():
    return "Stock Sight AI Telegram Bot is running successfully!"

@app.route("/health")
def health():
    return {"status": "healthy"}

@app.route("/status")
def status():
    return {
        "version": main.VERSION,
        "telegram_token_configured": bool(os.getenv("TELEGRAM_BOT_TOKEN")),
    }

def run_flask():
    port = int(os.environ.get("PORT", 8080))
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        threaded=True,
        use_reloader=False
    )

def run_bot_with_restart():
    """Run Telegram bot with auto-restart on crash"""
    while True:
        try:
            print("[INFO] Starting Telegram bot...")
            main.main()   # this blocks until bot stops/crashes
        except Exception as e:
            print(f"[ERROR] Bot crashed: {e}")
            print(traceback.format_exc())
        # wait a bit before restart to avoid crash loop
        print("[INFO] Restarting bot in 5 seconds...")
        time.sleep(5)

if __name__ == "__main__":
    # Start Flask in background thread
    threading.Thread(target=run_flask, daemon=True).start()

    # Keep bot in main thread with restart safety
    run_bot_with_restart()
