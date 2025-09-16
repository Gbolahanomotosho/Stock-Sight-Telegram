# web.py - stable for Render: Flask main thread, bot in background thread
import os
import time
import threading
import traceback
from flask import Flask, jsonify

app = Flask(__name__)

@app.route("/")
def home():
    return "Stock Sight AI Telegram Bot is running successfully!"

@app.route("/health")
def health():
    return {"status": "healthy", "service": "stock-sight-telegram-bot"}

@app.route("/status")
def status():
    import main
    return {
        "version": getattr(main, "VERSION", "unknown"),
        "telegram_token_configured": bool(os.getenv("TELEGRAM_BOT_TOKEN")),
        "admin_ids_configured": bool(getattr(main, "ADMIN_IDS", [])),
        "subscription_url_configured": bool(getattr(main, "SUBSCRIBE_URL", "")),
    }

def run_bot_with_restart():
    """
    Runs the bot (main.main) in a loop. main.main must call run_polling(stop_signals=None).
    We restart on crash with a brief sleep to avoid crash loops.
    """
    while True:
        try:
            import main
            print("[BOT] Starting bot (main.main) ...")
            main.main()  # main.main should call run_polling(..., stop_signals=None)
            # If main.main returns, break (clean exit)
            print("[BOT] main.main exited cleanly.")
            break
        except Exception as e:
            print("[BOT] Bot crashed:", e)
            traceback.print_exc()
            print("[BOT] Restarting bot in 5 seconds...")
            time.sleep(5)

if __name__ == "__main__":
    # Ensure token exists
    missing = [v for v in ("TELEGRAM_BOT_TOKEN",) if not os.getenv(v)]
    if missing:
        print(f"[ERROR] Missing required env vars: {missing}")
        raise SystemExit(1)

    # Start bot thread (daemon so Flask can still stop the process)
    bot_thread = threading.Thread(target=run_bot_with_restart, daemon=True)
    bot_thread.start()

    # Start Flask in the main thread (Render needs this)
    port = int(os.environ.get("PORT", 8080))
    host = "0.0.0.0"
    print(f"[INFO] Starting Flask web server on {host}:{port}")
    app.run(host=host, port=port, debug=False, threaded=True, use_reloader=False)
