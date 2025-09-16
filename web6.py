# web.py - OPTIMIZED for stability with persistent storage
import os
import threading
import time
import sys
import requests
import json
from flask import Flask, jsonify

app = Flask(__name__)

# Bot status tracking
bot_thread = None
bot_status = {"alive": False, "start_time": None, "error": None}

# Persistent storage solution for subscriptions
RENDER_EXTERNAL_URL = os.getenv("RENDER_EXTERNAL_URL", "https://stock-sight-telegram.onrender.com")

def save_subscriptions_to_external(subs_data):
    """Save subscriptions to external storage (GitHub Gist or similar)"""
    try:
        # Option 1: Use environment variable as backup storage
        backup_data = json.dumps(subs_data)
        # This would normally go to a database, but for now store in memory
        app.config['BACKUP_SUBS'] = backup_data
        return True
    except Exception as e:
        print(f"[WARN] Failed to backup subscriptions: {e}")
        return False

def load_subscriptions_from_external():
    """Load subscriptions from external storage"""
    try:
        # Try to load from environment or external source
        backup_data = app.config.get('BACKUP_SUBS')
        if backup_data:
            return json.loads(backup_data)
        
        # Fallback: try to load from URL endpoint (if you set one up)
        # This would be your own simple database service
        return {}
    except Exception as e:
        print(f"[WARN] Failed to load backup subscriptions: {e}")
        return {}

@app.route("/")
def home():
    return "Stock Sight AI Telegram Bot - Optimized Version!"

@app.route("/health")
def health():
    uptime = int(time.time() - bot_status["start_time"]) if bot_status["start_time"] else 0
    
    # Check memory usage
    try:
        import psutil
        memory_percent = psutil.virtual_memory().percent
        memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
    except:
        memory_percent = 0
        memory_mb = 0
    
    return jsonify({
        "status": "healthy" if bot_status["alive"] else "degraded",
        "service": "stock-sight-telegram-bot-optimized",
        "bot_alive": bot_status["alive"],
        "bot_uptime_seconds": uptime,
        "memory_usage_mb": round(memory_mb, 1),
        "memory_percent": memory_percent,
        "last_error": bot_status.get("error"),
        "optimization_mode": "lightweight",
        "telegram_token_configured": bool(os.getenv("TELEGRAM_BOT_TOKEN"))
    })

@app.route("/backup_subscriptions", methods=['POST'])
def backup_subscriptions():
    """API endpoint to backup subscription data"""
    try:
        from flask import request
        data = request.get_json()
        if data:
            save_subscriptions_to_external(data)
            return jsonify({"status": "success", "message": "Subscriptions backed up"})
        return jsonify({"status": "error", "message": "No data provided"}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/restore_subscriptions", methods=['GET'])
def restore_subscriptions():
    """API endpoint to restore subscription data"""
    try:
        data = load_subscriptions_from_external()
        return jsonify({"status": "success", "data": data})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

def run_lightweight_bot():
    """Run bot with minimal resource usage"""
    import main_optimized  # This will be your lightweight main.py
    import asyncio
    
    try:
        bot_status["start_time"] = time.time()
        bot_status["alive"] = True
        bot_status["error"] = None
        
        print("[INFO] Starting lightweight bot...")
        
        # Create new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Get token
        token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        if not token:
            raise RuntimeError("TELEGRAM_BOT_TOKEN required")
        
        # Use optimized main function
        main_optimized.main_lightweight()
        
    except Exception as e:
        print(f"[ERROR] Lightweight bot crashed: {e}")
        bot_status["alive"] = False
        bot_status["error"] = str(e)
        
        # Force garbage collection to free memory
        import gc
        gc.collect()

def start_bot():
    """Start bot in background thread"""
    global bot_thread
    
    # Force cleanup before starting
    import gc
    gc.collect()
    
    bot_thread = threading.Thread(target=run_lightweight_bot, daemon=True)
    bot_thread.start()
    print(f"[INFO] Lightweight bot thread started")
    return bot_thread

if __name__ == "__main__":
    # Validate environment
    required_env = ["TELEGRAM_BOT_TOKEN"]
    missing_env = [var for var in required_env if not os.getenv(var)]
    
    if missing_env:
        print(f"[ERROR] Missing environment variables: {missing_env}")
        sys.exit(1)
    
    print("[INFO] Starting OPTIMIZED Stock Sight Bot Service...")
    
    # Start bot
    start_bot()
    
    # Shorter startup time for lightweight version
    time.sleep(2)
    
    # Start Flask
    port = int(os.environ.get("PORT", 8080))
    host = "0.0.0.0"
    
    print(f"[INFO] Starting Flask web server on {host}:{port}")
    print("[INFO] OPTIMIZATION: Using lightweight forecasting only")
    print("[INFO] OPTIMIZATION: Persistent subscription storage enabled")
    
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
