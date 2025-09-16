# web.py - Production-Ready 24/7 Telegram Bot Service
import os
import sys
import threading
import time
import signal
import atexit
import logging
from contextlib import contextmanager
from flask import Flask, jsonify, request
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('bot_service.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global bot state with thread-safe operations
class BotManager:
    def __init__(self):
        self._lock = threading.RLock()
        self._bot_thread = None
        self._bot_application = None
        self._event_loop = None
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="bot-")
        self._restart_count = 0
        self._max_restarts = 10
        self._last_restart = 0
        self._restart_cooldown = 60  # seconds
        self._shutdown_event = threading.Event()
        
        self.status = {
            "alive": False,
            "start_time": None,
            "error": None,
            "last_heartbeat": None,
            "restart_count": 0,
            "commands_processed": 0,
            "uptime_seconds": 0
        }
    
    def _can_restart(self):
        """Check if bot can be restarted (rate limiting)"""
        current_time = time.time()
        if current_time - self._last_restart < self._restart_cooldown:
            return False
        if self._restart_count >= self._max_restarts:
            logger.error("Maximum restart attempts reached")
            return False
        return True
    
    def _update_status(self, **kwargs):
        """Thread-safe status update"""
        with self._lock:
            self.status.update(kwargs)
            if self.status.get("start_time"):
                self.status["uptime_seconds"] = int(time.time() - self.status["start_time"])
    
    def _bot_runner(self):
        """Main bot thread function with comprehensive error handling"""
        try:
            logger.info("Initializing bot thread...")
            
            # Set up async environment for this thread
            if sys.platform == 'win32':
                asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
            
            # Create new event loop for this thread
            self._event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._event_loop)
            
            # Import and setup bot
            import main
            from telegram.ext import Application
            from telegram import Update
            from telegram.ext import ContextTypes
            
            # Validate environment
            token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
            if not token:
                raise RuntimeError("TELEGRAM_BOT_TOKEN environment variable is required")
            
            logger.info(f"Token configured: {len(token)} chars")
            
            # Create application with optimal settings
            self._bot_application = (
                Application.builder()
                .token(token)
                .concurrent_updates(True)  # Enable concurrent updates
                .pool_timeout(30.0)
                .connection_pool_size(8)
                .read_timeout(30.0)
                .write_timeout(30.0)
                .build()
            )
            
            # Wrap handlers with monitoring
            def wrap_command_handler(handler_func, handler_name):
                async def wrapped_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
                    try:
                        logger.info(f"Processing command: {handler_name}")
                        self._update_status(
                            commands_processed=self.status["commands_processed"] + 1,
                            last_heartbeat=time.time()
                        )
                        result = await handler_func(update, context)
                        logger.info(f"Command {handler_name} completed successfully")
                        return result
                    except Exception as e:
                        logger.error(f"Handler {handler_name} failed: {e}")
                        if update and update.message:
                            try:
                                await update.message.reply_text(
                                    f"⚠️ Command failed: {str(e)[:100]}. Please try again."
                                )
                            except:
                                pass
                        raise
                return wrapped_handler
            
            # Register handlers with monitoring
            from telegram.ext import CommandHandler
            
            handlers = [
                ("start", main.start),
                ("forecast", main.forecast_cmd),
                ("subscribe", main.subscribe_cmd),
                ("paid", main.paid_cmd),
                ("activate", main.activate_cmd),
                ("deactivate", main.deactivate_cmd),
                ("status", main.status_cmd)
            ]
            
            for cmd_name, handler_func in handlers:
                try:
                    if hasattr(main, handler_func.__name__):
                        wrapped_handler = wrap_command_handler(handler_func, cmd_name)
                        self._bot_application.add_handler(CommandHandler(cmd_name, wrapped_handler))
                        logger.info(f"Registered handler: {cmd_name}")
                    else:
                        logger.warning(f"Handler not found: {cmd_name}")
                except Exception as e:
                    logger.error(f"Failed to register handler {cmd_name}: {e}")
            
            # Add error handler
            async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
                logger.error(f"Update {update} caused error: {context.error}")
                
            self._bot_application.add_error_handler(error_handler)
            
            # Update status
            self._update_status(
                alive=True,
                start_time=time.time(),
                error=None,
                last_heartbeat=time.time()
            )
            
            logger.info("Starting bot polling...")
            
            # Start polling with optimal settings
            self._bot_application.run_polling(
                poll_interval=2.0,  # Poll every 2 seconds
                timeout=20,  # 20 second timeout for long polling
                bootstrap_retries=-1,  # Infinite retries for bootstrap
                read_timeout=20,
                write_timeout=20,
                connect_timeout=20,
                pool_timeout=20,
                stop_signals=None,  # Disable signal handlers in thread
                close_loop=False,  # Don't close the event loop
                drop_pending_updates=True  # Drop pending updates on startup
            )
            
        except Exception as e:
            error_msg = f"Bot thread crashed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self._update_status(alive=False, error=error_msg)
            
            # Schedule restart if conditions allow
            if self._can_restart() and not self._shutdown_event.is_set():
                logger.info(f"Scheduling bot restart in {self._restart_cooldown} seconds...")
                self._schedule_restart()
        
        finally:
            # Cleanup
            if self._bot_application:
                try:
                    if self._bot_application.running:
                        asyncio.run_coroutine_threadsafe(
                            self._bot_application.stop(), self._event_loop
                        ).result(timeout=10)
                except Exception as e:
                    logger.error(f"Error stopping bot application: {e}")
            
            if self._event_loop and self._event_loop.is_running():
                try:
                    self._event_loop.stop()
                except Exception as e:
                    logger.error(f"Error stopping event loop: {e}")
    
    def _schedule_restart(self):
        """Schedule a bot restart after cooldown period"""
        def restart_after_delay():
            if not self._shutdown_event.wait(self._restart_cooldown):
                logger.info("Attempting bot restart...")
                self.start_bot()
        
        restart_thread = threading.Thread(target=restart_after_delay, daemon=True)
        restart_thread.start()
    
    def start_bot(self):
        """Start the bot with restart protection"""
        with self._lock:
            if self._bot_thread and self._bot_thread.is_alive():
                logger.warning("Bot thread already running")
                return False
            
            if not self._can_restart():
                logger.error("Cannot restart bot: rate limited or max attempts reached")
                return False
            
            try:
                self._restart_count += 1
                self._last_restart = time.time()
                
                # Create and start new bot thread
                self._bot_thread = threading.Thread(
                    target=self._bot_runner,
                    name=f"TelegramBot-{self._restart_count}",
                    daemon=False  # Don't make it daemon so it can restart
                )
                
                self._bot_thread.start()
                logger.info(f"Bot thread started (attempt #{self._restart_count})")
                
                # Start heartbeat monitor
                self._start_heartbeat_monitor()
                
                return True
                
            except Exception as e:
                logger.error(f"Failed to start bot thread: {e}")
                self._update_status(alive=False, error=str(e))
                return False
    
    def _start_heartbeat_monitor(self):
        """Monitor bot health and restart if needed"""
        def heartbeat_monitor():
            while not self._shutdown_event.is_set():
                try:
                    self._shutdown_event.wait(30)  # Check every 30 seconds
                    
                    if self._shutdown_event.is_set():
                        break
                    
                    with self._lock:
                        if not self.status["alive"]:
                            continue
                        
                        # Check if bot thread is still alive
                        if not self._bot_thread or not self._bot_thread.is_alive():
                            logger.warning("Bot thread died unexpectedly")
                            self._update_status(alive=False, error="Bot thread died")
                            
                            if self._can_restart():
                                logger.info("Restarting dead bot thread...")
                                self.start_bot()
                        
                        # Check heartbeat (commands should update last_heartbeat)
                        if self.status.get("last_heartbeat"):
                            time_since_heartbeat = time.time() - self.status["last_heartbeat"]
                            if time_since_heartbeat > 300:  # 5 minutes without activity
                                logger.warning(f"No heartbeat for {time_since_heartbeat:.0f}s")
                        
                except Exception as e:
                    logger.error(f"Heartbeat monitor error: {e}")
        
        monitor_thread = threading.Thread(target=heartbeat_monitor, daemon=True)
        monitor_thread.start()
    
    def stop_bot(self):
        """Gracefully stop the bot"""
        logger.info("Stopping bot service...")
        self._shutdown_event.set()
        
        with self._lock:
            if self._bot_application:
                try:
                    if self._event_loop and not self._event_loop.is_closed():
                        future = asyncio.run_coroutine_threadsafe(
                            self._bot_application.stop(), self._event_loop
                        )
                        future.result(timeout=10)
                except Exception as e:
                    logger.error(f"Error stopping bot application: {e}")
            
            if self._bot_thread and self._bot_thread.is_alive():
                self._bot_thread.join(timeout=15)
            
            if self._executor:
                self._executor.shutdown(wait=True, timeout=10)
        
        self._update_status(alive=False, error="Service stopped")
        logger.info("Bot service stopped")
    
    def get_status(self):
        """Get current bot status"""
        with self._lock:
            status = self.status.copy()
            status["thread_alive"] = bool(self._bot_thread and self._bot_thread.is_alive())
            status["restart_count"] = self._restart_count
            return status

# Global bot manager instance
bot_manager = BotManager()

# Flask routes
@app.route("/")
def home():
    return jsonify({
        "service": "Stock Sight AI Telegram Bot",
        "status": "running",
        "version": "2.0-production"
    })

@app.route("/health")
def health():
    """Comprehensive health check"""
    status = bot_manager.get_status()
    
    # Determine overall health
    is_healthy = (
        status.get("alive", False) and 
        status.get("thread_alive", False) and
        not status.get("error")
    )
    
    return jsonify({
        "status": "healthy" if is_healthy else "unhealthy",
        "service": "stock-sight-telegram-bot",
        "bot_status": status,
        "environment": {
            "token_configured": bool(os.getenv("TELEGRAM_BOT_TOKEN")),
            "admin_ids_configured": bool(os.getenv("ADMIN_IDS")),
            "subscribe_url_configured": bool(os.getenv("SUBSCRIBE_URL"))
        },
        "system": {
            "python_version": sys.version,
            "platform": sys.platform,
            "pid": os.getpid()
        }
    }), 200 if is_healthy else 503

@app.route("/restart", methods=["POST"])
def restart_bot():
    """Manual bot restart endpoint"""
    try:
        logger.info("Manual restart requested")
        bot_manager.stop_bot()
        time.sleep(2)
        
        if bot_manager.start_bot():
            return jsonify({"status": "success", "message": "Bot restarted"})
        else:
            return jsonify({"status": "error", "message": "Failed to restart bot"}), 500
    except Exception as e:
        logger.error(f"Manual restart failed: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/logs")
def get_logs():
    """Get recent log entries"""
    try:
        log_file = "bot_service.log"
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                lines = f.readlines()
                return jsonify({
                    "logs": lines[-100:],  # Last 100 lines
                    "total_lines": len(lines)
                })
        else:
            return jsonify({"logs": [], "total_lines": 0})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Signal handlers for graceful shutdown
def signal_handler(signum, frame):
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    bot_manager.stop_bot()
    sys.exit(0)

def cleanup():
    """Cleanup function called on exit"""
    bot_manager.stop_bot()

# Register cleanup handlers
atexit.register(cleanup)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

if __name__ == "__main__":
    # Environment validation
    required_env = ["TELEGRAM_BOT_TOKEN"]
    missing_env = [var for var in required_env if not os.getenv(var)]
    
    if missing_env:
        logger.error(f"Missing required environment variables: {missing_env}")
        sys.exit(1)
    
    logger.info("Starting Stock Sight Telegram Bot Service...")
    logger.info(f"Python: {sys.version}")
    logger.info(f"Platform: {sys.platform}")
    logger.info(f"PID: {os.getpid()}")
    
    # Start bot manager
    if not bot_manager.start_bot():
        logger.error("Failed to start bot on startup")
        sys.exit(1)
    
    # Wait for bot to initialize
    time.sleep(3)
    
    # Start Flask server
    port = int(os.environ.get("PORT", 8080))
    host = "0.0.0.0"
    
    logger.info(f"Starting Flask server on {host}:{port}")
    
    try:
        # Production WSGI server configuration
        app.run(
            host=host,
            port=port,
            debug=False,
            use_reloader=False,  # Critical: disable reloader
            threaded=True,
            processes=1
        )
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Flask server error: {e}")
    finally:
        logger.info("Shutting down service...")
        bot_manager.stop_bot()
