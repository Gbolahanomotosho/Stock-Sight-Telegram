
# 📈 Stock-Sight-Telegram

**A real-time stock market monitoring Telegram Bot** that delivers instant price alerts, trend analysis, and custom watchlists directly to your chat.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Telegram](https://img.shields.io/badge/Telegram-Bot_API-blue)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🚀 Live Demo

| Platform | Link |
|----------|------|
| Telegram Bot | [@StockSightBot](https://t.me/your_bot_username) *(replace with your bot's username)* |

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔔 **Real-time Alerts** | Get notified when a stock hits your target price |
| 📊 **Trend Analysis** | View price movement and volume trends |
| ⭐ **Custom Watchlist** | Track your favorite stocks |
| 💬 **Natural Language** | Type "show AAPL" or "alert me when TSLA drops below 200" |
| 📈 **Price History** | Fetch historical data for any ticker |

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|--------------|
| **Language** | Python 3.11 |
| **Bot Framework** | python-telegram-bot |
| **Data Source** | Yahoo Finance (yfinance) |
| **Web Framework** | Flask (for webhook) |
| **Deployment** | Render / Railway |

---

## 📂 Project Structure

```

Stock-Sight-Telegram/
├── main.py              # Main bot entry point
├── web.py / web2.py     # Webhook handlers for deployment
├── debug_bot.py         # Debugging utilities
├── requirements.txt     # Dependencies
├── runtime.txt          # Python version for deployment
├── subscriptions.json   # User watchlist storage
└── src/                 # Core modules

```

---

## 🔧 Installation & Local Testing

```bash
# Clone the repository
git clone https://github.com/Gbolahanomotosho/Stock-Sight-Telegram.git
cd Stock-Sight-Telegram

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set your Telegram Bot Token (get from @BotFather)
export TELEGRAM_BOT_TOKEN="your_token_here"

# Run the bot
python main.py
```

---

🤖 Example Bot Commands

Command What it does
/start Welcome message and instructions
/watch AAPL Add Apple to your watchlist
/alert TSLA 250 Alert me when Tesla hits $250
/price MSFT Get current Microsoft stock price
/watchlist Show your tracked stocks
/remove AAPL Remove from watchlist

---

🧠 What I Built (My Contribution)

· Complete Telegram bot logic – handling commands, user sessions, and error recovery
· Yahoo Finance integration – fetching real-time and historical stock data
· Alert system – background job that checks prices and notifies users
· JSON-based storage – lightweight user preference management
· Webhook deployment – Flask endpoints for hosting on Render/Railway

---

🚧 Current Status & Improvements

Component Status
Real-time price fetching ✅ Complete
Custom alerts ✅ Complete
Watchlist management ✅ Complete
Webhook deployment ⚠️ Needs documentation cleanup
Database migration (PostgreSQL) 🔄 Planned (replaces JSON)
Multi-language support 🔄 Planned (German, Yoruba)

---

📈 Why This Matters for German Employers

This project demonstrates:

· ✅ API integration (Telegram + Yahoo Finance)
· ✅ Asynchronous programming (handling multiple users)
· ✅ Real-world deployment (works on cloud platforms)
· ✅ Clean error handling (no bot crashes during my tests)
· ✅ User-centric design (natural language commands)

---

📫 Contact & Visa Status

Omotosho Gbolahan Hammed

· GitHub: Gbolahanomotosho
· Email: hammedg621@gmail.com
· 🛂 German IT Specialist Visa Eligible – 7+ years IT experience. No degree recognition required.

---

📜 License

MIT License – free for personal and commercial use with attribution.
