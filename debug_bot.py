# debug_bot.py - Simple script to test your bot token and basic connectivity
import os
import asyncio
import requests
from telegram import Bot

def test_bot_token():
    """Test if the bot token is valid and working"""
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    
    if not token:
        print("❌ TELEGRAM_BOT_TOKEN not found in environment")
        return False
    
    print(f"🔑 Testing token: {token[:10]}...{token[-10:]}")
    
    try:
        # Test using requests first
        url = f"https://api.telegram.org/bot{token}/getMe"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('ok'):
                bot_info = data.get('result', {})
                print(f"✅ Bot token valid!")
                print(f"   Bot name: {bot_info.get('first_name', 'Unknown')}")
                print(f"   Username: @{bot_info.get('username', 'Unknown')}")
                print(f"   Bot ID: {bot_info.get('id', 'Unknown')}")
                return True
            else:
                print(f"❌ Telegram API error: {data.get('description', 'Unknown')}")
                return False
        else:
            print(f"❌ HTTP error: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_webhook_info():
    """Check webhook configuration"""
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    
    try:
        url = f"https://api.telegram.org/bot{token}/getWebhookInfo"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('ok'):
                webhook_info = data.get('result', {})
                webhook_url = webhook_info.get('url', '')
                
                if webhook_url:
                    print(f"⚠️  Webhook is set: {webhook_url}")
                    print("   This might conflict with polling mode!")
                    print("   Consider deleting webhook with: /deleteWebhook")
                else:
                    print("✅ No webhook set (good for polling)")
                
                pending_updates = webhook_info.get('pending_update_count', 0)
                if pending_updates > 0:
                    print(f"📬 Pending updates: {pending_updates}")
                
                return True
        return False
        
    except Exception as e:
        print(f"❌ Error checking webhook: {e}")
        return False

async def test_bot_async():
    """Test bot using python-telegram-bot library"""
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    
    try:
        bot = Bot(token=token)
        me = await bot.get_me()
        print(f"✅ Async bot test successful!")
        print(f"   Bot: @{me.username} ({me.first_name})")
        return True
    except Exception as e:
        print(f"❌ Async bot test failed: {e}")
        return False

def main():
    print("🤖 Telegram Bot Diagnostics")
    print("=" * 40)
    
    # Test 1: Bot token validity
    print("\n1. Testing bot token...")
    if not test_bot_token():
        print("❌ Bot token test failed - check your TELEGRAM_BOT_TOKEN")
        return
    
    # Test 2: Webhook info
    print("\n2. Checking webhook configuration...")
    test_webhook_info()
    
    # Test 3: Async bot test
    print("\n3. Testing async bot connection...")
    try:
        asyncio.run(test_bot_async())
    except Exception as e:
        print(f"❌ Async test failed: {e}")
    
    print("\n" + "=" * 40)
    print("✅ Diagnostics complete!")
    print("\nIf all tests pass but bot still doesn't work:")
    print("1. Check Render logs for specific errors")
    print("2. Verify admin IDs are correctly set")
    print("3. Test with a simple /start command")
    print("4. Consider redeploying with the fixed web.py")

if __name__ == "__main__":
    main()
