# API Key Troubleshooting Guide

## 🔴 Problem Identified

Your API key is returning **403 Forbidden** errors for all authentication methods. This means:

❌ **API key is invalid, expired, or subscription is inactive**

---

## ✅ Quick Fixes

### Option 1: Get a New API Key from API-Sports (Direct)

1. **Visit:** https://dashboard.api-football.com/register
2. **Sign up** for a free account
3. **Go to:** https://dashboard.api-football.com/
4. **Copy your API key** from the dashboard
5. **Update** your code with the new key

**Free Plan Includes:**
- 100 requests/day
- Access to most endpoints including player stats
- Perfect for testing and development

### Option 2: Use RapidAPI

1. **Visit:** https://rapidapi.com/api-sports/api/api-basketball/pricing
2. **Subscribe** to a plan (they have free tier)
3. **Get your RapidAPI key** from the API page
4. **Use different headers:**

```python
HEADERS = {
    "X-RapidAPI-Key": "YOUR_RAPIDAPI_KEY_HERE",
    "X-RapidAPI-Host": "api-basketball.p.rapidapi.com"
}
BASE_URL = "https://api-basketball.p.rapidapi.com"  # Different URL!
```

---

## 🔍 Why Your Current Key Isn't Working

Your key format looks correct (32 characters, hexadecimal), but it's being rejected because:

1. **Expired Subscription** - Free trials expire after 7-30 days
2. **Rate Limit Exceeded** - You may have used all your daily requests
3. **Invalid Key** - The key may have been revoked or regenerated
4. **Account Issue** - Subscription may not be active

---

## 📊 Check Your API Status

### For Direct API-Sports:
1. Log in to: https://dashboard.api-football.com/
2. Check "API Requests" section to see:
   - Daily request limit
   - Requests used today
   - Subscription status
   - API key validity

### For RapidAPI:
1. Log in to: https://rapidapi.com/
2. Go to: "My Apps" → Your App → "API Hub"
3. Check subscription status for API-Basketball

---

## 🧪 Testing Your New Key

Once you have a new API key, update the script and test:

```python
# Update in your script:
API_KEY = "your_new_key_here"

# Then run:
python test_api_simple.py
```

You should see:
```
✅ Method X successful!
```

---

## 🆓 Alternative Free APIs (If you need immediate access)

If you need to test your code while waiting for a new key:

### 1. Balldontlie API (Completely Free, No Key Required)
```python
# Free NBA stats API
BASE_URL = "https://www.balldontlie.io/api/v1"

# Example: Get player stats
response = requests.get(f"{BASE_URL}/players")
# No API key needed!
```

### 2. NBA API (Unofficial, Free)
```python
# Direct from nba.com
from nba_api.stats.endpoints import playergamelog

# Free, no key needed
gamelog = playergamelog.PlayerGameLog(player_id='2544')
df = gamelog.get_data_frames()[0]
```

---

## 🔧 Quick Fix for Your Script

### Step 1: Update API Key

Open your script and find:
```python
API_KEY = "c47bd0c1e7c0d008a514ecba161b347f"  # ❌ This key doesn't work
```

Replace with your new key:
```python
API_KEY = "your_new_working_key_here"  # ✅ New key
```

### Step 2: Verify Headers Format

Make sure you're using the correct headers for your API source:

**For Direct API-Sports:**
```python
HEADERS = {
    "x-apisports-key": API_KEY
}
BASE_URL = "https://v1.basketball.api-sports.io"
```

**For RapidAPI:**
```python
HEADERS = {
    "X-RapidAPI-Key": API_KEY,
    "X-RapidAPI-Host": "api-basketball.p.rapidapi.com"
}
BASE_URL = "https://api-basketball.p.rapidapi.com"
```

### Step 3: Test
```bash
python test_api_simple.py
```

---

## 📞 Support Resources

### API-Sports Support:
- Email: contact@api-sports.io
- Documentation: https://www.api-football.com/documentation-v3

### RapidAPI Support:
- Help Center: https://docs.rapidapi.com/
- Contact: support@rapidapi.com

---

## ⚡ Emergency Solution: Use Mock Data

If you need to test your analyzer logic RIGHT NOW without waiting for API access:

### Create Mock Data Generator:

```python
# mock_api_data.py
import random
import numpy as np

def mock_player_stats(player_name, num_games=10):
    """Generate realistic mock player stats"""
    # Based on typical NBA player performance
    base_stats = {
        "LeBron James": {"points": 25, "assists": 7, "rebounds": 8, "threes": 2},
        "Stephen Curry": {"points": 28, "assists": 6, "rebounds": 5, "threes": 4},
        # Add more players...
    }

    base = base_stats.get(player_name, {"points": 15, "assists": 4, "rebounds": 5, "threes": 1})

    stats = []
    for _ in range(num_games):
        stats.append({
            "points": max(0, np.random.normal(base["points"], 5)),
            "assists": max(0, np.random.normal(base["assists"], 2)),
            "totReb": max(0, np.random.normal(base["rebounds"], 2)),
            "fgm": max(0, np.random.normal(base["threes"], 1))
        })

    return stats
```

Then modify your analyzer to use mock data during testing.

---

## 📝 Next Steps

1. ✅ **Get a new API key** from API-Sports or RapidAPI
2. ✅ **Update your script** with the new key
3. ✅ **Run test_api_simple.py** to verify it works
4. ✅ **Run your analyzer** and start getting predictions!

---

## ❓ Common Questions

**Q: Why did my key stop working?**
A: Free trials expire, or you may have exceeded your daily request limit.

**Q: How many requests do I need?**
A: For analyzing ~100 props, you'll make ~150-300 API calls. The free tier (100/day) might be tight.

**Q: Which plan should I get?**
A: Start with the free tier to test. Upgrade to Basic ($10-15/month) for serious use.

**Q: Can I use multiple API keys?**
A: Yes! You can rotate between keys to increase your daily limit.

---

## 🎯 Summary

**Current Status:** ❌ API key invalid
**Action Required:** Get new API key
**Estimated Time:** 5-10 minutes
**Cost:** Free tier available

**Your API key format is correct, it's just not active/valid anymore.**

Get a new key and you'll be up and running! 🚀
