# How to Get a Working Basketball API Key

## 🔴 Current Status

Your API key `4979ac5e1f7ae10b1d6b58f1bba01140` is being rejected (403 Forbidden) by all services.

**This could mean:**
1. The key is from the wrong API service (Football instead of Basketball)
2. The subscription hasn't been activated yet
3. There's an account issue that needs resolution
4. The key was copied incorrectly

---

## ✅ Step-by-Step: Get Working Basketball API Key

### Option 1: API-Sports Basketball (Recommended)

#### Step 1: Register for Basketball API
**Important:** This is DIFFERENT from API-Football!

1. **Go to:** https://dashboard.api-basketball.com/register
   - NOT api-football.com (that's for soccer)
   - Must be api-BASKETBALL.com

2. **Fill out registration:**
   - Email address
   - Password
   - Accept terms

3. **Verify your email** (check inbox/spam)

#### Step 2: Get Your API Key

1. **Log in to:** https://dashboard.api-basketball.com/
2. **Look for "Your API Key" section** on the dashboard
3. **Copy the key** (should be 32 characters)

#### Step 3: Check Your Plan

1. On the dashboard, verify you see:
   ```
   Plan: Free
   Requests: 100 per day
   Status: Active
   ```

2. If you see "Inactive" or no plan:
   - You may need to verify your email
   - Or wait a few minutes for activation
   - Or subscribe to the free plan

#### Step 4: Test Your Key

```bash
python test_api_simple.py
```

You should see:
```
✅ Method X successful!
```

---

### Option 2: RapidAPI (Alternative)

If API-Sports direct registration isn't working, try RapidAPI:

#### Step 1: Sign up for RapidAPI

1. **Go to:** https://rapidapi.com/auth/sign-up
2. **Create account** (can use Google/GitHub)

#### Step 2: Subscribe to Basketball API

1. **Go to:** https://rapidapi.com/api-sports/api/api-basketball
2. **Click "Subscribe to Test"**
3. **Select a plan:**
   - **Basic (Free):** 100 requests/day, $0/month
   - **Pro:** 1,000 requests/day, $10-15/month

4. **Click "Subscribe"**

#### Step 3: Get Your RapidAPI Key

1. On the API page, look for **"Code Snippets"** section
2. You'll see: `X-RapidAPI-Key: YOUR_KEY_HERE`
3. **Copy that key**

#### Step 4: Update Your Code

**Important:** RapidAPI uses different URLs and headers!

```python
# For RapidAPI, update these values:
API_KEY = "your_rapidapi_key_here"
BASE_URL = "https://api-basketball.p.rapidapi.com"  # Different URL!
HEADERS = {
    "X-RapidAPI-Key": API_KEY,
    "X-RapidAPI-Host": "api-basketball.p.rapidapi.com"
}
```

---

## 🧪 Testing Checklist

After getting your key, run these tests:

### Test 1: Basic Connection
```bash
python test_api_simple.py
```
**Expected:** ✅ One method shows "successful"

### Test 2: Full Diagnostics
```bash
python api_diagnostics.py
```
**Expected:**
- ✅ API connection works
- ✅ Games found
- ✅ Player found
- ✅ Player stats retrieved

### Test 3: Run Analyzer
```bash
python nba_prop_analyzer_optimized.py
```
**Expected:** Props analyzed and results saved

---

## 🚨 Troubleshooting

### Problem: "Access Denied" Even With New Key

**Check:**
1. Did you verify your email?
2. Is your subscription active?
3. Did you copy the ENTIRE key? (32 characters)
4. Are you using the right dashboard?
   - Basketball: dashboard.api-basketball.com ✅
   - Football: dashboard.api-football.com ❌

### Problem: "No Games Found"

**This is normal!** Could mean:
- No NBA games in the next 3 days (off-season, break)
- Season ended
- Games not yet scheduled

**Solution:** Wait for game days or adjust date range in code

### Problem: "Player Not Found"

**Possible causes:**
- Player name spelling
- Player retired/not in system
- API database not updated

**Solution:** Try common active players like:
- LeBron James
- Stephen Curry
- Kevin Durant
- Giannis Antetokounmpo

### Problem: Rate Limit Exceeded

Free tier = 100 requests/day. If exceeded:
- **Wait 24 hours** for reset
- **Upgrade plan** for more requests
- **Use multiple keys** (create multiple accounts)

---

## 💡 Quick Alternative: Free APIs (No Key Needed)

If you're still having issues and need to test your code NOW:

### Option A: Ball Don't Lie API (Free, No Key)

```python
import requests

# No API key needed!
response = requests.get("https://www.balldontlie.io/api/v1/players")
data = response.json()
print(data)
```

**Pros:**
- Completely free
- No registration
- No rate limits (reasonable use)

**Cons:**
- Different data structure
- Less detailed stats
- No betting odds

### Option B: nba_api Python Package (Free, No Key)

```bash
pip install nba_api
```

```python
from nba_api.stats.endpoints import playergamelog

# Direct from NBA.com, no key needed
gamelog = playergamelog.PlayerGameLog(player_id='2544')  # LeBron
df = gamelog.get_data_frames()[0]
print(df)
```

**Pros:**
- Official NBA data
- Very detailed stats
- Python-friendly

**Cons:**
- No betting odds
- Different API structure
- Requires code refactoring

---

## 📞 Support Contacts

### API-Sports Basketball Support
- **Website:** https://www.api-basketball.com
- **Email:** contact@api-sports.io
- **Docs:** https://www.api-basketball.com/documentation

### RapidAPI Support
- **Help Center:** https://docs.rapidapi.com/
- **Email:** support@rapidapi.com

---

## ✅ Verification Steps

Once you get a new key, verify it works:

1. **Update the key** in your scripts
2. **Run test:** `python test_api_simple.py`
3. **Look for:** `✅ Method X successful!`
4. **Check response:** Should show account info, not "Access denied"

---

## 🎯 Summary

**Current Issue:** Your API key doesn't work (403 Forbidden)

**Most Likely Cause:**
- Key is from wrong service (Football vs Basketball)
- Or key not activated yet
- Or account issue

**Solution:**
1. Register at **https://dashboard.api-basketball.com/register**
2. Verify email and activate account
3. Copy API key from dashboard
4. Update scripts with new key
5. Run `python test_api_simple.py` to verify

**Expected Time:** 5-10 minutes

**Cost:** Free (100 requests/day)

---

## 📝 Quick Checklist

- [ ] Registered at dashboard.api-BASKETBALL.com (not football)
- [ ] Verified email address
- [ ] Subscription shows "Active"
- [ ] Copied full 32-character API key
- [ ] Updated key in scripts
- [ ] Ran test_api_simple.py successfully
- [ ] Saw ✅ success message

**If all checked, your API is ready to use!** 🚀

---

## ❓ Common Questions

**Q: Why are there two different API-Sports dashboards?**
A: API-Sports offers multiple sport APIs (football, basketball, baseball, etc.). Each requires a separate subscription and API key.

**Q: Can I use my Football API key for Basketball?**
A: No, they're completely separate. You need a Basketball-specific key.

**Q: Is the free tier enough?**
A: For testing and small-scale analysis (10-20 props/day), yes. For production (100+ props multiple times/day), you'll need a paid plan.

**Q: How long does activation take?**
A: Usually instant after email verification. If not, wait 5-10 minutes or contact support.

**Q: Can I get more free requests?**
A: Create multiple accounts with different emails, or upgrade to a paid plan.

---

**Need Help?** Share the exact error message you're getting and which website you used to get the API key!
