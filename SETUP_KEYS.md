# Super Easy API Key Setup (One Time!)

## Why This Is Better

Instead of environment variables, you just edit ONE file with your keys and **never have to enter them again**.

✅ **No more:** `export API_SPORTS_KEY=...`
✅ **No more:** Setting up kaggle.json
✅ **No more:** Entering keys every time

Just edit `api_keys.py` once and you're done! 🎉

---

## Setup (Takes 30 seconds)

### Option 1: Edit the existing file
```bash
nano api_keys.py
```

Paste your keys:
```python
# API-Sports.io key
API_SPORTS_KEY = "4979ac5e1f7ae10b1d6b58f1bba01140"

# Kaggle key
KAGGLE_USERNAME = "your_username"  # Optional
KAGGLE_KEY = "f005fb2c580e2cbfd2b6b4b931e10dfc"
```

Save and exit (Ctrl+X, Y, Enter)

### Option 2: Use the template
```bash
cp api_keys.example.py api_keys.py
nano api_keys.py
# Add your keys
```

---

## That's It!

Now just run your scripts normally:

```bash
# No need to set environment variables!
python nba_prop_analyzer_fixed.py

# No need to setup Kaggle!
python train_auto.py
```

**All scripts automatically read from `api_keys.py`** ✨

---

## How It Works

Every script tries to import from `api_keys.py` first:

```python
try:
    from api_keys import API_SPORTS_KEY
    # Use the key from api_keys.py
except ImportError:
    # Fall back to environment variables
    API_KEY = os.getenv("API_SPORTS_KEY")
```

So you can use either method:
- **api_keys.py** (easiest - just edit once)
- **Environment variables** (traditional way)

---

## Security

**Is this safe?**

✅ Yes, because:
1. `api_keys.py` is in `.gitignore` (won't be committed)
2. File is only on your local machine
3. Not shared publicly

**DON'T:**
- ❌ Commit `api_keys.py` to git
- ❌ Share `api_keys.py` with others
- ❌ Push to public repositories

**The file is already .gitignore'd so you're safe!**

---

## Your Keys (for reference)

**API-Sports.io:** `4979ac5e1f7ae10b1d6b58f1bba01140`
**Kaggle:** `f005fb2c580e2cbfd2b6b4b931e10dfc`

These are already in `api_keys.py` - you don't need to do anything! 🎉

---

## Quick Test

```bash
# Test NBA analyzer
python nba_prop_analyzer_fixed.py

# Test ML training
python train_auto.py
```

If both work without asking for keys, you're all set! ✅
