# Running in a VENV (Isolated Environment)

## Your Situation

You're running scripts in a venv where you can't import from other Python files in the repo. This is common in isolated environments.

## ✅ Solution: Keys Are Already Hardcoded!

Your API keys are **already hardcoded directly** in the scripts:

- `nba_prop_analyzer_fixed.py` - Line 21: `API_KEY = "4979ac5e1f..."`
- `train_auto.py` - Line 61: `KAGGLE_KEY = "f005fb2c58..."`

## Just Run The Scripts!

```bash
# Activate your venv
source venv/bin/activate  # or however you activate it

# Run NBA analyzer (no setup needed!)
python nba_prop_analyzer_fixed.py

# Train ML models (no setup needed!)
python train_auto.py
```

**That's it!** No imports, no environment variables, no api_keys.py needed.

---

## How It Works

### nba_prop_analyzer_fixed.py
```python
# Line 21 - Your key is right here in the file:
API_KEY = "4979ac5e1f7ae10b1d6b58f1bba01140"
```

### train_auto.py
```python
# Line 61 - Your key is right here in the file:
KAGGLE_KEY = "f005fb2c580e2cbfd2b6b4b931e10dfc"
```

No imports = works in any venv! 🎉

---

## If You Clone Fresh From Git

If you clone the repo fresh, the keys will be placeholders:
```python
API_KEY = "YOUR_KEY_HERE"
```

### Quick Fix (One Command):
```bash
./setup_once.sh
```

This automatically injects your real keys into both scripts. Run it once, then you're done forever!

### Or Manual Fix:
Edit the files and replace `"YOUR_KEY_HERE"` with your actual keys:
- Line 21 in `nba_prop_analyzer_fixed.py`
- Line 61 in `train_auto.py`

---

## Current Status

✅ Your local copies **already have** the real keys hardcoded
✅ Ready to run immediately
✅ No imports or environment variables needed
✅ Works in any venv or isolated environment

---

## Security Note

⚠️ **Don't commit these files to git after adding keys!**

The files are tracked by git, but your local changes (with real keys) should stay local.

**Before committing:**
```bash
# Check what changed
git diff nba_prop_analyzer_fixed.py

# If it shows your real API key, DON'T commit it!
git restore nba_prop_analyzer_fixed.py
```

Or just don't run `git add` on these files after editing.

---

## Testing

```bash
# Quick test
python test_standalone.py

# Should show:
# ✅ Both scripts have hardcoded keys and can run standalone!
```

---

## Summary

**Question:** How do I run in a venv where I can't import other files?
**Answer:** Your keys are already hardcoded in the scripts - just run them!

```bash
python nba_prop_analyzer_fixed.py  # Works immediately
python train_auto.py                # Works immediately
```

No setup required! 🚀
