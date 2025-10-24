# Security Notice

## ⚠️ API Key Security

### What happened?
An API key was previously hardcoded in several test files and documentation. This has been **removed** and all files now use environment variables.

### Current status
✅ **All hardcoded API keys removed**
✅ **All scripts use environment variables**
✅ **Documentation redacted**

### Best practices going forward

#### ✅ DO:
```bash
# Set API key via environment variable
export API_SPORTS_KEY='your_actual_key'
python nba_prop_analyzer_fixed.py
```

Or create a `.env` file (already in `.gitignore`):
```bash
echo "API_SPORTS_KEY=your_actual_key" > .env
source .env
```

#### ❌ DON'T:
```python
# NEVER do this
API_KEY = "your_actual_key_hardcoded"
```

### If your key was exposed

If you previously pushed code with a hardcoded API key:

1. **Revoke the old key immediately**
   - Go to https://api-sports.io/account
   - Revoke/delete the exposed key
   - Generate a new key

2. **Update your local environment**
   ```bash
   export API_SPORTS_KEY='your_new_key'
   ```

3. **Verify no keys in git history**
   ```bash
   git log -S"your_old_key" --all
   ```

### Git history note

Even though we removed the keys from the current code, they may still exist in git history. If this repository is:

- **Public:** Consider the exposed key compromised - revoke it
- **Private:** Still best practice to revoke and rotate

### Automated detection

Consider using tools to prevent accidental commits:

**git-secrets:**
```bash
git secrets --install
git secrets --register-aws
git secrets --add '4979ac5e1f7ae10b1d6b58f1bba01140'  # Block old key
```

**pre-commit hooks:**
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/Yelp/detect-secrets
    rev: v1.4.0
    hooks:
      - id: detect-secrets
```

### What we do now

All scripts in this repository:

1. ✅ Use `os.getenv("API_SPORTS_KEY")` for API keys
2. ✅ Raise clear errors if key not set
3. ✅ Never hardcode credentials
4. ✅ Store keys in `.env` (which is `.gitignore`d)

### Questions?

See:
- `QUICK_START.md` - How to set up API keys correctly
- `README.md` - General setup instructions
- `.gitignore` - What files are excluded from git

---

**Remember:** Never commit API keys, passwords, or secrets to git! 🔒
