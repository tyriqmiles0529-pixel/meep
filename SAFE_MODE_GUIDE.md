# Safe Mode Guide 🛡️

## What is Safe Mode?

Safe Mode adds an extra buffer to betting lines for more conservative picks. When enabled, the system requires a larger edge before recommending a bet.

### Example

**Standard Mode**:
- Projection: LeBron 2.8 assists
- Line: 3.5 assists
- Pick: **UNDER** (2.8 < 3.5)
- Bet: UNDER 3.5

**Safe Mode (1.0 margin)**:
- Projection: LeBron 2.8 assists  
- Line: 3.5 assists
- Effective Line: 3.5 + 1.0 = **4.5 assists**
- Pick: **UNDER** (2.8 < 4.5)
- Bet: **UNDER 4.5** (requires finding a book with 4.5 line)

The system only shows picks where you can find the safer line!

## How to Enable

### Option 1: Environment Variables (Recommended)

**Windows (PowerShell)**:
```powershell
# Enable safe mode with 1.0 point margin
$env:SAFE_MODE="true"
$env:SAFE_MARGIN="1.0"

# Run analyzer
python riq_analyzer.py
```

**Windows (Command Prompt)**:
```cmd
set SAFE_MODE=true
set SAFE_MARGIN=1.0
python riq_analyzer.py
```

**Mac/Linux**:
```bash
export SAFE_MODE=true
export SAFE_MARGIN=1.0
python riq_analyzer.py
```

### Option 2: Edit File Directly

Edit `riq_analyzer.py` around line 69:

```python
# Change these lines:
SAFE_MODE = True  # Enable safe mode
SAFE_MARGIN = 1.0  # Extra points/rebounds/assists buffer
```

## Recommended Margins

### Conservative (Default)
```python
SAFE_MARGIN = 1.0  # 1 point/rebound/assist buffer
```
- **Points**: Line must be 1.0 point safer
- **Rebounds**: Line must be 1.0 rebound safer
- **Assists**: Line must be 1.0 assist safer
- **Threes**: Line must be 1.0 three-pointer safer

### Very Conservative
```python
SAFE_MARGIN = 1.5  # 1.5 point/rebound/assist buffer
```
- Even more room for error
- Fewer picks, but higher confidence

### Moderate
```python
SAFE_MARGIN = 0.5  # 0.5 point/rebound/assist buffer
```
- Still conservative, but less restrictive
- More picks than 1.0 margin

### Standard (No Safe Mode)
```python
SAFE_MODE = False
```
- Uses raw model projections
- Maximum number of picks

## How It Works

### UNDER Bets
When projection suggests UNDER:
- **Adds** margin to line
- Requires you to find a higher line

Example:
- Projection: 15.2 points
- Actual Line: 17.5 points  
- Safe Mode Line: 17.5 + 1.0 = **18.5 points**
- Pick: UNDER **18.5** (only if you can find this line)

### OVER Bets  
When projection suggests OVER:
- **Subtracts** margin from line
- Requires you to find a lower line

Example:
- Projection: 22.8 points
- Actual Line: 20.5 points
- Safe Mode Line: 20.5 - 1.0 = **19.5 points**
- Pick: OVER **19.5** (only if you can find this line)

## Output Examples

### Safe Mode ON
```
🛡️  SAFE MODE: ON (Margin: 1.0 points)
   Conservative betting - lines require extra 1.0 buffer

🏀 HIGH — PLAYER PROPS (3 picks)
=================================

✨ #1 — LeBron James
   Line:     3.5
   🛡️ Projection: 2.8 (Δ: -1.7, σ: 1.2) [Safe Mode]
   Pick:     UNDER @ -110
   
   ⚠️ Safe Mode active: Requires line at 4.5+ for UNDER
```

### Safe Mode OFF
```
⚡ SAFE MODE: OFF (Standard analysis)

🏀 HIGH — PLAYER PROPS (5 picks)
=================================

✨ #1 — LeBron James
   Line:     3.5
   Projection: 2.8 (Δ: -0.7, σ: 1.2)
   Pick:     UNDER @ -110
```

## When to Use Safe Mode

### Use Safe Mode When:
- ✅ You want very conservative picks
- ✅ You're risk-averse
- ✅ You have a smaller bankroll
- ✅ You're testing the system
- ✅ You want fewer, higher-confidence bets

### Don't Use Safe Mode When:
- ❌ You want maximum number of picks
- ❌ You trust the model fully
- ❌ You have a large bankroll
- ❌ You want standard edges

## Pro Tips

### 1. Line Shopping is Critical
Safe mode often requires finding alternate lines at different books:
- Check 3-5 different sportsbooks
- Use line shopping tools
- Some books offer alternate lines (+0.5, +1.0, etc.)

### 2. Combine with Kelly Sizing
Safe mode reduces picks but increases confidence:
- Higher Kelly fractions on remaining picks
- Better risk/reward ratio

### 3. Adjust Margin by Stat Type
You can customize margins per stat type (requires code edit):

```python
# Around line 3440 in riq_analyzer.py
if SAFE_MODE:
    # Custom margins per stat type
    margins = {
        "points": 1.5,    # More conservative for points
        "rebounds": 1.0,  # Standard for rebounds
        "assists": 0.5,   # Less conservative for assists
        "threes": 1.0     # Standard for threes
    }
    margin = margins.get(prop["prop_type"], SAFE_MARGIN)
    
    if projection < prop["line"]:
        effective_line = prop["line"] + margin
    else:
        effective_line = prop["line"] - margin
```

### 4. Track Performance
Compare results with and without safe mode:
- Run with safe mode for 1 week
- Run without for 1 week  
- Compare win rate, ROI, drawdowns

## FAQ

**Q: Will safe mode reduce my number of picks?**
A: Yes, significantly. Expect 30-50% fewer picks but higher win rate.

**Q: Does safe mode guarantee profits?**
A: No - it just adds an extra buffer. Still need disciplined bankroll management.

**Q: Can I use different margins for different bets?**
A: Yes, but requires code customization (see Pro Tips #3 above).

**Q: What if I can't find the safer line?**
A: Skip the bet. Safe mode is only useful if you can actually get the safer line.

**Q: How do I know if a pick is affected by safe mode?**
A: Look for the 🛡️ icon in the output and "[Safe Mode]" label.

## Testing Safe Mode

Quick test with debug output:

```powershell
$env:SAFE_MODE="true"
$env:SAFE_MARGIN="1.0"
$env:DEBUG_MODE="true"
python riq_analyzer.py
```

Look for output like:
```
[SAFE MODE] Original line: 3.5, Effective line: 4.5, Margin: 1.0
```

## Disabling Safe Mode

```powershell
# PowerShell
$env:SAFE_MODE="false"

# Or just unset it
Remove-Item Env:\SAFE_MODE
Remove-Item Env:\SAFE_MARGIN
```

---

**Recommendation**: Start with `SAFE_MARGIN=1.0` and adjust based on your risk tolerance and line availability at your sportsbooks.

**Remember**: Safe mode is only effective if you can actually find the safer lines. Always shop around!
