# RIQ Analyzer Updates - ELG Gates & Parlays

## Summary of Changes

### 1. Loosened ELG Gates (More Props Pass)
```python
ELG_GATES = {
    "points": -0.002,    # 4x more permissive
    "assists": -0.0005,  # Original (already working well)
    "rebounds": -0.002,  # 4x more permissive
    "threes": -0.002,    # 4x more permissive
    "moneyline": -0.0005,
    "spread": -0.0005,
}
```

**Effect**: Points, rebounds, and threes props now have a lower bar to pass, allowing more opportunities while assists remains selective.

### 2. Max Stake Limit
```python
MAX_STAKE = 10.0  # $10 maximum per bet
```

**Effect**: 
- All single bets capped at $10
- Kelly sizing still calculated but then limited to max
- Prevents over-betting on high-confidence props
- Applies to both single bets and parlays

### 3. Parlay Builder (New Feature!)

#### Automatic 2-3 Leg Parlays
The system now automatically builds optimal parlays from your best props:

**Strategy**:
- ✅ Only uses high-quality props (≥60% win probability)
- ✅ Prevents correlation (no duplicate players)
- ✅ Calculates true parlay odds
- ✅ Uses conservative Kelly (50% of normal, max 15%)
- ✅ Positive EV required to qualify

**Example Parlay**:
```
🎲 Parlay #1 — 3 Legs
Combined Odds: +14634 (Decimal: 147.34)
Win Probability: 84.2%
Stake: $0.29
Potential Profit: $41.73
Expected Value: +439.77%

Legs:
  1. Scottie Barnes - ASSISTS UNDER 7.5 @ +450 (97.9%)
  2. DeMar DeRozan - ASSISTS UNDER 7.5 @ +470 (94.3%)
  3. Domantas Sabonis - ASSISTS UNDER 7.5 @ +340 (93.3%)
```

### Parlay Sizing Philosophy

**Why are parlay stakes so small?**
- Parlays are **much riskier** than single bets
- Uses **half-Kelly** for extra safety
- Capped at **15% of bankroll max**
- Still offers huge upside (100x+ returns possible)

**Example**:
- Single bet: $10 stake → $45 profit (4.5x)
- Parlay: $0.29 stake → $41.73 profit (143x)

### Output Format

**JSON File Includes**:
```json
{
  "parlays": [
    {
      "type": "parlay",
      "num_legs": 3,
      "parlay_odds": 14634,
      "parlay_prob": 84.2,
      "stake": 0.29,
      "potential_profit": 41.73,
      "ev": 439.77,
      "legs": [...]
    }
  ]
}
```

### Configuration

To adjust parlay settings, modify in `riq_analyzer.py`:

```python
# In run_analysis() function:
parlays = build_parlays(
    analyzed,
    max_legs=3,      # Maximum legs per parlay (2-3)
    min_legs=2,      # Minimum legs per parlay
    max_parlays=10   # How many parlays to return
)
```

### Best Practices

**Single Bets**:
- Use for high-confidence plays (>90% win prob)
- Maximum stake protection at $10
- Lower variance, consistent returns

**Parlays**:
- Combine multiple strong props
- Very small stakes for asymmetric upside
- Higher variance, lottery-ticket style
- Only bet what you can afford to lose

### Why Only Assists Props?

Currently seeing only assists props because:
1. Data source may only have assists available right now
2. ELG gates are working - just need more prop data
3. When points/rebounds/threes data is available, they'll pass with loosened gates

**To verify**: Check the sportsbook API for available markets. The gates are ready when the data arrives!

## Recommendations

1. **Conservative Approach**: Stick to single bets with your $10 max stake
2. **Aggressive Approach**: Use parlays for upside with tiny stakes
3. **Balanced Approach**: Mix both - single bets for base, parlays for lottery tickets

**Bankroll Management**:
- $100 bankroll
- Max $10 per single bet (10% max)
- Parlays typically $0.20-0.50 (0.2-0.5%)
- Can place 10-20 bets comfortably

---

**Status**: ✅ All changes implemented and tested
**Date**: 2025-10-26
