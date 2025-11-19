# Proper NBA Data Aggregation - Plan

## The Real Problem

Your existing `aggregated_nba_data.parquet` was created with the correct approach using `create_aggregated_dataset_final.py`. That script:

1. ✅ Uses fuzzy name matching (rapidfuzz, 85% threshold)
2. ✅ Merges season-aggregated Basketball Reference stats onto game-level data
3. ✅ Properly handles the schema mismatch (game-level vs season-level)

**So why did Modal training show 0.0% match rate?**

The issue is that `shared/csv_aggregation.py` (which we just wrote) tries to merge CSVs differently than the original aggregation script. It doesn't use fuzzy matching!

## Two Paths Forward

### Option 1: Use Your Existing Parquet (Recommended)
Your `aggregated_nba_data.parquet` was created with the CORRECT fuzzy-matching script. It should already have all advanced stats properly merged.

**Test this**:
```bash
# Upload existing Parquet to Modal
py -3.12 -m modal run modal_upload_parquet.py

# Train with it
py -3.12 -m modal run modal_train.py --window-start 2022 --window-end 2024
```

The advanced stats should already be there! Check the training output for columns like:
- `adv_per`, `adv_bpm`, `adv_vorp`
- `per100_pts`, `per100_ast`
- `shoot_avg_dist_fga`
- `pbp_plus_minus`

### Option 2: Re-aggregate on Modal with Fuzzy Matching
If the Parquet is missing advanced stats, we need to:

1. Copy `create_aggregated_dataset_final.py` to Modal
2. Run it on Modal with 64GB RAM
3. This creates a fresh Parquet with fuzzy matching

**Steps**:
```bash
# Create Modal aggregation script (uses the fuzzy-match approach)
py -3.12 -m modal run modal_aggregate_fuzzy.py
```

## Why Fuzzy Matching is Needed

Basketball Reference CSVs have player names like:
- "LeBron James"
- "Stephen Curry"
- "Giannis Antetokounmpo"

PlayerStatistics has:
- firstName: "LeBron", lastName: "James" → "LeBron James" ✅
- firstName: "Stephen", lastName: "Curry" → "Stephen Curry" ✅
- firstName: "Giannis", lastName: "Antetokounmpo" → Might have accents/spelling variations ❌

Fuzzy matching handles:
- Accent differences (é vs e)
- Spelling variations
- Name order differences
- Nicknames

## My Recommendation

**Try Option 1 first!** Your existing Parquet should already be good. The 0.0% match rate you saw was from the CSV loading attempt on Modal, not from the Parquet file itself.

Just upload and train:
```bash
py -3.12 -m modal run modal_upload_parquet.py
py -3.12 -m modal run modal_train.py --window-start 2022 --window-end 2024
```

Look at the column count in training output. If you see ~180 columns with `adv_`, `per100_`, `shoot_`, `pbp_` prefixes, you're golden! 🎉
