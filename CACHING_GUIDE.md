# 5-Year Window Caching System

## Overview

The training pipeline now implements an intelligent 5-year window caching system that dramatically reduces RAM usage and training time by avoiding redundant retraining of historical data.

## How It Works

### Key Concepts

1. **5-Year Windows**: Data is split into non-overlapping 5-year windows (e.g., 2002-2006, 2007-2011, etc.)
2. **Smart Caching**: Each window's trained model is cached with metadata validation
3. **Incremental Training**: Only trains missing/invalid windows and the current season window
4. **Sequential Processing**: Processes windows one at a time to minimize RAM usage

### Benefits

- **RAM Savings**: Only loads one 5-year window at a time instead of all historical data
- **Time Savings**: Skips training for cached historical windows
- **Incremental Updates**: New data (current season) doesn't require retraining from 2002
- **Cache Validation**: Metadata ensures cache integrity (detects incomplete/corrupted caches)

## Usage

### Enable Window Ensemble Mode

Add the `--enable-window-ensemble` flag when running training:

```bash
python train_auto.py --enable-window-ensemble --dataset "eoinamoore/historical-nba-data-and-player-box-scores" --verbose
```

### First Run (No Cache)

On the first run, all windows will be trained and cached:

```
======================================================================
5-YEAR WINDOW TRAINING (RAM-Efficient Mode)
Data range: 2002-2025
======================================================================

[TRAIN] Window 2002-2006: Cache missing - will train
[TRAIN] Window 2007-2011: Cache missing - will train
[TRAIN] Window 2012-2016: Cache missing - will train
[TRAIN] Window 2017-2021: Cache missing - will train
[TRAIN] Window 2022-2025: Current season - will train

======================================================================
Will process 5 window(s) sequentially to minimize RAM
======================================================================

Training window 1/5: 2002-2006
...
[OK] Window 2002-2006 complete and cached
Memory freed for next window
```

### Subsequent Runs (With Cache)

On subsequent runs, only the current season window is retrained:

```
======================================================================
5-YEAR WINDOW TRAINING (RAM-Efficient Mode)
Data range: 2002-2025
======================================================================

[OK] Window 2002-2006: Valid cache found
[OK] Window 2007-2011: Valid cache found
[OK] Window 2012-2016: Valid cache found
[OK] Window 2017-2021: Valid cache found
[TRAIN] Window 2022-2025: Current season - will train

======================================================================
Will process 1 window(s) sequentially to minimize RAM
======================================================================

Training window 1/1: 2022-2025
...
```

### Adding New Season Data

When new season data arrives (e.g., 2026 season):

1. The system automatically detects the new current season
2. Only the window containing 2026 is retrained
3. All historical windows remain cached
4. No need to retrain from 2002 onwards!

## Cache Storage

### Location

Cached models are stored in the `model_cache/` directory:

```
model_cache/
├── ensemble_2002_2006.pkl          # Trained model
├── ensemble_2002_2006_meta.json    # Metadata for validation
├── ensemble_2007_2011.pkl
├── ensemble_2007_2011_meta.json
├── ...
```

### Metadata Format

Each cache has a JSON metadata file:

```json
{
  "seasons": [2002, 2003, 2004, 2005, 2006],
  "start_year": 2002,
  "end_year": 2006,
  "trained_date": "2025-01-15T10:30:00",
  "num_games": 6150,
  "is_current_season": false
}
```

### Cache Validation

The system validates caches by checking:
- File exists and is not empty
- Metadata file exists
- Metadata contains all expected seasons for the window

If validation fails, the window is automatically retrained.

## Cache Management

### Clearing Cache

To force retraining of all windows, delete the cache directory:

```bash
rm -rf model_cache/
```

Or on Windows:
```powershell
Remove-Item -Recurse -Force model_cache
```

### Selective Cache Clearing

To retrain specific windows, delete their cache files:

```bash
# Retrain 2017-2021 window only
rm model_cache/ensemble_2017_2021.pkl
rm model_cache/ensemble_2017_2021_meta.json
```

### Cache Corruption

If a cache is corrupted:
1. The system will detect it during validation
2. The window will be marked as invalid
3. It will be automatically retrained

## Memory Management

### Sequential Processing

Windows are processed one at a time with explicit memory cleanup:

```python
# After each window completes:
del games_window, result, game_weights
gc.collect()
```

This ensures:
- Only one window's data is in memory at a time
- Python garbage collector frees memory before next window
- RAM usage stays constant regardless of total data size

### Expected RAM Usage

- **Without caching**: Loads all data (2002-2025) → ~8-16GB RAM
- **With caching**: Loads one 5-year window → ~2-4GB RAM

## Testing

Run the simulation to see how the caching logic works:

```bash
python test_window_caching.py
```

This will show which windows would be trained without actually training them.

## Troubleshooting

### Issue: "Cache missing" for all windows on every run

**Cause**: Cache files not being written
**Solution**: Check write permissions for `model_cache/` directory

### Issue: "Cache invalid" warnings

**Cause**: Metadata validation failing
**Solution**: Delete the specific cache files to force retraining

### Issue: Out of memory during training

**Cause**: Individual window too large
**Solution**:
- Reduce `--n-jobs` to use fewer threads
- Consider smaller window sizes (modify `window_size = 5` in code)

### Issue: Current season not retraining

**Cause**: System doesn't detect it as current
**Solution**: Check `season_end_year` in your data matches expected format

## Performance Tips

1. **Keep Cache**: Don't delete cache unless necessary
2. **Monitor Disk Space**: Each cache file is ~100MB-500MB
3. **Use SSD**: Faster cache loading from SSD vs HDD
4. **Parallel Historical**: If RAM permits, could modify to train historical windows in parallel (advanced)

## Technical Details

### Why 5 Years?

- Balances model stability vs recency
- NBA rule changes typically gradual over 5-year periods
- Optimal trade-off between cache size and training time

### Why Retrain Current Season?

- New games added frequently during season
- Player performance evolving
- Ensures predictions use latest data
- Cache still saved for faster reruns

### Cache Key Design

Format: `ensemble_{start_year}_{end_year}.pkl`

- Includes start/end years for clarity
- Easy to identify which windows exist
- Consistent with metadata naming

## Future Enhancements

Potential improvements:
- [ ] Configurable window size via CLI flag
- [ ] Parallel historical window training (if RAM available)
- [ ] Incremental updates within current window (don't retrain all 5 years)
- [ ] Cache compression to reduce disk usage
- [ ] Expiration policy for very old caches
