import pandas as pd

print("Checking if advanced stats are matched to players in Parquet file...")
df = pd.read_parquet('aggregated_nba_data.parquet')

print(f'Total rows: {len(df):,}')

# Get advanced stat columns
adv_cols = [c for c in df.columns if c.startswith('adv_')]
print(f'\nAdvanced stat columns: {len(adv_cols)}')
print(f'Examples: {adv_cols[:10]}')

# Check a sample
print(f'\nSample of advanced stats (first 10 rows):')
sample = df[['firstName', 'lastName', 'points', 'adv_per', 'adv_ts_percent', 'adv_bpm']].head(10)
print(sample.to_string())

# Check null rates
print(f'\n\nChecking null rates for advanced stats:')
for col in ['adv_per', 'adv_ts_percent', 'adv_bpm', 'adv_vorp', 'adv_ws']:
    if col in df.columns:
        null_rate = df[col].isna().mean() * 100
        print(f'  {col}: {null_rate:.1f}% null')

# Check a specific player across multiple games
print('\n\nChecking LeBron James across games (are his advanced stats consistent?):')
lebron = df[(df['firstName'] == 'LeBron') & (df['lastName'] == 'James')].head(10)
if len(lebron) > 0:
    print(lebron[['gameDate', 'points', 'assists', 'adv_per', 'adv_ts_percent', 'adv_bpm']].to_string())
else:
    print("LeBron James not found (might use different name format)")
