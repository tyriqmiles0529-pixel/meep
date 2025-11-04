import pickle
import pandas as pd

with open('bets_ledger.pkl', 'rb') as f:
    ledger = pickle.load(f)

bets = ledger['bets'] if isinstance(ledger, dict) else ledger
df = pd.DataFrame(bets)

# Filter to games that should be completed (before Nov 4)
df['game_datetime'] = pd.to_datetime(df['game_date'], utc=True)
cutoff = pd.Timestamp('2025-11-04', tz='UTC')

should_be_completed = df[df['game_datetime'] < cutoff]
unsettled_completable = should_be_completed[should_be_completed['settled'] == False]

print(f'Total predictions: {len(df)}')
print(f'Already settled: {(df["settled"] == True).sum()}')
print()
print(f'Games before Nov 4 (should be completed): {len(should_be_completed)}')
print(f'  - Already settled: {(should_be_completed["settled"] == True).sum()}')
print(f'  - Unsettled (FETCHABLE NOW): {len(unsettled_completable)}')
print()
print(f'Games on/after Nov 4 (not yet played): {len(df[df["game_datetime"] >= cutoff])}')
print()
print('Breakdown by date:')
for date in sorted(df['game_datetime'].dt.date.unique()):
    date_df = df[df['game_datetime'].dt.date == date]
    settled_count = (date_df['settled'] == True).sum()
    total_count = len(date_df)
    print(f'  {date}: {settled_count}/{total_count} settled ({total_count - settled_count} fetchable)')
