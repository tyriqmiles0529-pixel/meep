import pandas as pd
import numpy as np

master_file = 'final_feature_matrix_with_per_min_1997_onward.csv'
print(f"Cleaning {master_file}...")

# Load Master
df = pd.read_csv(master_file, low_memory=False)

def unify_column(df, target, variants):
    """Safely unify multiple potential column names into one target name."""
    cols_to_pull = [v for v in variants if v in df.columns]
    if not cols_to_pull:
        return df
    
    # Start with the first available column
    new_series = df[cols_to_pull[0]].copy()
    # Fill from the rest
    for col in cols_to_pull[1:]:
        new_series = new_series.fillna(df[col])
    
    df[target] = new_series
    
    # Drop variants only if they aren't the target itself
    cols_to_drop = [v for v in cols_to_pull if v != target]
    df.drop(columns=cols_to_drop, inplace=True)
    return df

# Apply unification
df = unify_column(df, 'player_id', ['PLAYER_ID', 'player_id', 'personId'])
df = unify_column(df, 'player_name', ['PLAYER_NAME', 'player_name', 'player'])
df = unify_column(df, 'team', ['TEAM_ABBREVIATION', 'team', 'TEAM_ABBR'])
df = unify_column(df, 'matchup', ['MATCHUP', 'matchup'])
df = unify_column(df, 'gameDate', ['GAME_DATE', 'date', 'gameDate', 'game_date'])

# Final Formatting
if 'player_id' in df.columns:
    df['player_id'] = pd.to_numeric(df['player_id'], errors='coerce').fillna(0).astype(int).astype(str)

if 'gameDate' in df.columns:
    df['gameDate'] = pd.to_datetime(df['gameDate'], errors='coerce', format='mixed').dt.strftime('%Y-%m-%d')

# Sort
if 'gameDate' in df.columns and 'player_name' in df.columns:
    df = df.sort_values(['gameDate', 'player_name'])

# Save back
df.to_csv(master_file, index=False)
print(f"Cleanup complete. Columns: {list(df.columns[:10])}")
