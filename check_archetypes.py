import pandas as pd
from pathlib import Path

# Load data
df = pd.read_csv("final_feature_matrix_with_per_min_1997_onward.csv", usecols=['player_id', 'player_name', 'season_start_year'])
latest_season = df['season_start_year'].max()
print(f"Latest season: {latest_season}")

# Load archetypes
mapping_path = Path("models/archetypes/player_archetypes.csv")
if mapping_path.exists():
    arch_df = pd.read_csv(mapping_path)
    arch_df['player_id'] = arch_df['player_id'].astype(str).str.replace(r'\.0$', '', regex=True)
    archetype_map = arch_df.set_index('player_id')['archetype_id'].to_dict()
else:
    archetype_map = {}

# Check recent players
recent_players = df[df['season_start_year'] == latest_season].copy()
recent_players['player_id_str'] = recent_players['player_id'].astype(str).str.replace(r'\.0$', '', regex=True)
recent_players['has_archetype'] = recent_players['player_id_str'].isin(archetype_map)

print(f"Total entries in recent season: {len(recent_players)}")
print(f"Unique players in recent season: {recent_players['player_name'].nunique()}")
print(f"Players with archetype: {recent_players[recent_players['has_archetype'] == True]['player_name'].nunique()}")
print(f"Players without archetype: {recent_players[recent_players['has_archetype'] == False]['player_name'].nunique()}")

print("\nSample players without archetype:")
print(recent_players[recent_players['has_archetype'] == False][['player_id', 'player_name']].drop_duplicates().head(10))
