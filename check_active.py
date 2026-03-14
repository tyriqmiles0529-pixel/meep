import pandas as pd
from meep_terminal.core.engine import TerminalEngine
engine = TerminalEngine()
latest_season = engine.engine.aggregated_data['season_start_year'].max()
print(f"Latest season: {latest_season}")
players_this_season = engine.engine.aggregated_data[engine.engine.aggregated_data['season_start_year'] == latest_season]['player_name'].unique()
print(f"Count of active players: {len(players_this_season)}")
