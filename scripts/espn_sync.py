import requests
import pandas as pd
import json
import os
from datetime import datetime, timedelta

def get_espn_games(date_str):
    """Fetch all game IDs for a specific date from ESPN."""
    # format of date_str: YYYYMMDD
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={date_str}"
    try:
        r = requests.get(url, timeout=20)
        data = r.json()
        events = data.get('events', [])
        return [e.get('id') for e in events]
    except Exception as e:
        print(f"Error fetching ESPN games for {date_str}: {e}")
        return []

def get_espn_boxscore(game_id, date_str):
    """Fetch and parse boxscore for a game."""
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={game_id}"
    try:
        r = requests.get(url, timeout=20)
        if r.status_code != 200:
            print(f"  [WARN] Game {game_id} returned status {r.status_code}")
            return []
            
        try:
            data = r.json()
        except json.JSONDecodeError:
            print(f"  [WARN] Game {game_id} returned invalid JSON")
            return []
            
        box = data.get('boxscore', {})
        players_info = box.get('players', [])
        
        game_logs = []
        for team_box in players_info:
            team_info = team_box.get('team', {})
            team_abbr = team_info.get('abbreviation')
            
            # Find opponent
            opp_box = next((t for t in players_info if t.get('team', {}).get('abbreviation') != team_abbr), None)
            opp_abbr = opp_box.get('team', {}).get('abbreviation') if opp_box else "UNK"
            
            # Matchup string
            # ESPN doesn't easily say who is home/away in the boxscore team object 
            # but we can check the scoreboard or just use vs for now
            matchup = f"{team_abbr} vs. {opp_abbr}"
            
            # Get stat labels and athletes from the main statistics block
            stats_list = team_box.get('statistics', [])
            if not stats_list:
                continue
                
            main_stats_block = stats_list[0]
            labels = main_stats_block.get('names', [])
            athletes = main_stats_block.get('athletes', [])
            
            # Map labels to indices
            label_map = {l: i for i, l in enumerate(labels)}
            
            # Parse players
            for ath_stats_obj in athletes:
                # athletes in this block are usually objects with 'athlete' and 'stats'
                ath = ath_stats_obj.get('athlete', {})
                name = ath.get('displayName')
                stats = ath_stats_obj.get('stats', [])
                
                if not stats or len(stats) < len(labels):
                    continue
                
                # Helper to get stat safe
                def gs(label, get_attempt=False):
                    idx = label_map.get(label)
                    if idx is not None and idx < len(stats):
                        val = stats[idx]
                        if any(s in val for s in ['-', '/']):
                            parts = val.replace('/', '-').split('-')
                            return parts[1] if get_attempt else parts[0]
                        return val
                    return 0

                # Reconstruct into engine format
                log = {
                    'SEASON_ID': '22025',
                    'season_start_year': 2025,
                    'PLAYER_NAME': name,
                    'TEAM_ABBR': team_abbr,
                    'GAME_ID': game_id,
                    'GAME_DATE': f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}",
                    'MATCHUP': matchup,
                    'MIN': gs('MIN'),
                    'PTS': gs('PTS'),
                    'AST': gs('AST'),
                    'REB': gs('REB'),
                    'FG3M': gs('3PT'),
                    'FGA': gs('FG', get_attempt=True),
                    'FG3A': gs('3PT', get_attempt=True),
                    'TOV': gs('TO'),
                    'FTA': gs('FT', get_attempt=True),
                    'WL': 'UNK' 
                }
                game_logs.append(log)
        return game_logs
    except Exception as e:
        print(f"Error parsing ESPN boxscore {game_id}: {e}")
        return []

def run_sync(start_date_str, end_date_str):
    start = datetime.strptime(start_date_str, "%Y%m%d")
    end = datetime.strptime(end_date_str, "%Y%m%d")
    
    all_logs = []
    curr = start
    while curr <= end:
        ds = curr.strftime("%Y%m%d")
        print(f"Syncing {ds} via ESPN...")
        gids = get_espn_games(ds)
        for gid in gids:
            print(f"  Scraping Game {gid}...")
            all_logs.extend(get_espn_boxscore(gid, ds))
        curr += timedelta(days=1)
    
    if not all_logs:
        print("No logs found.")
        return
        
    df_new = pd.DataFrame(all_logs)
    
    # Map IDs
    print("\nMapping IDs from Master Matrix...")
    master_file = "final_feature_matrix_with_per_min_1997_onward.csv"
    if os.path.exists(master_file):
        # We only need player info
        # Smart column detection
        peek = pd.read_csv(master_file, nrows=0)
        p_id_col = next((c for c in ['player_id', 'PLAYER_ID'] if c in peek.columns), 'player_id')
        p_name_col = next((c for c in ['player_name', 'PLAYER_NAME'] if c in peek.columns), 'player_name')
        t_id_col = next((c for c in ['TEAM_ID', 'team_id'] if c in peek.columns), 'TEAM_ID')
        
        # Determine team abbreviation column
        t_abbr_col = next((c for c in ['team', 'TEAM_ABBR', 'TEAM_ABBREVIATION'] if c in peek.columns), None)
        
        cols_to_use = [p_id_col, p_name_col, t_id_col]
        if t_abbr_col:
            cols_to_use.append(t_abbr_col)
        else:
            if 'matchup' in peek.columns:
                cols_to_use.append('matchup')
            elif 'MATCHUP' in peek.columns:
                cols_to_use.append('MATCHUP')
        
        df_m = pd.read_csv(master_file, usecols=cols_to_use, low_memory=False)
        p_map = df_m[[p_id_col, p_name_col]].drop_duplicates().set_index(p_name_col)[p_id_col].to_dict()
        
        if t_abbr_col:
            t_map = df_m[[t_abbr_col, t_id_col]].drop_duplicates().set_index(t_abbr_col)[t_id_col].to_dict()
        else:
            # Derive from matchup (Abbr @ Abbr or Abbr vs. Abbr)
            m_col = 'matchup' if 'matchup' in df_m.columns else 'MATCHUP'
            df_m['derived_team'] = df_m[m_col].str.split(' ').str[0]
            t_map = df_m[['derived_team', t_id_col]].drop_duplicates().set_index('derived_team')[t_id_col].to_dict()
        
        df_new['PLAYER_ID'] = df_new['PLAYER_NAME'].map(p_map)
        
        # Filter to ONLY known NBA teams (filters out All-Star teams like WORLD/STARS)
        df_new = df_new[df_new['TEAM_ABBR'].isin(t_map.keys())].copy()
        df_new['TEAM_ID'] = df_new['TEAM_ABBR'].map(t_map)
    else:
        print("CRITICAL: Master matrix not found. Cannot map IDs.")
        return

    # Save to temp file for feature engine
    temp_file = "data/espn_fallback_logs.csv"
    os.makedirs('data', exist_ok=True)
    df_new.to_csv(temp_file, index=False)
    print(f"Saved {len(df_new)} logs to {temp_file}")
    
    # Now trigger the regular update_feature_matrix logic
    import sys
    sys.path.append(os.getcwd()) # Root is usually where this is run from
    try:
        from update_feature_matrix import update_dataset
    except ImportError:
        # If run from scripts/
        sys.path.append(os.path.dirname(os.getcwd()))
        from update_feature_matrix import update_dataset
        
    master = "final_feature_matrix_with_per_min_1997_onward.csv"
    
    # Prepare update_dataset expects specific naming
    # We want to match the NBA API columns as much as possible
    df_standard = df_new.rename(columns={
        'PLAYER_NAME': 'player_name',
        'GAME_DATE': 'date',
        'TEAM_ABBR': 'team',
        'PTS': 'points',
        'AST': 'assists',
        'REB': 'reboundsTotal',
        'MIN': 'minutes',
        'FG3M': 'three_pointers'
    })
    # If PLAYER_ID is missing (new players), we'll use their name as ID for now or drop them
    # But for unique_key, we NEED something.
    df_standard = df_standard.dropna(subset=['PLAYER_ID', 'GAME_ID'])
    
    std_file = "data/standardized_espn_logs.csv"
    df_standard.to_csv(std_file, index=False)
    
    print(f"Running master matrix update with {len(df_standard)} rows...")
    update_dataset(std_file, master)
    print("DONE.")

if __name__ == "__main__":
    # Sync from Feb 14 up to today
    today = datetime.now().strftime("%Y%m%d")
    run_sync("20260214", today)
