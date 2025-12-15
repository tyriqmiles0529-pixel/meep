"""
Team Stats Integration Module
Merges team advanced stats and opponent per game stats with player predictions.
"""
import pandas as pd
import os

def load_team_stats(season=2025):
    """Load team advanced stats and opponent per game stats."""
    
    # Paths to team stats files
    adv_path = f"team_stats_{season}.csv"
    opp_path = f"team_opp_stats_{season}.csv"
    
    team_data = {}
    
    # Load advanced stats (DRtg, Pace, ORtg)
    if os.path.exists(adv_path):
        adv = pd.read_csv(adv_path)
        for _, row in adv.iterrows():
            team = row['Team'].replace('*', '').strip()
            team_data[team] = {
                'opp_drtg': row['DRtg'],
                'opp_ortg': row['ORtg'],
                'opp_pace': row['Pace'],
                'opp_net_rtg': row['NRtg']
            }
        print(f"Loaded advanced stats for {len(team_data)} teams")
    else:
        print(f"Warning: {adv_path} not found")
    
    # Load opponent per game stats
    if os.path.exists(opp_path):
        opp = pd.read_csv(opp_path)
        for _, row in opp.iterrows():
            team = row['Team'].replace('*', '').strip()
            if team in team_data:
                team_data[team].update({
                    'opp_pts_allowed': row['PTS'],
                    'opp_reb_allowed': row['TRB'],
                    'opp_ast_allowed': row['AST'],
                    'opp_3p_allowed': row['3P'],
                    'opp_fg_pct_allowed': row['FG%']
                })
            else:
                team_data[team] = {
                    'opp_pts_allowed': row['PTS'],
                    'opp_reb_allowed': row['TRB'],
                    'opp_ast_allowed': row['AST'],
                    'opp_3p_allowed': row['3P'],
                    'opp_fg_pct_allowed': row['FG%']
                }
        print(f"Loaded opponent per game stats")
    else:
        print(f"Warning: {opp_path} not found")
    
    return team_data

def enrich_predictions(df, team_stats):
    """
    Add team stats columns to prediction dataframe.
    
    Args:
        df: Predictions dataframe with 'opponent' or 'opponentteamName' column
        team_stats: Dict from load_team_stats()
    
    Returns:
        DataFrame with added opponent stats columns
    """
    # Find opponent column
    opp_col = None
    for col in ['opponent', 'opponentteamName', 'opp_team', 'opp']:
        if col in df.columns:
            opp_col = col
            break
    
    if opp_col is None:
        print("Warning: No opponent column found. Returning original df.")
        return df
    
    print(f"Enriching predictions with team stats (using {opp_col})...")
    
    # Create mapping columns
    new_cols = ['opp_drtg', 'opp_ortg', 'opp_pace', 'opp_net_rtg', 
                'opp_pts_allowed', 'opp_reb_allowed', 'opp_ast_allowed', 
                'opp_3p_allowed', 'opp_fg_pct_allowed']
    
    for col in new_cols:
        df[col] = df[opp_col].apply(
            lambda x: team_stats.get(str(x).replace('*', '').strip(), {}).get(col, None)
        )
    
    # Fill missing with league average
    league_avg = {
        'opp_drtg': 114.5,
        'opp_ortg': 114.5,
        'opp_pace': 98.8,
        'opp_net_rtg': 0.0,
        'opp_pts_allowed': 113.8,
        'opp_reb_allowed': 44.1,
        'opp_ast_allowed': 26.5,
        'opp_3p_allowed': 13.5,
        'opp_fg_pct_allowed': 0.467
    }
    
    for col, avg in league_avg.items():
        if col in df.columns:
            df[col] = df[col].fillna(avg)
    
    matched = df[new_cols[0]].notna().sum()
    print(f"Matched {matched}/{len(df)} predictions with team stats")
    
    return df

if __name__ == "__main__":
    # Test loading
    stats = load_team_stats(2025)
    print(f"\nSample team data: {list(stats.items())[:3]}")
