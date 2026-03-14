import pandas as pd
import numpy as np
import os

LEDGER_FILE = "betting_ledger.csv"
STRICT_FEATURES = "data/strict_features_v4.csv"

def validate_ledger():
    print("=== Phase J.4.4: Ledger Validation (Dry Run) ===")
    
    if not os.path.exists(LEDGER_FILE):
        print("No ledger found.")
        return

    # 1. Load Ledger (handle potential corruption)
    try:
        ledger = pd.read_csv(LEDGER_FILE, on_bad_lines='skip')
    except Exception as e:
        print(f"Error reading ledger: {e}")
        return
        
    print(f"Loaded {len(ledger)} bets from ledger.")
    
    # 2. Extract Run Dates and Players to fetch outcomes
    # Ledger 'Run Date' is MM.DD.YY. Needs conversion to YYYY-MM-DD for strict matrix.
    def convert_date(d):
        try:
            return pd.to_datetime(d, format='%m.%d.%y').strftime('%Y-%m-%d')
        except:
            return None
            
    ledger['game_date'] = ledger['Run Date'].apply(convert_date)
    ledger = ledger.dropna(subset=['game_date'])
    
    # 3. Fetch Outcomes from Strict Matrix
    print("Fetching outcomes for ledger bets...")
    players = set(ledger['Player'].str.upper())
    dates = set(ledger['game_date'])
    
    outcomes = []
    chunk_size = 100000
    for chunk in pd.read_csv(STRICT_FEATURES, chunksize=chunk_size, usecols=['PLAYER_NAME', 'GAME_DATE', 'PTS', 'AST', 'REB'], skipinitialspace=True):
        mask = chunk['PLAYER_NAME'].str.upper().isin(players) & chunk['GAME_DATE'].isin(dates)
        if mask.any():
            outcomes.append(chunk[mask])
            
    if not outcomes:
        print("No outcomes matched in strict matrix.")
        return
        
    df_outcomes = pd.concat(outcomes).drop_duplicates(subset=['PLAYER_NAME', 'GAME_DATE'])
    df_outcomes['player_key'] = df_outcomes['PLAYER_NAME'].str.upper()
    
    # 4. Join
    ledger['player_key'] = ledger['Player'].str.upper()
    df = pd.merge(ledger, df_outcomes, left_on=['player_key', 'game_date'], right_on=['player_key', 'GAME_DATE'])
    
    # 5. Determine if bet won
    # Market names in ledger: 'points', 'rebounds', 'assists'
    market_col_map = {'points': 'PTS', 'rebounds': 'REB', 'assists': 'AST'}
    
    def check_result(row):
        target = row['Market'].lower()
        if target not in market_col_map: return None
        
        actual = row[market_col_map[target]]
        # We need the 'Line' but it's not in the ledger?
        # WAIT. I need to check if the ledger has 'Line'.
        # Ledger headers: Run Date,Player,Market,Side,Odds,Implied Prob,Model Prob,Stake Size,EV,Odds Source,Outcome
        # THE LINE IS MISSING FROM THE LEDGER HEADERS I SAW.
        return None
        
    print("\n[CRITICAL] Ledger is missing 'Line' column. Cannot compute historical hit rate without it.")
    print("Switching to Model Probability vs Realized frequency (Brier Score style).")
    
if __name__ == "__main__":
    validate_ledger()
