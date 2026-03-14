import pandas as pd
import os
import sys
from datetime import datetime
from betting_strategy import BettingStrategy

# --- CONFIGURATION ---
PREDICTIONS_PATH = "predictions/live_ensemble_2025.csv"
HISTORICAL_ODDS_PATH = "historical_data/the_odds_api_historical.csv"
ELIGIBILITY_PATH = "data/eligibility_lookup.csv"
LEDGER_PATH = "betting_ledger.csv"

def load_and_pivot_historical_odds(target_date):
    """Loads historical odds from CSV and pivots to the internal strategy format."""
    if not os.path.exists(HISTORICAL_ODDS_PATH):
        print(f"[ERROR] Historical odds file not found: {HISTORICAL_ODDS_PATH}")
        return pd.DataFrame()
    
    print(f"Loading historical odds from {HISTORICAL_ODDS_PATH}...")
    df = pd.read_csv(HISTORICAL_ODDS_PATH)
    
    # Filter for the specific date
    df_day = df[df['game_date'] == target_date].copy()
    if df_day.empty:
        print(f"[WARN] No historical odds found for {target_date}")
        return pd.DataFrame()

    # Map API market names to internal names
    market_map = {
        'player_points': 'points',
        'player_rebounds': 'rebounds',
        'player_assists': 'assists',
        'player_threes': 'three_pointers'
    }
    df_day['market_clean'] = df_day['market'].map(market_map)
    df_day = df_day.dropna(subset=['market_clean'])

    # Pivot: We usually prefer DraftKings for consistency, or the first available
    # For backfilling/validation, we'll take the mean line/odds or a specific book if available
    # Let's prioritize DraftKings
    df_dk = df_day[df_day['book'] == 'draftkings']
    if df_dk.empty:
        df_final = df_day.drop_duplicates(subset=['player_name', 'market_clean'])
    else:
        df_final = df_dk

    pivoted = []
    for player in df_final['player_name'].unique():
        p_rows = df_final[df_final['player_name'] == player]
        row_dict = {'player_name': player}
        for _, r in p_rows.iterrows():
            m = r['market_clean']
            row_dict[f'line_{m}'] = float(r['line'])
            row_dict[f'odds_{m}'] = int(r['odds'])
        pivoted.append(row_dict)
    
    return pd.DataFrame(pivoted)

def run_phase_j2():
    print("--- Phase J.2: Historical Odds + Edge Filter Integration ---")
    
    # 1. Load Predictions
    if not os.path.exists(PREDICTIONS_PATH):
        print(f"[FAIL] Predictions missing: {PREDICTIONS_PATH}")
        return
    
    preds = pd.read_csv(PREDICTIONS_PATH)
    # Ensure standard Renaming for BettingStrategy
    col_map = {
        'PLAYER_NAME': 'player_name',
        'TEAM_ABBREVIATION': 'playerteamName',
        'pred_PTS': 'pred_points', 
        'pred_AST': 'pred_assists', 
        'pred_REB': 'pred_rebounds', 
        'pred_3PM': 'pred_three_pointers'
    }
    preds = preds.rename(columns=col_map)
    
    # Get the date from the predictions file (assume same date for all rows)
    if 'GAME_DATE' in preds.columns:
        target_date = preds['GAME_DATE'].iloc[0]
    else:
        target_date = datetime.now().strftime("%Y-%m-%d")
    
    print(f"Target Date: {target_date}")

    # 2. Apply Eligibility Filter (Strict >= 20 MPG)
    if os.path.exists(ELIGIBILITY_PATH):
        print(f"Filtering by eligibility (>= 20 MPG) from {ELIGIBILITY_PATH}...")
        lookup = pd.read_csv(ELIGIBILITY_PATH)
        # Filter lookup for target date
        lookup_day = lookup[lookup['game_date_str'] == target_date]
        eligible_players = lookup_day[lookup_day['eligible'] == True]['PLAYER_NAME'].tolist()
        
        initial_count = len(preds)
        preds = preds[preds['player_name'].isin(eligible_players)]
        print(f"  Eligible players: {len(preds)} (Dropped {initial_count - len(preds)})")
    else:
        print("[WARN] Eligibility lookup missing. Proceeding with all players.")

    # 3. Merge with Historical Odds
    odds_pivoted = load_and_pivot_historical_odds(target_date)
    if odds_pivoted.empty:
        print("[FAIL] No odds found for this date. Cannot proceed.")
        return
    
    merged = pd.merge(preds, odds_pivoted, on='player_name', how='inner')
    print(f"Matched {len(merged)} players with odds.")

    # 4. Generate Bets (±1.5 Delta is now inside BettingStrategy)
    bs = BettingStrategy(load_models=False)
    # Bankroll, Conf, etc.
    bets = bs.generate_bets(merged, bankroll=1000.0, confidence_threshold=10, min_ev=0.3)
    
    if bets.empty:
        print("[INFO] No bets met the ±1.5 delta and EV thresholds today.")
    else:
        print(f"Found {len(bets)} qualified bets.")

    # 5. Output Markdown & Ledger
    short_date = datetime.strptime(target_date, "%Y-%m-%d").strftime('%m.%d.%y')
    run_time = datetime.now().strftime("%I:%M %p")
    md_filename = f"{short_date} - Riq's Picks.md"
    
    top_props_df = bs.select_top_props(bets, n=5) if not bets.empty else pd.DataFrame()
    # Generate Round Robins (Size 2, 3, 4)
    rrs = bs.generate_calibrated_round_robins(bets, n_candidates=5) if not bets.empty else []

    with open(md_filename, 'w', encoding='utf-8') as f:
        f.write(f"# Riq's Picks — {short_date} @ {run_time} (Phase J.2)\n\n")
        f.write(f"**Strategy:** ±1.5 Delta Edge Filter | **Odds:** Historical Backfill\n\n")
        
        f.write("## Top Bets Per Prop (Delta ≥ 1.5)\n")
        if bets.empty:
            f.write("None of today's props met the strict delta requirement.\n")
        else:
            for target in bs.targets:
                tgt_bets = top_props_df[top_props_df['target'] == target] if not top_props_df.empty else pd.DataFrame()
                if not tgt_bets.empty:
                    f.write(f"### {target.replace('_', ' ').title()}\n")
                    f.write("| Player | Side | Line | Prediction | Delta | Odds | EV | Stake |\n")
                    f.write("|---|---|---|---|---|---|---|---|\n")
                    for _, row in tgt_bets.iterrows():
                        odds_str = f"+{row['odds']}" if row['odds'] > 0 else f"{row['odds']}"
                        player_display = f"{row['player']} ({row['team']})"
                        f.write(f"| {player_display} | {row['side']} | {row['line']} | {row['prediction']:.2f} | {row['delta']} | {odds_str} | {row['ev']:.2f} | ${row['stake_amt']:.2f} |\n")
                    f.write("\n")

        if rrs:
            f.write("## Round Robin Parlays\n")
            rr_df = pd.DataFrame(rrs)
            for t in rr_df['type'].unique():
                f.write(f"### {t}\n")
                f.write("| Legs | Odds | Prob | EV |\n")
                f.write("|---|---|---|---|\n")
                for _, row in rr_df[rr_df['type'] == t].iterrows():
                    legs_str = "<br>".join(row['legs'])
                    odds_str = f"+{row['combined_odds']}" if row['combined_odds'] > 0 else f"{row['combined_odds']}"
                    f.write(f"| {legs_str} | {odds_str} | {row['combined_prob']:.1%} | {row['ev']:.2f} |\n")
                f.write("\n")

    # 6. Update Ledger
    if not bets.empty:
        ledger_rows = []
        # Add Singles
        for _, row in bets.iterrows():
            o = row['odds']
            imp = 100/(o+100) if o > 0 else abs(o)/(abs(o)+100)
            ledger_rows.append({
                'Run Date': target_date,
                'Player': row['player'],
                'Market': row['target'],
                'Side': row['side'],
                'Odds': o,
                'Implied Prob': round(imp, 4),
                'Model Prob': round(row['win_prob'], 4),
                'Stake Size': round(row['stake_amt'], 2),
                'EV': round(row['ev'], 4),
                'Outcome': 'Pending',
                'Bet Type': 'Single'
            })
        
        # Add Parlays
        for rr in rrs:
            ledger_rows.append({
                'Run Date': target_date,
                'Player': "Parlay", # Multi-player
                'Market': "RR",
                'Side': "Multiple",
                'Odds': rr['combined_odds'],
                'Implied Prob': round(1 / (rr['combined_odds']/100 + 1) if rr['combined_odds'] > 0 else abs(rr['combined_odds'])/(abs(rr['combined_odds'])+100), 4),
                'Model Prob': round(rr['combined_prob'], 4),
                'Stake Size': 0.0, # Strategy for RR sizing can be added later
                'EV': round(rr['ev'], 4),
                'Outcome': 'Pending',
                'Bet Type': rr['type']
            })
        
        new_ledger_df = pd.DataFrame(ledger_rows)
        # Use simple append
        if os.path.exists(LEDGER_PATH):
            new_ledger_df.to_csv(LEDGER_PATH, mode='a', header=False, index=False)
        else:
            new_ledger_df.to_csv(LEDGER_PATH, index=False)

    print(f"[SUCCESS] Phase J.2 run complete. Report: {md_filename}")

if __name__ == "__main__":
    run_phase_j2()
