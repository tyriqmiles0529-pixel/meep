"""
Generate Today's Picks with ±1.5 Delta Filter using Live Odds
"""
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats import skewnorm

# Configuration
TARGET_DATE = "2025-12-22"
LIVE_ODDS_FILE = f"historical_data/odds_{TARGET_DATE}.csv"
PREDICTIONS_FILE = "predictions/live_ensemble_2025.csv"
MIN_DELTA = 1.5  # Edge filter threshold

# J.6 Production Calibrated RMSE values (includes 3PM)
RMSES = {'points': 5.2, 'rebounds': 2.4, 'assists': 2.1, 'three_pointers': 0.9}
SKEWS = {'points': 2.0, 'rebounds': 2.5, 'assists': 2.2, 'three_pointers': 1.5}

def calculate_win_prob(pred, line, market):
    """Calculate probability of winning the bet"""
    rmse = RMSES.get(market, 4.5)
    skew_a = SKEWS.get(market, 2.0)
    
    # Probability of Over
    prob_over = 1 - skewnorm.cdf(line, skew_a, loc=pred, scale=rmse)
    prob_under = skewnorm.cdf(line, skew_a, loc=pred, scale=rmse)
    
    return prob_over, prob_under

def calculate_ev(win_prob, odds):
    """Calculate expected value"""
    # Convert to decimal odds
    if odds < 0:
        dec_odds = 1 + (100 / abs(odds))
    else:
        dec_odds = 1 + (odds / 100)
    
    ev = (win_prob * (dec_odds - 1)) - (1 - win_prob)
    return ev

def main():
    print(f"=== Generating Picks for {TARGET_DATE} with ±{MIN_DELTA} Delta Filter ===\n")
    
    # Load live odds
    if not pd.io.common.file_exists(LIVE_ODDS_FILE):
        print(f"ERROR: Live odds file not found: {LIVE_ODDS_FILE}")
        return
    
    odds_df = pd.read_csv(LIVE_ODDS_FILE)
    print(f"Loaded {len(odds_df)} props from live odds")
    
    # Load predictions
    if not pd.io.common.file_exists(PREDICTIONS_FILE):
        print(f"ERROR: Predictions file not found: {PREDICTIONS_FILE}")
        return
    
    preds_df = pd.read_csv(PREDICTIONS_FILE)
    
    # Standardize column names
    col_map = {
        'player': 'player_name',
        'proj_PTS': 'pred_points',
        'proj_AST': 'pred_assists',
        'proj_REB': 'pred_rebounds'
    }
    preds_df = preds_df.rename(columns=col_map)
    
    # Map market names (only include markets we have models for)
    market_map = {
        'Points': 'points',
        'Rebounds': 'rebounds',
        'Assists': 'assists'
    }
    odds_df['market_clean'] = odds_df['market'].map(market_map)
    
    # Merge predictions with odds
    merged = []
    for _, odds_row in odds_df.iterrows():
        player = odds_row['player_name']
        market = odds_row['market_clean']
        
        if market is None:
            continue
        
        # Find prediction for this player
        pred_row = preds_df[preds_df['player_name'] == player]
        if pred_row.empty:
            continue
        
        pred_col = f'pred_{market}'
        if pred_col not in pred_row.columns:
            continue
        
        prediction = pred_row[pred_col].iloc[0]
        line = odds_row['line']
        odds = odds_row['over_odds']
        
        if odds_row['line'] <= 0:
            continue

        # Convert odds to American format if needed
        if odds < 2.0:  # Decimal odds < 2.0 are favorites
            american_odds = -100 / (odds - 1)
        else:
            american_odds = (odds - 1) * 100
        
        delta = prediction - line
        
        # Apply ±1.5 delta filter
        if abs(delta) < MIN_DELTA:
            continue
        
        # Calculate probabilities
        prob_over, prob_under = calculate_win_prob(prediction, line, market)
        
        # Determine side and win probability
        if delta >= MIN_DELTA:
            side = 'OVER'
            win_prob = prob_over
        else:
            side = 'UNDER'
            win_prob = prob_under
        
        # Calculate EV
        ev = calculate_ev(win_prob, american_odds)
        
        # Only include positive EV bets (min 0.3 as requested)
        if ev >= 0.3:
            merged.append({
                'player': player,
                'team': odds_row['team'],
                'opponent': odds_row['opponent'],
                'market': market,
                'line': line,
                'prediction': prediction,
                'delta': round(delta, 2),
                'side': side,
                'odds': int(american_odds),
                'win_prob': win_prob,
                'ev': ev
            })
    
    bets_df = pd.DataFrame(merged)
    
    if bets_df.empty:
        print(f"No bets met the ±{MIN_DELTA} delta and positive EV threshold")
        # Write empty markdown
        short_date = datetime.strptime(TARGET_DATE, "%Y-%m-%d").strftime('%m.%d.%y')
        run_time = datetime.now().strftime("%I:%M %p")
        md_filename = f"{short_date} - Riq's Picks.md"
        
        with open(md_filename, 'w', encoding='utf-8') as f:
            f.write(f"# Riq's Picks — {short_date} @ {run_time}\\n\\n")
            f.write(f"**Strategy:** ±{MIN_DELTA} Delta Edge Filter | **Odds:** Live RapidAPI\\n\\n")
            f.write(f"## Top Bets Per Prop (Delta ≥ {MIN_DELTA})\\n")
            f.write("None of today's props met the strict delta requirement.\\n")
        
        print(f"Empty report written to {md_filename}")
        return
    
    # Sort by EV
    bets_df = bets_df.sort_values('ev', ascending=False)
    
    print(f"\\nFound {len(bets_df)} qualifying bets!")
    print("\\nTop 10 Bets by EV:")
    print(bets_df.head(10)[['player', 'market', 'side', 'line', 'prediction', 'delta', 'ev']])
    
    # Generate Markdown Report
    short_date = datetime.strptime(TARGET_DATE, "%Y-%m-%d").strftime('%m.%d.%y')
    run_time = datetime.now().strftime("%I:%M %p")
    md_filename = f"{short_date} - Riq's Picks.md"
    
    with open(md_filename, 'w', encoding='utf-8') as f:
        f.write(f"# Riq's Picks — {short_date} @ {run_time}\\n\\n")
        f.write(f"**Strategy:** ±{MIN_DELTA} Delta Edge Filter | **Odds:** Live RapidAPI\\n\\n")
        
        f.write(f"## Top Bets Per Prop (Delta ≥ {MIN_DELTA})\\n\\n")
        
        # Group by market (only the three we have models for)
        for market in ['points', 'rebounds', 'assists']:
            market_bets = bets_df[bets_df['market'] == market].head(5)
            
            if not market_bets.empty:
                f.write(f"### {market.replace('_', ' ').title()}\\n\\n")
                f.write("| Player | Team | Side | Line | Prediction | Delta | Odds | Win Prob | EV |\\n")
                f.write("|--------|------|------|------|------------|-------|------|----------|-----|\\n")
                
                for _, bet in market_bets.iterrows():
                    odds_str = f"+{bet['odds']}" if bet['odds'] > 0 else str(int(bet['odds']))
                    f.write(f"| {bet['player']} | {bet['team']} | {bet['side']} | {bet['line']} | "
                           f"{bet['prediction']:.1f} | {bet['delta']:+.1f} | {odds_str} | "
                           f"{bet['win_prob']:.1%} | {bet['ev']:.3f} |\\n")
                
                f.write("\\n")
        
        f.write("---\\n\\n")
        f.write(f"**Total Qualifying Bets:** {len(bets_df)}\\n\\n")
        f.write(f"*Generated on {datetime.now().strftime('%Y-%m-%d at %I:%M %p')}*\\n")
    
    print(f"\\n[SUCCESS] Report generated: {md_filename}")

if __name__ == "__main__":
    main()
