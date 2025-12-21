
import pandas as pd
import json
import os

# Configuration
INPUT_CSV = "predictions/bets_today_2025-12-15.csv"
OUTPUT_JSON = "predictions/bets_today_2025-12-15.json"
OUTPUT_MD = "predictions/bets_today_2025-12-15.md"

def convert_output():
    if not os.path.exists(INPUT_CSV):
        print(f"File {INPUT_CSV} not found.")
        return

    df = pd.read_csv(INPUT_CSV)
    
    # 1. JSON
    df.to_json(OUTPUT_JSON, orient='records', indent=2)
    print(f"Saved JSON to {OUTPUT_JSON}")
    
    # 2. Markdown
    # Clean up formatting for readability
    # Round floats
    cols = ['pred_PTS', 'pred_AST', 'pred_REB']
    for c in cols:
        if c in df.columns:
            df[c] = df[c].round(1)
            
    # Select Columns for readable table
    display_cols = ['PLAYER_NAME', 'TEAM_ABBREVIATION', 'MATCHUP', 'pred_PTS', 'pred_AST', 'pred_REB']
    # Filter only columns that exist
    display_cols = [c for c in display_cols if c in df.columns]
    
    md_table = df[display_cols].to_markdown(index=False)
    
    with open(OUTPUT_MD, 'w') as f:
        f.write(f"# NBA Predictions for {df['GAME_DATE'].iloc[0] if 'GAME_DATE' in df.columns else 'Today'}\n\n")
        f.write(md_table)
        
    print(f"Saved Markdown to {OUTPUT_MD}")

if __name__ == "__main__":
    convert_output()
