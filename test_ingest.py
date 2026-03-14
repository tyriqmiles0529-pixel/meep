import pandas as pd
import re

df = pd.read_csv('betting_ledger.csv')
recent = df.tail(30)
for _, row in recent.iterrows():
    side_raw = str(row['Side'])
    if "Parlay" in str(row['Player']) or "Round Robin" in str(row['Player']) or "RR (" in str(row['Player']) or "Lotto" in str(row['Player']):
        continue

    player_raw = str(row['Player'])
    market_raw = str(row['Market'])

    # Format 1: "Josh Giddey (CHI) - rebounds Over 6.5 (-146.0)"
    match1 = re.search(r'^(.*?)\s*\(([A-Z]{2,4})\)\s*-\s*([\w\s]+)\s+(Over|Under)\s+([\d.]+)\s+\((.*?)\)', side_raw, re.I)
    match2 = re.search(r'^(.*?)\s*\(([A-Z]{2,4})\)\s*\((.*?)\s+(Over|Under)\s+@\s+(.*?)\)', side_raw, re.I) if not match1 else None
    
    final_match = match1 or match2
    if final_match:
        p_name = final_match.group(1).strip()
        team_abbr = final_match.group(2).strip()
        
        if match1:
            prop_details = final_match.group(3).strip().lower()
            side = final_match.group(4).upper()
            line = float(final_match.group(5))
            odds_str = final_match.group(6)
        else:
            prop_details = final_match.group(3).strip().lower()
            side = final_match.group(4).upper()
            odds_str = final_match.group(5)
            line = 0.5 

        if prop_details:
            line_m = re.search(r'([\d.]+)$', prop_details)
            if line_m:
                line = float(line_m.group(1))
                prop_details = prop_details[:line_m.start()].strip()
        
        prop_type = prop_details
        print(f"MATCH: {p_name} | {team_abbr} | {prop_type} | {line} | {side}")
    else:
        print(f"NO MATCH: {player_raw} - {side_raw}")

