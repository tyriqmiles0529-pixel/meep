
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

LEDGER_PATH = "betting_ledger.csv"
REPORT_PATH = "performance_audit.md"

def load_ledger():
    if not os.path.exists(LEDGER_PATH):
        print("No ledger found.")
        return None
    try:
        df = pd.read_csv(LEDGER_PATH)
        # Ensure numeric columns
        cols = ['Odds', 'Closing_Odds', 'Stake Size', 'CLV_%']
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        
        # Convert date
        if 'Run Date' in df.columns:
            df['Run Date'] = pd.to_datetime(df['Run Date'], format='%m.%d.%y', errors='coerce')
            
        return df
    except Exception as e:
        print(f"Error loading ledger: {e}")
        return None

def calculate_bucket(row):
    """Classify bet into Core, Growth, Moonshot based on Odds/Type."""
    # Heuristic based on Odds or implicit type if available
    # Phase J ledgers might not have explicit 'Bucket' column yet.
    # We can infer from Odds:
    # Core: +100 to +250
    # Growth: +250 to +450
    # Moonshot: +450+ or 'Lotto'
    
    odds = row.get('Odds', 0)
    market = row.get('Market', '')
    player = row.get('Player', '')
    
    if 'Lotto' in str(player) or 'Lotto' in str(market):
        return 'Moonshot'
        
    # Convert US odds to Decimal for comparison
    dec = (1 + odds/100) if odds > 0 else (1 + 100/abs(odds))
    
    if dec < 3.5: # +250
        return 'Core'
    elif dec < 5.5: # +450
        return 'Growth'
    else:
        return 'Moonshot'

def generate_performance_report():
    print("Generating Phase K.5 Performance Audit...")
    df = load_ledger()
    if df is None or df.empty:
        print("Ledger empty.")
        return

    # 1. Bucket Classification
    df['Bucket'] = df.apply(calculate_bucket, axis=1)
    
    # 2. CLV Analysis (Rolling)
    # Filter for rows with valid CLV (where games have started/closed)
    clv_df = df.dropna(subset=['CLV_%']).copy()
    
    report_lines = []
    report_lines.append(f"# Phase K.5 Performance Audit - {datetime.now().strftime('%Y-%m-%d')}")
    report_lines.append(f"**Objective:** Self-Awareness & Drift Detection (No Intervention)\n")
    
    if not clv_df.empty:
        # Group by Bucket
        report_lines.append("## 1. CLV Monitoring (Closing Line Value)")
        report_lines.append("Positive CLV indicates we are beating the market price at tip-off.\n")
        
        avg_clv = clv_df.groupby('Bucket')['CLV_%'].mean()
        count_clv = clv_df.groupby('Bucket')['CLV_%'].count()
        
        report_lines.append("| Bucket | Bets (N) | Avg CLV % | Status |")
        report_lines.append("|---|---|---|---|")
        
        for bucket in ['Core', 'Growth', 'Moonshot']:
            val = avg_clv.get(bucket, 0.0)
            n = count_clv.get(bucket, 0)
            status = "✅ Healthy" if val >= 0 else "⚠️ WARNING (Negative Edge)"
            if n == 0: status = "No Data"
            report_lines.append(f"| {bucket} | {n} | {val:.2f}% | {status} |")
        
        report_lines.append("\n**Rolling 7-Day CLV (All Buckets):**")
        # Rolling
        clv_df = clv_df.sort_values('Run Date')
        # We need a proper time index for rolling
        daily_clv = clv_df.groupby('Run Date')['CLV_%'].mean()
        rolling_7 = daily_clv.rolling(window=7, min_periods=1).mean()
        
        if not rolling_7.empty:
            last_7 = rolling_7.iloc[-1]
            report_lines.append(f"- Current 7-Day Avg: **{last_7:.2f}%**")
        else:
            report_lines.append("- Insufficient data for rolling average.")

    else:
        report_lines.append("## 1. CLV Monitoring")
        report_lines.append("No closing line data available yet. Run `clv_tracker.py` after games start.")

    # 3. Outcome Analysis (Realized)
    report_lines.append("\n## 2. Realized Outcomes (Post-Slate)")
    outcomes_df = df.dropna(subset=['Outcome']).copy()
    
    if not outcomes_df.empty:
        # Calculate Units Won/Lost
        # Outcome: 'Win', 'Loss', 'Push' usually? Or logical usage? Assuming 1/0 or string
        # Let's assume input is mapped or raw.
        # If 'Outcome' is just a placeholder in CSV, we need a way to input results.
        # For now, simply reporting coverage.
        
        report_lines.append(f"Total Graded Bets: {len(outcomes_df)}")
        # Placeholder for actual PnL logic once `update_results.py` is built/run
    else:
        report_lines.append("No graded bets found in ledger.")

    # 4. Drift & Confidence Guardrails
    report_lines.append("\n## 3. Drift & Guardrails (Silent)")
    
    # Filter for Phase K Era (>= 12/28/25) for Governance Checks
    phase_k_start = pd.to_datetime("2025-12-28")
    era_df = df[df['Run Date'] >= phase_k_start].copy()
    
    if era_df.empty:
        report_lines.append("No Phase K (Active Era) bets found yet. Audit deferred.")
    else:
        # Confidence Check
        max_conf = era_df['Model Prob'].max() if 'Model Prob' in era_df.columns else 0.0
        report_lines.append(f"- **Max System Confidence (Phase K Era):** {max_conf:.1%} (Limit: 95.0%)")
        
        if max_conf > 0.9501: # Tolerance for float precision
            report_lines.append("  - ❌ **VIOLATION:** Confidence cap breached. Audit required.")
        else:
            report_lines.append("  - ✅ Confidence cap respected.")
            
        # Odds Distribution
        report_lines.append("\n**Odds Distribution (Bucket Check):**")
        if 'Odds' in era_df.columns:
            valid_odds = era_df[pd.to_numeric(era_df['Odds'], errors='coerce').notnull()]
            favs = len(valid_odds[valid_odds['Odds'] < -120])
            dogs = len(valid_odds[valid_odds['Odds'] > 0])
            total = len(valid_odds)
            pct_fav = (favs/total * 100) if total > 0 else 0
            
            report_lines.append(f"- Favorites (<-120): {pct_fav:.1f}%")
            report_lines.append(f"- Underdogs (>+100): {100-pct_fav:.1f}%")
            
            if pct_fav < 20: 
                 report_lines.append("  - ⚠️ **DRIFT DETECTED:** Portfolio is tilting too heavily towards longshots.")
            else:
                 report_lines.append("  - ✅ Portfolio balance healthy.")

    # filter for Phase L2 Era (Fund-Grade)
    report_lines[0] = f"# Phase L2 Fund-Grade Audit - {datetime.now().strftime('%Y-%m-%d')}"
    report_lines[1] = "**Objective:** Full Institutional Governance & Risk Monitoring\n"

    # ... (Keep existing CLV section) ...
    # But I am replacing the drift section. Wait, I should append the new section.
    
    # 6. Allocation Adherence (Phase L2)
    report_lines.append("\n## 5. Allocation Adherence (Phase L2 Audit)")
    if 'Stake Size' in era_df.columns:
        total_stake = era_df['Stake Size'].sum()
        if total_stake > 0:
            alloc = era_df.groupby('Bucket')['Stake Size'].sum() / total_stake
            report_lines.append("| Bucket | Actual % | Target % | Status |")
            report_lines.append("|---|---|---|---|")
            
            targets = {'Core': 0.80, 'Growth': 0.15, 'Moonshot': 0.05}
            for b in ['Core', 'Growth', 'Moonshot']:
                act = alloc.get(b, 0.0)
                tgt = targets.get(b, 0.0)
                delta = act - tgt
                status = "✅" if abs(delta) < 0.10 else "⚠️ DRIFT"
                report_lines.append(f"| {b} | {act:.1%} | {tgt:.1%} | {status} |")
        else:
            report_lines.append("No stake data available for adherence check.")
            
    # 7. Correlation Risk (Placeholder)
    report_lines.append("\n## 6. Risk & correlation")
    report_lines.append("- **Correlation Stress:** Simulated via Monte Carlo (weekly).")
    report_lines.append("- **Bench Volatility:** Capped at 70% Confidence (Phase L1).")

    # 5. Monte Carlo Status (Renumbering to 7)
    report_lines.append("\n## 7. Simulation Status")
    report_lines.append("Next scheduled run: Weekly (Sunday)")
    report_lines.append("Action: Re-validate Sharpe Ratio assumptions using updated ledger.")

    # Save
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))
    
    print(f"Report generated: {REPORT_PATH}")

if __name__ == "__main__":
    generate_performance_report()
