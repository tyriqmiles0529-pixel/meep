import pandas as pd

df = pd.read_csv("simulation_results.csv")

print("=== PARLAY ODDS DISTRIBUTION ===")
print(df["odds"].describe())

print("\n=== SAMPLE PARLAYS ===")
print(df[["player", "target", "odds", "outcome", "pnl"]].head(20).to_string())

# Check how many parlays have odds > 4.00 (+300 American)
high_odds = df[df["odds"] >= 4.0]
print(f"\nParlays with odds >= 4.00 (+300): {len(high_odds)} / {len(df)} ({len(high_odds)/len(df)*100:.1f}%)")

# Check individual leg odds range
print("\n=== ODDS BREAKDOWN ===")
print(f"Min parlay odds: {df['odds'].min():.2f}")
print(f"Max parlay odds: {df['odds'].max():.2f}")
print(f"Mean parlay odds: {df['odds'].mean():.2f}")
