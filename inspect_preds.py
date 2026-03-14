import pandas as pd
import os
from betting_strategy import BettingStrategy

# Minimal run to get 'bets'
preds = pd.read_csv("predictions/live_ensemble_2025.csv")
# Use the recently generated bets if possible or simulate a merge
# For now, let's just inspect the predictions and see where we stand
print("Predictions Sample:")
print(preds.head())

# In a real run, run_phase_i merges these. 
# Let's check common favorite lines in the current NBA market.
# Usually, props are centered around -110. -200 is rare for standard props.
