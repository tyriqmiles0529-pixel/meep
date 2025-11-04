import pandas as pd
from pathlib import Path

# Adjust the path if your CSV lives elsewhere
p = Path(r"C:\Users\tmiles11\nba_predictor\TeamStatistics.csv")

use = ["gameId","gameDate","teamId","opponentTeamId","home","teamScore","opponentScore"]

hdr = pd.read_csv(p, nrows=0).columns
df = pd.read_csv(p, low_memory=False, usecols=[c for c in use if c in hdr])

# Parse all dates as UTC to avoid mixed-tz object dtype, then drop tz to get naive datetimes
d_utc = pd.to_datetime(df["gameDate"], errors="coerce", utc=True)
d = d_utc.dt.tz_convert(None)

print("rows:", len(df), "unique gameIds:", df["gameId"].nunique())
print("date dtype:", d.dtype, "nulls:", int(d.isna().sum()))
print("date min/max:", d.min(), d.max())
print("home values:", df["home"].value_counts(dropna=False).head().to_dict())
print("\nhead:\n", df.head(3).to_string(index=False))