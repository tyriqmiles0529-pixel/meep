from data_processor import BasketballDataProcessor

p = BasketballDataProcessor("final_feature_matrix_with_per_min_1997_onward.csv")
p.load_data(nrows=10000)
p.preprocess(target="points")

print("=== NEW FEATURES SAMPLE ===")
new_features = ['days_rest', 'is_back_to_back', 'games_into_season', 'season_pct']
opp_features = ['opp_def_points', 'opp_def_rebounds', 'matchup_pts_adj']

# Check which features exist
found = [f for f in new_features + opp_features if f in p.df.columns]
missing = [f for f in new_features + opp_features if f not in p.df.columns]

print(f"Found features: {found}")
print(f"Missing features: {missing}")

if found:
    print()
    print(p.df[found].head(10).to_string())
    print()
    print(p.df[found].describe().to_string())
