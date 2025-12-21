import pandas as pd
import numpy as np
import os
import joblib
try:
    import torch
    from ft_transformer import FTTransformer
except ImportError:
    torch = None
    FTTransformer = None
    print("Warning: Torch not found. Embedding generation disabled.")

class StrictFeatureEngine:
    def __init__(self, data):
        self.data_source = data
        self.df = None
        self.processed_df = None

    def load_and_clean(self):
        if isinstance(self.data_source, str):
            print(f"Loading {self.data_source}...")
            self.df = pd.read_csv(self.data_source)
        else:
            print("Using provided DataFrame...")
            self.df = self.data_source.copy()
        
        # Ensure UTC Date
        self.df['GAME_DATE'] = pd.to_datetime(self.df['GAME_DATE'])
        self.df = self.df.sort_values(['GAME_DATE', 'GAME_ID'])
        
        # Filter Garbage (but be careful not to drop Inference rows which have 0/NaN stats)
        # Inference rows have SEASON_ID='22025', MIN=0.
        # We must NOT drop them.
        # Original logic: self.df = self.df.dropna(subset=['PTS', 'MIN'])
        # If MIN is 0, it might be dropped if we are strict?
        # But inference rows have MIN=0 placeholder. 
        # Actually inference rows will have NaN PTS? In `daily_inference` we set PTS=0.
        # So dropna might be safe if we rely on valid history.
        # BUT we shouldn't drop the target inference rows.
        # Let's drop only if 'PTS' is NaN.
        # In `daily_inference` we fill PTS=0. 
        # But wait, logic: `dropna(subset=['PTS', 'MIN'])`.
        # If I have historic rows with NaN, drop them.
        # Inference rows have filled 0.
        # So it should be fine.
        
        self.df = self.df.dropna(subset=['PTS', 'MIN'])
        self.df['MIN'] = self.df['MIN'].astype(str).apply(self._parse_minutes)
        print(f"Loaded {len(self.df)} rows.")

    def _parse_minutes(self, min_str):
        if ':' in min_str:
            parts = min_str.split(':')
            return float(parts[0]) + float(parts[1])/60.0
        try:
            return float(min_str)
        except:
            return 0.0

    def compute_rolling_stats(self, windows=[3, 5, 10]):
        """
        CRITICAL: We group by Player, Shift(1), THEN Rolling.
        This ensures Game T's features only know about T-1, T-2...
        """
        print("Computing Rolling Stats...")
        self.df = self.df.sort_values(['PLAYER_ID', 'GAME_DATE'])
        
        metrics = ['PTS', 'AST', 'REB', 'MIN', 'FGA', 'FG3A']
        
        for m in metrics:
            # 1. Shift first to move T to T+1 (so T sees T-1)
            # Row T gets value of T-1.
            shifted_col = self.df.groupby('PLAYER_ID')[m].shift(1)
            
            for w in windows:
                # 2. Rolling on the shifted column
                # min_periods=1 allows early season data (expanding-ish)
                self.df[f'roll_{m}_{w}'] = shifted_col.rolling(window=w, min_periods=1).mean()
                
            # Expanding Season Mean using same shifted logic
            # Group by [Player, Season] -> Shift(1) -> Expanding Mean
            self.df[f'season_{m}_avg'] = self.df.groupby(['PLAYER_ID', 'SEASON_ID'])[m]\
                                              .apply(lambda x: x.shift(1).expanding().mean())\
                                              .reset_index(level=[0,1], drop=True)
                                              
        print(f"Rolling stats computed for {metrics}")

    def compute_rest_days(self):
        print("Computing Rest Days...")
        # Sort is already guaranteed
        self.df['prev_date'] = self.df.groupby('PLAYER_ID')['GAME_DATE'].shift(1)
        self.df['rest_days'] = (self.df['GAME_DATE'] - self.df['prev_date']).dt.days
        self.df['rest_days'] = self.df['rest_days'].fillna(7).clip(0, 14) # Fill first game with 7 days rest
        self.df['is_b2b'] = (self.df['rest_days'] == 1).astype(int)
        
    def add_lag_features(self):
        print("Adding Lags...")
        metrics = ['PTS', 'AST', 'REB', 'MIN']
        for m in metrics:
            self.df[f'lag_{m}_1'] = self.df.groupby('PLAYER_ID')[m].shift(1).fillna(0)
            self.df[f'lag_{m}_2'] = self.df.groupby('PLAYER_ID')[m].shift(2).fillna(0)
            
    def compute_opponent_strength(self):
        print("Computing Opponent Defensive Strength...")
        # 1. Create a Team-Level Game Log (One row per Team-Game)
        # We need to aggregate stats allowed by the OPPONENT.
        # But 'df' is player level.
        # First, let's get unique team games to calculate what THEY scored.
        # Actually, simpler: For each game, we know the MATCHUP.
        # We need to calculate: For Team A, what is the avg PTS allowed to opponents in previous games?
        
        # Unique Game-Team rows
        # We have TEAM_ID and GAME_ID.
        team_games = self.df.groupby(['TEAM_ID', 'GAME_ID', 'GAME_DATE', 'MATCHUP']).agg({
             'PTS': 'sum', # Total points scored BY this team
             'FGA': 'sum',
             'FG_PCT': 'mean' # approximate
        }).reset_index()
        
        # Now, we need to link this to the OPPONENT.
        # self.df has 'MATCHUP'. We need to parse it to find Opponent Team ID?
        # Or usually there are exactly 2 teams per GAME_ID.
        
        # Self-join on GAME_ID where Team != Team
        game_summary = team_games.merge(
            team_games, 
            on='GAME_ID', 
            suffixes=('', '_OPP')
        )
        game_summary = game_summary[game_summary['TEAM_ID'] != game_summary['TEAM_ID_OPP']]
        
        # Now check: TEAM_ID vs TEAM_ID_OPP
        # PTS_OPP is points scored BY the opponent (which is Points ALLOWED by TEAM_ID).
        # We want to know: How many points does TEAM_ID allow? -> PTS_OPP.
        
        # Strict Causality:
        # For TEAM_ID at Date T, we want average(PTS_OPP) from T-1, T-2...
        
        game_summary = game_summary.sort_values(['TEAM_ID', 'GAME_DATE'])
        
        # Calculate Rolling Points Allowed
        windows = [5, 10, 20]
        # Shift(1) first!
        metric = 'PTS_OPP'
        shifted_col = game_summary.groupby('TEAM_ID')[metric].shift(1)
        
        for w in windows:
            game_summary[f'opp_allow_pts_roll_{w}'] = shifted_col.rolling(window=w, min_periods=1).mean()
            
        # Select columns to merge back
        # We want to merge these 'opp_allow' stats into the MAIN dataframe.
        # In main df: We have a row for Player P (Team A) vs Opponent B.
        # We want Opponent B's defensive stats.
        # So we merge on [Opponent_ID, Game_ID]? 
        # But main df doesn't have Opponent_ID explicit column usually, check?
        # We can merge on GAME_ID.
        # For Player P (Team A), we want the row from game_summary where TEAM_ID == Team B (Opponent).
        # Easier: In game_summary, 'TEAM_ID' is the 'Subject' team.
        # If we have 'TEAM_ID_OPP' in main df, we merge on that.
        
        # Let's add Opponent ID to main df
        # We can extract it from the same game_summary logic
        # Map (Game_ID, Team_ID) -> Opponent_ID
        matchup_map = game_summary[['GAME_ID', 'TEAM_ID', 'TEAM_ID_OPP']].copy()
        
        self.df = self.df.merge(matchup_map, on=['GAME_ID', 'TEAM_ID'], how='left')
        
        # Now merge the stats
        # We want stats for TEAM_ID_OPP
        stats_to_merge = game_summary[['GAME_ID', 'TEAM_ID'] + [f'opp_allow_pts_roll_{w}' for w in windows]]
        
        self.df = self.df.merge(
            stats_to_merge,
            left_on=['GAME_ID', 'TEAM_ID_OPP'],
            right_on=['GAME_ID', 'TEAM_ID'],
            how='left',
            suffixes=('', '_trash')
        )
        
        # Cleanup
        self.df = self.df.drop(columns=['TEAM_ID_trash'])
        print("Computed Opponent Strength (Pts Allowed).")

    def compute_advanced_rolling_stats(self):
        print("Computing Advanced Rolling Stats (EWMA)...")
        # Exponential Weighted Moving Average (EWMA)
        # Shift(1) must still be applied first!
        self.df = self.df.sort_values(['PLAYER_ID', 'GAME_DATE'])
        
        metrics = ['PTS', 'AST', 'REB']
        span_short = 3
        span_med = 10
        
        for m in metrics:
            shifted = self.df.groupby('PLAYER_ID')[m].shift(1)
            # EWMA
            self.df[f'ewma_{m}_{span_short}'] = shifted.ewm(span=span_short, adjust=False, min_periods=1).mean()
            self.df[f'ewma_{m}_{span_med}'] = shifted.ewm(span=span_med, adjust=False, min_periods=1).mean()
            
            # Variance (Consistency)
            self.df[f'roll_{m}_std_10'] = shifted.rolling(window=10, min_periods=3).std()

    def compute_advanced_player_metrics(self):
        print("Computing Advanced Player Metrics (TS%, Usage, Roles)...")
        # Need to handle division by zero safely
        
        # 1. Rolling TS% (True Shooting)
        # TS% = PTS / (2 * (FGA + 0.44 * FTA))
        # We need Rolling PTS, Rolling FGA, Rolling FTA.
        # Shift(1) first
        grouped = self.df.groupby('PLAYER_ID')
        s_pts = grouped['PTS'].shift(1)
        s_fga = grouped['FGA'].shift(1)
        s_fta = grouped['FTA'].shift(1)
        
        # Rolling Sums for 10 games to stabilize
        w = 10
        r_pts = s_pts.rolling(w, min_periods=1).sum()
        r_fga = s_fga.rolling(w, min_periods=1).sum()
        r_fta = s_fta.rolling(w, min_periods=1).sum()
        
        tsa = 2 * (r_fga + 0.44 * r_fta)
        # Avoid division by zero
        self.df['roll_TS_pct_10'] = (r_pts / tsa).replace([np.inf, -np.inf], 0).fillna(0)
        
        # 2. Rolling Usage Proxy
        # Usage ~ (FGA + 0.44*FTA + TOV)
        # We don't have Team Totals easily at player row level without heavy merge.
        # Proxy: Player's FGA+TOV relative to their own history? 
        # Or just Raw Volume Metric: FGA + TOV + 0.44*FTA
        s_tov = grouped['TOV'].shift(1)
        r_tov = s_tov.rolling(w, min_periods=1).sum()
        
        self.df['roll_usage_proxy_10'] = (r_fga + 0.44 * r_fta + r_tov)
        
        # 3. Role / Minutes Percentile
        # Is this player trending up in minutes?
        # Current Rolling 5 vs Rolling 20
        s_min = grouped['MIN'].shift(1)
        r_min_5 = s_min.rolling(5, min_periods=1).mean()
        r_min_20 = s_min.rolling(20, min_periods=1).mean()
        
        self.df['roll_MIN_5'] = r_min_5 # Ensure this exists
        self.df['role_trend_min'] = (r_min_5 - r_min_20).fillna(0)
        
        # 4. AST/TO Ratio
        # Rolling Sums
        r_ast = grouped['AST'].shift(1).rolling(w, min_periods=1).sum()
        self.df['roll_AST_TO_10'] = (r_ast / r_tov).replace([np.inf, -np.inf], 0).fillna(0)

    def compute_contextual_features(self):
        print("Computing Contextual Features (Home/Away, Rest)...")
        # Home/Away
        # MATCHUP column: "PHX vs. DET" (Home) or "PHX @ DET" (Away)
        # If contains '@', it's Away?
        # Usually: "Team @ Opp" = Road. "Team vs Opp" = Home.
        # Let's verify standard NBA API format.
        # Yes: @ is road, vs. is home.
        
        self.df['is_home'] = self.df['MATCHUP'].str.contains('vs.').astype(int)
        
        # Rest Days already done in `compute_rest_days`
        # Add Opponent Position metrics?
        # We need Player Position first. Do we have it?
        # Dataset columns don't show Position.
        # If no Position, we can't do Position-Specific Defense.
        # Skipping Opponent-Position for now (Complexity/Missing Data).
        pass

    def compute_opponent_strength_advanced(self):
        print("Computing Advanced Opponent Metrics (Pace, Efficiency)...")
        # Reuse team aggregation if possible.
        pass

    def compute_availability_features(self):
        print("Computing Injury/Availability Features (Teammate Missing Usage)...")
        # Logic:
        # 1. Calculate each player's Rolling Season Avg PTS/Usage (as proxy for importance).
        # 2. For each Game+Team, calculate "Total Available Avg PTS" and "Total Missing Avg PTS".
        # 3. For a specific player, feature = "Sum of Avg PTS of Teammates who are OUT".
        
        # Ensure we have `season_PTS_avg` (expanding mean).
        if 'season_PTS_avg' not in self.df.columns:
            # Recompute if needed or rely on order. 
            # It should be there from previous steps.
            pass
            
        # We need a robust "Who Played This Game" map.
        # self.df contains only players who PLAYED (Min > 0).
        # So "Missing" = Roster - Played.
        # Problem: We don't have full Roster history.
        # Approximation: "Top 8 Rotation Players".
        # If a "Top 8" player (by Avg PTS) is not in the current Game's rows for that Team, they are OUT.
        
        # Step A: Identify "Rotation Players" per Team/Season
        # We can look at the whole season to find the "Main Cast" or do it dynamically.
        # Dynamic: Look at rolling 20 team games. Identify list of players who played >10 mins avg.
        # Simple/Fast: For each Game, look at the Team's previous 10 games. Collect unique players.
        # Identify "High Impact" players (e.g. >10 avg pts).
        # Check if they are in current game.
        
        # This is computationally heavy for pandas groupby.
        # Vectorized Approach:
        # 1. Work on Team-Game level.
        # 2. Agg: List of Player_IDs present in Game G, Team T.
        # 3. Join with Previous Games to find who *usually* plays.
        
        # Let's simplify: "Team Total Season Avg PTS" vs "Current Game Available Season Avg PTS"
        # 1. Calc `season_PTS_avg` for every row.
        # 2. Group by GameID, TeamID -> Sum of `season_PTS_avg` = `team_available_avg_pts`.
        # 3. We assume `team_available_avg_pts` fluctuates based on who plays.
        # 4. We need a baseline "Expected Playing Total".
        #    e.g. Rolling Mean of `team_available_avg_pts` over last 5 games?
        #    If `team_available_avg_pts` drops by 25, it means a ~25ppg player is missing.
        
        # Implementation:
        # 1. Sum up `season_PTS_avg` for all players in the game (Team Total Potential).
        # 2. Feature = `team_avail_pts`.
        # 3. Feature = `roll_team_avail_pts_10` (What is normal for this team?).
        # 4. Feature = `missing_production` = `roll_team_avail_pts_10` - `team_avail_pts`.
        
        # This captures "LeBron is missing" without needing to know it's LeBron specifically,
        # just that the "Typical Total Talent" is lower.
        
        # Group by Game, Team
        team_game_stats = self.df.groupby(['GAME_ID', 'TEAM_ID'])['season_PTS_avg'].sum().reset_index()
        team_game_stats.rename(columns={'season_PTS_avg': 'team_avail_pts'}, inplace=True)
        
        # We need to compute rolling avg of `team_avail_pts` per TEAM (strictly past).
        # Add Date for sorting
        game_dates = self.df[['GAME_ID', 'GAME_DATE']].drop_duplicates()
        team_game_stats = team_game_stats.merge(game_dates, on='GAME_ID', how='left')
        team_game_stats = team_game_stats.sort_values(['TEAM_ID', 'GAME_DATE'])
        
        # Shift 1 to ensure we expect what *was* normal, not including today (if today is weird).
        # Actually we compare Today's Sum vs Rolling Sum of Previous.
        # Rolling mean of LAST 10 games' available points.
        team_game_stats['roll_team_avail_pts_10'] = team_game_stats.groupby('TEAM_ID')['team_avail_pts'].shift(1).rolling(10, min_periods=1).mean()
        
        # Missing Production metric
        # If Rolling is 110, and Today is 80, Missing = 30.
        team_game_stats['missing_high_impact_production'] = (team_game_stats['roll_team_avail_pts_10'] - team_game_stats['team_avail_pts']).fillna(0)
        
        # Clip negative (sometimes available is higher than average -> full squad back)
        # Actually positive delta means "More available than usual".
        # We want "Missingness". 
        # Missing = Normal - Current. 
        # If Missing > 0, we are missing people. 
        # If Missing < 0, we have more people than avg (e.g. injuries healing).
        
        # Merge back to Main DF
        self.df = self.df.merge(team_game_stats[['GAME_ID', 'TEAM_ID', 'missing_high_impact_production']], on=['GAME_ID', 'TEAM_ID'], how='left')
        print(" Availability features attached.")

    def compute_per_minute_features(self):
        print("Computing Per-Minute Efficiency Features...")
        self.df = self.df.sort_values(['PLAYER_ID', 'GAME_DATE'])
        
        # We need Rolling Sums of Numerators and Denominators to avoid noisy single-game spikes
        grouped = self.df.groupby('PLAYER_ID')
        w = 10
        
        # Helper: Rolling Sum Shifted
        def roll_sum_shifted(col):
            return grouped[col].shift(1).rolling(w, min_periods=1).sum()
            
        r_pts = roll_sum_shifted('PTS')
        r_ast = roll_sum_shifted('AST')
        r_reb = roll_sum_shifted('REB')
        r_min = roll_sum_shifted('MIN')
        
        # Avoid zero division
        r_min_safe = r_min.replace(0, np.nan)
        
        self.df['roll_PTS_per_MIN_10'] = (r_pts / r_min_safe).fillna(0)
        self.df['roll_AST_per_MIN_10'] = (r_ast / r_min_safe).fillna(0)
        self.df['roll_REB_per_MIN_10'] = (r_reb / r_min_safe).fillna(0)
        
        # Usage Density: (FGA + 0.44*FTA + TOV) / MIN
        # We need sum(FGA), sum(FTA), sum(TOV)
        r_fga = roll_sum_shifted('FGA')
        r_fta = roll_sum_shifted('FTA')
        r_tov = roll_sum_shifted('TOV')
        
        usage_num = r_fga + 0.44 * r_fta + r_tov
        self.df['roll_usage_density_10'] = (usage_num / r_min_safe).fillna(0)
        
    def merge_embeddings(self, embedding_path):
        if not os.path.exists(embedding_path):
            print(f"Warning: Embeddings not found at {embedding_path}. Skipping.")
            return
            
        print(f"Merging Embeddings from {embedding_path}...")
        emb_df = pd.read_parquet(embedding_path)
        
        # Merge on GAME_ID, PLAYER_ID
        # Ensure types (sometimes IDs are int vs str)
        # self.df IDs are usually int (from CSV) or whatever pandas inferred.
        # emb_df IDs came from same source.
        
        # Check columns of emb_df
        # It has GAME_ID, PLAYER_ID, GAME_DATE, emb_0...
        
        # Drop GAME_DATE from emb_df to avoid conflict/duplication if needed, 
        # but validation is good.
        emb_df = emb_df.drop(columns=['GAME_DATE'], errors='ignore')
        
        self.df = self.df.merge(emb_df, on=['GAME_ID', 'PLAYER_ID'], how='left')
        
        # Fill missing embeddings (e.g. new players not in training set) with 0
        emb_cols = [c for c in emb_df.columns if c.startswith('emb_')]
        self.df[emb_cols] = self.df[emb_cols].fillna(0.0)
        print(f"Merged {len(emb_cols)} embedding columns.")

    def generate_embeddings(self, model_path, encoder_path, scaler_path, feature_list_path):
        """
        Generate embeddings on the fly using saved FT-Transformer artifacts.
        Useful for live inference where pre-computed embeddings don't exist.
        """
        if torch is None or FTTransformer is None:
            print("Torch not available. Skipping embedding generation.")
            return

        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}")
            return
            
        print("Generating embeddings on-the-fly...")
        
        # Load Artifacts
        le = joblib.load(encoder_path)
        scaler = joblib.load(scaler_path)
        cont_features = joblib.load(feature_list_path)
        
        # Preprocess Features
        # Ensure we have the continuous features calculated
        # The caller (daily_inference) should have run rolling stats already.
        
        # Missing handling - Ensure cols exist
        missing_cols = [c for c in cont_features if c not in self.df.columns]
        if missing_cols:
             print(f"Warning: Missing columns for embedding generation: {missing_cols}. Filling with 0.")
             for c in missing_cols:
                 self.df[c] = 0
                 
        X_cont_raw = self.df[cont_features].fillna(0).values
        X_cont = scaler.transform(X_cont_raw)
        
        # Strings for Encoder (Force string)
        player_ids = self.df['PLAYER_ID'].astype(str).values
        
        # Handle Unknown Players - Map safely
        # Create a lookup array or map
        # Speed optimization: Create dict mapping
        player_map = {p: i+1 for i, p in enumerate(le.classes_)}
        
        # Map using list comp (vectorized map is tricky with string keys)
        player_indices = np.array([player_map.get(pid, 0) for pid in player_ids])
        
        # Prepare Tensors
        device = 'cpu' # Inference is fast enough on CPU usually
        X_cat_t = torch.LongTensor(player_indices).to(device)
        X_cont_t = torch.FloatTensor(X_cont).to(device)
        
        # Load Model
        num_players = len(le.classes_) + 1
        model = FTTransformer(num_players, len(cont_features), embed_dim=16).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        # Batch Inference
        batch_size = 4096
        embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(self.df), batch_size):
                b_cat = X_cat_t[i:i+batch_size]
                b_cont = X_cont_t[i:i+batch_size]
                _, rep = model(b_cat, b_cont)
                embeddings.append(rep.numpy())
                
        if embeddings:
            embeddings = np.vstack(embeddings)
            # Assign to DataFrame
            for i in range(16):
                self.df[f'emb_{i}'] = embeddings[:, i]
        
        print("Embeddings generated and assigned.")

    def save_features(self, output_path):
        # Update feature selector to include new types
        feature_cols = [c for c in self.df.columns if 
                        'roll' in c or 
                        'lag' in c or 
                        'season' in c or 
                        'ewma' in c or 
                        'emb_' in c or
                        c in ['is_home', 'rest_days', 'is_b2b', 'missing_high_impact_production', 'role_trend_min']]
                        
        # Fill NaNs
        self.df[feature_cols] = self.df[feature_cols].fillna(0)
        
        print(f"Saving {len(feature_cols)} features to {output_path}...")
        # Keep explicit columns + identifiers
        save_cols = ['SEASON_ID', 'PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'GAME_ID', 'GAME_DATE', 'MATCHUP', 'WL', 'MIN', 'PTS', 'AST', 'REB'] + feature_cols
        
        # Ensure all save_cols exist (some might be missing if steps skipped?)
        valid_cols = [c for c in save_cols if c in self.df.columns]
        
        self.df[valid_cols].to_csv(output_path, index=False)
        print("Done.")

if __name__ == "__main__":
    # Wait for extraction to finish or use what we have
    if os.path.exists("data/nba_game_logs_1997_2024.csv"):
        engine = StrictFeatureEngine("data/nba_game_logs_1997_2024.csv")
        engine.load_and_clean()
        engine.compute_rolling_stats()
        engine.compute_rest_days()
        engine.add_lag_features()
        engine.compute_opponent_strength()
        engine.compute_advanced_rolling_stats()
        engine.compute_advanced_player_metrics()
        engine.compute_contextual_features()
        engine.compute_availability_features()
        
        # Phase G Additions
        engine.compute_per_minute_features()
        engine.merge_embeddings("features/player_embedding_v1.parquet")
        
        engine.save_features("data/strict_features_v4.csv")
    else:
        print("Waiting for data extraction...")
