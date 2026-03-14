import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class BasketballDataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.label_encoders = {}
        self.feature_columns = None
        self.target_column = None

    def load_data(self, nrows=None):
        """Loads data with memory optimization."""
        print(f"Loading data from {self.file_path}...")
        # low_memory=False to avoid DtypeWarning on mixed types
        self.df = pd.read_csv(self.file_path, nrows=nrows, low_memory=False)
        
        if 'date' in self.df.columns:
            # Handle mixed formats and timezones by forcing UTC, then removing timezone
            self.df['date'] = pd.to_datetime(self.df['date'], utc=True, errors='coerce')
            self.df['date'] = self.df['date'].dt.tz_localize(None)
            self.df['date'] = self.df['date'].dt.normalize()
        
        # Optimize types to save memory
        for col in self.df.select_dtypes(include=['float64']).columns:
            self.df[col] = self.df[col].astype('float32')
        
        # Ensure season is integer for splitting
        if 'season' in self.df.columns:
            self.df['season'] = self.df['season'].fillna(0).astype(int)
            
        print(f"Data loaded: {self.df.shape}")
        return self.df

    def preprocess(self, target='points'):
        """
        Prepares data for tree-based models.
        - Handles categorical encoding (Label Encoding).
        - Sorts by time (Season, Game Number).
        """
        
        self.target_column = target
        
        # Drop rows with missing target
        if self.target_column in self.df.columns:
            initial_len = len(self.df)
            self.df = self.df.dropna(subset=[self.target_column])
            print(f"Dropped {initial_len - len(self.df)} rows with missing target '{self.target_column}'")

        
        # Sort by time to ensure correct time-series splits
        sort_cols = []
        if 'season' in self.df.columns: sort_cols.append('season')
        if 'date' in self.df.columns: sort_cols.append('date')
        elif 'game_number_in_season' in self.df.columns: sort_cols.append('game_number_in_season')
        
        if sort_cols:
            self.df = self.df.sort_values(by=sort_cols).reset_index(drop=True)

        # Identify categorical columns
        self.cat_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        print(f"Categorical columns to encode: {self.cat_cols}")

        for col in self.cat_cols:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col].astype(str))
            self.label_encoders[col] = le

        # Feature Engineering
        self.add_rest_features()
        self.add_game_context()
        self.add_opponent_adjustments()
        self.add_pace_adjustments()
        self.add_interactions()
        self.add_lag_features() # RE-ENABLED: Critical for capturing recent form
        
        # Add Consistency Metrics (Rolling Std Dev) - Research suggestion
        if 'points' in self.df.columns and 'player_name' in self.df.columns:
             self.df = self.df.sort_values(['player_name', 'date'])
             self.df['points_volatility_5'] = self.df.groupby('player_name')['points'].transform(lambda x: x.shift(1).rolling(5).std())
             self.df['points_volatility_5'] = self.df['points_volatility_5'].fillna(0)

        # Define feature columns (exclude target and non-predictive metadata if any)
        # CRITICAL: Exclude current game stats that leak the result (e.g., points, minutes, fg_percent)
        # We only want PRE-GAME knowledge (rolling averages, season averages, schedule info)
        
        leakage_cols = [
            'points', 'assists', 'rebounds', 'minutes', 'fg_percent', 'x3p_percent', 'ft_percent',
            'steals', 'blocks', 'turnovers', 'foulspersonal', 'plus_minus', 'game_score',
            'fg_per_game', 'fga_per_game', 'x3p_per_game', 'x3pa_per_game', 'ft_per_game', 'fta_per_game',
            'mp', 'fg', 'fga', 'fg%', '3p', '3pa', '3p%', 'ft', 'fta', 'ft%', 'orb', 'drb', 'trb', 'ast', 'stl', 'blk', 'tov', 'pf', 'pts', 'gmsc', '+/-',
            'fieldGoalsAttempted', 'fieldGoalsMade', 'fieldGoalsPercentage',
            'threePointersAttempted', 'threePointersMade', 'threePointersPercentage',
            'freeThrowsAttempted', 'freeThrowsMade', 'freeThrowsPercentage',
            'reboundsDefensive', 'reboundsOffensive', 'reboundsTotal',
            'foulsPersonal', 'turnovers', 'plusMinusPoints'
        ]
        
        exclude_cols = [target, 'player_name', 'date', 'game_id', 'team_score', 'opp_score', 'win', 'loss'] + leakage_cols
        
        # Filter out columns that are in the dataframe
        actual_exclude = [c for c in exclude_cols if c in self.df.columns]
        
        self.feature_columns = [c for c in self.df.columns if c not in actual_exclude]
        
        print(f"Preprocessing complete. {len(self.feature_columns)} features identified.")
        print(f"Excluded {len(actual_exclude)} potential leakage columns.")

    def add_pace_adjustments(self):
        """
        Normalizes volume stats by 'pace' to create per-100-possession metrics.
        Formula: (Stat / Pace) * 100
        """
        if 'pace' not in self.df.columns:
            print("Warning: 'pace' column not found. Skipping pace adjustments.")
            return

        print("Adding Pace-Adjusted Metrics...")
        # Rolling stats are good candidates for pace adjustment
        volume_cols = [c for c in self.df.columns if 'roll' in c and ('points' in c or 'assists' in c or 'rebounds' in c)]
        
        for col in volume_cols:
            new_col = f"{col}_per100"
            # Avoid division by zero
            self.df[new_col] = (self.df[col] / self.df['pace'].replace(0, 90)) * 100
            self.df[new_col] = self.df[new_col].fillna(0)
            
        print(f"Added {len(volume_cols)} pace-adjusted features.")

    def add_rest_features(self):
        """
        Adds rest-related features:
        - days_rest: Days since player's last game
        - is_back_to_back: 1 if playing on consecutive days
        """
        if 'player_name' not in self.df.columns or 'date' not in self.df.columns:
            print("Warning: Missing player_name or date. Skipping rest features.")
            return
        
        print("Adding Rest Day Features...")
        
        # Sort by player and date
        self.df = self.df.sort_values(['player_name', 'date'])
        
        # Days since last game per player
        self.df['prev_game_date'] = self.df.groupby('player_name')['date'].shift(1)
        self.df['days_rest'] = (self.df['date'] - self.df['prev_game_date']).dt.days
        self.df['days_rest'] = self.df['days_rest'].fillna(3).clip(0, 14)
        
        # Back-to-back indicator
        self.df['is_back_to_back'] = (self.df['days_rest'] == 1).astype(int)
        
        # Drop helper column
        self.df = self.df.drop(columns=['prev_game_date'], errors='ignore')
        
        print(f"Added: days_rest, is_back_to_back")

    def add_game_context(self):
        """
        Adds game context features:
        - road_trip_game: Games into consecutive road trip
        - season_pct: % of season completed
        - games_into_season: Game number for this player this season
        """
        if 'player_name' not in self.df.columns:
            print("Warning: Missing player_name. Skipping game context features.")
            return
        
        print("Adding Game Context Features...")
        
        # Games into season for player
        if 'season' in self.df.columns:
            self.df = self.df.sort_values(['player_name', 'season', 'date'])
            self.df['games_into_season'] = self.df.groupby(['player_name', 'season']).cumcount()
            self.df['season_pct'] = self.df['games_into_season'] / 82.0
            self.df['season_pct'] = self.df['season_pct'].clip(0, 1)
        
        # Road trip indicator (if home_team available)
        if 'home' in self.df.columns:
            # Already have home indicator
            self.df = self.df.sort_values(['player_name', 'date'])
            # Calculate consecutive away games
            self.df['is_away'] = (self.df['home'] == 0).astype(int)
            
            # Road trip game number
            self.df['road_trip_start'] = (self.df['is_away'] != self.df.groupby('player_name')['is_away'].shift(1)).astype(int) & self.df['is_away']
            self.df['road_trip_id'] = self.df.groupby('player_name')['road_trip_start'].cumsum()
            self.df['road_trip_game'] = self.df.groupby(['player_name', 'road_trip_id']).cumcount() + 1
            self.df.loc[self.df['is_away'] == 0, 'road_trip_game'] = 0
            
            # Cleanup
            self.df = self.df.drop(columns=['is_away', 'road_trip_start', 'road_trip_id'], errors='ignore')
        
        print(f"Added: games_into_season, season_pct, road_trip_game")

    def add_opponent_adjustments(self):
        """
        Calculates opponent defensive strength and adjusts player stats.
        Creates: opp_def_points, opp_def_rebounds, matchup_adj
        """
        # Find opponent column (case-insensitive)
        opp_col = None
        for col in self.df.columns:
            if col.lower() in ['opponentteamname', 'opponent']:
                opp_col = col
                break
        
        if opp_col is None or 'season' not in self.df.columns or 'points' not in self.df.columns:
            print("Warning: Missing opponent/season/points columns. Skipping opponent adjustments.")
            return

        print("Adding Opponent-Adjusted Metrics...")
        
        # Calculate opponent's average points allowed (to all players)
        opp_def = self.df.groupby([opp_col, 'season']).agg({
            'points': 'mean',
            'rebounds': 'mean',
            'assists': 'mean'
        }).rename(columns={
            'points': 'opp_def_points',
            'rebounds': 'opp_def_rebounds',
            'assists': 'opp_def_assists'
        })
        
        # Merge back
        self.df = self.df.merge(
            opp_def, 
            left_on=[opp_col, 'season'], 
            right_index=True, 
            how='left',
            suffixes=('', '_opp')
        )
        
        # Matchup adjustment: player's rolling avg vs opponent's defense
        if 'numpoints_roll_5' in self.df.columns and 'opp_def_points' in self.df.columns:
            self.df['matchup_pts_adj'] = self.df['numpoints_roll_5'] / self.df['opp_def_points'].replace(0, 1)
        
        print(f"Added: opp_def_points, opp_def_rebounds, opp_def_assists, matchup_pts_adj")

    def add_interactions(self):
        """
        Creates key interaction terms based on research.
        """
        print("Adding Interaction Terms...")
        # Usage * Efficiency (proxy)
        if 'usg_percent_season_avg' in self.df.columns and 'ts_percent_season_avg' in self.df.columns:
            self.df['usage_x_efficiency'] = self.df['usg_percent_season_avg'] * self.df['ts_percent_season_avg']
            
        # Minutes * Production (Total Output Potential)
        if 'numminutes_roll_5' in self.df.columns and 'bpm_season_avg' in self.df.columns:
             self.df['minutes_x_bpm'] = self.df['numminutes_roll_5'] * self.df['bpm_season_avg']

        # Research-Driven Interactions:
        # 1. Scoring * Playmaking (Offensive Load)
        if 'numpoints_roll_5' in self.df.columns and 'numassists_roll_5' in self.df.columns:
            self.df['off_load_index'] = self.df['numpoints_roll_5'] * self.df['numassists_roll_5']

        # 2. Rebounding * Matchup (Exploitation)
        # High rebounder vs weak rebounding defense
        if 'numrebounds_roll_10' in self.df.columns and 'opp_def_rebounds' in self.df.columns:
             self.df['reb_exploitation'] = self.df['numrebounds_roll_10'] * self.df['opp_def_rebounds']

    def add_lag_features(self):
        """
        Adds specific previous-game values (Lags).
        Requires sorting by player and date/game_number.
        """
        print("Adding Lag Features (Last 1, 2, 3 games)...")
        
        # Ensure we have necessary columns
        if 'player_name' not in self.df.columns:
            print("Warning: 'player_name' not found. Skipping lag features.")
            return

        # Sort by player and time to ensure correct shifting
        # We assume 'season' and 'game_number_in_season' are available and correct
        sort_cols = ['player_name', 'season', 'game_number_in_season']
        # Check if they exist
        if not all(col in self.df.columns for col in sort_cols):
             print(f"Warning: Missing sort columns for lags. Have: {self.df.columns.tolist()}")
             return
             
        self.df = self.df.sort_values(by=sort_cols)
        
        # Define targets to lag
        # We want to lag the TARGET variable (e.g. points) but we must be careful.
        # The 'points' column in the current row is the target (Y).
        # We want 'points' from the PREVIOUS row (t-1).
        # But wait, 'points' is in leakage_cols and might be dropped or not loaded?
        # In preprocess(), we define leakage_cols but we haven't dropped them yet from self.df, 
        # we just excluded them from self.feature_columns.
        # So 'points' should still be in self.df.
        
        targets_to_lag = ['points', 'assists', 'rebounds', 'minutes']
        # Only lag if they exist
        targets_to_lag = [t for t in targets_to_lag if t in self.df.columns]
        
        for target in targets_to_lag:
            # Group by player and shift
            # We use transform to keep the index aligned
            self.df[f'{target}_last_game'] = self.df.groupby('player_name')[target].shift(1)
            self.df[f'{target}_last_2_games'] = self.df.groupby('player_name')[target].shift(2)
            self.df[f'{target}_last_3_games'] = self.df.groupby('player_name')[target].shift(3)
            
            # Fill NaNs?
            # First game of season or career will be NaN.
            # We can fill with 0 or the player's season average so far (expanding mean).
            # For simplicity and tree models, -1 or 0 is often okay, but 0 makes sense for stats.
            self.df[f'{target}_last_game'] = self.df[f'{target}_last_game'].fillna(0)
            self.df[f'{target}_last_2_games'] = self.df[f'{target}_last_2_games'].fillna(0)
            self.df[f'{target}_last_3_games'] = self.df[f'{target}_last_3_games'].fillna(0)
            
        print(f"Added lag features for {targets_to_lag}")

    def get_time_series_splits(self, val_season=2022, test_season=2023, window_size=None):
        """
        Returns (X_train, y_train), (X_val, y_val), (X_test, y_test)
        based on season cutoffs.
        
        window_size: If set (int), train on only the last N seasons before val_season.
        """
        if 'season' not in self.df.columns:
            raise ValueError("Season column required for time-series split.")

        if window_size:
            start_season = val_season - window_size
            train_mask = (self.df['season'] >= start_season) & (self.df['season'] < val_season)
            print(f"Rolling Window Training: Seasons {start_season} to {val_season-1}")
        else:
            train_mask = self.df['season'] < val_season
            print(f"Expanding Window Training: Start to {val_season-1}")

        val_mask = (self.df['season'] == val_season)
        test_mask = self.df['season'] >= test_season

        X = self.df[self.feature_columns]
        y = self.df[self.target_column]

        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        print(f"Split sizes - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def get_cat_cols(self):
        """Returns list of categorical column names."""
        # Return only those that are actually in feature_columns
        if self.feature_columns and hasattr(self, 'cat_cols'):
            return [c for c in self.cat_cols if c in self.feature_columns]
        return []
