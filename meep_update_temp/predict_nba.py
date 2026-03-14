import pandas as pd
import lightgbm as lgb
import numpy as np
from pathlib import Path
import json
import sys
from datetime import datetime
from typing import Dict, List, Optional, Union

class NBAPredictor:
    def __init__(self, model_dir: str = "meta_models_production"):
        self.models: Dict[str, lgb.Booster] = {}
        self.features: List[str] = []
        self.windows = [
            "2004_2008", "2005_2009", "2006_2010", "2007_2011", "2008_2012",
            "2009_2013", "2010_2014", "2011_2015", "2012_2016", "2013_2017",
            "2014_2018", "2015_2019", "2016_2020", "2017_2021", "2018_2022",
            "2019_2023", "2020_2024", "2021_2025", "2022_2026"
        ]
        self.stats = ["points", "assists", "rebounds", "minutes", "threes"]
        self._load_models(model_dir)
    
    def _load_models(self, model_dir: str) -> None:
        \"\"\"Load all trained models and initialize features\"\"\"
        try:
            # Define model files
            model_files = {
                "points": "meta_points.txt",
                "assists": "meta_assists.txt",
                "rebounds": "meta_rebounds.txt",
                "minutes": "meta_minutes.txt",
                "threes": "meta_threes.txt"
            }
            
            # Generate expected features
            self._generate_feature_names()
            
            # Load each model
            for target, filename in model_files.items():
                model_path = Path(model_dir) / filename
                if model_path.exists():
                    self.models[target] = lgb.Booster(
                        params={'predict_disable_shape_check': True},
                        model_file=str(model_path)
                    )
            
            if not self.models:
                raise FileNotFoundError(\"No model files found\")
                
        except Exception as e:
            raise RuntimeError(f\"Failed to load models: {str(e)}\")
    
    def _generate_feature_names(self) -> None:
        \"\"\"Generate the exact feature names used in training\"\"\"
        # Window-based features
        features = []
        for stat in self.stats:
            for window in self.windows:
                features.append(f\"{stat}_win_{window}\")
        
        # Statistical features
        for stat in self.stats:
            features.extend([
                f\"{stat}_win_std\",
                f\"{stat}_win_range\",
                f\"{stat}_win_recent_vs_oldest_diff\",
                f\"{stat}_win_recent_only\"
            ])
        
        # Contextual features
        features.extend([
            \"player_season_avg_points\",
            \"player_season_avg_assists\",
            \"player_season_avg_rebounds\",
            \"player_season_avg_minutes\",
            \"player_season_avg_threes\",
            \"is_home\",
            \"opp_defensive_rating\"
        ])
        
        self.features = features
    
    def _calculate_derived_features(self, window_predictions: Dict[str, float]) -> Dict[str, float]:
        \"\"\"Calculate derived statistical features\"\"\"
        features = {}
        
        for stat in self.stats:
            # Get all window predictions for this stat
            stat_values = [v for k, v in window_predictions.items() 
                          if k.startswith(f\"{stat}_win_\") and k.count('_') == 2]
            
            if not stat_values:
                continue
                
            # Calculate statistics
            values = np.array(stat_values)
            features[f\"{stat}_win_std\"] = float(np.std(values))
            features[f\"{stat}_win_range\"] = float(np.max(values) - np.min(values))
            
            # Recent vs oldest difference (last 5 windows vs first 5)
            recent = np.mean(values[-5:]) if len(values) >= 5 else 0
            oldest = np.mean(values[:5]) if len(values) >= 5 else 0
            features[f\"{stat}_win_recent_vs_oldest_diff\"] = float(recent - oldest)
            
            # Most recent window
            features[f\"{stat}_win_recent_only\"] = float(values[-1] if len(values) > 0 else 0)
        
        return features
    
    def predict(self, player_data: Dict[str, float]) -> Dict[str, float]:
        \"\"\"
        Make predictions for a player
        
        Args:
            player_data: Dictionary containing:
                - window_predictions: Dict of window predictions
                - season_averages: Dict of player's season averages
                - game_context: Dict of game context (is_home, opp_defensive_rating)
                
        Returns:
            Dictionary of predictions for each target
        \"\"\"
        # Prepare window predictions
        window_data = player_data.get(\"window_predictions\", {})
        season_avgs = player_data.get(\"season_averages\", {})
        game_ctx = player_data.get(\"game_context\", {})
        
        # Calculate derived features
        derived_features = self._calculate_derived_features(window_data)
        
        # Prepare final feature vector
        features = {}
        
        # 1. Add window predictions
        for feature in self.features:
            if feature in window_data:
                features[feature] = window_data[feature]
        
        # 2. Add derived features
        features.update(derived_features)
        
        # 3. Add season averages
        for stat in self.stats:
            features[f\"player_season_avg_{stat}\"] = season_avgs.get(stat, 0.0)
        
        # 4. Add game context
        features[\"is_home\"] = game_ctx.get(\"is_home\", 0)
        features[\"opp_defensive_rating\"] = game_ctx.get(\"opp_defensive_rating\", 105.0)
        
        # Ensure all features are present
        for feat in self.features:
            if feat not in features:
                features[feat] = 0.0
        
        # Convert to DataFrame in correct order
        X = pd.DataFrame([features])[self.features]
        
        # Make predictions
        predictions = {}
        for target, model in self.models.items():
            try:
                predictions[target] = float(model.predict(X)[0])
            except Exception as e:
                print(f\"Warning: Prediction failed for {target}: {str(e)}\")
                predictions[target] = 0.0
        
        return predictions

def main():
    if len(sys.argv) < 2:
        print(\"Usage: python predict_nba.py 'Player Name'\")
        sys.exit(1)
    
    player_name = \" \".join(sys.argv[1:])
    print(f\"\\n🔍 Generating predictions for {player_name}...\")
    
    try:
        # Initialize predictor
        predictor = NBAPredictor()
        
        # Get player data (implement this function)
        player_data = get_player_data(player_name)
        if not player_data:
            raise ValueError(f\"Could not load data for {player_name}\")
        
        # Make predictions
        predictions = predictor.predict(player_data)
        
        # Display results
        print(\"\\n\" + \"=\" * 60)
        print(f\"PREDICTIONS FOR: {player_name.upper()}\")
        print(f\"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\")
        print(\"=\" * 60)
        print(f\"POINTS:    {predictions.get('points', 'N/A'):.1f}\")
        print(f\"ASSISTS:   {predictions.get('assists', 'N/A'):.1f}\") 
        print(f\"REBOUNDS:  {predictions.get('rebounds', 'N/A'):.1f}\")
        print(f\"3-POINTERS: {predictions.get('threes', 'N/A'):.1f}\")
        print(f\"MINUTES:   {predictions.get('minutes', 'N/A'):.1f}\")
        print(\"=\" * 60)
        
    except Exception as e:
        print(f\"\\n❌ Error: {str(e)}\")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def get_player_data(player_name: str) -> Dict:
    \"\"\"
    Get data for a player.
    Replace this with your actual data loading logic.
    \"\"\"
    # Example data - replace with actual data loading
    return {
        \"window_predictions\": {
            # These would come from your window models
            \"points_win_2022_2026\": 25.5,
            \"assists_win_2022_2026\": 7.2,
            # ... add all window predictions
        },
        \"season_averages\": {
            \"points\": 24.8,
            \"assists\": 7.1,
            \"rebounds\": 8.2,
            \"minutes\": 36.5,
            \"threes\": 2.7
        },
        \"game_context\": {
            \"is_home\": 1,
            \"opp_defensive_rating\": 108.3
        }
    }

if __name__ == \"__main__\":
    main()
