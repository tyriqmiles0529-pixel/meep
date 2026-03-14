import pandas as pd
import numpy as np
import os
import json
import joblib
import torch
from ensemble_model import EnsemblePredictor
from ft_transformer import FTTransformerFeatureExtractor
from data_processor import BasketballDataProcessor
# Add Odds Ingestion
try:
    from odds_ingestion.mock import MockOddsProvider
    from odds_ingestion.the_odds_api import TheOddsAPIProvider
    from odds_ingestion.rapid_api import RapidAPIProvider
except ImportError:
    MockOddsProvider = None
    TheOddsAPIProvider = None
    RapidAPIProvider = None

class BettingStrategy:
    def __init__(self, models_dir='models', data_path='final_feature_matrix_with_per_min_1997_onward.csv', provider='mock'):
        self.models_dir = models_dir
        self.data_path = data_path
        self.targets = ['points', 'rebounds', 'assists', 'three_pointers']
        self.predictors = {}
        self.ft_extractors = {}
        self.processor = None
        
        self.odds_provider = None
        if provider == 'the-odds-api' and TheOddsAPIProvider:
            self.odds_provider = TheOddsAPIProvider()
        elif provider == 'rapid-api' and RapidAPIProvider:
            self.odds_provider = RapidAPIProvider()
        elif MockOddsProvider:
            self.odds_provider = MockOddsProvider()
            if provider != 'mock':
                print(f"Warning: Provider '{provider}' not found or unavailable. Using Mock.")
        
    def load_resources(self):
        print("Loading Data Processor...")
        self.processor = BasketballDataProcessor(self.data_path)
        self.processor.load_data()
        # Preprocess for features (using 'points' as dummy target to generate features)
        self.processor.preprocess(target='points')
        
        print("Loading Models...")
        for target in self.targets:
            print(f"Loading {target} model...")
            # Load Ensemble
            pred = EnsemblePredictor()
            # Set model_dir to the target subdirectory (e.g. models/points)
            pred.model_dir = os.path.join(self.models_dir, target)
            # Load latest models (e.g. xgb_model_points.pkl)
            # Wait, ls showed xgb_model_points.pkl in models/points?
            # ls meep/nba_predictor/models/points showed xgb_model_2022.pkl etc.
            # It also showed xgb_model_points.pkl?
            # Step 3319 showed xgb_model_points.pkl in meep/nba_predictor/models (root models dir).
            # Step 3352 showed xgb_model_2022.pkl in meep/nba_predictor/models/points.
            
            # So for LATEST (production), we might use the ones in root models dir?
            # Or maybe we should use 2022 (latest available)?
            # Let's use 2022 as latest for now.
            
            # If using root models dir:
            # pred.model_dir = self.models_dir
            # pred.load_models(suffix=f"_{target}")
            
            # If using models/points/xgb_model_2022.pkl:
            # pred.model_dir = os.path.join(self.models_dir, target)
            # pred.load_models(suffix="_2022")
            
            # Let's assume we want to load the "latest" available season model.
            # For 2025 season (current), we might not have a model yet?
            # Or we use 2022 model for everything?
            # Let's try to load "_points" from root first (if it exists).
            # If not, try "_2022" from subdir.
            
            # Based on ls output:
            # models/xgb_model_points.pkl exists.
            # models/points/xgb_model_2022.pkl exists.
            
            # Let's use the root one for "default" loading.
            pred.model_dir = self.models_dir
            try:
                pred.load_models(suffix=f"_{target}")
                self.predictors[target] = pred
            except Exception as e:
                print(f"Failed to load default {target} model: {e}")
            
            # Load FT-Transformer (Global or per target? Train script saved global per season)
            # We need a strategy to pick the right FT model. 
            # For inference on new data, we should use the LATEST available model.
            # Let's assume we use the one from the latest trained season (e.g., 2025).
            ft_path = os.path.join(self.models_dir, f"global_ft_2025", "ft_transformer.pt")
            if os.path.exists(ft_path):
                # We need to know cardinalities to init the model structure first
                cat_cols = self.processor.get_cat_cols()
                cardinalities = [len(self.processor.label_encoders[col].classes_) for col in cat_cols]
                
                ft = FTTransformerFeatureExtractor(cardinalities, embed_dim=16, device='cpu')
                ft.load(ft_path)
                self.ft_extractors['global'] = ft
            else:
                print(f"Warning: FT-Transformer not found at {ft_path}")

    def generate_predictions(self, date_str):
        # Filter data for the specific date
        # This assumes we have data for that date in the csv (historical backtest)
        # For live, we'd need to fetch new data.
        
        df_day = self.processor.df[self.processor.df['date'] == date_str].copy()
        
        if df_day.empty:
            print(f"No games found for {date_str}")
            return None
            
        # Generate Embeddings
        if 'global' in self.ft_extractors:
            cat_cols = self.processor.get_cat_cols()
            X_cat = df_day[cat_cols].values
            embeddings = self.ft_extractors['global'].transform(X_cat)
            
            # Add to DF
            emb_cols = [f"emb_{i}" for i in range(embeddings.shape[1])]
            df_emb = pd.DataFrame(embeddings, columns=emb_cols, index=df_day.index)
            df_day = pd.concat([df_day, df_emb], axis=1)
            
        predictions = {}
        for target in self.targets:
            # Predict
            # We need to ensure columns match what the model expects
            # The EnsemblePredictor.predict method handles DMatrix conversion
            # But we need to pass the right columns.
            # We can use the processor's feature_columns + embedding columns
            
            # Filter for features only
            # Combine processor features + any embedding columns we added
            features_to_use = self.processor.feature_columns.copy()
            if 'global' in self.ft_extractors:
                 features_to_use += [c for c in df_day.columns if c.startswith('emb_')]
            
            # Ensure all features exist
            valid_features = [f for f in features_to_use if f in df_day.columns]
            X_pred = df_day[valid_features]
            
            if target in self.predictors:
                preds = self.predictors[target].predict(X_pred, use_stacking=True)
                predictions[target] = preds
            else:
                predictions[target] = np.zeros(len(df_day))
            
        # Combine into a results DF
        results = df_day[['player_name', 'playerteamName', 'opponentteamName', 'minutes']].copy()
        for t, p in predictions.items():
            results[f'pred_{t}'] = p
            
        # Add Confidence (Dummy for now, can be variance of ensemble members)
        # results['confidence'] = ...
        
        return results

    def calculate_ev(self, row, target, line, odds):
        # Simple EV calculation
        # Prob = Probability of winning bet
        # We need a distribution to get Prob(pred > line).
        # For now, assume normal distribution with std dev from validation RMSE.
        
        # Load RMSE from report
        rmses = {'points': 4.5, 'rebounds': 2.0, 'assists': 1.8, 'three_pointers': 0.8}
        skews = {'points': 2.0, 'rebounds': 2.5, 'assists': 2.2, 'three_pointers': 1.5}
        
        rmse = rmses.get(target, 4.5)
        skew_a = skews.get(target, 0)
        
        from scipy.stats import norm, skewnorm
        
        pred = row[f'pred_{target}']
        
        # Probability of Over
        prob_over = 1 - skewnorm.cdf(line, skew_a, loc=pred, scale=rmse)
        
        # Probability of Under
        prob_under = skewnorm.cdf(line, skew_a, loc=pred, scale=rmse)
        
        # Decimal Odds
        dec_odds = odds
        if odds < 0:
            dec_odds = 1 + (100 / abs(odds))
        else:
            dec_odds = 1 + (odds / 100)
            
        # EV
        # Assuming we bet OVER if pred > line
        if pred > line:
            win_prob = prob_over
        else:
            win_prob = prob_under
            
        ev = (win_prob * (dec_odds - 1)) - (1 - win_prob)
        
        return ev, win_prob

    def calculate_confidence(self, prediction, line, rmse):
        """
        Calculate confidence score (0-100) based on Z-score.
        """
        if rmse <= 0: return 0
        z_score = abs(prediction - line) / rmse
        # Map Z-score to 0-100. 
        # Z=1 (1 sigma) -> ~68% conf interval? 
        # Let's scale: Z=2 -> 100% confidence (arbitrary but practical)
        raw_score = (z_score / 2.0) * 100
        return min(100.0, max(0.0, raw_score))

    def load_season_models(self, season):
        """
        Load models trained on data prior to 'season'.
        E.g. if season=2022, load models trained on 1997-2021.
        Assumes models are saved with suffix '_{season}'.
        """
        print(f"Loading models for Season {season}...")
        
        # Determine model suffix/path based on season
        # Our training script saves models per season.
        # e.g. "points_2022" is trained on data UP TO 2022 (inclusive? or exclusive?).
        # Walk-forward logic: Train on 1997-2021, Validate on 2022.
        # So for predicting 2022 games, we need the model trained on 1997-2021.
        # Let's assume the model saved as '2021' is the one trained on data up to 2021.
        
        train_season = season - 1
        
        for target in self.targets:
            # Load Ensemble
            pred = EnsemblePredictor()
            # Models are in subdirectories: models/points/xgb_model_2021.pkl
            pred.model_dir = os.path.join(self.models_dir, target)
            
            # Try loading specific season model
            try:
                pred.load_models(suffix=f"_{train_season}") 
                self.predictors[target] = pred
            except Exception as e:
                print(f"Error loading {target} model for {train_season}: {e}")
                # Fallback to latest? Or skip?
                # For backtest validity, we should probably skip or warn.
                pass
                
        # Load FT-Transformer
        ft_path = os.path.join(self.models_dir, f"global_ft_{train_season}", "ft_transformer.pt")
        print(f"DEBUG: Attempting to load FT from {ft_path}")
        if os.path.exists(ft_path):
            try:
                cat_cols = self.processor.get_cat_cols()
                print(f"DEBUG: cat_cols for FT: {len(cat_cols)} columns")
                cardinalities = [len(self.processor.label_encoders[col].classes_) for col in cat_cols]
                
                ft = FTTransformerFeatureExtractor(cardinalities, embed_dim=16, device='cpu')
                ft.load(ft_path)
                self.ft_extractors['global'] = ft
                print(f"DEBUG: FT-Transformer loaded successfully for {train_season}")
            except Exception as e:
                print(f"DEBUG: Failed to init/load FT: {e}")
        else:
            print(f"Warning: FT-Transformer not found for {train_season} at {ft_path}")

    def generate_bets(self, predictions_df, bankroll=1000, confidence_threshold=10, kelly_fraction=0.25, min_ev=0.0):
        # 1. Filter by Minutes (Strict)
        min_col = 'pred_minutes' if 'pred_minutes' in predictions_df.columns else 'minutes'
        if min_col not in predictions_df.columns:
            mask_min = pd.Series([True] * len(predictions_df))
        else:
            mask_min = predictions_df[min_col] >= 20
            
        candidates = predictions_df[mask_min].copy()
        
        bets = []
        rmses = {'points': 4.5, 'rebounds': 2.0, 'assists': 1.8, 'three_pointers': 0.8}
        
        for idx, row in candidates.iterrows():
            for target in self.targets:
                line = row.get(f'line_{target}')
                odds = row.get(f'odds_{target}', -110)
                
                if line is None:
                    # Map full target name to dataset column abbreviation
                    # dataset: prior_pts, prior_reb, prior_ast, prior_three_pointers (maybe?)
                    abbr_map = {
                        'points': 'prior_pts',
                        'rebounds': 'prior_reb', 
                        'assists': 'prior_ast',
                        'three_pointers': 'prior_three_pointers' # Verify this one exists or fallback
                    }
                    
                    prior_col = abbr_map.get(target)
                    
                    if prior_col and prior_col in row:
                        line = row[prior_col]
                    else:
                        # Fallback for three pointers if not in prior
                        # or if prior_col is null
                        continue 
                        
                pred = row[f'pred_{target}']
                rmse = rmses.get(target, 4.5)
                
                ev, win_prob = self.calculate_ev(row, target, line, odds)
                conf = self.calculate_confidence(pred, line, rmse)
                
                # Filters
                if ev > min_ev and conf >= confidence_threshold:
                    bets.append({
                        'player': row['player_name'],
                        'team': row.get('playerteamName', 'N/A'),
                        'game_id': row.get('gameId', 'N/A'),
                        'target': target,
                        'line': line,
                        'prediction': pred,
                        'ev': ev,
                        'win_prob': win_prob,
                        'odds': odds,
                        'confidence': conf
                    })
                    
        bets_df = pd.DataFrame(bets)
        if bets_df.empty:
            return pd.DataFrame()
            
        # 3. Ranking
        bets_df = bets_df.sort_values('ev', ascending=False)
        
        # 4. Dynamic Kelly
        def get_kelly(row):
            b = 0.909 
            if row['odds'] < 0:
                b = 100 / abs(row['odds'])
            else:
                b = row['odds'] / 100
            p = row['win_prob']
            q = 1 - p
            f = (b * p - q) / b
            return max(0, f)
            
        bets_df['kelly_fraction'] = bets_df.apply(get_kelly, axis=1)
        
        # Scale by Confidence & User Fraction
        bets_df['adj_kelly'] = bets_df['kelly_fraction'] * (bets_df['confidence'] / 100.0)
        
        # Drawdown Protection (Placeholder)
        drawdown_pct = 0.0
        if drawdown_pct > 0.10:
            bets_df['adj_kelly'] *= 0.5
            
        bets_df['stake_pct'] = bets_df['adj_kelly'] * kelly_fraction # Use passed fraction (default 0.25 = 1/4)
        bets_df['stake_amt'] = bets_df['stake_pct'] * bankroll
        
        return bets_df

    def generate_parlays(self, bets_df, num_parlays=5):
        # Generate 5 parlays from top selections
        # Constraint: Uncorrelated legs (Max 1 leg per Game ID)
        
        parlays = []
        # Pool of best bets (Top 20)
        pool = bets_df.head(20).copy()
        
        if len(pool) < 2:
            return pd.DataFrame()
            
        import random
        
        attempts = 0
        while len(parlays) < num_parlays and attempts < 50:
            attempts += 1
            num_legs = random.randint(2, 5)
            
            # Iterative selection to ensure unique games
            current_parlay = []
            used_games = set()
            
            # Shuffle pool to randomize start
            shuffled_pool = pool.sample(frac=1)
            
            for _, bet in shuffled_pool.iterrows():
                if len(current_parlay) >= num_legs:
                    break
                    
                game_id = bet.get('game_id')
                # If game_id missing, assume unique? Or skip?
                # If we have game_id, check usage.
                if game_id and game_id != 'N/A':
                    if game_id in used_games:
                        continue
                    used_games.add(game_id)
                
                current_parlay.append(bet)
                
            if len(current_parlay) < 2:
                continue
                
            # Create Parlay Object
            legs = pd.DataFrame(current_parlay)
            
            combined_prob = legs['win_prob'].prod()
            # Odds calc: Convert all to decimal, multiply, convert back to American?
            # Or just store decimal.
            combined_dec_odds = legs['odds'].apply(lambda x: (1 + 100/abs(x)) if x < 0 else (1 + x/100)).prod()
            
            # EV
            parlay_ev = (combined_prob * (combined_dec_odds - 1)) - (1 - combined_prob)
            
            parlays.append({
                'legs': legs['player'].tolist(),
                'targets': legs['target'].tolist(),
                'combined_odds': combined_dec_odds, # Decimal
                'combined_prob': combined_prob,
                'ev': parlay_ev,
                'num_legs': len(legs)
            })
            
        return pd.DataFrame(parlays)

    def generate_round_robins(self, bets_df, num_rr=5):
        # Generate Round Robins (e.g. 3 bets, all 2-leg combos)
        rrs = []
        top_bets = bets_df.head(5) # Top 5 for RR
        
        if len(top_bets) < 3:
            return pd.DataFrame()
            
        # Example: 3x2 (3 selections, parlay size 2)
        from itertools import combinations
        
        combos = list(combinations(top_bets.index, 2))
        
        for idx_list in combos:
            legs = top_bets.loc[list(idx_list)]
            rrs.append({
                'type': '2-leg RR',
                'legs': legs['player'].tolist(),
                'combined_odds': combined_odds,
                'ev': parlay_ev
            })
            
        return pd.DataFrame(rrs)

    def backtest(self, start_season=2020, end_season=2026, confidence_threshold=10, kelly_fraction=0.25, min_ev=0.0):
        # Backtest Strategy with Season-Aware Model Loading
        print(f"Starting Backtest ({start_season}-{end_season}) | Conf: {confidence_threshold} | Kelly: {kelly_fraction} | MinEV: {min_ev}")
        
        # Filter data by season
        mask = (self.processor.df['season'] >= start_season) & (self.processor.df['season'] <= end_season)
        df_backtest = self.processor.df[mask].copy()
        
        # Get unique dates
        dates = df_backtest['date'].unique()
        dates = sorted(dates)
        
        bankroll = 1000
        history = []
        current_season = None
        
        for date in dates:
            # Determine season for this date
            # Ensure date is comparable if needed
            season = df_backtest[df_backtest['date'] == date]['season'].iloc[0]
            
            # Load correct model if season changes
            if season != current_season:
                self.load_season_models(season)
                current_season = season
            
            # Generate Predictions
            preds = self.generate_predictions(date)
            if preds is None or preds.empty:
                continue
                
            bets = self.generate_bets(preds, bankroll=bankroll, 
                                      confidence_threshold=confidence_threshold, 
                                      kelly_fraction=kelly_fraction, 
                                      min_ev=min_ev)
            
            if bets.empty:
                history.append({'date': date, 'bankroll': bankroll, 'pnl': 0, 'roi': 0})
                continue
                
            # Evaluate Bets
            day_pnl = 0
            day_stake = 0
            
            for _, bet in bets.iterrows():
                player = bet['player']
                target = bet['target']
                line = bet['line']
                
                # Get actual
                actual_row = df_backtest[(df_backtest['date'] == date) & (df_backtest['player_name'] == player)]
                if actual_row.empty:
                    continue
                    
                actual = actual_row[target].iloc[0]
                
                # Determine Win/Loss
                won = False
                if bet['prediction'] > line: # We bet OVER
                    if actual > line: won = True
                else: # We bet UNDER
                    if actual < line: won = True
                        
                # Calculate PnL
                stake = bet['stake_amt']
                day_stake += stake
                
                if won:
                    # Profit
                    dec_odds = (1 + bet['odds']/100) if bet['odds'] > 0 else (1 + 100/abs(bet['odds']))
                    profit = stake * (dec_odds - 1)
                    day_pnl += profit
                else:
                    day_pnl -= stake
            
            bankroll += day_pnl
            roi = (day_pnl / day_stake) if day_stake > 0 else 0
            history.append({'date': date, 'bankroll': bankroll, 'pnl': day_pnl, 'roi': roi})
            print(f"Date: {date} | Season: {season} | PnL: ${day_pnl:.2f} | Bankroll: ${bankroll:.2f}", end='\r')
            
        print("\nBacktest Complete.")
        
        # Calculate Metrics
        hist_df = pd.DataFrame(history)
        if hist_df.empty:
            return {'roi': 0, 'sharpe': 0, 'drawdown': 0, 'final_bankroll': 1000}
            
        total_roi = (bankroll - 1000) / 1000
        
        # Sharpe (Daily Returns)
        # We need daily % return relative to bankroll? Or just PnL?
        # Sharpe usually excess return / std dev.
        # Let's use daily PnL / Starting Bankroll as "daily return" approximation?
        # Or better: ln(today_br / yesterday_br)
        hist_df['daily_return'] = hist_df['bankroll'].pct_change().fillna(0)
        sharpe = hist_df['daily_return'].mean() / hist_df['daily_return'].std() * np.sqrt(252) if hist_df['daily_return'].std() != 0 else 0
        
        # Max Drawdown
        hist_df['peak'] = hist_df['bankroll'].cummax()
        hist_df['drawdown'] = (hist_df['bankroll'] - hist_df['peak']) / hist_df['peak']
        max_drawdown = hist_df['drawdown'].min()
        
        return {
            'roi': total_roi,
            'sharpe': sharpe,
            'drawdown': max_drawdown,
            'final_bankroll': bankroll,
            'history': hist_df
        }

if __name__ == "__main__":
    bs = BettingStrategy()
