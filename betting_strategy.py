import pandas as pd
import numpy as np
import os
import json
import joblib
import torch
from ensemble_predictor import EnsemblePredictor
from ft_transformer import FTTransformer
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
    def __init__(self, models_dir='models', data_path='final_feature_matrix_with_per_min_1997_onward.csv', provider='mock', load_models=True):
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

        if load_models:
            self.load_resources()
        
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
                # Do not fill with zeros! Skip or set to NaN.
                continue
            
        # Combine into a results DF
        results = df_day[['player_name', 'playerteamName', 'opponentteamName', 'minutes']].copy()
        for t, p in predictions.items():
            results[f'pred_{t}'] = p
            
        # Add Confidence (Dummy for now, can be variance of ensemble members)
        # results['confidence'] = ...
        
        return results

    def calculate_ev(self, row, target, line, odds, side='Over'):
        """
        [Phase J.9] Calculate EV and Win Prob for a specific line and side.
        """
        # Load RMSE from report (J.6 calibrated)
        rmses = {'points': 5.2, 'rebounds': 2.4, 'assists': 2.1, 'three_pointers': 0.9}
        skews = {'points': 2.0, 'rebounds': 2.5, 'assists': 2.2, 'three_pointers': 1.5}
        
        rmse = rmses.get(target, 4.5)
        skew_a = skews.get(target, 0)
        
        from scipy.stats import skewnorm
        pred = row[f'pred_{target}']
        
        if side == 'Over':
            win_prob = 1 - skewnorm.cdf(line, skew_a, loc=pred, scale=rmse)
        else:
            win_prob = skewnorm.cdf(line, skew_a, loc=pred, scale=rmse)
        
        # J.9: Cap win_prob at 99.9% for realism
        win_prob = min(0.999, max(0.001, win_prob))
        
        # Decimal Odds
        dec_odds = odds
        if odds < 0:
            dec_odds = 1 + (100 / abs(odds))
        else:
            dec_odds = 1 + (odds / 100)
            
        ev = (win_prob * (dec_odds - 1)) - (1 - win_prob)
        return ev, win_prob

    def calculate_confidence(self, prediction, line, rmse):
        """
        Calculate confidence score (0-100) based on Z-score.
        """
        if rmse <= 0: return 0
        z_score = abs(prediction - line) / rmse
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

    def generate_bets(self, merged_df, bankroll=1000, confidence_threshold=10, kelly_fraction=0.25, min_ev=0.05):
        """
        [Phase J.9] Generate bets from normalized prop lines.
        Expects: player_name, market, line, odds_over, odds_under
        """
        # 1. Filter by Minutes
        min_col = 'pred_minutes' if 'pred_minutes' in merged_df.columns else 'minutes'
        if min_col in merged_df.columns:
            candidates = merged_df[merged_df[min_col] >= 20].copy()
        else:
            candidates = merged_df.copy()
            
        bets = []
        rmses = {'points': 5.2, 'rebounds': 2.4, 'assists': 2.1, 'three_pointers': 0.9}
        
        for _, row in candidates.iterrows():
            target = row['market']
            line = row['line']
            pred = row.get(f'pred_{target}')
            if pred is None: continue
            
            rmse = rmses.get(target, 4.5)
            delta = pred - line
            
            # Check Edge Filter Constraint (±1.5 for most, ±0.5 for threes)
            edge_threshold = 0.5 if target == 'three_pointers' else 1.5
            
            # Check for Over and Under favorites
            for side, o_col in zip(['Over', 'Under'], ['odds_over', 'odds_under']):
                odds = row.get(o_col)
                if odds is None or pd.isna(odds): continue
                
                # J.9: Favorites (-200 to -500) per user request (Strong Favorites)
                if odds > -200: continue
                if odds < -500: continue
                
                # Check Edge Filter Constraint
                if side == 'Over' and delta < edge_threshold: continue
                if side == 'Under' and delta > -edge_threshold: continue
                
                ev, win_prob = self.calculate_ev(row, target, line, odds, side=side)
                conf = self.calculate_confidence(pred, line, rmse)
                
                if ev > min_ev and conf >= confidence_threshold:
                    bets.append({
                        'player': row['player_name'],
                        'team': row.get('team', row.get('playerteamName', 'N/A')),
                        'game_id': row.get('event_id', row.get('gameId', 'N/A')),
                        'target': target,
                        'line': line,
                        'prediction': pred,
                        'delta': round(delta, 2),
                        'side': side,
                        'ev': ev,
                        'win_prob': win_prob,
                        'odds': odds,
                        'confidence': conf
                    })
                    
        bets_df = pd.DataFrame(bets)
        if bets_df.empty: return pd.DataFrame()
            
        def get_kelly(row):
            # Decimal odds payout multiplier
            o = row['odds']
            b = (o / 100) if o > 0 else (100 / abs(o))
            p = row['win_prob']
            q = 1 - p
            f = (b * p - q) / b if b > 0 else 0
            return max(0, f)
            
        bets_df['kelly_fraction'] = bets_df.apply(get_kelly, axis=1)
        # Scale Kelly by confidence and the user's kelly_fraction (risk setting)
        bets_df['stake_pct'] = bets_df['kelly_fraction'] * (bets_df['confidence'] / 100.0) * kelly_fraction
        bets_df['stake_amt'] = bets_df['stake_pct'] * bankroll
        
        return bets_df

    def select_top_props(self, bets_df, n=7):
        """
        [Phase J.9] Select top n bets for each target (prop) based on EV.
        Constraint: Heavy Favorites only (Negative Odds).
        """
        if bets_df.empty:
            return pd.DataFrame()
            
        top_props = []
        # Filter for Target Range (Favorites -200 to -500)
        favorites_df = bets_df[(bets_df['odds'] <= -200) & (bets_df['odds'] >= -500)].copy()
        
        for target in self.targets:
            tgt_df = favorites_df[favorites_df['target'] == target].copy()
            if tgt_df.empty: continue
            
            # Balance Edge and EV (Composite Score)
            # Normalize EV and Edge (Abs Delta) to 0-1 scale within the group
            tgt_df['abs_delta'] = tgt_df['delta'].abs()
            
            max_ev = tgt_df['ev'].max()
            min_ev = tgt_df['ev'].min()
            denom_ev = max_ev - min_ev if max_ev != min_ev else 1.0
            
            max_edge = tgt_df['abs_delta'].max()
            min_edge = tgt_df['abs_delta'].min()
            denom_edge = max_edge - min_edge if max_edge != min_edge else 1.0
            
            tgt_df['norm_ev'] = (tgt_df['ev'] - min_ev) / denom_ev
            tgt_df['norm_edge'] = (tgt_df['abs_delta'] - min_edge) / denom_edge
            
            # 50% EV, 50% Edge
            tgt_df['score'] = 0.5 * tgt_df['norm_ev'] + 0.5 * tgt_df['norm_edge']
            
            # Sort by Composite Score
            tgt_df = tgt_df.sort_values('score', ascending=False)
            top_props.append(tgt_df.head(n))
            
        if not top_props: return pd.DataFrame()
        return pd.concat(top_props)

    def generate_calibrated_round_robins(self, bets_df, n_candidates=15):
        """
        [DEPRECATED by J.8] Standard RR generation. Use generate_optimal_targeted_parlays.
        """
        res = self.generate_optimal_targeted_parlays(bets_df)
        return res['rr']

    def generate_optimal_targeted_parlays(self, bets_df, bankroll=20.0, kelly_fraction=0.10):
        """
        Phase J.9: Targeted construction (+100 to +1000) using favorites only.
        All bankroll allocation (15% cap) applies here.
        """
        from itertools import combinations
        if bets_df.empty: return {'rr': [], 'traditional': []}
        
        # 1. Filter: Range [-500, -200]
        favorites = bets_df[(bets_df['odds'] <= -200) & (bets_df['odds'] >= -500)].copy()
        if len(favorites) < 3:
            print("[J.9] Not enough -200+ favorites for parlay construction.")
            return {'rr': [], 'traditional': []}
            
        # 2. Generate all valid combinations (3-6 legs)
        all_valid_combos = []
        # J.9: Increase candidates to 50 to handle strict usage cap
        candidates = favorites.sort_values('ev', ascending=False).head(50).to_dict('records')
        
        for size in [3, 4, 5, 6]:
            if len(candidates) < size: continue
            for combo in combinations(candidates, size):
                gids = {c['game_id'] for c in combo}
                if len(gids) < size: continue
                pnames = {c['player'] for c in combo}
                if len(pnames) < size: continue
                
                dec_odds = 1.0
                combo_prob = 1.0
                legs_desc = []
                leg_ids = []
                avg_conf = 0
                for leg in combo:
                    o = leg['odds']
                    d = (1 + o/100) if o > 0 else (1 + 100/abs(o))
                    dec_odds *= d
                    combo_prob *= leg['win_prob']
                    legs_desc.append(f"{leg['player']} ({leg.get('team', 'UNK')}) ({leg['target']} {leg['side']} @ {leg['odds']})")
                    leg_ids.append(f"{leg['player']}_{leg['target']}_{leg['side']}")
                    avg_conf += leg.get('confidence', 50)
                
                avg_conf /= len(combo)
                combo_ev = (combo_prob * (dec_odds - 1)) - (1 - combo_prob)
                us_odds = int((dec_odds - 1) * 100) if dec_odds >= 2.0 else int(-100 / (dec_odds - 1))
                
                # Check Max Odds Constraint (+700)
                if us_odds > 700: continue
                
                # Kelly Calculation for Parlay
                b = (dec_odds - 1)
                p = combo_prob
                q = 1 - p
                f = (b * p - q) / b if b > 0 else 0
                kelly_stake = max(0, f) * (avg_conf / 100.0) * kelly_fraction
                
                all_valid_combos.append({
                    'size': size,
                    'legs': legs_desc,
                    'combined_odds': us_odds,
                    'combined_prob': combo_prob,
                    'ev': combo_ev,
                    'leg_data': list(combo),
                    'combo_hash': tuple(sorted(leg_ids)),
                    'stake_pct': kelly_stake
                })
        
        if not all_valid_combos: return {'rr': [], 'traditional': []}
        
        results = {'rr': [], 'traditional': []}
        used_hashes = set()
        leg_usage = {} # Prop ID -> Count
        MAX_USAGE = 1 # Each prop used in exactly one parlay MAX for ultra-diversity

        def can_use_combo(combo_hashes):
            """Diversity check: Ensure no leg exceeds MAX_USAGE."""
            for h in combo_hashes:
                if leg_usage.get(h, 0) >= MAX_USAGE:
                    return False
            return True

        def increment_usage(combo_hashes):
            for h in combo_hashes:
                leg_usage[h] = leg_usage.get(h, 0) + 1

        # A. Round Robin (Targeted up to +1000)
        rr_targets = [(150, 3), (300, 3), (500, 4), (750, 5), (1000, 6)]
        for target, size in rr_targets:
            best_match = None
            min_diff = 999999
            for c in all_valid_combos:
                if c['size'] != size or c['combo_hash'] in used_hashes: continue
                if not can_use_combo(c['combo_hash']): continue # Diversity
                
                diff = abs(c['combined_odds'] - target)
                if diff < min_diff:
                    min_diff = diff
                    best_match = c
            if best_match:
                rr_entry = best_match.copy()
                rr_entry['type'] = f"{size}-Leg RR"
                results['rr'].append(rr_entry)
                used_hashes.add(best_match['combo_hash'])
                increment_usage(best_match['combo_hash'])
        
        # B. Traditional (5 total, scaled 100-1000)
        targets = [100, 250, 500, 750, 1000]
        for target in targets:
            best_match = None
            min_diff = 999999
            for c in all_valid_combos:
                if c['combo_hash'] in used_hashes: continue
                if not can_use_combo(c['combo_hash']): continue # Diversity
                if c['combined_odds'] < 0: continue # Strict Positive Odds only for Traditional
                
                diff = abs(c['combined_odds'] - target)
                if diff < min_diff:
                    min_diff = diff
                    best_match = c
            if best_match:
                best_match['type'] = f"{best_match['size']}-Leg"
                results['traditional'].append(best_match)
                used_hashes.add(best_match['combo_hash'])
                increment_usage(best_match['combo_hash'])
        
        # C. BANKROLL ALLOCATION (J.9)
        all_parlays = results['rr'] + results['traditional']
        total_p_exposure = sum(p['stake_pct'] for p in all_parlays)
        DAILY_CAP = 0.15
        
        scale = 1.0
        if total_p_exposure > DAILY_CAP:
            scale = DAILY_CAP / total_p_exposure
            print(f"[J.9] Parlay exposure capped: {total_p_exposure:.1%} -> {DAILY_CAP:.0%}")
            
        for p in all_parlays:
            p['stake_pct'] *= scale
            p['stake_amt'] = p['stake_pct'] * bankroll
                
        return results

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

    def generate_lotto_parlays(self, top_props_df, n=3):
        """
        [Phase J.9] Generate 'Lotto Slips' - high odds (+1000+) parlays.
        - Start odds >= +1000.
        - No odds cap for upper bound.
        - Max 10 legs.
        - Disregard usage limits (can reuse props from main card).
        - Must be unique among lotto slips (disjoint legs).
        - Source: top_props_df (Top 7 props).
        """
        from itertools import combinations
        if top_props_df.empty: return []

        candidates = top_props_df.to_dict('records')
        
        # Need at least ~5-6 legs to hit +1000 with strong favorites (-200 to -500 avg ~-300 = 1.33. 1.33^8 ~ 9.7)
        # We will try sizes 5 to 10
        sizes = [5, 6, 7, 8, 9, 10]
        
        potential_lottos = []

        for size in sizes:
            if len(candidates) < size: continue
            # Limit iterations
            combos = combinations(candidates, size)
            
            count = 0 
            for combo in combos:
                count += 1
                if count > 5000: break
                
                # Check Unique Games/Players? 
                # Standard logic: unique player at least?
                players = {c['player'] for c in combo}
                if len(players) < size: continue # Ensure different players for safety/simplicity
                
                dec_odds = 1.0
                combo_prob = 1.0
                legs_desc = []
                leg_ids = []
                
                for leg in combo:
                    o = leg['odds']
                    d = (1 + o/100) if o > 0 else (1 + 100/abs(o))
                    dec_odds *= d
                    combo_prob *= leg['win_prob']
                    legs_desc.append(f"{leg['player']} ({leg.get('team', 'UNK')}) ({leg['target']} {leg['side']} @ {leg['odds']})")
                    leg_ids.append(leg['player'] + leg['target'])
                
                us_odds = int((dec_odds - 1) * 100) if dec_odds >= 2.0 else int(-100 / (dec_odds - 1))
                
                if us_odds < 1000: continue
                
                # EV Calc
                combo_ev = (combo_prob * (dec_odds - 1)) - (1 - combo_prob)
                
                potential_lottos.append({
                    'legs': legs_desc,
                    'combined_odds': us_odds,
                    'combined_prob': combo_prob,
                    'ev': combo_ev,
                    'combo_hash': set(leg_ids)
                })

        # Sort by EV descending
        potential_lottos.sort(key=lambda x: x['ev'], reverse=True)
        
        final_lottos = []
        
        for l in potential_lottos:
            if len(final_lottos) >= n: break
            
            # Disjoint check against existing lottos
            overlap = False
            for existing in final_lottos:
                if not l['combo_hash'].isdisjoint(existing['combo_hash']):
                    overlap = True
                    break
            
            if overlap: continue
            
            final_lottos.append(l)
        
        return final_lottos

    def generate_lotto_parlays(self, top_props_df, n=3):
        """
        [Phase J.9] Generate 'Lotto Slips' - high odds (+1000+) parlays.
        - Start odds >= +1000.
        - No odds cap for upper bound.
        - Max 10 legs.
        - Disregard usage limits (can reuse props from main card).
        - Must be unique among lotto slips (disjoint legs).
        - Source: top_props_df (Top 7 props).
        """
        from itertools import combinations
        if top_props_df.empty: return []

        candidates = top_props_df.to_dict('records')
        
        # Need at least ~5-6 legs to hit +1000 with strong favorites (-200 to -500 avg ~-300 = 1.33. 1.33^8 ~ 9.7)
        # We will try sizes 5 to 10
        sizes = [5, 6, 7, 8, 9, 10]
        
        potential_lottos = []

        for size in sizes:
            if len(candidates) < size: continue
            # Limit iterations
            combos = combinations(candidates, size)
            
            count = 0 
            for combo in combos:
                count += 1
                if count > 5000: break
                
                # Check Unique Games/Players? 
                # Standard logic: unique player at least?
                players = {c['player'] for c in combo}
                if len(players) < size: continue # Ensure different players for safety/simplicity
                
                dec_odds = 1.0
                combo_prob = 1.0
                legs_desc = []
                leg_ids = []
                
                for leg in combo:
                    o = leg['odds']
                    d = (1 + o/100) if o > 0 else (1 + 100/abs(o))
                    dec_odds *= d
                    combo_prob *= leg['win_prob']
                    legs_desc.append(f"{leg['player']} ({leg.get('team', 'UNK')}) ({leg['target']} {leg['side']} @ {leg['odds']})")
                    leg_ids.append(leg['player'] + leg['target'])
                
                us_odds = int((dec_odds - 1) * 100) if dec_odds >= 2.0 else int(-100 / (dec_odds - 1))
                
                if us_odds < 1000: continue
                
                # EV Calc
                combo_ev = (combo_prob * (dec_odds - 1)) - (1 - combo_prob)
                
                potential_lottos.append({
                    'legs': legs_desc,
                    'combined_odds': us_odds,
                    'combined_prob': combo_prob,
                    'ev': combo_ev,
                    'combo_hash': set(leg_ids)
                })

        # Sort by EV descending
        potential_lottos.sort(key=lambda x: x['ev'], reverse=True)
        
        final_lottos = []
        
        for l in potential_lottos:
            if len(final_lottos) >= n: break
            
            # Disjoint check against existing lottos
            overlap = False
            for existing in final_lottos:
                if not l['combo_hash'].isdisjoint(existing['combo_hash']):
                    overlap = True
                    break
            
            if overlap: continue
            
            final_lottos.append(l)
        
        return final_lottos

if __name__ == "__main__":
    bs = BettingStrategy()
