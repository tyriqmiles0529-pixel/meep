import pandas as pd
import numpy as np
import os
import json
import joblib
import torch
from ensemble_predictor import EnsemblePredictor
from ft_transformer import FTTransformer, FTTransformerFeatureExtractor
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
            pred.model_dir = self.models_dir
            try:
                pred.load_models(suffix=f"_{target}")
                self.predictors[target] = pred
            except Exception as e:
                print(f"Failed to load default {target} model: {e}")
            
            # Load FT-Transformer
            ft_path = os.path.join(self.models_dir, f"global_ft_2025", "ft_transformer.pt")
            if os.path.exists(ft_path):
                # We need to know cardinalities to init the model structure first
                cat_cols = self.processor.get_cat_cols()
                cardinalities = [len(self.processor.label_encoders[col].classes_) for col in cat_cols]
                
                ft = FTTransformerFeatureExtractor(cardinalities, embed_dim=16, device='cpu')
                ft.load(ft_path)
                self.ft_extractors['global'] = ft
            else:
                pass 
                # print(f"Warning: FT-Transformer not found at {ft_path}")

    def generate_predictions(self, date_str):
        df_day = self.processor.df[self.processor.df['date'] == date_str].copy()
        
        if df_day.empty:
            print(f"No games found for {date_str}")
            return None
            
        # Generate Embeddings
        if 'global' in self.ft_extractors:
            cat_cols = self.processor.get_cat_cols()
            X_cat = df_day[cat_cols].values
            embeddings = self.ft_extractors['global'].transform(X_cat)
            
            emb_cols = [f"emb_{i}" for i in range(embeddings.shape[1])]
            df_emb = pd.DataFrame(embeddings, columns=emb_cols, index=df_day.index)
            df_day = pd.concat([df_day, df_emb], axis=1)
            
        predictions = {}
        for target in self.targets:
            features_to_use = self.processor.feature_columns.copy()
            if 'global' in self.ft_extractors:
                 features_to_use += [c for c in df_day.columns if c.startswith('emb_')]
            
            valid_features = [f for f in features_to_use if f in df_day.columns]
            X_pred = df_day[valid_features]
            
            if target in self.predictors:
                preds = self.predictors[target].predict(X_pred, use_stacking=True)
                predictions[target] = preds
            else:
                continue
            
        results = df_day[['player_name', 'playerteamName', 'opponentteamName', 'minutes']].copy()
        for t, p in predictions.items():
            results[f'pred_{t}'] = p
            
        return results

    def calculate_ev(self, row, target, line, odds, side='Over', is_rookie=False):
        """
        [Phase L1] Player Identity Refresh (NBA 2025-2026)
        Constraints:
        - Bench Correction (MPG < 24)
        - Minutes Gating
        - Overs Bias (Smart)
        - Early-Season Confidence Envelope
        - Rookie Tax (High Variance)
        - Breakout/Demotion Detection (Live Role vs Historical Model)
        """
        # Load RMSE from report (J.6 calibrated)
        rmses = {'points': 5.2, 'rebounds': 2.4, 'assists': 2.1, 'three_pointers': 0.9}
        skews = {'points': 2.0, 'rebounds': 2.5, 'assists': 2.2, 'three_pointers': 1.5}
        
        rmse = rmses.get(target, 4.5)
        skew_a = skews.get(target, 0)
        
        from scipy.stats import skewnorm
        pred = row[f'pred_{target}']
        minutes = row.get('minutes', 0)
        
        # Phase L1: Role Context
        hist_minutes = row.get('pred_minutes', minutes) 
        if pd.isna(hist_minutes): hist_minutes = minutes
        
        role_delta = minutes - hist_minutes
        
        if side == 'Over':
            win_prob = 1 - skewnorm.cdf(line, skew_a, loc=pred, scale=rmse)
        else:
            win_prob = skewnorm.cdf(line, skew_a, loc=pred, scale=rmse)
        
        # --- PHASE L1 CORRECTIONS ---
        
        # 1. Breakout / Demotion Detection (Dynamic Identity)
        # ENHANCEMENT: Role Persistence Gate
        # Only apply identity shift if trend is visible in Last 3 Games (L3).
        # Prevents reacting to one-off spot starts or blowouts.
        
        l3_mins = row.get('minutes_rolling_mean_3', row.get('minutes_mean_3', minutes))
        
        is_breakout_trend = (role_delta > 0) and (l3_mins > hist_minutes + 2.0)
        is_demotion_trend = (role_delta < 0) and (l3_mins < hist_minutes - 2.0)
        
        if abs(role_delta) >= 5.0:
            if role_delta > 0 and is_breakout_trend: # CONFIRMED BREAKOUT
                if side == 'Over': win_prob *= 1.10 
                if side == 'Under': win_prob *= 0.85 
            elif role_delta < 0 and is_demotion_trend: # CONFIRMED DEMOTION
                if side == 'Under': win_prob *= 1.05 
                if side == 'Over': win_prob *= 0.85 
        
        # 2. Minutes Gating & Bench Correction
        if minutes < 18:
            win_prob *= 0.80 
        elif minutes < 24:
            win_prob *= 0.90 
            
        # 3. Overs Bias (The Right Way)
        if side == 'Over' and minutes >= 24:
            win_prob = min(0.99, win_prob * 1.05) 
        elif side == 'Under':
            win_prob *= 0.92 
        
        # 4. Rookie Tax (New)
        if is_rookie:
             win_prob *= 0.85 
             
        # 5. Early-Season Confidence Envelope (Governance)
        if minutes >= 24 and not is_rookie:
            cap = 0.90 
        else:
            cap = 0.70 
            
        win_prob = min(cap, max(0.001, win_prob))
        
        # Decimal Odds
        dec_odds = odds
        if odds < 0:
            dec_odds = 1 + (100 / abs(odds))
        else:
            dec_odds = 1 + (odds / 100)
            
        ev = (win_prob * (dec_odds - 1)) - (1 - win_prob)
        return ev, win_prob

    def calculate_confidence(self, prediction, line, rmse, minutes=30, is_rookie=False):
        """
        Calculate confidence score (0-100) based on Z-score.
        GOVERNANCE: Phase S (Survival Mode) Envelopes.
        """
        if rmse <= 0: return 0
        z_score = abs(prediction - line) / rmse
        raw_score = (z_score / 2.0) * 100
        
        # Phase S: Survival Caps (Strict Variance Control)
        if is_rookie:
            cap = 65.0
        elif minutes < 24:
            cap = 60.0 # Bench Players
        else:
            cap = 90.0
            
        return min(cap, max(0.0, raw_score))

    def load_season_models(self, season):
        print(f"Loading models for Season {season}...")
        train_season = season - 1
        for target in self.targets:
            pred = EnsemblePredictor()
            pred.model_dir = os.path.join(self.models_dir, target)
            try:
                pred.load_models(suffix=f"_{train_season}") 
                self.predictors[target] = pred
            except Exception as e:
                print(f"Error loading {target} model for {train_season}: {e}")
                pass
        
        ft_path = os.path.join(self.models_dir, f"global_ft_{train_season}", "ft_transformer.pt")
        if os.path.exists(ft_path):
            try:
                cat_cols = self.processor.get_cat_cols()
                cardinalities = [len(self.processor.label_encoders[col].classes_) for col in cat_cols]
                ft = FTTransformerFeatureExtractor(cardinalities, embed_dim=16, device='cpu')
                ft.load(ft_path)
                self.ft_extractors['global'] = ft
            except Exception as e:
                print(f"DEBUG: Failed to init/load FT: {e}")
        else:
            print(f"Warning: FT-Transformer not found for {train_season} at {ft_path}")

    def generate_bets(self, merged_df, bankroll=1000, confidence_threshold=10, kelly_fraction=0.25, min_ev=0.0):
        """
        [Phase J] Generate bets from normalized prop lines.
        Rules:
        - Filters: >= 20 mins.
        - Edge: >= 1.5 diff (Strict).
        - Bias: Overs First.
        - Unders only if Over fails AND Under is significantly better.
        """
        # 1. Filter by Minutes
        min_col = 'pred_minutes' if 'pred_minutes' in merged_df.columns else 'minutes'
        if min_col in merged_df.columns:
            candidates = merged_df[merged_df[min_col] >= 20].copy()
        else:
            candidates = merged_df.copy()
            
        bets = []
        rmses = {'points': 5.2, 'rebounds': 2.4, 'assists': 2.1, 'three_pointers': 0.9}
        
        # Verify Known Players for Rookie Logic
        known_players = set()
        if hasattr(self, 'processor') and hasattr(self.processor, 'label_encoders'):
             if 'player_name' in self.processor.label_encoders:
                 known_players = set(self.processor.label_encoders['player_name'].classes_)
        
        for _, row in candidates.iterrows():
            target = row['market']
            line = row['line']
            pred = row.get(f'pred_{target}')
            if pred is None: continue
            
            rmse = rmses.get(target, 4.5)
            
            minutes_val = row.get(min_col, 0)
            player_name = row['player_name']
            
            # Phase L: Rookie Detection
            # If player not in historical encoder classes, treat as Rookie (High Variance)
            is_rookie = player_name not in known_players
            
            # Phase J: Primary Directional Bias: OVERS
            # We strictly check Over first.
            
            over_odds = row.get('odds_over')
            under_odds = row.get('odds_under')
            
            # --- EVALUATE OVER ---
            over_valid = False
            over_bet_data = None
            
            if over_odds is not None and not pd.isna(over_odds):
                delta_over = pred - line
                
                # Edge Constraint: >= 1.5
                edge_threshold = 1.5
                
                if delta_over >= edge_threshold:
                    ev, prob = self.calculate_ev(row, target, line, over_odds, side='Over', is_rookie=is_rookie)
                    conf = self.calculate_confidence(pred, line, rmse, minutes=minutes_val, is_rookie=is_rookie)
                    if ev > min_ev and prob >= 0.50: # Enforce EV > 0
                        over_valid = True
                        over_bet_data = {
                            'player': player_name,
                            'team': row.get('team', row.get('playerteamName', 'N/A')),
                            'game_id': row.get('event_id', row.get('gameId', 'N/A')),
                            'target': target,
                            'line': line,
                            'prediction': pred,
                            'delta': round(delta_over, 2),
                            'side': 'Over',
                            'ev': ev,
                            'win_prob': prob,
                            'odds': over_odds,
                            'confidence': conf,
                            'minutes': minutes_val
                        }

            # --- EVALUATE UNDER ---
            under_valid = False
            under_bet_data = None
            
            if under_odds is not None and not pd.isna(under_odds):
                delta_under = line - pred
                if delta_under >= edge_threshold:
                    ev, prob = self.calculate_ev(row, target, line, under_odds, side='Under', is_rookie=is_rookie)
                    conf = self.calculate_confidence(pred, line, rmse, minutes=minutes_val, is_rookie=is_rookie)
                    if ev > min_ev and prob >= 0.50:
                        under_valid = True
                        under_bet_data = {
                            'player': player_name,
                            'team': row.get('team', row.get('playerteamName', 'N/A')),
                            'game_id': row.get('event_id', row.get('gameId', 'N/A')),
                            'target': target,
                            'line': line,
                            'prediction': pred,
                            'delta': round(delta_under, 2),
                            'side': 'Under',
                            'ev': ev,
                            'win_prob': prob,
                            'odds': under_odds,
                            'confidence': conf,
                            'minutes': minutes_val
                        }
            
            # --- SELECTION LOGIC ---
            selected_bet = None
            
            if over_valid:
                selected_bet = over_bet_data
            elif under_valid:
                selected_bet = under_bet_data
                
            if selected_bet:
                bets.append(selected_bet)

        return pd.DataFrame(bets)

    def select_top_props(self, bets_df, n=7):
        """
        [Phase J.9] Select top n bets for each target (prop) based on EV.
        Constraint: Relaxed Favorites (-120 to -500) enabling mathematically viable Round Robins.
        """
        if bets_df.empty:
            return pd.DataFrame()
            
        top_props = []
        # Filter for Viable Odds Range (-120 to -500)
        # We need slightly better pricing (-120 to -200) to make RR 2/3 hedges break even.
        favorites_df = bets_df[(bets_df['odds'] <= -120) & (bets_df['odds'] >= -500)].copy()
        
        for target in self.targets:
            tgt_df = favorites_df[favorites_df['target'] == target].copy()
            if tgt_df.empty: continue
            
            # Balance Edge and EV (Composite Score)
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

    def generate_optimal_targeted_parlays(self, bets_df, bankroll=20.0, kelly_fraction=0.10, target_units=10.0):
        """
        Phase J: Tiered Parlay Generation (Hit Rate First).
        Phase S2: Accepts target_units for Volume Scaling.
        """
        from itertools import combinations
        if bets_df.empty: return {'rr': [], 'traditional': []}
        
        # Phase S Diversity Patch: Split Pools to Minimize Correlation Risk
        # 1. Singles Pool (Core): Top 6 Best Bets.
        # 2. Parlay Pool (Growth/Moonshot): Next 12 Best Bets.
        # This ensures that if the Core thesis fails, the Parlays (acting as a hedge) are not automatically dead.
        
        sorted_candidates = bets_df.sort_values('win_prob', ascending=False)
        singles_pool = sorted_candidates.head(6).to_dict('records')
        parlay_pool = sorted_candidates.iloc[6:18].to_dict('records')
        
        used_leg_ids = set()
        results = {'traditional': [], 'rr': []}
        
        tiers = [
            {'name': 'Core 1',     'target_dec': 2.0, 'min_legs': 2, 'max_legs': 2},
            {'name': 'Core 2',     'target_dec': 3.0, 'min_legs': 2, 'max_legs': 3},
            # Phase S2 Constraint: MAX 3 LEGS (Hard Cap)
            {'name': 'Growth',     'target_dec': 4.0, 'min_legs': 3, 'max_legs': 3},
            {'name': 'Moonshot',   'target_dec': 5.0, 'min_legs': 3, 'max_legs': 3},
            {'name': 'Hail Mary',  'target_dec': 6.0, 'min_legs': 3, 'max_legs': 3}
        ]
        
        for tier in tiers:
            target = tier['target_dec']
            pool = parlay_pool # UPDATED: Use Diversity Pool for Parlays
            valid_combos = []
            
            for size in range(tier['min_legs'], tier['max_legs'] + 1):
                if len(pool) < size: continue
                
                for combo in combinations(pool, size):
                    players = set()
                    game_ids = set()
                    is_valid_corr = True
                    comb_prob = 1.0
                    comb_dec = 1.0
                    
                    for leg in combo:
                        p = leg['player']
                        gid = leg['game_id']
                        if p in players: is_valid_corr = False; break
                        if gid in game_ids: is_valid_corr = False; break 
                        players.add(p)
                        game_ids.add(gid)
                        
                        o = leg['odds']
                        d = (1 + o/100) if o > 0 else (1 + 100/abs(o))
                        comb_dec *= d
                        comb_prob *= leg['win_prob']
                        
                    if not is_valid_corr: continue
                    
                    dist = abs(comb_dec - target)
                    valid_combos.append({
                        'legs': combo,
                        'combined_dec': comb_dec,
                        'combined_prob': comb_prob,
                        'dist': dist
                    })
            
            if not valid_combos: continue
            
            valid_combos.sort(key=lambda x: x['dist'])
            top_candidates = valid_combos[:5]
            top_candidates.sort(key=lambda x: x['combined_prob'], reverse=True)
            
            best = top_candidates[0]
            
            legs_fmt = []
            avg_conf = 0.0
            for l in best['legs']:
                used_leg_ids.add(l['player'] + l['target'])
                legs_fmt.append(f"{l['player']} ({l['team']}) - {l['target']} {l['side']} {l['line']} ({l['odds']})")
                avg_conf += l.get('confidence', 50.0)
            
            avg_conf /= len(best['legs'])
                
            us_odds = int((best['combined_dec'] - 1) * 100) if best['combined_dec'] >= 2.0 else int(-100 / (best['combined_dec'] - 1))
            
            # EV & Kelly
            b = best['combined_dec'] - 1
            p = best['combined_prob']
            q = 1 - p
            f = (b * p - q) / b if b > 0 else 0
            
            stake_pct = max(0, f) * (avg_conf / 100.0) * kelly_fraction
            stake_amt = stake_pct * bankroll
            
            combo_ev = (p * b) - q
            
            results['traditional'].append({
                'name': tier['name'],
                'legs': legs_fmt,
                'combined_odds': us_odds,
                'prob': best['combined_prob'],
                'ev': combo_ev,
                'stake_amt': stake_amt,
                'stake_pct': stake_pct
            })

        # B. ROUND ROBIN GENERATION (Phase K.3: The Anchor)
        # 3 Legs, 2-way Combo (3x2)
        # Constraint: Odds -120 to -500
        # Phase L2: Use PARLAY POOL (Diversity) to ensure hedge against Core failure.
        
        rr_pool = [c for c in parlay_pool if -500 <= c['odds'] <= -120]
        
        if len(rr_pool) >= 3:
            rr_combos = combinations(rr_pool, 3)
            
            best_rr = None
            best_rr_ev = -9999
            
            for combo in rr_combos:
                # Validation
                players = set()
                gids = set()
                valid = True
                avg_prob = 0
                for leg in combo:
                    if leg['player'] in players: valid = False; break
                    if leg['game_id'] in gids: valid = False; break
                    players.add(leg['player'])
                    gids.add(leg['game_id'])
                    avg_prob += leg['win_prob']
                
                if not valid: continue
                avg_prob /= 3.0
                
                # Check Profit Gate (simplified for heuristic selection)
                # We want high win prob and decent odds
                
                # EV Approx: Just use Avg Prob for ranking
                if avg_prob > best_rr_ev:
                    best_rr_ev = avg_prob
                    best_rr = combo
            
            if best_rr:
                # Format RR
                legs_fmt = []
                avg_conf = 0.0
                comb_prob = 1.0 # Not really accurate for RR, checks all outcomes
                
                for l in best_rr:
                    legs_fmt.append(f"{l['player']} ({l['team']}) - {l['target']} {l['side']} {l['line']} ({l['odds']})")
                    avg_conf += l.get('confidence', 50.0)
                
                avg_conf /= 3.0
                
                # Fake stake_pct to get allocation
                stake_pct = (avg_conf / 100.0) * kelly_fraction
                
                results['rr'].append({
                    'name': 'Round Robin (3x2)',
                    'legs': legs_fmt,
                    'combined_odds': 0, # N/A
                    'prob': best_rr_ev,
                    'ev': 0,
                    'stake_amt': 0,
                    'stake_pct': stake_pct
                })

        # C. UNIT ALLOCATION (Phase S: Survival Mode)
        # Rule 1: Dynamic Units (Passed via Slate Filter)
        # Rule 2: Buckets (Phase S Updates)
        # Core (Singles): 65%
        # Growth (2-Leg Parlays): 25%
        # Moonshot (3-4 Legs): 10%
        
        bucket_alloc = {
            'core': 0.65 * target_units,
            'growth': 0.25 * target_units,
            'moonshot': 0.10 * target_units
        }
        
        # Extract Singles (Core)
        results['singles'] = []
        # Take All from Singles Pool (Top 6)
        for leg in singles_pool:
             results['singles'].append({
                 'name': f"Single: {leg['player']}",
                 'legs': [f"{leg['player']} ({leg['team']}) - {leg['target']} {leg['side']} {leg['line']} ({leg['odds']})"],
                 'prob': leg['win_prob'],
                 'stake_pct': 1.0 # Flat weight marker
             })

        # Classify Investments
        core_investments = results['singles']
        growth_investments = []
        moonshot_investments = []
        
        # 2-Legs go to Growth
        for p in results['traditional']:
            leg_count = len(p['legs'])
            if leg_count == 2:
                growth_investments.append(p)
            else:
                moonshot_investments.append(p)

        # RRs go to Moonshot (3-Leg structures)
        for p in results['rr']:
            moonshot_investments.append(p)
                
        # Helper to distribute bucket units FLATLY (Survival Rule)
        def distribute(items, bucket_units):
            if not items: return
            # Flat Sizing: Divide bucket equally
            val = bucket_units / len(items)
            for i in items: 
                i['units'] = val
                i['stake_amt'] = i['units'] * (bankroll / 15.0) # Visual approx for output

        distribute(core_investments, bucket_alloc['core'])
        distribute(growth_investments, bucket_alloc['growth'])
        distribute(moonshot_investments, bucket_alloc['moonshot'])

        return results

    def generate_lotto_parlays(self, top_props_df, n=3):
        """
        [Phase J.9] Generate 'Lotto Slips' - high odds (+1000+) parlays.
        """
        from itertools import combinations
        if top_props_df.empty: return []

        candidates = top_props_df.to_dict('records')
        sizes = [5, 6, 7, 8, 9, 10]
        potential_lottos = []

        for size in sizes:
            if len(candidates) < size: continue
            combos = combinations(candidates, size)
            
            count = 0 
            for combo in combos:
                count += 1
                if count > 5000: break
                
                players = {c['player'] for c in combo}
                if len(players) < size: continue
                
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
                
                combo_ev = (combo_prob * (dec_odds - 1)) - (1 - combo_prob)
                
                potential_lottos.append({
                    'legs': legs_desc,
                    'combined_odds': us_odds,
                    'combined_prob': combo_prob,
                    'ev': combo_ev,
                    'combo_hash': set(leg_ids)
                })

        potential_lottos.sort(key=lambda x: x['ev'], reverse=True)
        final_lottos = []
        
        for l in potential_lottos:
            if len(final_lottos) >= n: break
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
