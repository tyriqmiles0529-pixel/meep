import numpy as np
import pandas as pd
from scipy.stats import norm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SportsbookSimulator:
    """
    Simulates realistic sportsbook odds based on true (model) probabilities.
    Includes mechanisms for vig (overround), market noise, and favorite bias.
    """
    
    def __init__(self, vig=0.045, noise_sigma=0.03, favorite_bias=0.02, liquidity_max_odds=15.0):
        """
        :param vig: The theoretical overround (e.g., 0.045 for -110/-110 lines standard is ~4.5%)
        :param noise_sigma: Standard deviation of Gaussian noise added to probabilities.
        :param favorite_bias: Percentage to shade favorite odds (e.g., 0.02 = 2%).
        :param liquidity_max_odds: Cap on extreme odds (decimal).
        """
        self.vig = vig
        self.noise_sigma = noise_sigma
        self.favorite_bias = favorite_bias
        self.liquidity_max_odds = liquidity_max_odds

    def simulate_book_odds(self, true_prob, is_over=True):
        """
        Generates simulated decimal odds for a given probability.
        
        Formula Stack:
        1. Fair Odds: 1/p
        2. Overround: p_book = p * (1 + vig)
        3. Favorite Bias: if p > 0.5: p_book *= (1 + bias)
        4. Noise: p_final = p_book * (1 + N(0, sigma))
        5. Odds: 1 / p_final (clipped)
        """
        # 1. Base Book Probability with Overround
        # Distribute vig proportionally? 
        # Standard approach: Implied Prob = True Prob + (Vig / 2) ? 
        # User defined: p_book = p * (1 + vig)
        p_book = true_prob * (1 + self.vig)
        
        # 2. Favorite Bias
        # If this outcome is the favorite (p > 50%), the book shades it further
        if true_prob > 0.5:
             p_book = p_book * (1 + self.favorite_bias)
             
        # 3. Market Noise
        # Noise factor N(0, sigma)
        noise = np.random.normal(0, self.noise_sigma)
        p_noisy = p_book * (1 + noise)
        
        # Clip to valid probability range (leaving room for the other side)
        # e.g. 0.01 to 0.99
        p_clipped = np.clip(p_noisy, 0.01, 0.99)
        
        # 4. Convert to Decimal Odds
        raw_odds = 1.0 / p_clipped
        
        # 5. Rounding & Throttle
        # Standard American odds often convert to specific decimal steps, but 2 decimals is fine.
        final_odds = np.round(raw_odds, 2)
        
        # Liquidity Cap (Books rarely offer 100.0 odds on props)
        final_odds = min(final_odds, self.liquidity_max_odds)
        
        return final_odds

    def calculate_ev(self, model_prob, decimal_odds):
        """
        EV = (Prob_Win * (Odds - 1)) - (Prob_Loss * 1)
        EV = P(d-1) - (1-P)
        """
        if decimal_odds <= 1.0: return -1.0
        
        win_amt = decimal_odds - 1.0
        ev = (model_prob * win_amt) - (1 - model_prob)
        return ev

    def kelly_criterion(self, model_prob, decimal_odds, fraction=1.0):
        """
        Kelly = (b*p - q) / b
        where b = decimal_odds - 1
        """
        if decimal_odds <= 1.0: return 0.0
        
        b = decimal_odds - 1.0
        q = 1 - model_prob
        f = (b * model_prob - q) / b
        
        return max(0.0, f * fraction)


class BacktestEngine:
    """
    Executes the betting strategy over historical data using simulated odds.
    """
    
    def __init__(self, predictions_df, simulator: SportsbookSimulator):
        self.predictions = predictions_df
        self.simulator = simulator
        self.results = []
        self.bankroll_history = []
        
    def run(self, initial_bankroll=1000.0, min_ev=0.02, kelly_fraction=0.25, base_stake=10.0, max_stake=20.0, rr_pct=0.75):
        """
        Iterates through the dataframe, simulating odds and placing bets.
        Uses PROGRESSIVE STAKING with daily budget that scales with bankroll.
        
        :param base_stake: Starting daily stake (default $10)
        :param max_stake: Maximum daily stake cap (default $20)
        :param rr_pct: Percent of daily budget for Round Robin parlays (default 75%)
        """
        from itertools import combinations
        
        current_bankroll = initial_bankroll
        self.bankroll_history = [current_bankroll]
        self.results = [] # Reset
        
        # Ensure we have date sorted
        df = self.predictions.sort_values('date')
        
        for date, group in df.groupby('date'):
            # FULLY DYNAMIC STAKING
            # Daily budget = 1/5 of bankroll (20% - balanced risk/reward)
            # At $25 bankroll: $5/day → ~$0.55 per parlay
            # At $50 bankroll: $10/day → ~$1.11 per parlay
            daily_budget = current_bankroll / 5
            
            # Skip if bankroll too low
            if current_bankroll < 5:
                self.bankroll_history.append(current_bankroll)
                continue
                
            daily_bets = []
            
            # 1. Generate Odds & Evaluate EV for all candidates
            for idx, row in group.iterrows():
                # MINUTES FILTER: Skip players with < 20 predicted minutes
                pred_minutes = row.get('pred_minutes', row.get('minutes', 30))
                if pred_minutes < 20:
                    continue
                
                house_prob_over = row.get('prob_over')
                model_prob_over = row.get('true_prob', house_prob_over)
                house_prob_under = 1 - house_prob_over
                model_prob_under = 1 - model_prob_over
                
                odds_over = self.simulator.simulate_book_odds(house_prob_over)
                odds_under = self.simulator.simulate_book_odds(house_prob_under)
                
                ev_over = self.simulator.calculate_ev(model_prob_over, odds_over)
                ev_under = self.simulator.calculate_ev(model_prob_under, odds_under)
                
                # Odds Filter: -500 to +300 American = 1.20 to 4.00 Decimal
                MIN_ODDS = 1.20  # -500
                MAX_ODDS = 4.00  # +300
                
                # Get game_id for correlation filtering (use team as proxy if not available)
                game_id = row.get('game_id', row.get('team', str(idx)))
                
                if ev_over > min_ev and MIN_ODDS <= odds_over <= MAX_ODDS:
                    daily_bets.append({
                        'player': row['player_name'],
                        'target': row['target'],
                        'side': 'Over',
                        'prob': model_prob_over,
                        'odds': odds_over,
                        'ev': ev_over,
                        'outcome': 1 if row['actual_value'] > row['line_value'] else 0,
                        'game_id': game_id
                    })
                elif ev_under > min_ev and MIN_ODDS <= odds_under <= MAX_ODDS:
                    daily_bets.append({
                        'player': row['player_name'],
                        'target': row['target'],
                        'side': 'Under',
                        'prob': model_prob_under,
                        'odds': odds_under,
                        'ev': ev_under,
                        'outcome': 1 if row['actual_value'] < row['line_value'] else 0,
                        'game_id': game_id
                    })
            
            if not daily_bets:
                self.bankroll_history.append(current_bankroll)
                continue
                
            # Sort by EV
            daily_bets.sort(key=lambda x: x['ev'], reverse=True)
            
            day_pnl = 0
            used_indices = set()  # Track used picks to prevent repetition
            
            # ===== 2-3 LEG FOCUS (OPTIMAL) =====
            # 60% on 2-leg parlays, 40% on 3-leg parlays
            # Budget: 75% on parlays, 25% on straight bets
            parlay_budget = daily_budget * rr_pct
            straight_budget = daily_budget * (1 - rr_pct)
            
            leg2_budget = parlay_budget * 0.60
            leg3_budget = parlay_budget * 0.40
            
            top_picks = daily_bets[:10]
            parlays = []
            
            # Helper to build parlay with CORRELATION FILTERING
            def build_parlay(picks, num_legs, used):
                """Build parlay avoiding same-game correlation."""
                available = [(i, p) for i, p in enumerate(picks) if i not in used]
                if len(available) < num_legs:
                    return None, used
                
                # Select legs while avoiding same-game correlation
                selected = []
                used_games = set()
                
                for i, p in available:
                    game = p.get('game_id', str(i))
                    if game not in used_games:
                        selected.append((i, p))
                        used_games.add(game)
                        if len(selected) >= num_legs:
                            break
                
                if len(selected) < num_legs:
                    return None, used  # Not enough uncorrelated picks
                
                for s in selected:
                    used.add(s[0])
                parlay_picks = [s[1] for s in selected]
                combined_odds = 1.0
                all_won = True
                for p in parlay_picks:
                    combined_odds *= p['odds']
                    if p['outcome'] != 1:
                        all_won = False
                return {
                    'legs': num_legs,
                    'picks': parlay_picks,
                    'odds': combined_odds,
                    'won': all_won,
                    'prob': np.prod([p['prob'] for p in parlay_picks])
                }, used
            
            # Build 2x 2-leg parlays
            for _ in range(2):
                p, used_indices = build_parlay(top_picks, 2, used_indices)
                if p:
                    p['stake'] = leg2_budget / 2
                    parlays.append(p)
            
            # Build 1x 3-leg parlay
            p, used_indices = build_parlay(top_picks, 3, used_indices)
            if p:
                p['stake'] = leg3_budget
                parlays.append(p)
            
            # Place parlay bets
            for parlay in parlays:
                stake = parlay['stake']
                
                if parlay['won']:
                    pnl = stake * (parlay['odds'] - 1)
                else:
                    pnl = -stake
                
                day_pnl += pnl
                legs_str = '+'.join([p['player'][:10] for p in parlay['picks']])
                self.results.append({
                    'player': f"RR: {legs_str}",
                    'target': '2-leg',
                    'side': 'Parlay',
                    'prob': parlay['prob'],
                    'odds': parlay['odds'],
                    'ev': 0,
                    'outcome': 1 if parlay['won'] else 0,
                    'stake': stake,
                    'pnl': pnl,
                    'date': date,
                    'bankroll_before': current_bankroll
                })
            
            # ===== 4-LEG ROUND ROBIN (25% of budget) - Using remaining picks =====
            # Take next 4 unused picks, create all 2-leg combinations (6 parlays)
            remaining_picks = [(i, p) for i, p in enumerate(daily_bets) if i not in used_indices][:4]
            
            if len(remaining_picks) >= 2:
                from itertools import combinations
                rr4_combos = list(combinations(range(len(remaining_picks)), 2))
                stake_per_rr4 = straight_budget / len(rr4_combos) if rr4_combos else 0
                
                for i, j in rr4_combos:
                    leg1 = remaining_picks[i][1]
                    leg2 = remaining_picks[j][1]
                    combined_odds = leg1['odds'] * leg2['odds']
                    parlay_won = leg1['outcome'] == 1 and leg2['outcome'] == 1
                    
                    if parlay_won:
                        pnl = stake_per_rr4 * (combined_odds - 1)
                    else:
                        pnl = -stake_per_rr4
                    
                    day_pnl += pnl
                    legs_str = f"{leg1['player'][:10]}+{leg2['player'][:10]}"
                    self.results.append({
                        'player': f"RR4: {legs_str}",
                        'target': '2-leg-rr4',
                        'side': 'Parlay',
                        'prob': leg1['prob'] * leg2['prob'],
                        'odds': combined_odds,
                        'ev': 0,
                        'outcome': 1 if parlay_won else 0,
                        'stake': stake_per_rr4,
                        'pnl': pnl,
                        'date': date,
                        'bankroll_before': current_bankroll
                    })
                
            current_bankroll += day_pnl
            self.bankroll_history.append(current_bankroll)
            
            if current_bankroll <= 0:
                print("Bankruptcy!")
                break
                
        # Final Summary
        roi = ((current_bankroll - initial_bankroll) / initial_bankroll) * 100
        print(f"Simulation Complete. Final Bankroll: {current_bankroll:.2f} (ROI: {roi:.2f}%)")
        return pd.DataFrame(self.results)

