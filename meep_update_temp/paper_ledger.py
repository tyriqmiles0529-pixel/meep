import json
import os
from datetime import datetime

class PaperLedger:
    def __init__(self, ledger_file='paper_ledger.json', initial_bankroll=1000.0):
        self.ledger_file = ledger_file
        self.initial_bankroll = initial_bankroll
        self.state = self._load()

    def _load(self):
        if os.path.exists(self.ledger_file):
            with open(self.ledger_file, 'r') as f:
                return json.load(f)
        else:
            return {
                'bankroll': self.initial_bankroll,
                'bet_history': [],
                'active_bets': [],
                'metrics': {
                    'roi': 0.0,
                    'win_rate': 0.0,
                    'total_bets': 0,
                    'total_won': 0,
                    'total_lost': 0
                }
            }

    def save(self):
        with open(self.ledger_file, 'w') as f:
            json.dump(self.state, f, indent=4)

    def log_bet(self, bet):
        """
        bet: dict with keys [player, target, line, prediction, stake_amt, odds, date, type]
        """
        # Deduct stake
        self.state['bankroll'] -= bet['stake_amt']
        
        # Add to active
        bet['status'] = 'PENDING'
        bet['placed_at'] = datetime.now().isoformat()
        self.state['active_bets'].append(bet)
        self.save()

    def resolve_bets(self, actual_results_df):
        """
        actual_results_df: DataFrame with [player_name, date, points, rebounds, assists, three_pointers]
        """
        resolved_count = 0
        still_active = []
        
        # Map date format if needed. Assuming ISO YYYY-MM-DD string match.
        
        for bet in self.state['active_bets']:
            # Find matching result
            # Match on player name and date
            # Ensure bet['date'] matches row['date']
            
            match = actual_results_df[
                (actual_results_df['player_name'] == bet['player']) & 
                (actual_results_df['date'] == bet['date'])
            ]
            
            if match.empty:
                still_active.append(bet)
                continue
                
            # Determine Outcome
            actual_val = match.iloc[0].get(bet['target'])
            if actual_val is None:
                still_active.append(bet) # Target missing?
                continue
                
            won = False
            line = bet['line']
            prediction = bet['prediction']
            
            # Logic: If pred > line, we bet OVER. Else UNDER.
            # We need to store WHAT we bet. 'direction'?
            # Let's assume prediction logic:
            direction = 'OVER' if prediction > line else 'UNDER'
            
            if direction == 'OVER':
                if actual_val > line: won = True
            else:
                if actual_val < line: won = True
                
            # Payout
            profit = 0
            if won:
                # American Odds to Dec
                odds = bet['odds']
                dec_odds = (1 + odds/100) if odds > 0 else (1 + 100/abs(odds))
                payout = bet['stake_amt'] * dec_odds
                profit = payout - bet['stake_amt']
                
                self.state['bankroll'] += payout
                self.state['metrics']['total_won'] += 1
            else:
                self.state['metrics']['total_lost'] += 1
                
            bet['status'] = 'WON' if won else 'LOST'
            bet['actual'] = float(actual_val)
            bet['profit'] = profit
            bet['resolved_at'] = datetime.now().isoformat()
            
            self.state['bet_history'].append(bet)
            self.state['metrics']['total_bets'] += 1
            resolved_count += 1
            
        self.state['active_bets'] = still_active
        self._update_metrics()
        self.save()
        return resolved_count

    def _update_metrics(self):
        # Update ROI, Win Rate
        total = self.state['metrics']['total_bets']
        if total > 0:
            self.state['metrics']['win_rate'] = self.state['metrics']['total_won'] / total
            
        # ROI = (Current BR - Initial) / Initial
        # Or Sum(Profit) / Sum(Stake)
        # Using simple Bankroll growth for now
        self.state['metrics']['roi'] = (self.state['bankroll'] - self.initial_bankroll) / self.initial_bankroll
