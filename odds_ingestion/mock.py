from .base import OddsProvider
import pandas as pd
import numpy as np

class MockOddsProvider(OddsProvider):
    """
    Generates synthetic odds for testing/paper trading.
    Uses rolling averages or random variation around a baseline.
    """
    def __init__(self, games_df=None):
        self.games_df = games_df
        
    def fetch_odds(self, date=None):
        # In a real app, 'date' queries the API.
        # Here, we generate random odds for players in self.game_df
        if self.games_df is None:
            # For testing, return a dummy DF
            return pd.DataFrame()
            
        odds_data = []
        import random
        
        # Example: Iterate rows in games_df (which should be player-game rows)
        # We need player_id and name.
        if 'player_id' not in self.games_df.columns:
            return pd.DataFrame()
            
        markets = ['points', 'rebounds', 'assists', 'three_pointers']
        
        for idx, row in self.games_df.iterrows():
            player_id = row['player_id']
            player_name = row['player_name']
            game_id = row['gameId']
            
            for market in markets:
                # Synthetic Line: Random around 15-25 for pts, etc.
                if market == 'points':
                    line = random.randint(10, 30) + 0.5
                elif market == 'rebounds':
                    line = random.randint(4, 12) + 0.5
                elif market == 'assists':
                    line = random.randint(2, 10) + 0.5
                else: 
                    line = random.randint(1, 4) + 0.5
                    
                # Synthetic Odds: -110 standard
                over_odds = -110
                under_odds = -110
                
                odds_data.append({
                    'player_id': player_id,
                    'player_name': player_name,
                    'game_id': game_id,
                    'market': market,
                    'line': line,
                    'over_odds': over_odds,
                    'under_odds': under_odds,
                    'bookmaker': 'MOCK'
                })
                
        return pd.DataFrame(odds_data)
