import pandas as pd
import numpy as np
from data_processor import BasketballDataProcessor

# Mocking the class to isolate add_opponent_adjustments
class MockProcessor(BasketballDataProcessor):
    def __init__(self):
        self.df = None

    def load_mock_data(self):
        # Create a tiny dataset: 2 games for the same matchup
        # Season 2023. Knick vs Celtics.
        # Game 1: Early season. Celtics offense scores 100.
        # Game 2: Late season. Celtics offense scores 120.
        # Actual Def Rating of Knicks should evolve.
        # If 'opp_def_points' for Game 1 includes Game 2's high score, it's a leak.
        
        data = {
            'player_name': ['Tatum', 'Tatum'],
            'season': [2023, 2023],
            'date': pd.to_datetime(['2023-11-01', '2024-04-01']),
            'points': [25, 30], # Tatum scores
            'rebounds': [5, 8],
            'assists': [4, 6],
            'opponentteamname': ['Knicks', 'Knicks'],
            'team': ['Celtics', 'Celtics'],
            # The 'points' column here is what `add_opponent_adjustments` aggregates
            # In data_processor, it aggregates 'points' (player points) 
            # Wait, data_processor aggregates 'points' grouped by OPPONENT.
            # So it sums up ALL points scored against the Knicks.
            # Here we just have one player, but the logic holds.
        }
        self.df = pd.DataFrame(data)
        print("Mock Data Created:\n", self.df)

    def test_leakage(self):
        print("\n--- Testing Opponent Adjustment Leakage ---")
        self.add_opponent_adjustments()
        
        # Check if opp_def_points is the same for both games
        # Average of 25 and 30 is 27.5
        # If Game 1 opp_def_points is 27.5, it 'knows' about Game 2.
        # Correct expanding window value for Game 1 would be... well, undefined or just 25 (if strictly past).
        
        print("\nResulting Data:\n", self.df[['date', 'points', 'opp_def_points']])
        
        val1 = self.df.iloc[0]['opp_def_points']
        val2 = self.df.iloc[1]['opp_def_points']
        
        if val1 == val2:
            print(f"\n[CONFIRMED LEAK] Game 1 opp_def_points ({val1}) equals Game 2 ({val2}).")
            print("The model uses the FULL SEASON average for current games.")
        else:
            print("\n[NO LEAK] Values differ.")

if __name__ == "__main__":
    tester = MockProcessor()
    tester.load_mock_data()
    tester.test_leakage()
