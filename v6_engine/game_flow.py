import numpy as np
from datetime import datetime, timedelta
from typing import List

class GameFlowDynamics:
    """
    V6 Evolution: Dynamic State Management.
    Models rotations, substitutions, and behavioral momentum.
    """
    def __init__(self):
        self.clutch_time = False
        self.momentum_multiplier = 1.0
        self.rotations = {
             "starters": [0, 1, 2, 3, 4],
             "bench": [5, 6, 7, 8, 9]
        }
    
    def calculate_clutch_factor(self, score_diff: int, time_remaining: float) -> float:
        """
        Determines the intensity boost in clutch situations (final 5 mins, < 5 pts diff).
        Returns: [1.0 - 1.2]
        """
        if abs(score_diff) <= 5 and time_remaining <= 5.0:
            self.clutch_time = True
            return 1.15 # 15% intensity boost
        
        self.clutch_time = False
        return 1.0

    def get_rotation_lineup(self, minute: int) -> List[int]:
        """
        Mock: Dynamic Rotation Logic.
        Returns the active player indices (0-14) for a given minute.
        Typically: 
        1st & 3rd Qtrs: Starters (0-12 mins)
        2nd & 4th Qtrs: Bench transition (12-24, 36-40 mins)
        Closing: Starters (42-48 mins)
        """
        if 0 <= minute <= 8 or 24 <= minute <= 32 or 42 <= minute <= 48:
             return self.rotations["starters"]
        else:
             return self.rotations["bench"]

    def apply_momentum_shift(self, last_n_possessions_orating: float) -> float:
        """
        Calculates the "Hot Hand" or "Momentum" for a team.
        If recent ORtg > 125.0, boost ORtg slightly.
        """
        if last_n_possessions_orating > 125.0:
             self.momentum_multiplier = 1.05
        elif last_n_possessions_orating < 90.0:
             self.momentum_multiplier = 0.95
        else:
             self.momentum_multiplier = 1.0
        
        return self.momentum_multiplier
