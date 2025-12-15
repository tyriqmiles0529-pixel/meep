from abc import ABC, abstractmethod
import pandas as pd

class OddsProvider(ABC):
    """
    Abstract Base Class for Odds Providers.
    """
    
    @abstractmethod
    def fetch_odds(self, date=None):
        """
        Fetches odds for a given date (defaults to today/upcoming).
        Returns a DataFrame with columns:
        [player_name, team, game_id, market, line, over_odds, under_odds, bookmaker]
        """
        pass
    
    def standardize_odds(self, raw_data):
        """
        Optional helper to map specific API format to standard.
        """
        pass
