import requests
import pandas as pd
import os
from datetime import datetime
from .base import OddsProvider

class TheOddsAPIProvider(OddsProvider):
    """
    Odds Provider using The-Odds-API (the-odds-api.com).
    Requires 'THE_ODDS_API_KEY' environment variable.
    """
    BASE_URL = "https://api.the-odds-api.com/v4/sports"
    SPORT = "basketball_nba"
    REGIONS = "us" # us, uk, eu, au
    MARKETS = "player_points,player_rebounds,player_assists,player_threes" # supported markets
    ODDS_FORMAT = "american" # american, decimal
    
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("THE_ODDS_API_KEY")
        if not self.api_key:
            print("Warning: No API Key provided for TheOddsAPIProvider. Set THE_ODDS_API_KEY env var.")

    def fetch_odds(self, date=None):
        if not self.api_key:
            return pd.DataFrame()
            
        print("Fetching live odds from The-Odds-API...")
        
        # Endpoint: /v4/sports/{sport}/events/{eventId}/odds?markets={markets}
        # Actually, for getting all events, use /v4/sports/{sport}/odds with market specified?
        # Props are usually under a separate endpoint or require 'markets' param.
        # Note: Player props require a specific plan or endpoint usage.
        # "player_points" etc are valid markets in v4.
        
        # 1. Get Events (Games)
        events_url = f"{self.BASE_URL}/{self.SPORT}/events"
        params = {
            'apiKey': self.api_key,
            'regions': self.REGIONS,
            'markets': self.MARKETS,
            'oddsFormat': self.ODDS_FORMAT,
            'dateFormat': 'iso'
        }
        
        # Note: The 'odds' endpoint returns odds for games (h2h, spreads).
        # For player props, we need to iterate events and call the event-specific odds endpoint.
        # This consumes API quota fast. 
        # Strategy: Get list of events first.
        
        try:
            resp = requests.get(f"{self.BASE_URL}/{self.SPORT}/odds", params=params)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"Error fetching odds: {e}")
            return pd.DataFrame()
            
        # Parse logic
        # The-Odds-API structure is complex nested JSON.
        # [ { id, sport_key, bookmakers: [ { key, markets: [ { key: 'player_points', outcomes: [...] } ] } ] } ]
        
        parsed_data = []
        
        for event in data:
            game_id = event['id'] # internal ID, distinct from NBA ID
            teams = [event['home_team'], event['away_team']]
            
            for bookmaker in event['bookmakers']:
                bookie_name = bookmaker['title']
                
                for market in bookmaker['markets']:
                    market_key = market['key']
                    # Map market key to our internal keys
                    # player_points -> points
                    if 'player_points' in market_key: internal_market = 'points'
                    elif 'player_rebounds' in market_key: internal_market = 'rebounds'
                    elif 'player_assists' in market_key: internal_market = 'assists'
                    elif 'player_threes' in market_key: internal_market = 'three_pointers'
                    else: continue
                    
                    for outcome in market['outcomes']:
                        player_name = outcome['name']
                        # point (line)
                        line = outcome.get('point')
                        price = outcome.get('price') # Odds
                        
                        # We need Over/Under in one row?
                        # Outcome name is usually "Over" or "Under" for totals, 
                        # but for props it's the Player Name usually with a label?
                        # Actually for player props:
                        # outcomes: [ { name: 'LeBron James', description: 'Over', price: -110, point: 25.5 }, ... ]
                        
                        # We need to pivot this.
                        # Let's verify structure. 'name' is player? 'description' is Over/Under?
                        # Yes, typically.
                        
                        desc = outcome.get('description', '')
                        if 'Over' in desc: side = 'Over'
                        elif 'Under' in desc: side = 'Under'
                        else: continue
                        
                        # We want a row per player-market-line with Over/Under columns
                        # We can append raw and pivot later, or build dict.
                        
                        # Simplified: Store each side, group by later?
                        # Our betting strategy expects: [player, market, line, over_odds, under_odds]
                        
                        parsed_data.append({
                            'event_id': game_id,
                            'bookmaker': bookie_name,
                            'player_name': player_name,
                            'market': internal_market,
                            'line': line,
                            'side': side,
                            'odds': price
                        })
                        
        df = pd.DataFrame(parsed_data)
        if df.empty:
            return df
            
        # Pivot to get Over/Under in same row
        # Group by Player, Market, Bookmaker, Line
        pivot_df = df.pivot_table(
            index=['player_name', 'market', 'line', 'bookmaker'], 
            columns='side', 
            values='odds',
            aggfunc='first' # Should be unique
        ).reset_index()
        
        # Rename cols
        pivot_df.rename(columns={'Over': 'over_odds', 'Under': 'under_odds'}, inplace=True)
        
        # Fill missing?
        return pivot_df

