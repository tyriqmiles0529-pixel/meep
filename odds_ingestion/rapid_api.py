import http.client
import json
import pandas as pd
import os
from datetime import datetime
from .base import OddsProvider

class RapidAPIProvider(OddsProvider):
    """
    Provider for 'nba-player-props-odds' on RapidAPI.
    Requires 'RAPIDAPI_KEY' env var.
    """
    HOST = "nba-player-props-odds.p.rapidapi.com"
    
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("RAPIDAPI_KEY")
        if not self.api_key:
            print("Warning: RAPIDAPI_KEY not set.")
            
    def _get_headers(self):
        return {
            'x-rapidapi-key': self.api_key,
            'x-rapidapi-host': self.HOST
        }

    def get_events(self, date_str):
        """
        Fetches events for a specific date.
        Endpoint: /get-events-for-date (Seems to return today's events by default?)
        If we need a specific date, we might need to filter the result or check if param exists.
        The snippet provided: /get-events-for-date (no params).
        Let's assume it returns *upcoming* or *today's* events.
        """
        conn = http.client.HTTPSConnection(self.HOST)
        # Try passing date just in case, or filter later.
        # User snippet had no params.
        endpoint = "/get-events-for-date"
        
        try:
            conn.request("GET", endpoint, headers=self._get_headers())
            res = conn.getresponse()
            data = res.read()
            events = json.loads(data.decode("utf-8"))
            
            # Filter by date if possible
            # Event structure unknown, but likely has 'date' or 'startTime'.
            # We will return all and let fetch_odds filter if needed.
            return events
        except Exception as e:
            print(f"Error fetching events: {e}")
            return []

    def fetch_odds(self, date=None):
        if not self.api_key:
            return pd.DataFrame()
            
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
            
        print(f"Fetching RapidAPI odds for {date}...")
        
        # 1. Get Events
        events = self.get_events(date)
        if not events:
            print("No events found.")
            return pd.DataFrame()
            
        all_odds = []
        
        # Market mapping
        MARKET_MAP = {
            'Points': 'points',
            'Rebounds': 'rebounds',
            'Assists': 'assists',
            'Threes': 'three_pointers',
            '3-Pointers Made': 'three_pointers'
        }
        
        # 2. Iterate Events
        for event in events:
            event_id = event.get('id')
            if not event_id: continue
            
            home_team = event.get('teams', {}).get('home', {}).get('abbreviation', '')
            away_team = event.get('teams', {}).get('away', {}).get('abbreviation', '')
            
            # 3. Get Odds for Event
            conn = http.client.HTTPSConnection(self.HOST)
            # marketId: 1=Points, 2=Rebounds, 3=Assists, 4=Threes
            endpoint = f"/get-player-odds-for-event?eventId={event_id}&bookieId=1:4:5:6:7:8:9:10&marketId=1:2:3:4&decimal=true&best=true"
            
            try:
                conn.request("GET", endpoint, headers=self._get_headers())
                res = conn.getresponse()
                raw = res.read()
                odds_data = json.loads(raw.decode("utf-8"))
                
                # Parse actual API structure
                if isinstance(odds_data, list):
                    for item in odds_data:
                        market_label = item.get('market_label', '')
                        target = MARKET_MAP.get(market_label)
                        if not target:
                            continue  # Skip unsupported markets (Blocks, Steals, etc.)
                        
                        player_info = item.get('player', {})
                        player_name = player_info.get('name', '')
                        player_team = player_info.get('team', '')
                        
                        selections = item.get('selections', [])
                        over_odds = None
                        under_odds = None
                        line = None
                        
                        for sel in selections:
                            label = sel.get('label', '')
                            books = sel.get('books', [])
                            
                            # Get best odds from first available book
                            if books:
                                best = books[0].get('line', {})
                                if label == 'Over':
                                    over_odds = best.get('cost')
                                    if line is None:
                                        line = best.get('line')
                                elif label == 'Under':
                                    under_odds = best.get('cost')
                                    if line is None:
                                        line = best.get('line')
                        
                        if player_name and line is not None:
                            all_odds.append({
                                'player_name': player_name,
                                'team': player_team,
                                'target': target,
                                'line': line,
                                'over_odds': over_odds,
                                'under_odds': under_odds,
                                'event_id': event_id,
                                'home_team': home_team,
                                'away_team': away_team
                            })
                            
            except Exception as e:
                print(f"Error fetching odds for event {event_id}: {e}")
                
        print(f"Fetched {len(all_odds)} player props.")
        return pd.DataFrame(all_odds)

