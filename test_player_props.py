import requests

# Test what markets are available
url = 'https://api.the-odds-api.com/v4/sports/basketball_nba/odds'
params = {
    'apiKey': '50d574e850574016d2cce5fb07b4e954',
    'regions': 'us',
    'markets': 'player_points,player_rebounds,player_assists,player_threes',
    'oddsFormat': 'american'
}

try:
    resp = requests.get(url, params=params, timeout=10)
    print(f'Status: {resp.status_code}')
    print(f'Remaining: {resp.headers.get("x-requests-remaining", "unknown")}')
    
    if resp.status_code == 200:
        data = resp.json()
        print(f'Events returned: {len(data)}')
        
        if len(data) > 0:
            # Count markets
            total_markets = 0
            market_counts = {}
            
            for event in data:
                print(f'\nEvent: {event.get("away_team")} @ {event.get("home_team")}')
                for bookmaker in event.get('bookmakers', []):
                    print(f'  Bookmaker: {bookmaker.get("title")}')
                    for market in bookmaker.get('markets', []):
                        market_key = market.get('key')
                        outcomes = len(market.get('outcomes', []))
                        market_counts[market_key] = market_counts.get(market_key, 0) + outcomes
                        total_markets += 1
                        print(f'    Market: {market_key} ({outcomes} outcomes)')
                        
                        # Show first outcome as example
                        if outcomes > 0:
                            sample = market.get('outcomes', [])[0]
                            print(f'      Example: {sample.get("description")} {sample.get("name")} {sample.get("point")} @ {sample.get("price")}')
                
                break  # Just show first event in detail
            
            print(f'\nTotal market instances: {total_markets}')
            print(f'\nMarket breakdown:')
            for mk, count in market_counts.items():
                print(f'  {mk}: {count} outcomes')
    
    elif resp.status_code == 422:
        print(f'\n422 Error - Invalid request')
        print(f'Response: {resp.text}')
        print(f'\nThis likely means player props require a paid subscription')
    else:
        print(f'Error: {resp.status_code}')
        print(f'Response: {resp.text[:500]}')
        
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
