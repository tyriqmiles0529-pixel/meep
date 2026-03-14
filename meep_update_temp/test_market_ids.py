"""Test fetching player odds for specific market IDs to identify them."""
import http.client
import json

API_KEY = "9ef7289093msh76adf5ee5bedb5fp15e0d6jsnc2a0d0ed9abe"
HOST = "nba-player-props-odds.p.rapidapi.com"

conn = http.client.HTTPSConnection(HOST)

headers = {
    'x-rapidapi-key': API_KEY,
    'x-rapidapi-host': HOST
}

# Pelicans @ Nets (Dec 6 id)
event_id = 26821 

# Market IDs to test: 5, 6
market_ids = [5, 6]

for mk_id in market_ids:
    print(f"\n--- Testing Market ID: {mk_id} ---")
    
    # Request ONLY this market ID
    path = f"/get-player-odds-for-event?eventId={event_id}&marketId={mk_id}&decimal=true&best=true"
    
    try:
        conn.request("GET", path, headers=headers)
        res = conn.getresponse()
        data = res.read().decode("utf-8")
        parsed = json.loads(data)
        
        if isinstance(parsed, list):
            count = len(parsed)
            print(f"Returned {count} props")
            if count > 0:
                # Print the market label from the first item to identify it
                first_item = parsed[0]
                label = first_item.get('market_label', 'Unknown')
                print(f"Market Label: {label}")
                # Print a few player names to confirm variety
                players = [p.get('player', {}).get('name') for p in parsed[:3]]
                print(f"Sample players: {players}")
        else:
            print("Response is not a list")
            
    except Exception as e:
        print(f"Error: {e}")
