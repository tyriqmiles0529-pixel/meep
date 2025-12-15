"""Test the player odds endpoint directly."""
import http.client
import json

API_KEY = "9ef7289093msh76adf5ee5bedb5fp15e0d6jsnc2a0d0ed9abe"
HOST = "nba-player-props-odds.p.rapidapi.com"

conn = http.client.HTTPSConnection(HOST)

headers = {
    'x-rapidapi-key': API_KEY,
    'x-rapidapi-host': HOST
}

# Test with event ID from Dec 6
event_id = 26821  # Pelicans @ Nets

print(f"Fetching player odds for event {event_id}...")
conn.request("GET", f"/get-player-odds-for-event?eventId={event_id}&bookieId=1%3A4%3A5%3A6%3A7%3A8%3A9%3A10&marketId=1%3A2%3A3%3A4%3A5%3A6&decimal=true&best=true", headers=headers)

res = conn.getresponse()
data = res.read().decode("utf-8")

print(f"Response: {data[:500]}")

try:
    parsed = json.loads(data)
    print(f"\nParsed type: {type(parsed)}")
    if isinstance(parsed, list):
        print(f"Number of items: {len(parsed)}")
        if parsed:
            print(f"First item keys: {parsed[0].keys() if isinstance(parsed[0], dict) else parsed[0]}")
    elif isinstance(parsed, dict):
        print(f"Keys: {parsed.keys()}")
except Exception as e:
    print(f"Parse error: {e}")
