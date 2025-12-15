import http.client
import json

conn = http.client.HTTPSConnection("nba-player-props-odds.p.rapidapi.com")

headers = {
    'x-rapidapi-key': "9ef7289093msh76adf5ee5bedb5fp15e0d6jsnc2a0d0ed9abe",
    'x-rapidapi-host': "nba-player-props-odds.p.rapidapi.com"
}

print("Fetching events for today...")
conn.request("GET", "/get-events-for-date", headers=headers)

res = conn.getresponse()
data = res.read()

result = data.decode("utf-8")
print(result)

# Parse and show nicely
try:
    events = json.loads(result)
    print(f"\n=== Found {len(events)} events ===")
    for event in events:
        print(f"  {event}")
except:
    print("Could not parse as JSON")
