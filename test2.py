import http.client
import json

# ===== CONFIG =====
API_KEY = "c47bd0c1e7c0d008a514ecba161b347f"
LEAGUE_ID = 12  # NBA
SEASON = "2025-2026"
BOOKMAKER_ID = 4  # DraftKings
DESIRED_PROPS = ["points", "assists", "rebounds", "3pm"]

# ===== CONNECTION =====
conn = http.client.HTTPSConnection("v1.basketball.api-sports.io")

headers = {
    'x-rapidapi-host': "v1.basketball.api-sports.io",
    'x-rapidapi-key': API_KEY
}

# ===== REQUEST ODDS =====
endpoint = f"/odds?league={LEAGUE_ID}&season={SEASON}&bookmaker={BOOKMAKER_ID}"
conn.request("GET", endpoint, headers=headers)

res = conn.getresponse()
data = res.read()
response = json.loads(data.decode("utf-8"))

print("Status Code:", res.status)
print("Results:", response.get("results", 0))

# ===== VALIDATION =====
if response.get("results", 0) == 0:
    print("⚠️ No odds returned — check if season is active or data is available yet.")
else:
    all_odds = response.get("response", [])
    found = []

    for game in all_odds:
       
