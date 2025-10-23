"""
Diagnose player name search issues
"""

import requests

API_KEY = "4979ac5e1f7ae10b1d6b58f1bba01140"
BASE_URL = "https://v1.basketball.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}

print("=" * 80)
print("PLAYER SEARCH DIAGNOSTIC")
print("=" * 80)

# Test with common NBA players
test_players = [
    "LeBron James",
    "LeBron",
    "James",
    "Stephen Curry",
    "Curry",
    "Stephen",
    "Kevin Durant",
    "Durant",
    "Kevin",
    "Giannis Antetokounmpo",
    "Giannis",
    "Antetokounmpo",
    "Luka Doncic",
    "Luka",
    "Doncic"
]

print("\nTesting different search formats:\n")

for player_name in test_players:
    url = f"{BASE_URL}/players"
    params = {"search": player_name}

    try:
        response = requests.get(url, headers=HEADERS, params=params, timeout=10)

        if response.status_code == 200:
            data = response.json()
            results = data.get("results", 0)

            if results > 0:
                print(f"✅ '{player_name}' → Found {results} results")
                # Show first match
                first_player = data["response"][0]
                print(f"   First match: {first_player.get('name')} (ID: {first_player.get('id')})")
            else:
                print(f"❌ '{player_name}' → No results")
        else:
            print(f"⚠️  '{player_name}' → API Error {response.status_code}")

    except Exception as e:
        print(f"❌ '{player_name}' → Error: {e}")

print("\n" + "=" * 80)
print("TESTING ACTUAL ODDS DATA")
print("=" * 80)

# Fetch a real game and check what player names are in the odds
print("\nFetching a game to see actual player names in odds...\n")

# Get recent games
games_response = requests.get(
    f"{BASE_URL}/games",
    headers=HEADERS,
    params={"league": 12, "season": "2024-2025"},
    timeout=10
)

if games_response.status_code == 200:
    games_data = games_response.json()
    if games_data.get("results", 0) > 0:
        # Get first game
        game = games_data["response"][0]
        game_id = game["id"]
        game_name = f"{game['teams']['home']['name']} vs {game['teams']['away']['name']}"

        print(f"Game: {game_name} (ID: {game_id})")

        # Get odds for this game
        odds_response = requests.get(
            f"{BASE_URL}/odds",
            headers=HEADERS,
            params={"game": game_id, "bookmaker": 4},
            timeout=10
        )

        if odds_response.status_code == 200:
            odds_data = odds_response.json()

            if odds_data.get("results", 0) > 0:
                bookmakers = odds_data["response"][0].get("bookmakers", [])

                if bookmakers:
                    print("\nPlayer props found in odds:\n")

                    player_names_in_odds = set()

                    for bet in bookmakers[0].get("bets", []):
                        bet_name = bet.get("name", "").lower()

                        # Check if it's a player prop
                        if any(x in bet_name for x in ["point", "assist", "rebound", "three"]):
                            if "spread" not in bet_name and "total" not in bet_name.replace("total rebounds", ""):
                                for value in bet.get("values", []):
                                    prop_text = value.get("value", "")
                                    parts = prop_text.split()

                                    if len(parts) >= 2:
                                        # Extract player name (everything except last part which is the line)
                                        player_name = " ".join(parts[:-1])
                                        player_names_in_odds.add(player_name)

                    if player_names_in_odds:
                        print(f"Found {len(player_names_in_odds)} unique players in odds:")

                        for player_name in sorted(list(player_names_in_odds))[:10]:  # Show first 10
                            print(f"\n  Testing: '{player_name}'")

                            # Try searching for this exact name
                            search_response = requests.get(
                                f"{BASE_URL}/players",
                                headers=HEADERS,
                                params={"search": player_name},
                                timeout=10
                            )

                            if search_response.status_code == 200:
                                search_data = search_response.json()
                                results = search_data.get("results", 0)

                                if results > 0:
                                    first = search_data["response"][0]
                                    print(f"    ✅ Found: {first.get('name')} (ID: {first.get('id')})")
                                else:
                                    print(f"    ❌ NOT FOUND")

                                    # Try with just last name
                                    last_name = player_name.split()[-1]
                                    print(f"    🔄 Trying last name only: '{last_name}'")

                                    retry_response = requests.get(
                                        f"{BASE_URL}/players",
                                        headers=HEADERS,
                                        params={"search": last_name},
                                        timeout=10
                                    )

                                    if retry_response.status_code == 200:
                                        retry_data = retry_response.json()
                                        retry_results = retry_data.get("results", 0)

                                        if retry_results > 0:
                                            print(f"    ✅ Found with last name: {retry_results} results")
                                            for i, p in enumerate(retry_data["response"][:3]):
                                                print(f"       {i+1}. {p.get('name')} (ID: {p.get('id')})")
                                        else:
                                            print(f"    ❌ Still not found")
                    else:
                        print("No player props found in odds")
                else:
                    print("No bookmakers found in odds")
            else:
                print("No odds data found")
        else:
            print(f"Failed to fetch odds: {odds_response.status_code}")
    else:
        print("No games found")
else:
    print(f"Failed to fetch games: {games_response.status_code}")

print("\n" + "=" * 80)
print("RECOMMENDATIONS")
print("=" * 80)

print("""
If players are not being found, possible solutions:

1. **Name Format Issue**
   - Odds might use "L. James" but API needs "LeBron James"
   - Try searching by last name only
   - Implement fuzzy matching

2. **API Database**
   - Player might not be in the API database
   - Try with team ID filtering

3. **Search Strategy**
   - Parse name from odds differently
   - Try multiple search variations
   - Fall back to last name only

4. **Debug Mode**
   - Run analyzer with DEBUG_MODE=True
   - Check what names are being searched
   - See exact API responses
""")

print("=" * 80)
