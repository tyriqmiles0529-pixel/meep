import os
"""
Quick test for Christian Braun (Player ID: 623)
"""

import requests
import json

API_KEY = os.getenv("API_SPORTS_KEY", "")
BASE_URL = "https://v1.basketball.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}

print("=" * 80)
print("CHRISTIAN BRAUN - PLAYER STATS TEST (ID: 623)")
print("=" * 80)

# Fetch stats for Christian Braun from 2024-2025 season
player_id = 623
season = "2024-2025"

print(f"\n📊 Fetching stats for player ID {player_id} from {season} season...")
print(f"Endpoint: /games/statistics/players?player={player_id}&season={season}")

response = requests.get(
    f"{BASE_URL}/games/statistics/players",
    headers=HEADERS,
    params={"player": player_id, "season": season},
    timeout=10
)

print(f"\nStatus Code: {response.status_code}")

if response.status_code == 200:
    data = response.json()

    print(f"✅ Success!")
    print(f"\nResponse Summary:")
    print(f"  Results: {data.get('results', 0)} games found")

    if data.get("results", 0) > 0:
        games = data["response"][:10]  # Show last 10 games

        print(f"\n{'Game':<6} {'PTS':<6} {'AST':<6} {'REB':<6} {'3PM':<6} {'MIN':<8}")
        print("-" * 45)

        total_pts = 0
        total_ast = 0
        total_reb = 0
        total_3pm = 0

        for i, game in enumerate(games, 1):
            points = game.get("points", 0) or 0
            assists = game.get("assists", 0) or 0

            # Rebounds
            rebounds_data = game.get("rebounds", {})
            if isinstance(rebounds_data, dict):
                rebounds = rebounds_data.get("total", 0) or 0
            else:
                rebounds = rebounds_data or 0

            # Three-pointers
            threes_data = game.get("threepoint_goals", {})
            if isinstance(threes_data, dict):
                threes = threes_data.get("total", 0) or 0
            else:
                threes = 0

            minutes = game.get("minutes", "0:00")

            print(f"{i:<6} {points:<6} {assists:<6} {rebounds:<6} {threes:<6} {minutes:<8}")

            total_pts += points
            total_ast += assists
            total_reb += rebounds
            total_3pm += threes

        # Calculate averages
        num_games = len(games)
        print("-" * 45)
        print(f"{'Avg':<6} {total_pts/num_games:<6.1f} {total_ast/num_games:<6.1f} {total_reb/num_games:<6.1f} {total_3pm/num_games:<6.1f}")

        print(f"\n📈 Statistics Summary:")
        print(f"  Games analyzed: {num_games}")
        print(f"  Points per game: {total_pts/num_games:.1f}")
        print(f"  Assists per game: {total_ast/num_games:.1f}")
        print(f"  Rebounds per game: {total_reb/num_games:.1f}")
        print(f"  3-Pointers per game: {total_3pm/num_games:.1f}")

        print(f"\n✅ Data looks good! Player stats are accessible and parsing correctly.")

        # Show raw data structure for first game
        print(f"\n📋 Raw data structure (first game):")
        print(json.dumps(games[0], indent=2))

    else:
        print("\n⚠️ No games found for this player in this season")
        print("Possible reasons:")
        print("  - Player didn't play this season")
        print("  - Wrong season specified")
        print("  - Player ID incorrect")
else:
    print(f"❌ API Error: {response.status_code}")
    print(f"Response: {response.text}")

print("\n" + "=" * 80)
