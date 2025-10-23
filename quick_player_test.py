"""
Quick test of player name matching logic - Finds players and shows last 7 games stats
"""

import requests
import pandas as pd

API_KEY = "4979ac5e1f7ae10b1d6b58f1bba01140"
BASE_URL = "https://v1.basketball.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}
STATS_SEASON = "2024-2025"

def search_player_smart(player_name):
    """
    Smart player search with last name fallback
    """
    print(f"\n🔍 Searching for: {player_name}")

    # Try full name first
    response = requests.get(
        f"{BASE_URL}/players",
        headers=HEADERS,
        params={"search": player_name},
        timeout=10
    )

    if response.status_code == 200:
        data = response.json()

        if data.get("results", 0) > 0:
            player = data["response"][0]
            print(f"   ✅ Found directly: {player.get('name')} (ID: {player.get('id')})")
            return player.get('id')
        else:
            print(f"   ⚠️  Full name not found, trying last name...")

            # Extract last name
            name_parts = player_name.strip().split()
            if len(name_parts) >= 2:
                last_name = name_parts[-1]
                print(f"   🔍 Searching by last name: {last_name}")

                response2 = requests.get(
                    f"{BASE_URL}/players",
                    headers=HEADERS,
                    params={"search": last_name},
                    timeout=10
                )

                if response2.status_code == 200:
                    data2 = response2.json()

                    if data2.get("results", 0) > 0:
                        # Find best match
                        best_match = None
                        for result in data2["response"]:
                            api_name = result.get("name", "").lower()
                            if last_name.lower() in api_name:
                                best_match = result
                                break

                        if not best_match:
                            best_match = data2["response"][0]

                        print(f"   ✅ Matched to: {best_match.get('name')} (ID: {best_match.get('id')})")
                        return best_match.get('id')
                    else:
                        print(f"   ❌ Not found even by last name")
                        return None
            else:
                print(f"   ❌ Cannot extract last name")
                return None
    else:
        print(f"   ❌ API Error: {response.status_code}")
        return None


def get_player_last_games(player_id, player_name, num_games=7):
    """
    Fetch last N games stats for a player
    """
    print(f"\n   📊 Fetching last {num_games} games for {player_name}...")

    response = requests.get(
        f"{BASE_URL}/games/statistics/players",
        headers=HEADERS,
        params={"player": player_id, "season": STATS_SEASON},
        timeout=10
    )

    if response.status_code == 200:
        data = response.json()

        if data.get("results", 0) > 0:
            games = data["response"][:num_games]  # Get last N games

            print(f"   ✅ Found {len(games)} games")
            print(f"\n   {'Game':<8} {'PTS':<5} {'AST':<5} {'REB':<5} {'3PM':<5}")
            print(f"   {'-'*35}")

            for i, game in enumerate(games, 1):
                points = game.get("points", 0) or 0
                assists = game.get("assists", 0) or 0

                # Handle rebounds
                rebounds_data = game.get("rebounds", {})
                if isinstance(rebounds_data, dict):
                    rebounds = rebounds_data.get("total", 0) or 0
                else:
                    rebounds = rebounds_data or 0

                # Handle three-pointers
                threes_data = game.get("threepoint_goals", {})
                if isinstance(threes_data, dict):
                    threes = threes_data.get("total", 0) or 0
                else:
                    threes = 0

                print(f"   Game {i:<3} {points:<5} {assists:<5} {rebounds:<5} {threes:<5}")

            # Calculate averages
            if len(games) > 0:
                avg_pts = sum(g.get("points", 0) or 0 for g in games) / len(games)
                avg_ast = sum(g.get("assists", 0) or 0 for g in games) / len(games)

                avg_reb = 0
                for g in games:
                    reb_data = g.get("rebounds", {})
                    if isinstance(reb_data, dict):
                        avg_reb += reb_data.get("total", 0) or 0
                    else:
                        avg_reb += reb_data or 0
                avg_reb /= len(games)

                avg_3pm = 0
                for g in games:
                    three_data = g.get("threepoint_goals", {})
                    if isinstance(three_data, dict):
                        avg_3pm += three_data.get("total", 0) or 0
                avg_3pm /= len(games)

                print(f"   {'-'*35}")
                print(f"   Average  {avg_pts:<5.1f} {avg_ast:<5.1f} {avg_reb:<5.1f} {avg_3pm:<5.1f}")

            return True
        else:
            print(f"   ⚠️  No game stats found")
            return False
    else:
        print(f"   ❌ API Error: {response.status_code}")
        return False


# Test with common player names that would appear in odds
print("=" * 80)
print("QUICK PLAYER NAME MATCHING TEST")
print("=" * 80)

test_names = [
    "LeBron James",
    "Stephen Curry",
    "Kevin Durant",
    "Luka Doncic",
    "Giannis Antetokounmpo",
    "Jayson Tatum",
    "Nikola Jokic",
    "Joel Embiid"
]

found_count = 0
not_found_count = 0
stats_found_count = 0

for name in test_names:
    player_id = search_player_smart(name)
    if player_id:
        found_count += 1
        # Also fetch and show last 7 games stats
        if get_player_last_games(player_id, name, num_games=7):
            stats_found_count += 1
    else:
        not_found_count += 1
    print()  # Blank line between players

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"✅ Players Found: {found_count}/{len(test_names)}")
print(f"📊 Stats Retrieved: {stats_found_count}/{found_count}" if found_count > 0 else "📊 Stats Retrieved: 0/0")
print(f"❌ Not Found: {not_found_count}/{len(test_names)}")

if found_count == len(test_names) and stats_found_count == found_count:
    print("\n🎉 SUCCESS! All players found and stats retrieved!")
    print("   The analyzer is ready to use!")
elif found_count > 0:
    print(f"\n✅ Partial success! {found_count} out of {len(test_names)} players found.")
    if stats_found_count < found_count:
        print(f"⚠️  But only {stats_found_count} have stats available.")
else:
    print("\n❌ No players found - there may be another issue.")

print("=" * 80)
