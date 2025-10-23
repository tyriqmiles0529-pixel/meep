"""
NBA API Diagnostics Script
Tests API connectivity and data retrieval
"""

import requests
import json
from pprint import pprint

# Configuration
API_KEY = "c47bd0c1e7c0d008a514ecba161b347f"
BASE_URL = "https://v1.basketball.api-sports.io"
HEADERS = {
    "x-rapidapi-host": "v1.basketball.api-sports.io",
    "x-rapidapi-key": API_KEY
}

LEAGUE_ID = 12
SEASON = "2025-2026"
STATS_SEASON = "2024-2025"

def print_section(title):
    """Print section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def test_api_connection():
    """Test basic API connectivity"""
    print_section("TEST 1: API Connection")

    try:
        response = requests.get(
            f"{BASE_URL}/status",
            headers=HEADERS,
            timeout=10
        )

        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print("✅ API Connection Successful!")
            print(f"\nAPI Status:")
            pprint(data)
            return True
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Response: {response.text[:500]}")
            return False

    except Exception as e:
        print(f"❌ Connection Error: {e}")
        return False

def test_games_endpoint():
    """Test games endpoint"""
    print_section("TEST 2: Games Endpoint")

    params = {
        "league": LEAGUE_ID,
        "season": SEASON
    }

    try:
        response = requests.get(
            f"{BASE_URL}/games",
            headers=HEADERS,
            params=params,
            timeout=10
        )

        print(f"URL: {response.url}")
        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()

            if "response" in data and len(data["response"]) > 0:
                print(f"✅ Found {len(data['response'])} games")
                print(f"\nFirst game sample:")
                pprint(data["response"][0])
                return data["response"][0]  # Return first game for further testing
            else:
                print("⚠️ No games found in response")
                print(f"Full response:")
                pprint(data)
                return None
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text[:500]}")
            return None

    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_player_search(player_name="LeBron James"):
    """Test player search endpoint"""
    print_section(f"TEST 3: Player Search - '{player_name}'")

    params = {
        "search": player_name
    }

    try:
        response = requests.get(
            f"{BASE_URL}/players",
            headers=HEADERS,
            params=params,
            timeout=10
        )

        print(f"URL: {response.url}")
        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()

            if "response" in data and len(data["response"]) > 0:
                player = data["response"][0]
                player_id = player.get("id")
                player_name = player.get("name") or player.get("firstname", "") + " " + player.get("lastname", "")

                print(f"✅ Found player: {player_name}")
                print(f"   Player ID: {player_id}")
                print(f"\nFull player data:")
                pprint(player)
                return player_id
            else:
                print("⚠️ Player not found")
                print(f"Full response:")
                pprint(data)
                return None
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text[:500]}")
            return None

    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_player_stats(player_id, season):
    """Test player statistics endpoint"""
    print_section(f"TEST 4: Player Stats - ID {player_id}, Season {season}")

    params = {
        "season": season,
        "player": player_id
    }

    try:
        response = requests.get(
            f"{BASE_URL}/players/statistics",
            headers=HEADERS,
            params=params,
            timeout=10
        )

        print(f"URL: {response.url}")
        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()

            print(f"\nResponse keys: {data.keys()}")

            if "response" in data:
                print(f"Number of game entries: {len(data['response'])}")

                if len(data["response"]) > 0:
                    print(f"✅ Found {len(data['response'])} games")

                    # Show first game in detail
                    print(f"\n📊 First game structure:")
                    first_game = data["response"][0]
                    pprint(first_game)

                    # Try to extract stats
                    print(f"\n📈 Attempting to extract stats...")
                    stats = first_game.get("statistics", first_game)

                    if isinstance(stats, list) and len(stats) > 0:
                        stats = stats[0]
                        print("Stats are in a list, extracted first element")

                    if isinstance(stats, dict):
                        print(f"\nAvailable stat fields:")
                        for key in sorted(stats.keys()):
                            print(f"   {key}: {stats[key]}")

                        # Try to get common stats
                        points = stats.get("points")
                        assists = stats.get("assists")
                        rebounds = stats.get("totReb")
                        threes = stats.get("tpm")

                        print(f"\n🏀 Key Stats from first game:")
                        print(f"   Points: {points}")
                        print(f"   Assists: {assists}")
                        print(f"   Total Rebounds: {rebounds}")
                        print(f"   3-Pointers Made: {threes}")

                        if points is None and assists is None:
                            print("\n⚠️ WARNING: Standard stat fields are NULL/None")
                            print("This might indicate:")
                            print("   1. Player didn't play in this game")
                            print("   2. Stats not yet recorded")
                            print("   3. Different API response structure")

                    return data["response"]
                else:
                    print("⚠️ No games found in response")
                    print(f"Full response:")
                    pprint(data)
                    return None
            else:
                print("❌ No 'response' key in data")
                print(f"Full response:")
                pprint(data)
                return None
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text[:500]}")
            return None

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_team_stats(team_id=132):
    """Test team statistics endpoint"""
    print_section(f"TEST 5: Team Stats - Team ID {team_id}, Season {STATS_SEASON}")

    params = {
        "league": LEAGUE_ID,
        "season": STATS_SEASON,
        "team": team_id
    }

    try:
        response = requests.get(
            f"{BASE_URL}/statistics",
            headers=HEADERS,
            params=params,
            timeout=10
        )

        print(f"URL: {response.url}")
        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()

            if "response" in data and len(data["response"]) > 0:
                print(f"✅ Found {len(data['response'])} team stat entries")
                print(f"\nFirst entry:")
                pprint(data["response"][0])
                return data["response"]
            else:
                print("⚠️ No team stats found")
                print(f"Full response:")
                pprint(data)
                return None
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text[:500]}")
            return None

    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_odds(game_id, bookmaker_id=4):
    """Test odds endpoint"""
    print_section(f"TEST 6: Odds - Game ID {game_id}, Bookmaker {bookmaker_id}")

    params = {
        "game": game_id,
        "bookmaker": bookmaker_id
    }

    try:
        response = requests.get(
            f"{BASE_URL}/odds",
            headers=HEADERS,
            params=params,
            timeout=10
        )

        print(f"URL: {response.url}")
        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()

            if "response" in data and len(data["response"]) > 0:
                print(f"✅ Found odds data")
                print(f"\nOdds structure:")
                pprint(data["response"][0])
                return data["response"][0]
            else:
                print("⚠️ No odds found")
                print(f"Full response:")
                pprint(data)
                return None
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text[:500]}")
            return None

    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def run_full_diagnostics():
    """Run all diagnostic tests"""
    print("\n" + "🔍" * 35)
    print("NBA API DIAGNOSTICS TOOL")
    print("🔍" * 35)

    # Test 1: Connection
    if not test_api_connection():
        print("\n❌ API connection failed. Check your API key and internet connection.")
        return

    # Test 2: Games
    game = test_games_endpoint()
    game_id = game.get("id") if game else None

    # Test 3: Player Search
    player_id = test_player_search("LeBron James")

    if not player_id:
        print("\n⚠️ Trying alternative player: Stephen Curry")
        player_id = test_player_search("Stephen Curry")

    if not player_id:
        print("\n⚠️ Trying alternative player: Kevin Durant")
        player_id = test_player_search("Kevin Durant")

    # Test 4: Player Stats (try both seasons)
    if player_id:
        print(f"\n🔄 Testing CURRENT season ({SEASON})...")
        stats_current = test_player_stats(player_id, SEASON)

        print(f"\n🔄 Testing LAST season ({STATS_SEASON})...")
        stats_last = test_player_stats(player_id, STATS_SEASON)

        if not stats_current and not stats_last:
            print("\n❌ No stats found in either season!")
            print("\nPossible issues:")
            print("   1. API subscription doesn't include player stats")
            print("   2. Season format is incorrect")
            print("   3. Player ID is invalid")
            print("   4. Rate limiting in effect")
    else:
        print("\n⚠️ Skipping player stats test (no player ID)")

    # Test 5: Team Stats
    test_team_stats(132)  # Boston Celtics

    # Test 6: Odds
    if game_id:
        test_odds(game_id)
    else:
        print("\n⚠️ Skipping odds test (no game ID)")

    # Summary
    print_section("DIAGNOSTICS SUMMARY")
    print("\n✅ = Working correctly")
    print("⚠️  = Warning or partial data")
    print("❌ = Failed or no data")
    print("\nIf player stats are showing as NULL/None, this indicates:")
    print("   • Your API plan may not include detailed player statistics")
    print("   • Or the season format needs adjustment")
    print("   • Or the specific endpoint requires a different subscription tier")
    print("\nCheck your API plan at: https://rapidapi.com/api-sports/api/api-basketball")
    print("=" * 70)

if __name__ == "__main__":
    run_full_diagnostics()
