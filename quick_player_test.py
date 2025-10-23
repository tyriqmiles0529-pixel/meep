"""
Quick test of player name matching logic - Just finds players, no full analysis
"""

import requests

API_KEY = "4979ac5e1f7ae10b1d6b58f1bba01140"
BASE_URL = "https://v1.basketball.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}

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

for name in test_names:
    player_id = search_player_smart(name)
    if player_id:
        found_count += 1
    else:
        not_found_count += 1

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"✅ Found: {found_count}/{len(test_names)}")
print(f"❌ Not Found: {not_found_count}/{len(test_names)}")

if found_count == len(test_names):
    print("\n🎉 SUCCESS! All players can be found with the new logic!")
elif found_count > 0:
    print(f"\n✅ Partial success! {found_count} out of {len(test_names)} players found.")
else:
    print("\n❌ No players found - there may be another issue.")

print("=" * 80)
