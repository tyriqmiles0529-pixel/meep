"""
Test which endpoints work with your current API plan
"""

import requests
import json
from pprint import pprint

API_KEY = "4979ac5e1f7ae10b1d6b58f1bba01140"
BASE_URL = "https://v1.basketball.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}

print("=" * 80)
print("TESTING WHICH ENDPOINTS WORK WITH YOUR PLAN")
print("=" * 80)

# Test different endpoint categories
test_endpoints = {
    "General Endpoints": [
        "/status",
        "/timezone",
        "/countries",
        "/seasons",
        "/leagues",
    ],
    "Game Data": [
        "/games?league=12&season=2024-2025",
        "/standings?league=12&season=2024-2025",
    ],
    "Team Data": [
        "/teams?league=12",
        "/statistics?league=12&season=2024-2025&team=132",
    ],
    "Player Data (Requires Higher Plan)": [
        "/players?search=LeBron",
        "/players/statistics?season=2024-2025&player=265",
    ],
    "Betting Data (May Require Higher Plan)": [
        "/odds?game=1",
        "/odds/bookmakers",
    ],
}

working_endpoints = []
forbidden_endpoints = []

for category, endpoints in test_endpoints.items():
    print(f"\n{'='*80}")
    print(f"{category}")
    print('='*80)

    for endpoint in endpoints:
        url = f"{BASE_URL}{endpoint}"

        try:
            response = requests.get(url, headers=HEADERS, timeout=10)

            if response.status_code == 200:
                print(f"✅ {endpoint}")
                print(f"   Status: 200 OK")

                try:
                    data = response.json()
                    if "response" in data:
                        print(f"   Results: {len(data['response'])} items")
                        if data["response"] and len(data["response"]) > 0:
                            print(f"   Sample: {str(data['response'][0])[:100]}...")
                    else:
                        print(f"   Data: {str(data)[:200]}...")
                except:
                    print(f"   Response: {response.text[:100]}")

                working_endpoints.append(endpoint)

            elif response.status_code == 403:
                print(f"❌ {endpoint}")
                print(f"   Status: 403 Forbidden (Not included in your plan)")
                forbidden_endpoints.append(endpoint)

            elif response.status_code == 404:
                print(f"⚠️  {endpoint}")
                print(f"   Status: 404 Not Found (May need valid parameters)")

            elif response.status_code == 429:
                print(f"⚠️  {endpoint}")
                print(f"   Status: 429 Rate Limited")

            else:
                print(f"❓ {endpoint}")
                print(f"   Status: {response.status_code}")
                print(f"   Response: {response.text[:100]}")

        except Exception as e:
            print(f"❌ {endpoint}")
            print(f"   Error: {e}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"\n✅ Working Endpoints ({len(working_endpoints)}):")
for ep in working_endpoints:
    print(f"   {ep}")

print(f"\n❌ Forbidden/Not Included ({len(forbidden_endpoints)}):")
for ep in forbidden_endpoints:
    print(f"   {ep}")

print("\n" + "=" * 80)
print("DIAGNOSIS")
print("=" * 80)

if "/games" in str(working_endpoints) and "/players" in str(forbidden_endpoints):
    print("""
✅ YOUR API KEY IS WORKING!

✅ What's Working:
   - Game data (schedules, scores, etc.)
   - Team data
   - General endpoints

❌ What's NOT Working:
   - Player statistics
   - Detailed player data

🎯 ROOT CAUSE:
   Your current API plan does NOT include player statistics endpoints.
   This is common with free/basic plans.

💡 SOLUTIONS:

Option 1: Upgrade API Plan (Recommended for Full Functionality)
   → Go to your dashboard and upgrade to a plan that includes player stats
   → Usually "Pro" or "Premium" tier
   → Cost: Typically $10-30/month depending on provider

Option 2: Use Alternative Free Player Stats API
   → Use nba_api Python package (free, direct from NBA)
   → Install: pip install nba_api
   → Get player stats from NBA directly
   → Keep using API-Sports for games/odds

Option 3: Modify Your Analyzer (Quick Fix)
   → Use only team-level statistics instead of player stats
   → Analyze game totals and team performance
   → Skip player prop bets, focus on game props

Option 4: Hybrid Approach (Best Free Solution)
   → Use API-Sports for game schedules and odds ✅
   → Use nba_api for player statistics ✅
   → Combine both data sources
   → All free, no upgrade needed!

RECOMMENDED: Option 4 (Hybrid Approach)
   I can modify your analyzer to pull player stats from nba_api
   while still using your current API key for games and odds.
   This gives you full functionality at no cost!

Want me to implement the hybrid solution?
""")

elif len(working_endpoints) == 0:
    print("""
❌ NO ENDPOINTS WORKING

Your API key is being rejected for ALL endpoints.
This means:
   - Key is not valid/activated
   - Or wrong API service
   - Or no plan selected

Check your dashboard again!
""")

elif "/players" in str(working_endpoints):
    print("""
✅ EVERYTHING IS WORKING!

Your plan includes player statistics.
The analyzer should work perfectly.

If you're still seeing errors, it might be:
   - Rate limiting (too many requests)
   - Specific players not in database
   - Season format issues
""")

else:
    print("""
⚠️  MIXED RESULTS

Some endpoints work, others don't.
This suggests a partial plan or specific restrictions.

Working: """ + str(working_endpoints) + """
Not Working: """ + str(forbidden_endpoints) + """

Check your dashboard to see which endpoints are included in your plan.
""")

print("\n" + "=" * 80)
