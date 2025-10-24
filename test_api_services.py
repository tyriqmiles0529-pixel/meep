import os
"""
Test API key against different API-Sports services
"""

import requests

API_KEY = os.getenv("API_SPORTS_KEY", "")

services = {
    "Basketball v1": "https://v1.basketball.api-sports.io",
    "Football v3": "https://v3.football.api-sports.io",
    "NBA (alternative)": "https://api-nba-v1.p.rapidapi.com",
}

headers_apisports = {
    "x-apisports-key": API_KEY
}

print("=" * 70)
print("TESTING API KEY AGAINST DIFFERENT SERVICES")
print("=" * 70)

for service_name, base_url in services.items():
    print(f"\n🔍 Testing: {service_name}")
    print(f"   URL: {base_url}/status")

    try:
        response = requests.get(f"{base_url}/status", headers=headers_apisports, timeout=10)
        print(f"   Status: {response.status_code}")

        if response.status_code == 200:
            print(f"   ✅ SUCCESS!")
            data = response.json()
            print(f"   Response: {data}")
        elif response.status_code == 403:
            print(f"   ❌ Access Denied (403)")
        elif response.status_code == 404:
            print(f"   ⚠️ Endpoint not found (404)")
        else:
            print(f"   ⚠️ Status: {response.status_code}")
            print(f"   Response: {response.text[:200]}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

print("\n" + "=" * 70)
print("DIAGNOSIS")
print("=" * 70)

# Try checking account info from API-Football dashboard
print("\n📝 Where did you get this API key from?")
print("\n1. From dashboard.api-football.com")
print("   → This is for FOOTBALL (soccer), not basketball!")
print("   → You need a separate key for basketball")
print("   → Go to: https://dashboard.api-basketball.com/")
print("\n2. From RapidAPI.com")
print("   → Check which API you subscribed to")
print("   → Basketball API URL: https://rapidapi.com/api-sports/api/api-basketball")
print("\n3. Both services require SEPARATE subscriptions!")
print("   → API-Football ≠ API-Basketball")
print("   → Same company, different APIs, different keys")

print("\n🎯 SOLUTION:")
print("   Go to: https://dashboard.api-basketball.com/register")
print("   (Note: api-BASKETBALL, not api-football)")
print("   Create account and get basketball-specific API key")
print("=" * 70)
