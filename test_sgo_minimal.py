#!/usr/bin/env python3
"""
Minimal test to isolate the exact SportsGameOdds API issue
"""

import requests
import os

SGO_API_KEY = os.getenv("SGO_API_KEY") or "3ee00eb314b80853c6c77920c5bf74f7"

print("="*60)
print("MINIMAL SPORTSGAMEODDS API TEST")
print("="*60)
print(f"API Key: {SGO_API_KEY[:15]}...{SGO_API_KEY[-10:]}\n")

# Exact minimal request
url = 'https://api.sportsgameodds.com/v2/events'
params = {
    'leagueID': 'NBA',
    'marketOddsAvailable': 'true',
    'limit': 50,
}
headers = {
    'x-api-key': SGO_API_KEY
}

print("Request Details:")
print(f"  URL: {url}")
print(f"  Params: {params}")
print(f"  Headers: {{'x-api-key': '{SGO_API_KEY[:10]}...'}}\n")

try:
    response = requests.get(url, params=params, headers=headers, timeout=10)

    print(f"Response:")
    print(f"  Status Code: {response.status_code}")
    print(f"  Full URL: {response.url}")
    print(f"  Headers Sent: {response.request.headers}")
    print(f"\nResponse Body:")
    print(response.text[:500])
    print()

    if response.status_code == 200:
        data = response.json()
        events = data.get('events', [])
        print(f"✅ SUCCESS! Got {len(events)} events")
        if events:
            print(f"\nFirst event sample:")
            print(f"  ID: {events[0].get('id')}")
            print(f"  Teams: {events[0].get('homeTeam')} vs {events[0].get('awayTeam')}")
    else:
        print(f"❌ FAILED with status {response.status_code}")

except Exception as e:
    print(f"❌ ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("If this fails, paste the FULL output above")
print("="*60)
