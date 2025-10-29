#!/usr/bin/env python3
"""
Try ALL possible combinations to find what works with SportsGameOdds API
"""

import requests
import os

SGO_API_KEY = os.getenv("SGO_API_KEY") or "3ee00eb314b80853c6c77920c5bf74f7"

print("="*70)
print("COMPREHENSIVE SPORTSGAMEODDS API TEST")
print("="*70)
print(f"Testing different combinations to find what works...\n")

# Try different league IDs
league_ids = [
    "NBA",
    "nba",
    "basketball_nba",
    "BASKETBALL_NBA",
    "Basketball_NBA",
    "americanbasketball_nba",
    "1",  # Numeric ID
    "4",  # Another common numeric ID for NBA
]

# Try different parameter formats
test_cases = []

for league_id in league_ids:
    # Test 1: Basic with x-api-key header
    test_cases.append({
        "name": f"LeagueID='{league_id}' with header auth",
        "url": "https://api.sportsgameodds.com/v2/events",
        "params": {
            "leagueID": league_id,
            "marketOddsAvailable": "true",
            "limit": 5,
        },
        "headers": {"x-api-key": SGO_API_KEY}
    })

# Also try without marketOddsAvailable
for league_id in ["NBA", "nba", "basketball_nba"]:
    test_cases.append({
        "name": f"LeagueID='{league_id}' WITHOUT marketOddsAvailable",
        "url": "https://api.sportsgameodds.com/v2/events",
        "params": {
            "leagueID": league_id,
            "limit": 5,
        },
        "headers": {"x-api-key": SGO_API_KEY}
    })

# Try with different boolean formats
for bool_val in ["true", "True", "1", True]:
    test_cases.append({
        "name": f"marketOddsAvailable={repr(bool_val)}",
        "url": "https://api.sportsgameodds.com/v2/events",
        "params": {
            "leagueID": "NBA",
            "marketOddsAvailable": bool_val,
            "limit": 5,
        },
        "headers": {"x-api-key": SGO_API_KEY}
    })

successful_tests = []
failed_tests = []

for i, test in enumerate(test_cases, 1):
    print(f"\n[{i}/{len(test_cases)}] {test['name']}")
    print("-" * 70)

    try:
        response = requests.get(
            test['url'],
            params=test['params'],
            headers=test['headers'],
            timeout=10
        )

        status = response.status_code

        if status == 200:
            data = response.json()
            events = data.get('events', [])
            print(f"  ✅ SUCCESS! Status: {status}, Events: {len(events)}")
            print(f"  URL: {response.url}")
            successful_tests.append({
                "test": test['name'],
                "url": response.url,
                "events": len(events)
            })
        else:
            print(f"  ❌ FAILED! Status: {status}")
            print(f"  URL: {response.url}")
            error_msg = response.text[:200]
            print(f"  Error: {error_msg}")
            failed_tests.append({
                "test": test['name'],
                "status": status,
                "error": error_msg
            })

    except Exception as e:
        print(f"  ❌ EXCEPTION: {type(e).__name__}: {str(e)[:100]}")
        failed_tests.append({
            "test": test['name'],
            "status": "Exception",
            "error": str(e)[:100]
        })

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

if successful_tests:
    print(f"\n✅ {len(successful_tests)} SUCCESSFUL COMBINATION(S):")
    for s in successful_tests:
        print(f"\n  Test: {s['test']}")
        print(f"  URL:  {s['url']}")
        print(f"  Events: {s['events']}")
    print("\n⭐ USE THE FIRST SUCCESSFUL COMBINATION IN riq_analyzer.py!")
else:
    print("\n❌ NO SUCCESSFUL COMBINATIONS FOUND")
    print("\nMost common errors:")
    from collections import Counter
    error_types = Counter([f['status'] for f in failed_tests])
    for error_type, count in error_types.most_common(3):
        print(f"  - {error_type}: {count} times")

print("\n" + "="*70)
