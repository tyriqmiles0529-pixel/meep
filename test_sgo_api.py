#!/usr/bin/env python3
"""
Test SportsGameOdds API with the exact format from their documentation
"""

import requests
import os

# Get API key
SGO_API_KEY = os.getenv("SGO_API_KEY") or "3ee00eb314b80853c6c77920c5bf74f7"

print("Testing SportsGameOdds API...")
print(f"API Key: {SGO_API_KEY[:10]}...{SGO_API_KEY[-10:]}")
print()

# Test 1: Exact format from docs (no auth shown in docs sample)
print("="*60)
print("TEST 1: Exact format from docs (no auth)")
print("="*60)

try:
    response = requests.get('https://api.sportsgameodds.com/v2/events', params={
        'leagueID': 'NBA',
        'marketOddsAvailable': 'true',  # lowercase string
        'limit': 50,
    }, timeout=10)

    print(f"URL: {response.url}")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text[:300]}")

    if response.status_code == 200:
        data = response.json()
        print(f"✓ SUCCESS: Got {len(data.get('events', []))} events")
    else:
        print(f"✗ FAILED: {response.status_code}")

except Exception as e:
    print(f"✗ ERROR: {e}")

print()

# Test 2: With x-api-key header (RECOMMENDED)
print("="*60)
print("TEST 2: With x-api-key in header (CORRECT METHOD)")
print("="*60)

try:
    response = requests.get('https://api.sportsgameodds.com/v2/events',
        params={
            'leagueID': 'NBA',
            'marketOddsAvailable': 'true',  # lowercase string
            'limit': 50,
        },
        headers={
            'x-api-key': SGO_API_KEY
        },
        timeout=10
    )

    print(f"URL: {response.url}")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text[:300]}")

    if response.status_code == 200:
        data = response.json()
        print(f"✓ SUCCESS: Got {len(data.get('events', []))} events")
        print(f"   Sample event: {data.get('events', [{}])[0].get('id', 'N/A')}")
    else:
        print(f"✗ FAILED: {response.status_code}")

except Exception as e:
    print(f"✗ ERROR: {e}")

print()

# Test 3: With API key as query param (NOT RECOMMENDED - causes 400)
print("="*60)
print("TEST 3: With apiKey in query params (WRONG - for comparison)")
print("="*60)

try:
    response = requests.get('https://api.sportsgameodds.com/v2/events',
        params={
            'apiKey': SGO_API_KEY,
            'leagueID': 'NBA',
            'marketOddsAvailable': 'true',  # lowercase string
            'limit': 50,
        },
        timeout=10
    )

    print(f"URL (truncated): {response.url[:100]}...")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text[:300]}")

    if response.status_code == 200:
        data = response.json()
        print(f"✓ SUCCESS: Got {len(data.get('events', []))} events")
    else:
        print(f"✗ FAILED: {response.status_code}")

except Exception as e:
    print(f"✗ ERROR: {e}")

print()
print("="*60)
print("SUMMARY")
print("="*60)
print("Check which test succeeded above.")
print("That's the format that should be used in riq_analyzer.py")
