import os
"""
Deep diagnostic for active API key returning 403
"""

import requests
import json
from pprint import pprint

API_KEY = os.getenv("API_SPORTS_KEY", "")

print("=" * 80)
print("DEEP DIAGNOSTIC - ACTIVE API KEY RETURNING 403")
print("=" * 80)

# Test 1: Get detailed error response
print("\n" + "=" * 80)
print("TEST 1: Examining Detailed Error Response")
print("=" * 80)

url = "https://v1.basketball.api-sports.io/status"
headers = {"x-apisports-key": API_KEY}

response = requests.get(url, headers=headers)

print(f"\nStatus Code: {response.status_code}")
print(f"\nResponse Headers:")
for key, value in response.headers.items():
    print(f"  {key}: {value}")

print(f"\nResponse Body:")
print(response.text)

print(f"\nResponse Length: {len(response.text)} bytes")

# Test 2: Try without any parameters
print("\n" + "=" * 80)
print("TEST 2: Testing Different Header Formats")
print("=" * 80)

test_cases = [
    {
        "name": "x-apisports-key",
        "headers": {"x-apisports-key": API_KEY}
    },
    {
        "name": "x-rapidapi-key (basketball host)",
        "headers": {
            "x-rapidapi-key": API_KEY,
            "x-rapidapi-host": "v1.basketball.api-sports.io"
        }
    },
    {
        "name": "x-rapidapi-key (p.rapidapi host)",
        "headers": {
            "x-rapidapi-key": API_KEY,
            "x-rapidapi-host": "api-basketball.p.rapidapi.com"
        },
        "url": "https://api-basketball.p.rapidapi.com/status"
    },
    {
        "name": "X-RapidAPI-Key (capitalized)",
        "headers": {
            "X-RapidAPI-Key": API_KEY,
            "X-RapidAPI-Host": "api-basketball.p.rapidapi.com"
        },
        "url": "https://api-basketball.p.rapidapi.com/status"
    },
]

for test in test_cases:
    print(f"\n📝 Testing: {test['name']}")
    test_url = test.get('url', url)
    print(f"   URL: {test_url}")
    print(f"   Headers: {test['headers']}")

    try:
        resp = requests.get(test_url, headers=test['headers'], timeout=10)
        print(f"   Status: {resp.status_code}")

        if resp.status_code == 200:
            print(f"   ✅ SUCCESS!")
            try:
                data = resp.json()
                print(f"   Data: {json.dumps(data, indent=6)[:500]}")
            except:
                print(f"   Response: {resp.text[:200]}")
        else:
            print(f"   Response: {resp.text[:200]}")
    except Exception as e:
        print(f"   Error: {e}")

# Test 3: Check if it's a subscription/plan issue
print("\n" + "=" * 80)
print("TEST 3: Checking Error Message for Clues")
print("=" * 80)

error_text = response.text.lower()

if "subscription" in error_text:
    print("⚠️  ERROR MENTIONS 'SUBSCRIPTION'")
    print("   → Your account may need an active subscription plan")
    print("   → Check dashboard to ensure a plan is selected")

if "plan" in error_text:
    print("⚠️  ERROR MENTIONS 'PLAN'")
    print("   → You may need to select a plan (even the free one)")

if "limit" in error_text or "quota" in error_text:
    print("⚠️  ERROR MENTIONS 'LIMIT' OR 'QUOTA'")
    print("   → You may have exceeded your daily request limit")
    print("   → Check dashboard for usage statistics")

if "ip" in error_text or "whitelist" in error_text:
    print("⚠️  ERROR MENTIONS 'IP' OR 'WHITELIST'")
    print("   → Your IP address may need to be whitelisted")
    print("   → Check dashboard security settings")

if "rapidapi" in error_text:
    print("⚠️  ERROR MENTIONS 'RAPIDAPI'")
    print("   → This key might be from RapidAPI")
    print("   → Try using RapidAPI URLs and headers")

if error_text == "access denied":
    print("⚠️  GENERIC 'ACCESS DENIED'")
    print("   → Key format is recognized but not authorized")
    print("   → Possible causes:")
    print("     1. Key is valid but plan is not activated")
    print("     2. Key is for a different API service")
    print("     3. Account needs additional verification")

# Test 4: Check account info endpoint
print("\n" + "=" * 80)
print("TEST 4: Trying Alternative Endpoints")
print("=" * 80)

alternative_endpoints = [
    "/status",
    "/timezone",
    "/countries",
    "/leagues",
    "/seasons",
]

for endpoint in alternative_endpoints:
    test_url = f"https://v1.basketball.api-sports.io{endpoint}"
    resp = requests.get(test_url, headers={"x-apisports-key": API_KEY}, timeout=5)

    print(f"\n{endpoint}")
    print(f"  Status: {resp.status_code}")

    if resp.status_code != 403:
        print(f"  ✅ Different response!")
        print(f"  Body: {resp.text[:200]}")

# Test 5: Where did you get this key from?
print("\n" + "=" * 80)
print("TEST 5: Source Identification")
print("=" * 80)

print("\n❓ WHERE DID YOU GET THIS API KEY?")
print("\n1. From dashboard.api-basketball.com")
print("   If YES:")
print("   → Check 'Subscription' section for active plan")
print("   → Check 'Requests' to see if quota is exceeded")
print("   → Verify email is confirmed")
print("   → Check if any IP restrictions are set")

print("\n2. From RapidAPI.com")
print("   If YES:")
print("   → This key won't work with api-sports.io URLs")
print("   → You MUST use api-basketball.p.rapidapi.com URLs")
print("   → You MUST use X-RapidAPI-Key and X-RapidAPI-Host headers")

print("\n3. From somewhere else")
print("   If YES:")
print("   → Please specify where you got it")

print("\n" + "=" * 80)
print("DIAGNOSTIC SUMMARY")
print("=" * 80)

print(f"""
API Key: {API_KEY}
Status Code: {response.status_code}
Error Message: {response.text}

✅ Key is active in dashboard (user confirmed)
❌ Key returns 403 Forbidden
❓ Mismatch between dashboard and actual usage

POSSIBLE EXPLANATIONS:

1. RapidAPI Key Mismatch
   → If key is from RapidAPI, it won't work with direct API-Sports URLs
   → Solution: Update to RapidAPI URLs and headers

2. Plan Not Activated
   → Dashboard shows account as active, but plan not selected
   → Solution: Select a plan in dashboard (even free tier)

3. IP Restriction
   → Your IP may not be whitelisted
   → Solution: Check security settings in dashboard

4. Service Mismatch
   → Key might be for Football API, not Basketball
   → Solution: Get Basketball-specific key

5. Pending Activation
   → Account created but full activation pending
   → Solution: Wait or contact support

NEXT STEPS:

1. Check your dashboard URL:
   - Is it dashboard.api-basketball.com? (correct)
   - Or dashboard.api-football.com? (wrong - that's soccer)
   - Or rapidapi.com? (needs different code)

2. Screenshot your dashboard showing:
   - API Key
   - Plan status
   - Request quota
   - Service name (should say "Basketball")

3. Let me know which dashboard you're using so I can provide
   the exact configuration needed!
""")

print("=" * 80)
