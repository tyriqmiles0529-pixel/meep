"""
Test various API endpoint combinations to find the correct format
"""

import requests
import json

API_KEY = "4979ac5e1f7ae10b1d6b58f1bba01140"

# Different possible base URLs
base_urls = [
    "https://v1.basketball.api-sports.io",
    "https://api-basketball.p.rapidapi.com",
    "https://v2.nba.api-sports.io",
    "https://api.api-basketball.com",
]

# Different possible endpoints
endpoints = [
    "/status",
    "/timezone",
    "/countries",
    "/leagues",
    "/seasons",
]

# Different header combinations
header_combos = [
    {"x-apisports-key": API_KEY},
    {"x-rapidapi-key": API_KEY, "x-rapidapi-host": "v1.basketball.api-sports.io"},
    {"x-rapidapi-key": API_KEY, "x-rapidapi-host": "api-basketball.p.rapidapi.com"},
    {"Authorization": f"Bearer {API_KEY}"},
    {"apikey": API_KEY},
]

print("=" * 80)
print("TESTING ALL ENDPOINT COMBINATIONS")
print("=" * 80)

working_combos = []

for base_url in base_urls:
    print(f"\n{'='*80}")
    print(f"Testing Base URL: {base_url}")
    print('='*80)

    for endpoint in endpoints:
        full_url = f"{base_url}{endpoint}"

        for idx, headers in enumerate(header_combos, 1):
            print(f"\n  🔍 {endpoint} with Headers #{idx}")

            try:
                response = requests.get(full_url, headers=headers, timeout=5)

                if response.status_code == 200:
                    print(f"     ✅ SUCCESS! Status: 200")
                    print(f"     URL: {full_url}")
                    print(f"     Headers: {headers}")

                    try:
                        data = response.json()
                        print(f"     Response preview: {json.dumps(data, indent=6)[:300]}")
                    except:
                        print(f"     Response: {response.text[:200]}")

                    working_combos.append({
                        "url": full_url,
                        "base": base_url,
                        "endpoint": endpoint,
                        "headers": headers
                    })

                elif response.status_code == 401:
                    print(f"     ⚠️  401 Unauthorized - Wrong auth format")
                elif response.status_code == 403:
                    print(f"     ❌ 403 Forbidden - Access denied")
                elif response.status_code == 404:
                    print(f"     ⚠️  404 Not Found - Wrong endpoint")
                elif response.status_code == 429:
                    print(f"     ⚠️  429 Rate Limited")
                else:
                    print(f"     ⚠️  Status: {response.status_code}")

            except requests.exceptions.Timeout:
                print(f"     ⏱️  Timeout")
            except Exception as e:
                print(f"     ❌ Error: {str(e)[:50]}")

print("\n" + "=" * 80)
print("SUMMARY OF WORKING COMBINATIONS")
print("=" * 80)

if working_combos:
    print(f"\n✅ Found {len(working_combos)} working combination(s)!\n")

    for combo in working_combos:
        print(f"Base URL: {combo['base']}")
        print(f"Endpoint: {combo['endpoint']}")
        print(f"Headers: {combo['headers']}")
        print(f"Full URL: {combo['url']}")
        print("-" * 80)

    print("\n💡 Update your scripts with:")
    print(f"   BASE_URL = \"{working_combos[0]['base']}\"")
    print(f"   HEADERS = {working_combos[0]['headers']}")

else:
    print("\n❌ No working combinations found!")
    print("\nPossible issues:")
    print("1. API key is invalid or not activated")
    print("2. Need to subscribe to a plan first")
    print("3. Email verification pending")
    print("4. API service changed URLs/authentication")
    print("\n📝 Verify at your API provider's dashboard:")
    print("   - API-Sports: https://dashboard.api-basketball.com/")
    print("   - RapidAPI: https://rapidapi.com/api-sports/api/api-basketball")

print("\n" + "=" * 80)
print("CHECKING FOR COMMON ISSUES")
print("=" * 80)

# Test if it's a network issue
print("\n1. Testing basic internet connectivity...")
try:
    response = requests.get("https://www.google.com", timeout=5)
    if response.status_code == 200:
        print("   ✅ Internet connection working")
    else:
        print("   ⚠️  Unusual response from google.com")
except:
    print("   ❌ No internet connection or network blocked")

# Test if we can reach the API server at all
print("\n2. Testing if API server is reachable...")
try:
    response = requests.get("https://v1.basketball.api-sports.io/status", timeout=5)
    print(f"   Server responded with: {response.status_code}")
    if response.status_code == 403:
        print("   ✅ Server is up (403 = auth issue, not server issue)")
    elif response.status_code == 401:
        print("   ✅ Server is up (401 = auth issue, not server issue)")
except:
    print("   ❌ Cannot reach API server")

print("\n3. API Key format check...")
print(f"   Key: {API_KEY}")
print(f"   Length: {len(API_KEY)} chars (should be 32)")
print(f"   Contains only hex: {all(c in '0123456789abcdef' for c in API_KEY)}")

print("\n" + "=" * 80)
