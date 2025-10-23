"""
Simple API test to diagnose authentication issues
"""

import requests
import json

API_KEY = "c47bd0c1e7c0d008a514ecba161b347f"
BASE_URL = "https://v1.basketball.api-sports.io"

print("=" * 70)
print("TESTING DIFFERENT AUTHENTICATION METHODS")
print("=" * 70)

# Method 1: x-rapidapi headers (what you're currently using)
print("\n1️⃣ Testing with x-rapidapi headers...")
headers1 = {
    "x-rapidapi-host": "v1.basketball.api-sports.io",
    "x-rapidapi-key": API_KEY
}

response1 = requests.get(f"{BASE_URL}/status", headers=headers1)
print(f"   Status: {response1.status_code}")
print(f"   Response: {response1.text[:200]}")

# Method 2: x-apisports-key header (direct API-Sports format)
print("\n2️⃣ Testing with x-apisports-key header...")
headers2 = {
    "x-apisports-key": API_KEY
}

response2 = requests.get(f"{BASE_URL}/status", headers=headers2)
print(f"   Status: {response2.status_code}")
print(f"   Response: {response2.text[:200]}")

# Method 3: Simple API key in URL params
print("\n3️⃣ Testing with API key as URL parameter...")
response3 = requests.get(f"{BASE_URL}/status?apikey={API_KEY}")
print(f"   Status: {response3.status_code}")
print(f"   Response: {response3.text[:200]}")

# Method 4: Authorization header
print("\n4️⃣ Testing with Authorization header...")
headers4 = {
    "Authorization": f"Bearer {API_KEY}"
}

response4 = requests.get(f"{BASE_URL}/status", headers=headers4)
print(f"   Status: {response4.status_code}")
print(f"   Response: {response4.text[:200]}")

print("\n" + "=" * 70)
print("DIAGNOSIS:")
print("=" * 70)

if response1.status_code == 403:
    print("\n❌ API Key Issue Detected!")
    print("\nPossible causes:")
    print("   1. API key is invalid or expired")
    print("   2. API subscription is not active")
    print("   3. Rate limit exceeded")
    print("   4. Wrong API service (RapidAPI vs direct API-Sports)")
    print("\n💡 Solutions:")
    print("   • Check if your API key is from RapidAPI or direct from API-Sports")
    print("   • Verify subscription status at: https://dashboard.api-football.com/")
    print("   • If using RapidAPI: https://rapidapi.com/api-sports/api/api-basketball")
    print("   • Generate a new API key if expired")
    print("   • Check if you've exceeded your daily quota")
elif response2.status_code == 200:
    print("\n✅ Method 2 works! Use 'x-apisports-key' header instead")
    print("\nUpdate your code:")
    print('   HEADERS = {"x-apisports-key": API_KEY}')
elif response1.status_code == 200:
    print("\n✅ Method 1 works! Your current setup is correct")
else:
    print(f"\n⚠️ All methods failed. Status codes: {response1.status_code}, {response2.status_code}, {response3.status_code}, {response4.status_code}")
    print("\nThis likely means:")
    print("   • Invalid API key")
    print("   • Subscription expired")
    print("   • Service is down")

print("\n" + "=" * 70)
print("CHECKING API KEY FORMAT")
print("=" * 70)
print(f"\nYour API key: {API_KEY}")
print(f"Length: {len(API_KEY)} characters")
print(f"Format: {'Valid' if len(API_KEY) == 32 else 'Unusual length'}")
print("\nTypical API-Sports key format:")
print("   • Length: 32 characters")
print("   • Contains: lowercase letters and numbers")
print("   • Example: 1234567890abcdef1234567890abcdef")

# Try to get account info if any method worked
for i, response in enumerate([response1, response2, response3, response4], 1):
    if response.status_code == 200:
        print(f"\n✅ Method {i} successful! Full response:")
        try:
            print(json.dumps(response.json(), indent=2))
        except:
            print(response.text)
        break
