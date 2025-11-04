import requests
import json

SPORTSBOOK_API_KEY = "3ee00eb314b80853c6c77920c5bf74f7"

def test_nba_events_endpoint():
    """
    Fetch NBA events with available odds to verify API connectivity.
    """
    url = "https://api.sportsgameodds.com/v2/events"
    headers = {"X-Api-Key": SPORTSBOOK_API_KEY}
    params = {
        "oddsAvailable": "true",
        "leagueID": "NBA"
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()  # Raises HTTPError for bad responses
        data = response.json()
        print("✅ API request successful!")
        print(f"Number of events returned: {len(data.get('data', []))}\n")
        print(json.dumps(data.get('data', [])[:5], indent=2))  # show first 5 events
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching NBA events: {e}")

# Run the test
if __name__ == "__main__":
    test_nba_events_endpoint()
