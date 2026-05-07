"""
Direct test of the ConfirmTkt API chain — no LLM, no browser, no orchestrator.
Hardcoded params to verify: auto-suggest → station codes → search → trains.
"""
import json
import httpx

BASE = "https://cttrainsapi.confirmtkt.com"
HEADERS = {
    "accept": "*/*",
    "accept-language": "en-US,en;q=0.9",
    "apikey": "ct-web!2$",
    "ct-token": "",
    "ct-userkey": "",
    "clientid": "ct-web",
    "content-type": "application/json",
    "deviceid": "0975e5d5-8374-4f1a-b004-efcf7d550f20",
    "origin": "https://www.confirmtkt.com",
    "referer": "https://www.confirmtkt.com/",
    "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36",
}


def auto_suggest(search_string: str) -> dict:
    """Node n0/n1: auto-suggestion endpoint."""
    params = {
        "searchString": search_string,
        "popularStnListLimit": "15",
        "preferredStnListLimit": "6",
        "channel": "mwebd",
        "language": "EN",
    }
    resp = httpx.get(
        f"{BASE}/api/v2/trains/stations/auto-suggestion",
        params=params,
        headers=HEADERS,
    )
    print(f"\n--- Auto-suggest '{search_string}' ---")
    print(f"  HTTP {resp.status_code}")
    body = resp.json()
    stations = body.get("data", {}).get("stationList", [])
    if stations:
        top = stations[0]
        print(f"  Top match: {top['stationCode']} — {top.get('stationName', '')}")
    else:
        print("  No stations found!")
    return body


def search_trains(source_code: str, dest_code: str, date: str) -> dict:
    """Node n2: train search endpoint."""
    params = {
        "sourceStationCode": source_code,
        "destinationStationCode": dest_code,
        "dateOfJourney": date,
        "sortBy": "DEFAULT",
        "addAvailabilityCache": "true",
        "excludeMultiTicketAlternates": "false",
        "excludeBoostAlternates": "false",
        "enableNearby": "true",
        "enableTG": "true",
        "tGPlan": "CTG-A40",
        "showTGPrediction": "false",
        "tgColor": "DEFAULT",
        "showPredictionGlobal": "true",
        "showNewAlternates": "false",
        "showNewAltText": "true",
    }
    resp = httpx.get(
        f"{BASE}/api/v1/trains/search",
        params=params,
        headers=HEADERS,
    )
    print(f"\n--- Search {source_code} → {dest_code} on {date} ---")
    print(f"  HTTP {resp.status_code}")
    body = resp.json()
    data = body.get("data", {})
    # API uses trainList OR trainsBtwCities depending on version
    trains = data.get("trainsBtwCities") or data.get("trainList") or []
    if trains:
        print(f"  Found {len(trains)} trains:")
        for t in trains[:5]:
            print(f"    {t.get('trainNumber', '?')} {t.get('trainName', '?')}")
    else:
        print(f"  No trains found.")
        print(f"  Error: {data.get('errorMessage', 'none')}")
        print(f"  Response keys: {list(data.keys())}")
        # Show trainList details if present but empty
        if "trainList" in data:
            print(f"  trainList type: {type(data['trainList']).__name__}, len: {len(data['trainList']) if data['trainList'] else 0}")
    return body


def main():
    print("=== Perfect Graph Direct Test ===")
    print("Task: Search for trains from Pune to Bangalore on 22 April 2026")
    print()

    # Step 1: Auto-suggest source city
    source_resp = auto_suggest("Pune")
    source_code = source_resp["data"]["stationList"][0]["stationCode"]
    print(f"  → Source station code: {source_code}")

    # Step 2: Auto-suggest destination city
    dest_resp = auto_suggest("Bangalore")
    dest_code = dest_resp["data"]["stationList"][0]["stationCode"]
    print(f"  → Destination station code: {dest_code}")

    # Step 3: Search trains (chaining station codes from steps 1 & 2)
    search_resp = search_trains(source_code, dest_code, "22-04-2026")

    # Verdict
    data = search_resp.get("data", {})
    trains = data.get("trainsBtwCities") or data.get("trainList") or []
    print(f"\n=== Result: {'SUCCESS' if trains else 'FAILED'} ===")
    if trains:
        t = trains[0]
        print(f"Answer: {t.get('trainName', '?')} (Train No. {t.get('trainNumber', '?')})")


if __name__ == "__main__":
    main()
