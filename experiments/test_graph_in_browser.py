"""
Test the perfect graph inside a real browser session.
Launches Chrome (same stealth args as session_manager) → navigates to ConfirmTkt
→ syncs cookies → executes API chain with hardcoded params → navigates to results.
"""
import asyncio
import json
import subprocess
import sys
import os
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from morphnet.session_manager import SessionManager

PORT = 9333


def launch_chrome(port: int) -> subprocess.Popen:
    """Launch Chrome with the same stealth args as session_manager."""
    chrome_bin = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    project_root = Path(__file__).parent.parent
    tmp_dir = project_root / ".tmp" / "chrome-profiles"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    profile_dir = tmp_dir / f"chrome-morphnet-{port}"
    cmd = [
        chrome_bin,
        f"--remote-debugging-port={port}",
        f"--user-data-dir={profile_dir}",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-default-apps",
        "--disable-component-update",
        "--disable-breakpad",
        "--disable-sync",
        "--metrics-recording-only",
        "--disable-dev-shm-usage",
        "--disable-features=Translate,OptimizationHints,MediaRouter",
        "--disable-blink-features=AutomationControlled",
        "--exclude-switches=enable-automation",
        "--use-gl=angle",
        "--use-angle=default",
        "--window-size=1920,1080",
        "--window-position=0,0",
        "--password-store=basic",
        "--use-mock-keychain",
        "--force-color-profile=srgb",
        "--lang=en-US",
    ]
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def wait_for_cdp(port: int, timeout: int = 15):
    url = f"http://localhost:{port}/json/version"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return
        except Exception:
            pass
        time.sleep(0.25)
    raise TimeoutError(f"Chrome CDP not responding on port {port}")


async def main():
    print("=== Graph-in-Browser Test ===")
    print("Launching Chrome with stealth args...")
    chrome_proc = launch_chrome(PORT)
    wait_for_cdp(PORT)
    print("Chrome ready.\n")

    session = SessionManager(
        start_url="https://www.confirmtkt.com",
        task_prompt="Search for trains from Pune to Bangalore on 22 April 2026",
        chrome_cdp_url=f"http://localhost:{PORT}",
        site_name="confirmtkt_com",
        headless=False,
        viewport_width=1280,
        viewport_height=900,
    )

    try:
        await session.start()
        print(f"Browser at: {session.page.url}")
        await asyncio.sleep(2)  # Let page fully render

        # Sync browser cookies to curl_cffi HTTP session
        await session.sync_cookies_to_http_session()
        http = session.http_session

        headers = {
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
        }

        base = "https://cttrainsapi.confirmtkt.com"

        # ── Node n0: Auto-suggest source city ────────────────────────
        print("\n── n0: Auto-suggest 'Pune' ──")
        resp = http.get(
            f"{base}/api/v2/trains/stations/auto-suggestion",
            params={
                "searchString": "Pune",
                "popularStnListLimit": "15",
                "preferredStnListLimit": "6",
                "channel": "mwebd",
                "language": "EN",
            },
            headers=headers,
        )
        print(f"  HTTP {resp.status_code}")
        n0_body = resp.json()
        stations = n0_body.get("data", {}).get("stationList", [])
        if not stations:
            print(f"  ERROR: No stations. Response: {json.dumps(n0_body)[:300]}")
            return
        source_code = stations[0]["stationCode"]
        print(f"  → {source_code} ({stations[0].get('stationName', '')})")

        # ── Node n1: Auto-suggest destination city ───────────────────
        print("\n── n1: Auto-suggest 'Bangalore' ──")
        resp = http.get(
            f"{base}/api/v2/trains/stations/auto-suggestion",
            params={
                "searchString": "Bangalore",
                "popularStnListLimit": "15",
                "preferredStnListLimit": "6",
                "channel": "mwebd",
                "language": "EN",
            },
            headers=headers,
        )
        print(f"  HTTP {resp.status_code}")
        n1_body = resp.json()
        stations = n1_body.get("data", {}).get("stationList", [])
        if not stations:
            print(f"  ERROR: No stations. Response: {json.dumps(n1_body)[:300]}")
            return
        dest_code = stations[0]["stationCode"]
        print(f"  → {dest_code} ({stations[0].get('stationName', '')})")

        # ── Node n2: Search trains (chained from n0 + n1) ───────────
        print(f"\n── n2: Search {source_code} → {dest_code} on 22-04-2026 ──")
        resp = http.get(
            f"{base}/api/v1/trains/search",
            params={
                "sourceStationCode": source_code,
                "destinationStationCode": dest_code,
                "dateOfJourney": "22-04-2026",
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
            },
            headers=headers,
        )
        print(f"  HTTP {resp.status_code}")
        n2_body = resp.json()
        data = n2_body.get("data", {})
        trains = data.get("trainsBtwCities") or data.get("trainList") or []

        if trains:
            print(f"  Found {len(trains)} trains:")
            for t in trains[:5]:
                print(f"    {t.get('trainNumber', '?')} {t.get('trainName', '?')} "
                      f"(dep {t.get('departureTime', '?')} → arr {t.get('arrivalTime', '?')})")
        else:
            print(f"  No trains returned.")
            print(f"  Error message: {data.get('errorMessage', 'none')}")
            print(f"  Data keys: {list(data.keys())}")
            # Dump first 500 chars of response for debugging
            print(f"  Raw: {json.dumps(n2_body)[:500]}")

        # Navigate browser to show results (SPA route)
        results_url = (
            f"https://www.confirmtkt.com/rbooking/trains/"
            f"from/{source_code}/to/{dest_code}/22-04-2026"
        )
        print(f"\n── Navigating browser to results ──")
        print(f"  {results_url}")
        await session.page.goto(results_url, wait_until="domcontentloaded", timeout=15000)
        await asyncio.sleep(3)
        title = await session.page.title()
        print(f"  Title: {title}")
        print(f"  URL:   {session.page.url}")

        # ── Verdict ──────────────────────────────────────────────────
        print(f"\n{'='*50}")
        if trains:
            t = trains[0]
            print(f"SUCCESS — {t.get('trainName', '?')} (Train No. {t.get('trainNumber', '?')})")
        else:
            print("FAILED — No trains from API")
        print(f"{'='*50}")

        print("\nBrowser open for inspection. Ctrl+C to exit.")
        await asyncio.sleep(300)

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        await session.close()
        chrome_proc.terminate()
        try:
            chrome_proc.wait(timeout=5)
        except Exception:
            chrome_proc.kill()


if __name__ == "__main__":
    asyncio.run(main())
