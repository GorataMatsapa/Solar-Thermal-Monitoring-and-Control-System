"""
weather_to_firebase.py - copies the current weather into Firebase for the STMaCS dashboard.

Writes (using Open-Meteo, free, no API key):
    site/weather/latest          {ts, ambient_temp, irradiance, source}   (overwritten)
    site/weather/history/<id>    same record, appended on every run

irradiance is shortwave radiation on a HORIZONTAL plane (W/m2), a weather-service estimate,
not a measurement on the collector. The dashboard labels it that way.

Run it every ~15 minutes (see weather-to-firebase.yml for a free GitHub Actions schedule, or use
Windows Task Scheduler / cron on any computer that is always on).

Settings (environment variables):
    FIREBASE_DATABASE_URL            e.g. https://your-project-default-rtdb.firebaseio.com
    FIREBASE_SERVICE_ACCOUNT_JSON    the service-account key file CONTENT (used by GitHub Actions)
      or GOOGLE_APPLICATION_CREDENTIALS = path to the key file (use this on your own computer)
    STMACS_LAT / STMACS_LON          site coordinates (default: Gaborone)

Install once:  pip install firebase-admin
"""

import json
import os
import sys
import time
import urllib.parse
import urllib.request

import firebase_admin
from firebase_admin import credentials, db

LAT = float(os.environ.get("STMACS_LAT", "-24.6581"))
LON = float(os.environ.get("STMACS_LON", "25.9088"))


def fetch_weather(retries=3):
    query = urllib.parse.urlencode({
        "latitude": LAT, "longitude": LON,
        "current": "temperature_2m,shortwave_radiation",
    })
    url = "https://api.open-meteo.com/v1/forecast?" + query
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=15) as r:
                cur = json.load(r)["current"]
            return float(cur["temperature_2m"]), float(cur["shortwave_radiation"])
        except Exception as exc:                      # network blip, bad JSON ...
            last = exc
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"Open-Meteo request failed: {last}")


def init_firebase():
    raw = os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON")
    cred = credentials.Certificate(json.loads(raw)) if raw else credentials.Certificate(
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"])
    firebase_admin.initialize_app(cred, {"databaseURL": os.environ["FIREBASE_DATABASE_URL"]})


def main():
    ambient, irradiance = fetch_weather()
    record = {
        "ts": int(time.time()),
        "ambient_temp": round(ambient, 1),
        "irradiance": round(irradiance, 0),
        "source": "open-meteo",
    }
    init_firebase()
    db.reference("site/weather/latest").set(record)
    db.reference("site/weather/history").push(record)
    print("Wrote", record)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("FAILED:", exc, file=sys.stderr)
        sys.exit(1)
