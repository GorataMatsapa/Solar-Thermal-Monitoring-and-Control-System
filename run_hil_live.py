"""
Live HIL bridge: connects Firebase, the ESP32 and the digital twin.

    python run_hil_live.py                 # 10x time: a 40-minute test takes about 4 minutes
    python run_hil_live.py --time-scale 1  # real time

Pick the test from the dashboard (or write hil/control in the Firebase console:
  cmd_id (a number that changes every time), scenario_cmd ("T1".."T6", "T4A", "T4B", "stop"),
  s1_source ("simulated" | "physical"), flow_source ("simulated" | "physical")).
Press Ctrl+C to stop; the relays are commanded OFF on exit.
"""

import argparse
import warnings

warnings.filterwarnings("ignore")

from firebase_admin import db

import utils.firebase  # noqa: F401  (initialises firebase_admin with your service account)
from hil.live import LiveBridge

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-scale", type=float, default=None)
    ap.add_argument("--demo", action="store_true",
                    help="short demo tests (about 1 minute each): 20x speed, compressed tests, and a clearly "
                         "labelled SIMULATED crew notification unless you choose another --sms")
    ap.add_argument("--sms", choices=["unavailable", "demo", "board", "twilio", "ntfy"], default=None,
                    help="how to notify the maintenance crew (board = your ESP32-S3 + SIM7670G, twilio = SMS over the internet, ntfy = free phone push)")
    args = ap.parse_args()
    if args.time_scale is None:
        args.time_scale = 20.0 if args.demo else 10.0
    if args.sms is None:
        args.sms = "demo" if args.demo else "unavailable"
    from hil.gateway import make_sender
    LiveBridge(db, time_scale=args.time_scale, sms_backend=make_sender(args.sms, db),
               profile="demo" if args.demo else "standard").run_forever()
