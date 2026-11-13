"""
Ways to notify the maintenance crew WITHOUT the GSM board. Run from the computer that runs
run_hil_live.py (needs internet). Secrets come from environment variables, never from code.

  TwilioSender  real SMS through the Twilio web service
                  TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM (your Twilio number), STMACS_SMS_TO (crew number, +267...)
                  A free trial only sends to numbers you have verified in Twilio, and can ONLY send Twilio's
                  predefined template messages . To use a trial set
                  TWILIO_TRIAL_TEMPLATE=sms_internal_alerts : the phone then receives that template's fixed
                  wording, NOT the fault details (the details stay on the dashboard and in Fault Logs).
  NtfySender    a PUSH notification (not an SMS) to the free ntfy phone app; no account needed
                  STMACS_NTFY_TOPIC (a hard-to-guess topic name you subscribe to in the app)

Statuses are only ever what the service reported:
  pending -> queued -> sent -> delivered      or      failed
A notification is never reported as sent unless the service confirmed it.
"""

import os
import time

import requests


class TwilioSender:
    name, channel = "twilio", "sms"

    def __init__(self, sid=None, token=None, from_number=None, to_number=None, timeout=8, trial_template=None):
        self.trial_template = trial_template or os.environ.get("TWILIO_TRIAL_TEMPLATE")
        self.sid = sid or os.environ.get("TWILIO_ACCOUNT_SID")
        self.token = token or os.environ.get("TWILIO_AUTH_TOKEN")
        self.from_number = from_number or os.environ.get("TWILIO_FROM")
        self.to_number = to_number or os.environ.get("STMACS_SMS_TO")
        self.timeout = timeout
        missing = [n for n, v in (("TWILIO_ACCOUNT_SID", self.sid), ("TWILIO_AUTH_TOKEN", self.token),
                                  ("TWILIO_FROM", self.from_number), ("STMACS_SMS_TO", self.to_number)) if not v]
        if missing:
            raise RuntimeError("missing environment variables: " + ", ".join(missing))

    def send(self, text, update):
        base = f"https://api.twilio.com/2010-04-01/Accounts/{self.sid}/Messages"
        try:
            r = requests.post(base + ".json", auth=(self.sid, self.token), timeout=self.timeout,
                              data={"From": self.from_number, "To": self.to_number,
                                    "Body": self.trial_template or text})
            body = r.json() if r.content else {}
        except Exception as exc:
            update("failed", f"could not reach Twilio: {type(exc).__name__}")
            return
        if r.status_code >= 300:
            update("failed", f"Twilio refused it: {body.get('message', r.status_code)}")
            return
        note = (f" (trial template '{self.trial_template}': generic wording, no fault details)"
                if self.trial_template else "")
        update("queued", "accepted by Twilio, waiting for the network" + note)
        msg_url = f"{base}/{body.get('sid')}.json"
        for _ in range(12):                                  # follow the delivery report for ~1 minute
            time.sleep(5)
            try:
                st = requests.get(msg_url, auth=(self.sid, self.token), timeout=self.timeout).json().get("status")
            except Exception:
                continue
            if st in ("sent", "delivered"):
                update(st, f"Twilio reports: {st}")
                if st == "delivered":
                    return
            elif st in ("failed", "undelivered"):
                update("failed", f"Twilio reports: {st}")
                return


class NtfySender:
    name, channel = "ntfy", "push"

    def __init__(self, topic=None, server="https://ntfy.sh", timeout=8):
        self.topic = topic or os.environ.get("STMACS_NTFY_TOPIC")
        self.server, self.timeout = server.rstrip("/"), timeout
        if not self.topic:
            raise RuntimeError("missing environment variable: STMACS_NTFY_TOPIC")

    def send(self, text, update):
        try:
            r = requests.post(f"{self.server}/{self.topic}", data=text.encode("utf-8"), timeout=self.timeout,
                              headers={"Title": "STMaCS HIL alert", "Priority": "high"})
        except Exception as exc:
            update("failed", f"could not reach ntfy: {type(exc).__name__}")
            return
        if r.status_code == 200:
            update("sent", "published to the ntfy topic (push notification, NOT an SMS; phone delivery not confirmed)")
        else:
            update("failed", f"ntfy answered {r.status_code}")


class BoardSender:
    """
    SMS through YOUR ESP32-S3 + SIM7670G board. The bridge writes a job to hil/sms_outbox/<id> (status
    "pending"); the board (stmacs_sms_board firmware) picks it up over WiFi, sends it with its SIM and
    writes back "sent" ONLY when the modem accepted it, or "failed". If no board answers, it is "failed".
    "sent" means the modem accepted the SMS; delivery to the handset is not confirmed.
    """
    name, channel = "board", "sms"

    def __init__(self, db, root="hil", wait_s=150, poll_s=3, clock=time.time):
        self.db, self.root, self.wait_s, self.poll_s, self.clock = db, root, wait_s, poll_s, clock
        self._n = 0

    def send(self, text, update):
        self._n += 1
        job_id = f"J{int(self.clock() * 1000)}_{self._n}"
        clean = text.encode("ascii", "replace").decode("ascii").replace('"', "'").replace("\\", "/")[:160]
        ref = self.db.reference(f"{self.root}/sms_outbox/{job_id}")
        try:
            ref.set({"text": clean, "status": "pending", "created_epoch": self.clock(), "source": "hil_bridge"})
        except Exception as exc:
            update("failed", f"could not queue the SMS job: {type(exc).__name__}")
            return
        update("queued", "waiting for the SMS board (ESP32-S3) to pick it up")
        deadline = self.clock() + self.wait_s
        while self.clock() < deadline:
            time.sleep(self.poll_s)
            try:
                job = ref.get() or {}
            except Exception:
                continue
            st = job.get("status")
            if st == "sending":
                update("queued", "the SMS board is sending it now")
            elif st == "sent":
                update("sent", job.get("detail") or "modem accepted the SMS")
                return
            elif st == "failed":
                update("failed", job.get("detail") or "the SMS board could not send it")
                return
        update("failed", f"no SMS board picked it up within {self.wait_s} s (is the S3 board on and on WiFi?)")


class DemoSender:
    """
    DEMO ONLY: no message is sent anywhere. Marks the notification "simulated" at once so the dashboard can
    show a clearly labelled "maintenance crew notified (demo, simulated)" popup. The status is "simulated",
    never "sent" or "delivered", in the popup, the records and Fault Logs.
    """
    name, channel = "demo", "demo"

    def send(self, text, update):
        update("simulated", "DEMO: crew notification simulated, no SMS was sent")


def make_sender(kind, db=None):
    if kind == "twilio":
        return TwilioSender()
    if kind == "ntfy":
        return NtfySender()
    if kind == "board":
        return BoardSender(db)
    if kind == "demo":
        return DemoSender()
    return kind                      # "unavailable" / "firebase" stay as plain strings
