"""
Send ONE test SMS through Twilio, to check your account before running a HIL test.

    python test_sms.py

Needs the four environment variables set in the same PowerShell window:
  TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM, STMACS_SMS_TO
"""
import time

from hil.gateway import TwilioSender

sender = TwilioSender()
print(f"Sending a test SMS from {sender.from_number} to {sender.to_number} ...")
status = {"s": "pending", "d": ""}


def update(s, d):
    status["s"], status["d"] = s, d
    print(f"  status: {s} - {d}")


t = time.time()
sender.send("STMaCS test message. If you can read this, SMS alerts work.", update)
print(f"Final status: {status['s']} ({time.time() - t:.0f} s). {status['d']}")



 