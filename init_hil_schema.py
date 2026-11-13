"""
One-time creation of the HIL structure in your Firebase Realtime Database.

    python init_hil_schema.py

Creates (only if absent) under hil/ : _schema, config/thresholds, control, status.
It never overwrites what already exists and never deletes anything. Nothing here
expires: you remove HIL data yourself, in the Firebase console.
"""

from firebase_admin import db

import utils.firebase  # noqa: F401  (initialises firebase_admin with your service account)
from hil.config import HILConfig, config_as_dict
from hil.sinks import FirebaseSink

if __name__ == "__main__":
    created, kept = FirebaseSink(db).init_schema(config_as_dict(HILConfig()))
    print("created:", *created, sep="\n  ")
    print("already there (left untouched):", *kept, sep="\n  ")
