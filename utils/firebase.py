import firebase_admin
from firebase_admin import credentials, db


# Initialize Firebase only once
if not firebase_admin._apps:
    cred = credentials.Certificate("firebase-service-account.json")

    firebase_admin.initialize_app(cred, {
        "databaseURL": "https://stmacs-default-rtdb.europe-west1.firebasedatabase.app"
    })


# Reference to the root of the Realtime Database
db_ref = db.reference("/")