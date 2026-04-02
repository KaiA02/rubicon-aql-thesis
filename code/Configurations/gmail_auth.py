import os
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from google.auth.transport.requests import Request


SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLIENT_SECRET_PATH = os.path.join(BASE_DIR, "client_secret.json")
TOKEN_PATH = os.path.join(BASE_DIR, "token.json")

def get_gmail_service(trace: list[str]):
    trace.append("GMAIL: AUTH START")

    creds = None

    if os.path.exists(TOKEN_PATH):
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
        trace.append("GMAIL: TOKEN LOADED FROM FILE")

    # AUTO-REFRESH
    if creds and creds.expired and creds.refresh_token:
        trace.append("GMAIL: TOKEN EXPIRED → REFRESHING")
        creds.refresh(Request())

        with open(TOKEN_PATH, "w") as f:
            f.write(creds.to_json())

        trace.append("GMAIL: TOKEN REFRESHED & SAVED")

    # ONLY IF REALLY NECESSARY
    if not creds or not creds.valid:
        trace.append("GMAIL: NO VALID TOKEN → BROWSER LOGIN")
        flow = InstalledAppFlow.from_client_secrets_file(
            CLIENT_SECRET_PATH,
            SCOPES
        )
        creds = flow.run_local_server(port=0)

        with open(TOKEN_PATH, "w") as f:
            f.write(creds.to_json())

        trace.append("GMAIL: NEW TOKEN SAVED")

    trace.append("GMAIL: AUTH OK")
    return build("gmail", "v1", credentials=creds)
