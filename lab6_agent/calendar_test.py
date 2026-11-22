from pathlib import Path
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from datetime import datetime, timezone


# Google Calendar permissions
SCOPES = ["https://www.googleapis.com/auth/calendar.events"]

# File locations
BASE_DIR = Path(__file__).parent
CREDENTIALS_FILE = BASE_DIR / "credentials.json"
TOKEN_FILE = BASE_DIR / "token.json"


def get_calendar_service():

    creds = None

    # Check whether an existing login is available
    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(
            str(TOKEN_FILE), SCOPES
        )

    # Refresh or request authentication if needed
    if not creds or not creds.valid:

        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())

        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                str(CREDENTIALS_FILE), SCOPES
            )

            creds = flow.run_local_server(port=0)

        # Save login information securely
        TOKEN_FILE.write_text(creds.to_json())

    # Connect to Google Calendar
    service = build("calendar", "v3", credentials=creds)

    return service

def create_calendar_event(title, start_time, end_time, description=""):

    service = get_calendar_service()

    event = {
        "summary": title,
        "description": description,
        "start": {
            "dateTime": start_time,
            "timeZone": "Asia/Colombo",
        },
        "end": {
            "dateTime": end_time,
            "timeZone": "Asia/Colombo",
        },
    }

    created_event = service.events().insert(
        calendarId="primary",
        body=event
    ).execute()

    return created_event.get("htmlLink")


