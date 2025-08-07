import os.path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# This scope allows reading, sending and modifying emails,
# as well as managing push notifications.
SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]

# --- PROJECT SETTINGS ---
PROJECT_ID = "newtestproject-323113"
TOPIC_NAME = "email-notifications"
# --------------------------------

def main():
    """
    Starts the authentication process and sends a .watch() request to Gmail API.
    """
    creds = None
    # The token.json file will be created automatically to store the access token.
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)

    # If there's no valid token, start the browser login process.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            # Use the credentials.json file you downloaded
            flow = InstalledAppFlow.from_client_secrets_file(
                "../client_secret_2_1071029075877-begchd2oilv47uvie754phl60blj1ae0.apps.googleusercontent.com.json", SCOPES
            )
            creds = flow.run_local_server(port=0)

        # Save the token for future runs
        with open("token.json", "w") as token:
            token.write(creds.to_json())

    try:
        # Create client for working with Gmail API
        service = build("gmail", "v1", credentials=creds)

        # Form the request body for .watch()
        request = {
            "labelIds": ["INBOX", "UNREAD"], # Monitor unread emails in Inbox
            "topicName": f"projects/{PROJECT_ID}/topics/{TOPIC_NAME}",
            "labelFilterAction": "include"
        }

        print("Sending .watch() request to Gmail API...")

        # Send the request
        response = service.users().watch(userId="me", body=request).execute()

        print("Success! Gmail will now send notifications.")
        print(f"History ID: {response['historyId']}")
        print(f"Subscription expires: {response['expiration']}") # Usually after 7 days, for production in roadmap we need implement cloud function with cloud scheduler to renew it automatically

    except HttpError as error:
        print(f"An error occurred: {error}")

if __name__ == "__main__":
    main()