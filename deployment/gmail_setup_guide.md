# Gmail API Setup Guide for Alza Email Bot

## Prerequisites

1. Google Cloud Project with billing enabled
2. Gmail account for the bot (e.g., alza-ai-assistant@gmail.com)
3. GCP CLI installed and authenticated

## Step 1: Enable Gmail API

```bash
# Enable Gmail API
gcloud services enable gmail.googleapis.com
```

## Step 2: Create OAuth 2.0 Credentials

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Navigate to **APIs & Services > Credentials**
3. Click **+ CREATE CREDENTIALS > OAuth client ID**
4. Choose **Desktop application**
5. Name it "Alza Email Bot"
6. Download the JSON file as `gmail_credentials.json`

## Step 3: Set Up OAuth Consent Screen

1. Go to **APIs & Services > OAuth consent screen**
2. Choose **External** user type
3. Fill in required information:
   - App name: "Alza Email Bot"
   - User support email: your-email@gmail.com
   - Developer contact: your-email@gmail.com
4. Add scopes:
   - `https://www.googleapis.com/auth/gmail.readonly`
   - `https://www.googleapis.com/auth/gmail.send`
   - `https://www.googleapis.com/auth/gmail.modify`
5. Add test users (including your bot email)

## Step 4: Generate OAuth Token

Create a Python script to generate the initial token:

```python
# generate_token.py
import os
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = [
    'https://www.googleapis.com/auth/gmail.readonly',
    'https://www.googleapis.com/auth/gmail.send',
    'https://www.googleapis.com/auth/gmail.modify'
]

def generate_token():
    creds = None
    
    # Run OAuth flow
    flow = InstalledAppFlow.from_client_secrets_file(
        'gmail_credentials.json', SCOPES)
    creds = flow.run_local_server(port=0)
    
    # Save the credentials for future use
    with open('gmail_token.json', 'w') as token:
        token.write(creds.to_json())
    
    print("✅ Token generated successfully!")
    print("📁 Saved as gmail_token.json")

if __name__ == '__main__':
    generate_token()
```

Run the script:
```bash
python generate_token.py
```

## Step 5: Set Up Gmail Push Notifications

### Create Pub/Sub Topic (if not done by setup script)

```bash
gcloud pubsub topics create email-notifications
```

### Configure Gmail Watch

```python
# setup_gmail_watch.py
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials

def setup_gmail_watch():
    # Load credentials
    creds = Credentials.from_authorized_user_file('gmail_token.json')
    service = build('gmail', 'v1', credentials=creds)
    
    # Set up watch request
    request = {
        'labelIds': ['INBOX'],
        'topicName': 'projects/your-project-id/topics/email-notifications'
    }
    
    # Start watching
    result = service.users().watch(userId='me', body=request).execute()
    print(f"✅ Gmail watch set up: {result}")

if __name__ == '__main__':
    setup_gmail_watch()
```

## Step 6: File Structure

Organize your credentials:

```
credentials/
├── gmail_credentials.json  # OAuth client credentials
└── gmail_token.json       # Generated token
```

Update your `.env` file:
```env
GMAIL_CREDENTIALS_PATH=credentials/gmail_credentials.json
GMAIL_TOKEN_PATH=credentials/gmail_token.json
BOT_EMAIL_ADDRESS=alza-ai-assistant@gmail.com
```

## Step 7: Test Gmail Access

```python
# test_gmail.py
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials

def test_gmail_access():
    try:
        creds = Credentials.from_authorized_user_file('credentials/gmail_token.json')
        service = build('gmail', 'v1', credentials=creds)
        
        # Get profile info
        profile = service.users().getProfile(userId='me').execute()
        print(f"✅ Connected to Gmail: {profile['emailAddress']}")
        
        # List recent messages
        results = service.users().messages().list(userId='me', maxResults=5).execute()
        messages = results.get('messages', [])
        print(f"📬 Found {len(messages)} recent messages")
        
    except Exception as e:
        print(f"❌ Gmail access failed: {e}")

if __name__ == '__main__':
    test_gmail_access()
```

## Security Notes

1. **Never commit credentials to Git**
2. **Use service accounts in production**
3. **Limit OAuth scopes to minimum required**
4. **Regularly rotate credentials**
5. **Monitor API usage and quotas**

## Troubleshooting

### Common Issues

**"insufficient_scope" error**:
- Delete `gmail_token.json` and regenerate with correct scopes

**"invalid_grant" error**:
- Token may be expired, regenerate the token

**"quota_exceeded" error**:
- Check Gmail API quotas in Cloud Console

**Push notifications not working**:
- Verify Pub/Sub topic exists and has correct permissions
- Check that Gmail watch is active (expires after 7 days)

### Verification Commands

```bash
# Check if APIs are enabled
gcloud services list --enabled | grep gmail

# Check Pub/Sub topic
gcloud pubsub topics list

# Check function deployment
gcloud functions list --filter="name:alza-email"

# View function logs
gcloud functions logs read alza-email-processor --limit=20
```

## Production Considerations

1. **Use service account authentication** instead of OAuth for production
2. **Set up Gmail domain-wide delegation** for enterprise use
3. **Implement proper error handling** for expired tokens
4. **Monitor API quotas** and implement rate limiting
5. **Use Cloud Secret Manager** for storing credentials securely