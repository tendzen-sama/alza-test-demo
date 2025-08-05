# Deployment Scripts

**Automated Google Cloud Platform deployment for Alza Email Bot**

## 🚀 Quick Deploy

### Prerequisites
1. **Google Cloud SDK** installed and authenticated
2. **GCP Project** with billing enabled
3. **Environment file** configured (see below)

### Environment Setup
Create `.env` file in project root:

```env
# Required GCP Settings
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION_GEMINI=europe-west1

# Email Bot Configuration  
BOT_EMAIL_ADDRESS=your-bot@gmail.com
GCS_BUCKET_NAME=your-storage-bucket
FIRESTORE_DB_ID=email-history

# Pub/Sub Configuration
PUBSUB_TOPIC_NAME=email-notifications
PUBSUB_SUBSCRIPTION_NAME=email-notifications-sub

# AI Configuration
GEMINI_MODEL=gemini-2.5-flash-lite-001  
GEMINI_MODEL_FINAL_ANSWER=gemini-2.5-pro
RAG_CORPUS_DISPLAY_NAME=alza-email-bot-knowledge

# Function Settings (for deployment script)
FUNCTION_TIMEOUT=540
```

### Deploy Commands

```bash
# 1. Setup GCP infrastructure
cd deployment
./setup_gcp.sh

# 2. Deploy Cloud Functions
./deploy.sh

# 3. Initialize Gmail monitoring
cd ../src
python start_watch.py
```

## 📋 What Gets Created

### Infrastructure
- **Cloud Functions**: Email processing (Pub/Sub + HTTP triggers)
- **Pub/Sub**: Email notification topic and subscription
- **Storage Bucket**: RAG documents and audio files
- **Service Account**: Proper IAM permissions
- **APIs**: Gmail, Vertex AI, Cloud Functions, Pub/Sub

### Permissions
- Cloud Functions invoker
- Pub/Sub subscriber  
- AI Platform user
- Storage object viewer
- Service account token creator

## 🔧 Manual Setup Alternative

If you prefer manual setup:

1. **Enable APIs** in Cloud Console
2. **Create Pub/Sub topic**: `email-notifications`
3. **Create Storage bucket** for your region
4. **Setup service account** with required roles
5. **Deploy function** using gcloud CLI

## 🧪 Testing

```bash
# Test HTTP endpoint
curl -X POST https://REGION-PROJECT.cloudfunctions.net/alza-email-processor-http

# Monitor logs
gcloud functions logs read alza-email-processor --limit=20 --region=europe-west1
```

## 📊 Production Considerations

- **Quotas**: Monitor Vertex AI and Gmail API limits
- **Scaling**: Functions auto-scale to handle email volume  
- **Security**: Service account follows least-privilege principle
- **Monitoring**: Cloud Logging enabled for debugging
- **Cost**: Serverless architecture minimizes idle costs

## 🔐 Security Notes

- Never commit `.env` files to Git
- Use Secret Manager for production credentials
- Service accounts have minimal required permissions
- HTTP endpoint allows unauthenticated access for testing only