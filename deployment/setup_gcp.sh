#!/bin/bash

# Alza Email Bot - GCP Infrastructure Setup Script
# Run this script to set up all required GCP services

set -e

echo "🚀 Setting up Alza Email Bot on Google Cloud Platform"
echo "=================================================="

# Load environment variables (user must create .env file)
if [ -f ../.env ]; then
    export $(cat ../.env | grep -v '^#' | xargs)
else
    echo "❌ .env file not found. Please create it with your GCP project settings."
    echo "See README.md for required environment variables."
    exit 1
fi

# Check if required variables are set
if [ -z "$GOOGLE_CLOUD_PROJECT" ]; then
    echo "❌ GOOGLE_CLOUD_PROJECT not set in .env"
    exit 1
fi

echo "📋 Project: $GOOGLE_CLOUD_PROJECT"
echo "📍 Location: $GOOGLE_CLOUD_LOCATION_GEMINI"

# Set the project
echo "🔧 Setting GCP project..."
gcloud config set project $GOOGLE_CLOUD_PROJECT

# Enable required APIs
echo "🔌 Enabling required Google Cloud APIs..."
gcloud services enable \
    cloudfunctions.googleapis.com \
    pubsub.googleapis.com \
    gmail.googleapis.com \
    aiplatform.googleapis.com \
    storage.googleapis.com \
    run.googleapis.com \
    eventarc.googleapis.com

echo "✅ APIs enabled successfully"

# Create Pub/Sub topic for Gmail notifications
echo "📨 Creating Pub/Sub topic: $PUBSUB_TOPIC_NAME"
gcloud pubsub topics create $PUBSUB_TOPIC_NAME || echo "Topic might already exist"

echo "📥 Creating Pub/Sub subscription: $PUBSUB_SUBSCRIPTION_NAME"
gcloud pubsub subscriptions create $PUBSUB_SUBSCRIPTION_NAME \
    --topic=$PUBSUB_TOPIC_NAME \
    --ack-deadline=60 || echo "Subscription might already exist"

# Create storage bucket for RAG documents
echo "🗂️ Creating storage bucket for RAG documents..."
BUCKET_NAME="${GOOGLE_CLOUD_PROJECT}-rag-docs"
gsutil mb -l $GOOGLE_CLOUD_LOCATION_GEMINI gs://$GCS_BUCKET_NAME || echo "Bucket might already exist"

# Upload knowledge base documents to storage
echo "📚 Uploading knowledge base documents..."
gsutil -m cp ../knowledge_base/* gs://$GCS_BUCKET_NAME/knowledge_base/

# Create service account for the Cloud Function
echo "👤 Creating service account..."
SERVICE_ACCOUNT_NAME="alza-email-bot"
SERVICE_ACCOUNT_EMAIL="${SERVICE_ACCOUNT_NAME}@${GOOGLE_CLOUD_PROJECT}.iam.gserviceaccount.com"

gcloud iam service-accounts create $SERVICE_ACCOUNT_NAME \
    --display-name="Alza Email Bot Service Account" \
    --description="Service account for Alza Email Bot Cloud Function" || echo "Service account might already exist"

# Grant necessary permissions to service account
echo "🔐 Granting permissions to service account..."
gcloud projects add-iam-policy-binding $GOOGLE_CLOUD_PROJECT \
    --member="serviceAccount:$SERVICE_ACCOUNT_EMAIL" \
    --role="roles/cloudfunctions.invoker"

gcloud projects add-iam-policy-binding $GOOGLE_CLOUD_PROJECT \
    --member="serviceAccount:$SERVICE_ACCOUNT_EMAIL" \
    --role="roles/pubsub.subscriber"

gcloud projects add-iam-policy-binding $GOOGLE_CLOUD_PROJECT \
    --member="serviceAccount:$SERVICE_ACCOUNT_EMAIL" \
    --role="roles/aiplatform.user"

gcloud projects add-iam-policy-binding $GOOGLE_CLOUD_PROJECT \
    --member="serviceAccount:$SERVICE_ACCOUNT_EMAIL" \
    --role="roles/storage.objectViewer"

# Add required role for Pub/Sub service account
echo "🔐 Adding IAM roles for Pub/Sub service account..."
PROJECT_NUMBER=$(gcloud projects describe $GOOGLE_CLOUD_PROJECT --format="value(projectNumber)")
PUBSUB_SERVICE_ACCOUNT="service-$PROJECT_NUMBER@gcp-sa-pubsub.iam.gserviceaccount.com"

gcloud projects add-iam-policy-binding $GOOGLE_CLOUD_PROJECT \
    --member="serviceAccount:$PUBSUB_SERVICE_ACCOUNT" \
    --role="roles/iam.serviceAccountTokenCreator"

# Note: Speech and Vision APIs not needed - using Gemini multimodal instead

echo "✅ GCP infrastructure setup completed!"
echo ""
echo "Next steps:"
echo "1. Set up Gmail API credentials (see gmail_setup_guide.md)"
echo "2. Upload RAG documents: ./upload_knowledge_base.sh"
echo "3. Deploy Cloud Function: ./deploy.sh"
echo "4. Test the system: python ../main.py"