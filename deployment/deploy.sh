#!/bin/bash

# Alza Email Bot - Cloud Function Deployment Script

set -e

echo "🚀 Deploying Alza Email Bot to Google Cloud Functions"
echo "==================================================="

# Load environment variables
if [ -f ../.env ]; then
    export $(cat ../.env | grep -v '^#' | xargs)
else
    echo "❌ .env file not found. Please create it first."
    exit 1
fi

# Check if required variables are set
if [ -z "$GOOGLE_CLOUD_PROJECT" ]; then
    echo "❌ GOOGLE_CLOUD_PROJECT not set in .env"
    exit 1  
fi

echo "📋 Project: $GOOGLE_CLOUD_PROJECT"
echo "📍 Location: $GOOGLE_CLOUD_LOCATION_GEMINI"

# Navigate to project root
cd ..

# Deploy the Pub/Sub triggered function
echo "📤 Deploying Pub/Sub triggered function..."
gcloud functions deploy alza-email-processor \
    --gen2 \
    --runtime=python311 \
    --region=$GOOGLE_CLOUD_LOCATION_GEMINI \
    --source=. \
    --entry-point=process_email_notification \
    --trigger-topic=$PUBSUB_TOPIC_NAME \
    --timeout=${FUNCTION_TIMEOUT}s \
    --memory=1GiB \
    --max-instances=10 \
    --service-account="alza-email-bot@${GOOGLE_CLOUD_PROJECT}.iam.gserviceaccount.com" \
    --set-env-vars="GOOGLE_CLOUD_PROJECT=${GOOGLE_CLOUD_PROJECT},GOOGLE_CLOUD_LOCATION_GEMINI=${GOOGLE_CLOUD_LOCATION_GEMINI},RAG_CORPUS_DISPLAY_NAME=${RAG_CORPUS_DISPLAY_NAME},GEMINI_MODEL=${GEMINI_MODEL},GEMINI_MODEL_FINAL_ANSWER=${GEMINI_MODEL_FINAL_ANSWER},BOT_EMAIL_ADDRESS=${BOT_EMAIL_ADDRESS},GCS_BUCKET_NAME=${GCS_BUCKET_NAME},FIRESTORE_DB_ID=${FIRESTORE_DB_ID}"

# Deploy the HTTP triggered function for testing
echo "🌐 Deploying HTTP triggered function for testing..."
gcloud functions deploy alza-email-processor-http \
    --gen2 \
    --runtime=python311 \
    --region=$GOOGLE_CLOUD_LOCATION_GEMINI \
    --source=. \
    --entry-point=process_emails_http \
    --trigger-http \
    --allow-unauthenticated \
    --timeout=${FUNCTION_TIMEOUT}s \
    --memory=1GiB \
    --max-instances=5 \
    --service-account="alza-email-bot@${GOOGLE_CLOUD_PROJECT}.iam.gserviceaccount.com" \
    --set-env-vars="GOOGLE_CLOUD_PROJECT=${GOOGLE_CLOUD_PROJECT},GOOGLE_CLOUD_LOCATION_GEMINI=${GOOGLE_CLOUD_LOCATION_GEMINI},RAG_CORPUS_DISPLAY_NAME=${RAG_CORPUS_DISPLAY_NAME},GEMINI_MODEL=${GEMINI_MODEL},GEMINI_MODEL_FINAL_ANSWER=${GEMINI_MODEL_FINAL_ANSWER},BOT_EMAIL_ADDRESS=${BOT_EMAIL_ADDRESS},GCS_BUCKET_NAME=${GCS_BUCKET_NAME},FIRESTORE_DB_ID=${FIRESTORE_DB_ID}"

echo "✅ Deployment completed successfully!"
echo ""

# Get function URLs
echo "📡 Function URLs:"
echo "Pub/Sub Function: alza-email-processor (triggered by Gmail notifications)"

HTTP_URL=$(gcloud functions describe alza-email-processor-http --region=$GOOGLE_CLOUD_LOCATION_GEMINI --format="value(serviceConfig.uri)")
echo "HTTP Function: $HTTP_URL"

echo ""
echo "🧪 Test the HTTP function:"
echo "curl -X POST $HTTP_URL"
echo ""
echo "📊 Monitor function logs:"
echo "gcloud functions logs read alza-email-processor --region=$GOOGLE_CLOUD_LOCATION_GEMINI --limit=50"
echo ""
echo "Next steps:"
echo "1. Configure Gmail push notifications to trigger Pub/Sub"
echo "2. Send a test email to your bot email address"
echo "3. Monitor the function logs to see processing"