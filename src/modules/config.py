"""
Shared configuration and client initialization for the Alza email bot.
"""
import os
import logging
from dotenv import load_dotenv
import vertexai
from google.cloud import firestore, secretmanager, storage

# Load environment
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger('googleapiclient.discovery_cache').setLevel(logging.ERROR)

# Environment Variables
GCP_PROJECT = os.getenv('GOOGLE_CLOUD_PROJECT')
GCP_LOCATION = os.getenv('GOOGLE_CLOUD_LOCATION_GEMINI')
BOT_EMAIL = os.getenv('BOT_EMAIL_ADDRESS')
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-1.5-flash-001')
GEMINI_MODEL_F = os.getenv('GEMINI_MODEL_FINAL_ANSWER', 'gemini-1.5-flash-001')
LLM_RANKER = os.getenv('LLM_RANKER')
GCS_BUCKET_NAME = os.getenv('GCS_BUCKET_NAME')
FIRESTORE_DB_ID = os.getenv('FIRESTORE_DB_ID')
RAG_CORPUS_NAME = os.getenv('RAG_CORPUS_DISPLAY_NAME', 'alza-email-bot-knowledge')

STATE_COLLECTION = "gmail_bot_state"
STATE_DOCUMENT = "last_run_status"

# Client Initialization
vertexai.init(project=GCP_PROJECT, location=GCP_LOCATION)
storage_client = storage.Client()
secret_manager_client = secretmanager.SecretManagerServiceClient()
firestore_client = firestore.Client(project=GCP_PROJECT, database=FIRESTORE_DB_ID)