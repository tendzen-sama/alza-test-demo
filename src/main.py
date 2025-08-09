"""
Lightweight Cloud Function orchestrator for Alza email bot.
"""
import base64
import binascii
import hashlib
import json
import logging
import os
import re
import time
import random
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

import functions_framework
from flask import Request
from google.cloud import firestore, secretmanager, storage
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.api_core.exceptions import AlreadyExists, ResourceExhausted
import vertexai
from dotenv import load_dotenv
from vertexai.generative_models import GenerativeModel, Part
from vertexai.generative_models import GenerationConfig
from modules.config import (
    firestore_client, secret_manager_client, 
    GCP_PROJECT, BOT_EMAIL,
    STATE_COLLECTION, STATE_DOCUMENT
)
from modules.gmail_service import _get_email_details, _process_attachments, _send_reply
from modules.ai_core import generate_search_queries_from_email, get_rag_context, generate_final_reply
from modules.security import _sanitize_for_prompt_injection

# --- Initialization ---
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger('googleapiclient.discovery_cache').setLevel(logging.ERROR)


def _add_responsible_ai_disclaimer(html_content: str, sources: list) -> str:
    """Add responsible AI disclaimer and source references to HTML email."""
    html_content = html_content.rstrip()
    if html_content.endswith('</div>'):
        html_content = html_content[:-6]

    # Add source references if sources exist
    sources_section = ""
    if sources:
        unique_sources = list(dict.fromkeys(sources))  # Remove duplicates
        sources_list = "\n".join([f"<li style='font-size: 12px; margin: 5px 0;'>[{i+1}] {source}</li>"
                                  for i, source in enumerate(unique_sources)])
        sources_section = f"""
<div style="margin-top: 25px; padding: 15px; background-color: #f8f9fa; border-radius: 5px;">
<p style="font-size: 13px; margin: 0 0 10px 0; font-weight: bold; color: #333;">📚 Zdroje informací:</p>
<ul style="margin: 0; padding-left: 20px;">
{sources_list}
</ul>
</div>"""

    # Add responsible AI disclaimer
    disclaimer = f"""{sources_section}

<hr style="margin: 20px 0; border: none; border-top: 1px solid #e0e0e0;">
<p style="font-size: 11px; color: #666; text-align: center; line-height: 1.4;">
<em>🤖 Tato odpověď byla vygenerována umělou inteligencí. AI modely mohou dělat chyby, proto prosím ověřte si důležité informace přímo s našimi experty.<br>
Google AI models may make mistakes, so double-check outputs.</em>
</p>
</div>"""

    return html_content + disclaimer

def _create_fallback_response(email_subject: str, rag_context: str) -> str:
    """Create intelligent fallback response when AI generation fails"""
    return f'''<p>Dobrý den,</p>

<p>děkujeme za váš dotaz ohledně "{email_subject}". V současné době dochází k technickým potížím s naším AI asistentem, ale váš email byl přijat a bude zpracován.</p>

<p>Náš tým zákaznické podpory vám odpoví do 24 hodin. Pro okamžitou pomoc nás prosím kontaktujte:</p>

<ul>
<li><strong>Zákaznická linka:</strong> +420 225 340 111 (24/7)</li>
<li><strong>Live chat:</strong> dostupný na www.alza.cz</li>
<li><strong>Email:</strong> info@alza.cz</li>
</ul>

<p>Děkujeme za pochopení.</p>

<p>S pozdravem,<br>
Tým Alza.cz</p>'''

def _create_technical_error_response() -> str:
    """Fallback HTML response for technical errors"""
    return '''<p>Dobrý den,</p>

<p>omlouváme se, ale v současné době dochází k technickým potížím s naším systémem. Váš dotaz byl přijat a náš tým se vám ozve do 24 hodin.</p>

<p>Pro naléhavé dotazy nás prosím kontaktujte na <strong>+420 225 340 111</strong>.</p>

<p>Děkujeme za pochopení.</p>

<p>S pozdravem,<br>
Tým Alza.cz</p>'''

def _mark_email_as_processed(message_id: str, status: str):
    """Mark email as processed in Firestore."""
    try:
        doc_ref = firestore_client.collection("processed_emails").document(message_id)
        doc_ref.set({
            "message_id": message_id,
            "status": status,
            "processed_at": datetime.now(),
            "expires_at": datetime.now() + timedelta(days=30)
        })
        logger.info(f"Marked email {message_id} as '{status}'.")
    except Exception as e:
        logger.error(f"Failed to mark email as processed: {e}")

def _get_last_history_id_from_firestore() -> str:
    """Get the last history ID from Firestore."""
    doc_ref = firestore_client.collection(STATE_COLLECTION).document(STATE_DOCUMENT)
    doc = doc_ref.get()
    return doc.to_dict().get('historyId') if doc.exists else None

def _save_last_history_id_to_firestore(history_id: str):
    """Save the last history ID to Firestore."""
    doc_ref = firestore_client.collection(STATE_COLLECTION).document(STATE_DOCUMENT)
    doc_ref.set({'historyId': str(history_id), 'updated_at': firestore.SERVER_TIMESTAMP})
    logger.info(f"Saved new state to Firestore. Last historyId is now: {history_id}")

def _get_gmail_credentials() -> Credentials:
    """Get Gmail credentials from Secret Manager."""
    token_secret_name = f"projects/{GCP_PROJECT}/secrets/gmail-token/versions/latest"
    token_response = secret_manager_client.access_secret_version(request={"name": token_secret_name})
    token_info = json.loads(token_response.payload.data.decode("UTF-8"))
    creds_secret_name = f"projects/{GCP_PROJECT}/secrets/gmail-credentials/versions/latest"
    creds_response = secret_manager_client.access_secret_version(request={"name": creds_secret_name})
    client_config = json.loads(creds_response.payload.data.decode("UTF-8"))['installed']
    return Credentials(token=token_info.get('token'), refresh_token=token_info.get('refresh_token'), token_uri=client_config.get('token_uri'), client_id=client_config.get('client_id'), client_secret=client_config.get('client_secret'), scopes=token_info.get('scopes'))

def _claim_email_for_processing(message_id: str) -> bool:
    """Claim email for processing to prevent duplicates."""
    doc_ref = firestore_client.collection('processed_emails').document(message_id)
    try:
        doc_ref.create({'claimed_at': firestore.SERVER_TIMESTAMP, 'status': 'processing'})
        logger.info(f"Successfully claimed email {message_id} for processing.")
        return True
    except AlreadyExists:
        logger.warning(f"Email {message_id} already claimed or processed. Skipping.")
        return False

@functions_framework.http
def process_email_http(request: Request):
    """Main Cloud Function entry point."""
    logger.info("Function triggered by Pub/Sub notification.")
    try:
        credentials = _get_gmail_credentials()
        gmail_service = build("gmail", "v1", credentials=credentials)
        start_history_id = _get_last_history_id_from_firestore()
        if not start_history_id:
            profile = gmail_service.users().getProfile(userId='me').execute()
            _save_last_history_id_to_firestore(profile['historyId'])
            return "OK - Initialized state.", 204

        history_response = gmail_service.users().history().list(userId='me', startHistoryId=start_history_id).execute()
        new_history_id = history_response.get('historyId', start_history_id)
        if not history_response.get('history'):
            if new_history_id != start_history_id: _save_last_history_id_to_firestore(new_history_id)
            return "OK", 204

        new_message_ids = {msg['message']['id'] for rec in history_response['history'] if 'messagesAdded' in rec for msg in rec['messagesAdded'] if 'INBOX' in msg['message']['labelIds']}
        if not new_message_ids:
            _save_last_history_id_to_firestore(new_history_id)
            return "OK", 204

        logger.info(f"Found {len(new_message_ids)} new message(s) to process.")
        for message_id in new_message_ids:
            # Initialize variables for exception handling
            headers = {}
            email_subject = ""
            thread_id = ""
            consolidated_rag_context = ""
            
            try:
                if not _claim_email_for_processing(message_id): continue

                message = gmail_service.users().messages().get(userId='me', id=message_id, format='full').execute()
                payload = message.get('payload', {})

                # Temporarily enable debug logging for email parsing
                original_level = logging.getLogger().level
                logging.getLogger().setLevel(logging.DEBUG)
                email_body, headers = _get_email_details(payload)
                logging.getLogger().setLevel(original_level)

                sender_email_match = re.search(r'<(.*?)>', headers.get('from', ''))
                sender_email = sender_email_match.group(1) if sender_email_match else headers.get('from', '')

                if sender_email == BOT_EMAIL:
                    logger.info(f"Ignoring self-sent message {message_id}.")
                    _mark_email_as_processed(message_id, status="ignored_self_sent")
                    continue

                thread_id = message.get('threadId')
                email_subject = headers.get('subject', '')

                # Sanitize user-provided content before sending to LLM
                email_subject = _sanitize_for_prompt_injection(email_subject)
                email_body = _sanitize_for_prompt_injection(email_body)

                attachment_parts = _process_attachments(gmail_service, message_id, payload.get('parts', []))
                search_queries = generate_search_queries_from_email(email_subject, email_body, attachment_parts)

                consolidated_rag_context = ""
                all_sources = []
                if search_queries:
                    rag_results = [get_rag_context(query) for query in search_queries if query]
                    all_contexts = [context for context, sources in rag_results if context]
                    for context, sources in rag_results:
                        all_sources.extend(sources)
                    unique_contexts = list(dict.fromkeys(all_contexts))  # Remove duplicates while preserving order
                    consolidated_rag_context = "\n\n---\n\n".join(unique_contexts)

                attachment_summary_text = "\n".join([f"- Soubor: '{part.get('filename')}'" for part in payload.get('parts', []) if part.get('filename')]) or "Žádné přílohy."

                response = generate_final_reply(
                    email_subject=email_subject,
                    email_body=email_body,
                    attachment_summary=attachment_summary_text,
                    consolidated_rag_context=consolidated_rag_context,
                    attachment_parts=attachment_parts
                )

                try:
                    reply_json = json.loads(response.text)
                    raw_html_body = reply_json.get('html_body', _create_technical_error_response())
                    reply_body = _add_responsible_ai_disclaimer(raw_html_body, all_sources)
                except json.JSONDecodeError:
                    logger.error(f"Failed to decode JSON from final reply model: {response.text}")
                    reply_body = _create_fallback_response(email_subject, consolidated_rag_context)

                logger.info(f"Generated reply: {reply_body[:200].replace(chr(10), ' ')}...")
                _send_reply(gmail_service, headers, thread_id, reply_body, sender_email)
                _mark_email_as_processed(message_id, status="replied")

            except ResourceExhausted as e:
                logger.error(f"🚨 CRITICAL: Resource exhausted while processing {message_id}: {e}")
                try:
                    # Extract sender email for fallback response
                    fallback_sender_match = re.search(r'<(.*?)>', headers.get('from', ''))
                    fallback_sender_email = fallback_sender_match.group(1) if fallback_sender_match else headers.get('from', '')
                    
                    fallback_reply = _create_fallback_response(email_subject, consolidated_rag_context)
                    _send_reply(gmail_service, headers, thread_id, fallback_reply, fallback_sender_email)
                    logger.info(f"✅ Sent fallback response due to quota exhaustion for {message_id}")
                    _mark_email_as_processed(message_id, status="replied_fallback")
                except Exception as send_error:
                    logger.error(f"Failed to send fallback response: {send_error}")
                    _mark_email_as_processed(message_id, status="failed_fallback")
            except Exception as e:
                logger.error(f"Failed to process message {message_id}: {e}")
                # Extract sender email for error response
                error_sender_match = re.search(r'<(.*?)>', headers.get('from', ''))
                error_sender_email = error_sender_match.group(1) if error_sender_match else headers.get('from', '')
                
                if error_sender_email != BOT_EMAIL:  # Don't reply to ourselves
                    fallback_reply = _create_fallback_response(email_subject, "")
                    _send_reply(gmail_service, headers, thread_id, fallback_reply, error_sender_email)
                    logger.info(f"✅ Sent fallback response due to processing error for {message_id}")
                    _mark_email_as_processed(message_id, status="replied_fallback")
                else:
                    _mark_email_as_processed(message_id, status="failed_self_sent")
        
        _save_last_history_id_to_firestore(new_history_id)
        return "Success", 200
        
    except Exception as e:
        logger.error(f"Critical error in main function: {e}")
        return "Error", 500

