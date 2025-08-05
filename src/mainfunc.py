import base64
import binascii
import logging
import os
import re
import json
import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Tuple, Optional

import functions_framework
from flask import Request
from google.cloud import firestore, secretmanager, storage
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.api_core.exceptions import AlreadyExists
import vertexai
from dotenv import load_dotenv
from vertexai.generative_models import GenerativeModel, Part

# --- Initialization ---
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger('googleapiclient.discovery_cache').setLevel(logging.ERROR)

# --- Environment Variables ---
GCP_PROJECT = os.getenv('GOOGLE_CLOUD_PROJECT')
# This is the location of the Cloud Function itself
GCP_LOCATION = os.getenv('GOOGLE_CLOUD_LOCATION_GEMINI')
BOT_EMAIL = os.getenv('BOT_EMAIL_ADDRESS')
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-2.0-flash-lite-001')
GEMINI_MODEL_F = os.getenv('GEMINI_MODEL_FINAL_ANSWER', 'gemini-2.0-flash-lite-001')
GCS_BUCKET_NAME = os.getenv('GCS_BUCKET_NAME')
FIRESTORE_DB_ID = os.getenv('FIRESTORE_DB_ID')
RAG_CORPUS_NAME = os.getenv('RAG_CORPUS_DISPLAY_NAME', 'alza-email-bot-knowledge')

STATE_COLLECTION = "gmail_bot_state"
STATE_DOCUMENT = "last_run_status"

# --- Client Initialization ---
# Initialize Vertex AI in the FUNCTION's location for general tasks
vertexai.init(project=GCP_PROJECT, location=GCP_LOCATION)
storage_client = storage.Client()
secret_manager_client = secretmanager.SecretManagerServiceClient()
firestore_client = firestore.Client(project=GCP_PROJECT, database=FIRESTORE_DB_ID)

# --- RAG Helper Function (Corrected with Explicit Region) ---
# --- RAG Helper Function (FINAL, CORRECTED VERSION) ---
def get_rag_context(query: str, max_length: int = 3000) -> str:
    """
    Gets relevant context from the knowledge base.
    This version correctly parses the RAG API response and limits the context
    to the top 2 most relevant chunks to manage length and quality.
    """
    try:
        if not query or len(query.strip()) < 10:
            logger.info(f"RAG query ('{query}') is too short, skipping.")
            return ""

        from vertexai import rag
        corpus_name = None
        try:
            corpora = rag.list_corpora()
            for corpus in corpora:
                if corpus.display_name == RAG_CORPUS_NAME:
                    corpus_name = corpus.name
                    break
        except Exception as e:
            logger.warning(f"Failed to list RAG corpora. Ensure the service is active in '{GCP_LOCATION}'. Error: {e}")
            return ""

        if not corpus_name:
            logger.warning(f"RAG corpus '{RAG_CORPUS_NAME}' not found in project.")
            return ""

        logger.info(f"RAG searching for: '{query}'")

        # The API returns a list of Context objects.
        response = rag.retrieval_query(
            rag_resources=[rag.RagResource(rag_corpus=corpus_name)],
            text=query.strip()
        )

        raw_contexts = []
        if hasattr(response, 'contexts'):
            context_list_or_obj = getattr(response, 'contexts')
            if hasattr(context_list_or_obj, 'contexts'):
                raw_contexts = context_list_or_obj.contexts
            else:
                raw_contexts = context_list_or_obj

        if not raw_contexts:
            logger.info("No relevant RAG context found.")
            return ""

        # Limit to the top 2 most relevant chunks to avoid context overload and keep the information targeted.
        top_contexts = [
            context.text.strip() for context in raw_contexts[:2] if hasattr(context, 'text') and context.text
        ]

        if top_contexts:
            combined_context = "\n\n---\n\n".join(top_contexts)
            logger.info(f"RAG found {len(raw_contexts)} contexts, using top {len(top_contexts)} for the final prompt (length: {len(combined_context)} chars).")
            return combined_context
        else:
            logger.info("No text could be extracted from the found RAG contexts.")
            return ""

    except Exception as e:
        logger.error(f"RAG search failed unexpectedly: {e}", exc_info=True)
        return ""


# --- Retry Helper for LLM Calls ---
def _retry_llm_call(func, max_retries: int = 2, delay: float = 1.0):
    """Retry LLM calls with exponential backoff for production reliability."""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"LLM call failed after {max_retries} attempts: {e}")
                raise
            logger.warning(f"LLM call failed (attempt {attempt + 1}/{max_retries}): {e}")
            time.sleep(delay * (2 ** attempt))

# --- Query Generation Helper Function ---
def generate_search_queries_from_email(email_subject: str, email_body: str, attachments: List[Part]) -> List[str]:
    logger.info("Generating multi-modal search queries using Gemini 2.0 Lite...")
    try:
        # Using the powerful Pro model for this complex multi-modal reasoning task with temperature for creativity
        from vertexai.generative_models import GenerationConfig
        
        model = GenerativeModel(
            GEMINI_MODEL,
            generation_config=GenerationConfig(
                temperature=0.8,  # Higher creativity for comprehensive query generation
                top_p=0.9,
                top_k=40,
                max_output_tokens=1024
            )
        )

        prompt = f"""<ROLE_AND_GOAL>
You are an expert query analysis engine. Your goal is to identify what the USER is specifically asking about and generate targeted search queries to answer their questions.
</ROLE_AND_GOAL>
<INSTRUCTIONS>
Follow these steps in order:
1.  **Identify USER QUESTIONS:** Read the email subject and body. What is the user specifically asking about? Focus on their explicit questions and requests.
2.  **Analyze Attachments for USER CONTEXT:** If attachments are provided, analyze them ONLY to understand what the user is asking about - not to extract all possible topics from the attachments.
3.  **Generate TARGETED QUERIES:** Create search queries that directly address the user's questions. Prioritize user needs over document content.
4.  **Limit and Prioritize:** Generate maximum 8 queries, ranked by relevance to user's actual questions.
5.  **Output JSON:** Your final output MUST be ONLY a valid JSON object: {{"queries": ["most_relevant_query", "second_most_relevant", ...]}}

IMPORTANT: Focus on what the USER wants to know, not everything that might be in the attachments.
</INSTRUCTIONS>
<EXAMPLE>
---
EMAIL SUBJECT: "Problém s objednávkou a dotaz"
EMAIL BODY: "Dobrý den, chci se zeptat na cenu AlzaConnect USB-C Hub. Jak ho můžu vrátit?"
ATTACHMENT: [Audio file where user says: "Mám problém s Alza, jak můžu vyřešit spor?"]
---
YOUR JSON RESPONSE:
{{"queries": ["cena produktu AlzaConnect USB-C Hub", "proces vrácení zboží", "řešení sporů s Alza"]}}
</EXAMPLE>
<CURRENT_TASK>
---
EMAIL SUBJECT: "{email_subject}"
EMAIL BODY: "{email_body}"
ATTACHMENT: [The user has attached one or more files. Analyze their content.]
---
YOUR JSON RESPONSE:
"""
        all_parts = [prompt] + attachments

        # Use retry logic for LLM call reliability
        def _generate_queries():
            return model.generate_content(all_parts)

        response = _retry_llm_call(_generate_queries)
        raw_text = response.text.strip()
        logger.info(f"Raw query generation response: {raw_text}")

        # Extract JSON from response, handling extra text
        cleaned_response_text = raw_text.replace("```json", "").replace("```", "")

        # Find JSON object in the text
        start_idx = cleaned_response_text.find('{')
        if start_idx == -1:
            logger.warning("No opening brace found in response")
            return []

        # Find the matching closing brace by counting braces
        brace_count = 0
        end_idx = -1
        for i in range(start_idx, len(cleaned_response_text)):
            if cleaned_response_text[i] == '{':
                brace_count += 1
            elif cleaned_response_text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i + 1
                    break

        if start_idx != -1 and end_idx > start_idx:
            json_text = cleaned_response_text[start_idx:end_idx]
            logger.info(f"Extracted JSON: {json_text}")
            parsed_json = json.loads(json_text)
        else:
            logger.warning("No valid JSON found in response")
            return []
        queries = parsed_json.get("queries", [])
        if not isinstance(queries, list):
            logger.warning("Model returned JSON but 'queries' key is not a list.")
            return []
        
        # Limit queries to prevent quota exhaustion (prioritized by model ranking)
        max_queries = 8
        if len(queries) > max_queries:
            logger.info(f"Limiting {len(queries)} queries to top {max_queries} most relevant")
            queries = queries[:max_queries]
        
        logger.info(f"Successfully generated {len(queries)} search queries: {queries}")
        return queries
    except Exception as e:
        logger.error(f"An unexpected error occurred during multi-modal query generation: {e}", exc_info=True)
        return []


# --- Other Helper Functions ---
def _get_email_details(payload: dict) -> Tuple[str, dict]:
    headers = {h['name'].lower(): h['value'] for h in payload.get('headers', [])}
    body = ""
    parts = payload.get('parts', [])
    
    logger.info(f"Email parsing - has parts: {len(parts)}, payload keys: {list(payload.keys())}")
    
    if parts:
        logger.info(f"Processing {len(parts)} email parts")
        for i, part in enumerate(parts):
            part_mime = part.get('mimeType', 'unknown')
            part_size = part.get('body', {}).get('size', 0)
            logger.info(f"  Part {i}: mimeType={part_mime}, size={part_size}")
            
            if part_mime == 'text/plain':
                body_data = part['body'].get('data', '')
                if body_data: 
                    body = base64.urlsafe_b64decode(body_data.encode('ASCII')).decode('utf-8')
                    logger.info(f"Extracted email body from part {i}: {len(body)} chars")
                    break
            elif part_mime == 'multipart/mixed' or part_mime == 'multipart/alternative':
                # Handle nested multipart
                nested_parts = part.get('parts', [])
                logger.info(f"  Found nested multipart with {len(nested_parts)} sub-parts")
                for j, nested_part in enumerate(nested_parts):
                    nested_mime = nested_part.get('mimeType', 'unknown')
                    logger.info(f"    Nested part {j}: mimeType={nested_mime}")
                    if nested_mime == 'text/plain':
                        nested_body_data = nested_part['body'].get('data', '')
                        if nested_body_data:
                            body = base64.urlsafe_b64decode(nested_body_data.encode('ASCII')).decode('utf-8')
                            logger.info(f"Extracted email body from nested part {j}: {len(body)} chars")
                            break
                if body:
                    break
    elif 'body' in payload and payload['body'].get('data'):
        body = base64.urlsafe_b64decode(payload['body']['data'].encode('ASCII')).decode('utf-8')
        logger.info(f"Extracted email body from main payload: {len(body)} chars")
    
    logger.info(f"Final email body length: {len(body)} chars")
    if body:
        # Log first 200 chars of body for debugging (be careful with sensitive data)
        preview = body[:200].replace('\n', '\\n').replace('\r', '\\r')
        logger.info(f"Email body preview: '{preview}{'...' if len(body) > 200 else ''}'")
    else:
        logger.warning("❌ NO EMAIL BODY EXTRACTED!")
    
    return body, headers

def _validate_attachment_security(filename: str, mime_type: str, size: int) -> tuple[bool, str]:
    """Validate attachment meets security requirements for Alza production."""
    MAX_SIZES = {
        'image': 10 * 1024 * 1024,  # 10MB for images
        'audio': 50 * 1024 * 1024,  # 50MB for audio
        'application': 25 * 1024 * 1024  # 25MB for PDFs
    }

    ALLOWED_TYPES = {
        'image/jpeg', 'image/jpg', 'image/png',
        'audio/mp3', 'audio/wav', 'audio/mpeg', 'audio/m4a',
        'application/pdf'
    }

    # Security checks
    if mime_type not in ALLOWED_TYPES:
        return False, f"Unsupported file type: {mime_type}"

    category = mime_type.split('/')[0]
    max_size = MAX_SIZES.get(category, 5 * 1024 * 1024)
    if size > max_size:
        return False, f"File too large: {size} bytes (max: {max_size})"

    # Path traversal protection
    if not filename or '..' in filename or '/' in filename or '\\' in filename:
        return False, "Invalid filename"

    return True, ""

def _process_attachments(gmail_service, message_id: str, parts: List[dict]) -> List[Part]:
    """Process email attachments with security validation for Alza production."""
    gemini_parts = []
    if not parts: return gemini_parts

    for part in parts:
        filename = part.get('filename')
        if not filename: continue

        try:
            mime_type = part.get('mimeType', 'unknown')
            size = part.get('body', {}).get('size', 0)

            # SECURITY: Validate attachment before processing
            is_valid, error_msg = _validate_attachment_security(filename, mime_type, size)
            if not is_valid:
                logger.warning(f"SECURITY: Blocked attachment {filename}: {error_msg}")
                continue

            attachment_id = part['body']['attachmentId']
            attachment_data = gmail_service.users().messages().attachments().get(userId='me', messageId=message_id, id=attachment_id).execute()['data']
            file_data = base64.urlsafe_b64decode(attachment_data.encode('UTF-8'))

            if mime_type.startswith('audio/'):
                # Sanitize filename for GCS
                safe_filename = re.sub(r'[^a-zA-Z0-9._-]', '_', os.path.basename(filename))
                blob_name = f"audio_attachments/{message_id}_{safe_filename}"
                bucket = storage_client.bucket(GCS_BUCKET_NAME)
                blob = bucket.blob(blob_name)
                blob.upload_from_string(file_data, content_type=mime_type)
                gcs_uri = f"gs://{GCS_BUCKET_NAME}/{blob_name}"
                gemini_parts.append(Part.from_uri(gcs_uri, mime_type=mime_type))
                logger.info(f"Processed audio attachment: {filename} ({size} bytes)")
            else:
                gemini_parts.append(Part.from_data(file_data, mime_type=mime_type))
                logger.info(f"Processed inline attachment: {filename} ({size} bytes)")

        except Exception as e:
            logger.error(f"Failed to process attachment {filename}: {e}")
    return gemini_parts

def _send_reply(gmail_service, headers: dict, thread_id: str, reply_body: str):
    try:
        original_subject = headers.get('subject', '(no subject)')
        original_from = headers.get('from')
        original_message_id = headers.get('message-id')
        reply_subject = f"Re: {original_subject}" if not original_subject.lower().startswith("re:") else original_subject
        message = MIMEMultipart()
        message['to'] = original_from
        message['from'] = BOT_EMAIL
        message['subject'] = reply_subject
        message['In-Reply-To'] = original_message_id
        message['References'] = original_message_id
        message.attach(MIMEText(reply_body, 'plain', 'utf-8'))
        encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
        create_message = {'raw': encoded_message, 'threadId': thread_id}
        gmail_service.users().messages().send(userId='me', body=create_message).execute()
        logger.info(f"Successfully sent reply to thread {thread_id}")
    except HttpError as error:
        logger.error(f"An error occurred while sending email: {error}")
        raise

def _get_last_history_id_from_firestore() -> Optional[str]:
    doc_ref = firestore_client.collection(STATE_COLLECTION).document(STATE_DOCUMENT)
    doc = doc_ref.get()
    return doc.to_dict().get('historyId') if doc.exists else None

def _save_last_history_id_to_firestore(history_id: str):
    doc_ref = firestore_client.collection(STATE_COLLECTION).document(STATE_DOCUMENT)
    doc_ref.set({'historyId': str(history_id), 'updated_at': firestore.SERVER_TIMESTAMP})
    logger.info(f"Saved new state to Firestore. Last historyId is now: {history_id}")

def _get_gmail_credentials() -> Credentials:
    token_secret_name = f"projects/{GCP_PROJECT}/secrets/gmail-token/versions/latest"
    token_response = secret_manager_client.access_secret_version(request={"name": token_secret_name})
    token_info = json.loads(token_response.payload.data.decode("UTF-8"))
    creds_secret_name = f"projects/{GCP_PROJECT}/secrets/gmail-credentials/versions/latest"
    creds_response = secret_manager_client.access_secret_version(request={"name": creds_secret_name})
    client_config = json.loads(creds_response.payload.data.decode("UTF-8"))['installed']
    return Credentials(token=token_info.get('token'), refresh_token=token_info.get('refresh_token'), token_uri=client_config.get('token_uri'), client_id=client_config.get('client_id'), client_secret=client_config.get('client_secret'), scopes=token_info.get('scopes'))

def _claim_email_for_processing(message_id: str) -> bool:
    doc_ref = firestore_client.collection('processed_emails').document(message_id)
    try:
        doc_ref.create({'claimed_at': firestore.SERVER_TIMESTAMP, 'status': 'processing'})
        logger.info(f"Successfully claimed email {message_id} for processing.")
        return True
    except AlreadyExists:
        logger.warning(f"Email {message_id} already claimed or processed. Skipping.")
        return False

def _mark_email_as_processed(message_id: str, status: str):
    doc_ref = firestore_client.collection('processed_emails').document(message_id)
    doc_ref.update({'processed_at': firestore.SERVER_TIMESTAMP, 'status': status})
    logger.info(f"Marked email {message_id} as '{status}'.")


# --- Main Function (Corrected) ---
@functions_framework.http
def process_email_http(request: Request):
    """Statefully processes new emails using a multi-query RAG approach."""
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
            try:
                if not _claim_email_for_processing(message_id): continue

                message = gmail_service.users().messages().get(userId='me', id=message_id, format='full').execute()
                payload = message.get('payload', {})
                email_body, headers = _get_email_details(payload)
                sender_email = re.search(r'<(.*?)>', headers.get('from', ''))
                sender_email = sender_email.group(1) if sender_email else headers.get('from', '')
                if sender_email == BOT_EMAIL:
                    logger.info(f"Ignoring self-sent message {message_id}.")
                    _mark_email_as_processed(message_id, status="ignored_self_sent")
                    continue

                thread_id = message.get('threadId')
                email_subject = headers.get('subject', '')
                logger.info(f"Processing email from '{sender_email}' with subject: '{email_subject}'")

                attachment_parts = _process_attachments(gmail_service, message_id, payload.get('parts', []))

                # Debug logging for query generation inputs
                logger.info(f"Query generation inputs - Subject: '{email_subject}', Body length: {len(email_body)}, Attachments: {len(attachment_parts)}")
                if email_body:
                    body_preview = email_body[:200].replace('\n', '\\n').replace('\r', '\\r')
                    logger.info(f"Email body for query generation: '{body_preview}{'...' if len(email_body) > 200 else ''}'")
                else:
                    logger.warning("❌ EMPTY EMAIL BODY being sent to query generation!")
                
                # ## FIX ## Pass the attachments to the query generation function to solve the TypeError.
                search_queries = generate_search_queries_from_email(email_subject, email_body, attachment_parts)

                consolidated_rag_context = ""
                if search_queries:
                    all_rag_contexts = [get_rag_context(query) for query in search_queries if query]
                    unique_contexts = {context for context in all_rag_contexts if context}
                    consolidated_rag_context = "\n\n---\n\n".join(unique_contexts)

                attachment_summary_text = "\n".join([f"- Soubor: '{part.get('filename')}'" for part in payload.get('parts', []) if part.get('filename')]) or "Žádné přílohy."

                prompt_template = """<ROLE_AND_GOAL>
Jsi "AlzaBot", špičkový AI asistent zákaznické podpory pro Alza.cz. Tvoje osobnost je profesionální, přátelská, a především extrémně nápomocná a efektivní. Tvým cílem je vyřešit dotaz zákazníka v jediné odpovědi, pokud je to možné. Odpovídej VŽDY v jazyce, ve kterém je napsán původní email zákazníka. Pokud je email v angličtině, odpověz anglicky. Pokud je v češtině, odpověz česky.
</ROLE_AND_GOAL>
<CONTEXT>
<customer_email>
<subject>{subject}</subject>
<body>
{email_body}
</body>
</customer_email>
<attachment_summary>
(Toto je seznam souborů, které zákazník poslal. Jejich obsah je přiložen k této zprávě pro tvou analýzu.)
{attachment_summary}
</attachment_summary>
<knowledge_base_search_results>
(Toto jsou relevantní informace z interní databáze Alza.cz, které se vztahují k otázkám zákazníka. Tyto informace mají absolutní přednost a musíš je použít k zodpovězení otázek týkajících se produktů, služeb nebo specifikací.)
{knowledge_base_search_results}
</knowledge_base_search_results>
</CONTEXT>
<RULES>
1.  **Priorita Znalostí:** Pokud sekce `<knowledge_base_search_results>` obsahuje odpověď, použij VÝHRADNĚ tyto informace. Nikdy si nevymýšlej specifikace produktů, firemní pravidla a postupy (například reklamační řád, podmínky pro vrácení zboží, záruční podmínky nebo obchodní podmínky), ceny ani skladovou dostupnost.
2.  **Kompletnost:** Odpověz na VŠECHNY otázky zákazníka, a to jak z těla emailu, tak z přiložených souborů (PDF, audio).
3.  **Tón Hlasu:** Vždy udržuj profesionální a přátelský tón Alza.cz. Buď empatický, pokud jde o stížnost.
4.  **Formát Odpovědi:** Generuj POUZE text samotného emailu. Nepřidávej žádné úvodní fráze. Začni přímo odpovědí.
</RULES>
<TASK>
Nyní, na základě VŠECH informací v sekci <CONTEXT> a dodržení VŠECH <RULES>, zformuluj finální, kompletní a nápomocnou odpověď, která bude odeslána zákazníkovi.
</TASK>"""
                final_prompt = prompt_template.format(
                    subject=email_subject,
                    email_body=email_body,
                    attachment_summary=attachment_summary_text,
                    knowledge_base_search_results=consolidated_rag_context or "Pro dotazy v emailu nebyly v databázi nalezeny žádné specifické informace."
                )

                # Log what we're sending to final Gemini call
                logger.info(f"Final prompt length: {len(final_prompt)} chars")
                logger.info(f"RAG context length: {len(consolidated_rag_context)} chars")
                logger.info(f"Email body in final prompt: {'YES' if email_body and email_body in final_prompt else 'NO'}")
                if consolidated_rag_context:
                    context_preview = consolidated_rag_context[:300].replace('\n', '\\n')
                    logger.info(f"RAG context preview: {context_preview}...")
                else:
                    logger.warning("❌ NO RAG CONTEXT for final response!")

                # Use the fast model for the final reply generation with retry logic and temperature
                from vertexai.generative_models import GenerationConfig
                
                reply_model = GenerativeModel(
                    GEMINI_MODEL_F,
                    generation_config=GenerationConfig(
                        temperature=0.7,  # Add creativity while maintaining accuracy
                        top_p=0.8,
                        top_k=40,
                        max_output_tokens=4096
                    )
                )

                def _generate_reply():
                    return reply_model.generate_content([final_prompt] + attachment_parts)

                response = _retry_llm_call(_generate_reply)
                reply_body = response.text

                logger.info(f"Generated reply: {reply_body[:200].replace(chr(10), ' ')}...")
                _send_reply(gmail_service, headers, thread_id, reply_body)
                _mark_email_as_processed(message_id, status="replied")
                gmail_service.users().messages().modify(userId='me', id=message_id, body={'removeLabelIds': ['UNREAD']}).execute()
                logger.info(f"Marked message {message_id} as read.")
            except Exception as e:
                logger.error(f"Error processing message {message_id}: {e}", exc_info=True)
                _mark_email_as_processed(message_id, status="failed")
                continue
        _save_last_history_id_to_firestore(new_history_id)
    except Exception as e:
        logger.error(f"Critical error in function execution: {e}", exc_info=True)
        return "Error", 500
    return "OK", 200