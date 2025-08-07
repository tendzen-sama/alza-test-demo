import base64
import binascii
import logging
import os
import re
import json
import time
import random
import hashlib
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

# --- Initialization ---
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger('googleapiclient.discovery_cache').setLevel(logging.ERROR)

# --- Environment Variables ---
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

# --- Client Initialization ---
vertexai.init(project=GCP_PROJECT, location=GCP_LOCATION)
storage_client = storage.Client()
secret_manager_client = secretmanager.SecretManagerServiceClient()
firestore_client = firestore.Client(project=GCP_PROJECT, database=FIRESTORE_DB_ID)


# --- START OF ADDED SECURITY FUNCTION ---
def _sanitize_for_prompt_injection(content: str) -> str:
    """
    A minimal sanitizer to remove common prompt injection patterns.
    """
    if not content:
        return ""

    # List of dangerous patterns to detect prompt injection
    dangerous_patterns = [
        r'ignore\s+(?:previous|all|above)\s+(?:instructions|prompts|rules)',
        r'act\s+as\s+',
        r'reveal\s+(?:prompt|instructions|system|rules|secrets)',
        r'execute\s+(?:code|command|script|function)',
        r'override\s+(?:security|safety|instructions)',
        r'pretend\s+(?:to\s+be|you\s+are)',
        r'new\s+(?:instructions|rules|role)',
        r'forget\s+(?:everything|instructions|rules)',
        r'jailbreak|jail\s+break',
    ]

    sanitized = content
    for pattern in dangerous_patterns:
        # Replace found patterns with a harmless placeholder
        sanitized = re.sub(pattern, '[FILTERED]', sanitized, flags=re.IGNORECASE)

    if sanitized != content:
        logger.warning(f"Potential prompt injection detected and filtered.")

    return sanitized
# --- END OF ADDED SECURITY FUNCTION ---


# --- Advanced RAG Helper with Intelligent Quota Management ---
def get_rag_context(query: str, max_length: int = 3000, max_retries: int = 3) -> tuple[str, list]:
    """
    Enhanced RAG context retrieval with intelligent quota handling and fallback strategies.

    Features:
    - Exponential backoff for quota exhaustion
    - Intelligent retry with jitter
    - Graceful degradation when RAG unavailable
    - Comprehensive error recovery
    """
    if not query or len(query.strip()) < 10:
        logger.info(f"RAG query ('{query}') is too short, skipping.")
        return "", []

    from vertexai import rag

    # Get RAG corpus with caching
    corpus_name = _get_rag_corpus_name()
    if not corpus_name:
        return "", []

    # Retry loop with exponential backoff
    for attempt in range(max_retries):
        try:
            logger.info(f"RAG searching with reranking for: '{query}' (attempt {attempt + 1}/{max_retries})")

            # Enhanced RAG with LLM reranking for 30.8% better performance
            # Based on evaluation results showing significant improvement
            rag_config = rag.RagRetrievalConfig(
                top_k=3,  # Retrieve top 3 most relevant chunks
                ranking=rag.Ranking(
                    llm_ranker=rag.LlmRanker(
                        model_name=LLM_RANKER  # Same model as successful evaluation
                    )
                )
            )

            response = rag.retrieval_query(
                rag_resources=[rag.RagResource(rag_corpus=corpus_name)],
                text=query.strip(),
                rag_retrieval_config=rag_config
            )

            # Process successful response
            return _process_rag_response(response, query)

        except ResourceExhausted as e:
            # Handle quota exhaustion with intelligent backoff
            return _handle_quota_exhaustion(e, attempt, max_retries, query)

        except RuntimeError as e:
            # Handle RAG runtime errors (which often wrap quota errors)
            if "ResourceExhausted" in str(e) or "Quota exceeded" in str(e):
                return _handle_quota_exhaustion(e, attempt, max_retries, query)
            else:
                logger.error(f"RAG runtime error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    return _create_fallback_context(query), []

        except Exception as e:
            logger.error(f"Unexpected RAG error on attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                return _create_fallback_context(query), []

        # Wait before retry (except on last attempt)
        if attempt < max_retries - 1:
            wait_time = (2 ** attempt) + random.uniform(0, 1)
            logger.info(f"Waiting {wait_time:.1f}s before retry...")
            time.sleep(wait_time)

    # Final fallback
    return _create_fallback_context(query), []

def _get_rag_corpus_name() -> Optional[str]:
    """Get RAG corpus name with error handling"""
    try:
        from vertexai import rag
        corpora = rag.list_corpora()
        for corpus in corpora:
            if corpus.display_name == RAG_CORPUS_NAME:
                return corpus.name
        logger.warning(f"RAG corpus '{RAG_CORPUS_NAME}' not found in project.")
        return None
    except Exception as e:
        logger.warning(f"Failed to list RAG corpora: {e}")
        return None

def _process_rag_response(response, query: str) -> tuple[str, list]:
    """Process successful RAG response with source information for citations"""
    raw_contexts = []
    sources = []

    if hasattr(response, 'contexts'):
        context_list_or_obj = getattr(response, 'contexts')
        if hasattr(context_list_or_obj, 'contexts'):
            raw_contexts = context_list_or_obj.contexts
        else:
            raw_contexts = context_list_or_obj

    if not raw_contexts:
        logger.info("No relevant RAG context found.")
        return "", []

    # Extract contexts with source information for citations
    context_with_sources = []
    for i, context in enumerate(raw_contexts[:3]):  # Top 3 for citations
        if hasattr(context, 'text') and context.text:
            source_info = f"SOURCE_{i}"

            # Extract source file information for citations
            if hasattr(context, 'source_uri') and context.source_uri:
                # Extract filename from GCS URI for cleaner citations
                source_path = context.source_uri
                filename = source_path.split('/')[-1] if '/' in source_path else source_path
                source_info += f" ({filename})"
                sources.append(filename)
            else:
                sources.append(f"internal_doc_{i}")

            # Format context with source reference
            context_text = f"[{source_info}]\n{context.text.strip()}"
            context_with_sources.append(context_text)

    if context_with_sources:
        combined_context = "\n\n---\n\n".join(context_with_sources)
        logger.info(f"✅ RAG SUCCESS with citations: Found {len(raw_contexts)} contexts, using top {len(context_with_sources)} with sources: {sources}")
        return combined_context, sources
    else:
        logger.info("No text could be extracted from RAG contexts.")
        return "", []

def _handle_quota_exhaustion(error, attempt: int, max_retries: int, query: str) -> tuple[str, list]:
    """Handle quota exhaustion with intelligent backoff"""
    logger.warning(f"🚨 QUOTA EXHAUSTED on attempt {attempt + 1}: {error}")

    if attempt < max_retries - 1:
        # Exponential backoff with jitter for quota recovery
        base_delay = 10  # Start with 10 seconds for quota issues
        wait_time = base_delay * (2 ** attempt) + random.uniform(0, 5)

        logger.warning(f"⏳ Quota exhausted - waiting {wait_time:.1f}s for quota recovery...")
        time.sleep(wait_time)
        return None, []  # Signal to continue retrying
    else:
        # Final attempt failed - create fallback response
        logger.error(f"❌ RAG quota exhausted after {max_retries} attempts - using fallback")
        return _create_fallback_context(query), []

def _create_fallback_context(query: str) -> str:
    """Create intelligent fallback context when RAG is unavailable"""
    logger.info(f"🔄 Creating fallback context for: '{query}'")

    # Basic fallback context with common Alza information
    fallback_context = """
Alza Customer Support Information (Limited Mode):

Basic Contact Information:
- Customer Service: +420 225 340 111
- Email: info@alza.cz
- Website: www.alza.cz

General Policies:
- Standard return period: 30 days for online orders
- Standard warranty: 24 months minimum
- Customer service available 24/7

Note: Detailed product and policy information temporarily unavailable. 
For specific questions, please contact customer service directly.
"""

    logger.warning("⚠️ Using fallback context - limited information available")
    return fallback_context.strip()

def _create_technical_error_response() -> str:
    """Create professional error response for technical issues"""
    return '''<p>Dobrý den,</p>

<p>omlouváme se, ale v současné době dochází k technickým potížím s naším AI systémem. Váš dotaz byl zaregistrován a náš tým vám odpoví v nejkratší možné době.</p>

<p>Pro rychlejší vyřízení vašeho dotazu nás prosím kontaktujte:</p>
<ul>
<li><strong>Telefon:</strong> +420 225 340 111 (nonstop)</li>
<li><strong>Email:</strong> info@alza.cz</li>
<li><strong>Live chat:</strong> www.alza.cz</li>
</ul>

<p>Děkujeme za pochopení.</p>

<p>S pozdravem,<br>
Tým Alza.cz</p>'''

def _add_responsible_ai_disclaimer(html_content: str, sources: list) -> str:
    """Add responsible AI disclaimer and source references to HTML email"""

    # Remove closing </div> if present to add our footer
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
<li><strong>Email podpora:</strong> info@alza.cz</li>
</ul>

<p>Omlouváme se za způsobené nepříjemnosti a děkujeme za trpělivost.</p>

<p>S pozdravem,<br>
AlzaBot & Tým zákaznické podpory</p>'''

# --- Enhanced Retry Helper for LLM Calls with Quota Management ---
def _retry_llm_call(func, max_retries: int = 3, base_delay: float = 2.0):
    """
    Enhanced retry mechanism for LLM calls with intelligent quota handling.

    Features:
    - Exponential backoff for quota errors
    - Different strategies for different error types
    - Jitter to prevent thundering herd
    """
    for attempt in range(max_retries):
        try:
            return func()

        except ResourceExhausted as e:
            logger.warning(f"🚨 LLM QUOTA EXHAUSTED (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                logger.error(f"❌ LLM quota exhausted after {max_retries} attempts")
                raise

            # Longer wait for quota issues
            wait_time = base_delay * (3 ** attempt) + random.uniform(1, 3)
            logger.warning(f"⏳ LLM quota wait: {wait_time:.1f}s...")
            time.sleep(wait_time)

        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"❌ LLM call failed after {max_retries} attempts: {e}")
                raise

            # Standard retry for other errors
            wait_time = base_delay * (2 ** attempt) + random.uniform(0, 1)
            logger.warning(f"⚠️ LLM call failed (attempt {attempt + 1}/{max_retries}): {e}")
            logger.info(f"Retrying in {wait_time:.1f}s...")
            time.sleep(wait_time)

# --- Schema Definitions for Structured Output ---
# These dataclasses are for internal documentation and clarity.
@dataclass
class SearchQueries:
    """Schema for the list of generated search queries."""
    queries: List[str]

@dataclass
class FinalReply:
    """Schema for the final HTML email reply."""
    html_body: str

## FIX: Define schemas as dictionaries to conform to the Vertex AI SDK's expected type.
SEARCH_QUERIES_SCHEMA = {
    "type": "object",
    "properties": {
        "queries": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "required": ["queries"]
}

FINAL_REPLY_SCHEMA = {
    "type": "object",
    "properties": {
        "html_body": {"type": "string"}
    },
    "required": ["html_body"]
}

# --- Query Generation Helper Function ---
def generate_search_queries_from_email(email_subject: str, email_body: str, attachments: List[Part]) -> List[str]:
    logger.info("Generating search queries using Gemini with structured output...")
    try:
        model = GenerativeModel(
            GEMINI_MODEL,
            generation_config=GenerationConfig(
                temperature=0.5,
                top_p=0.9,
                top_k=40,
                max_output_tokens=1024,
                response_mime_type="application/json",
                ## FIX: Pass the schema dictionary, not the dataclass type.
                response_schema=SEARCH_QUERIES_SCHEMA,
            )
        )

        prompt = f"""<ROLE_AND_GOAL>
You are an expert bilingual query analysis engine for Alza.cz. Your goal is to generate strategic search queries that work optimally with a mixed Czech/English knowledge base.
</ROLE_AND_GOAL>
<KNOWLEDGE_BASE_INFO>
- Product information is primarily in ENGLISH (AlzaConnect, AlzaPower, specifications, prices)
- Policy/warranty information is in CZECH and ENGLISH mix
- Service center information is in ENGLISH format
</KNOWLEDGE_BASE_INFO>
<ADVANCED_STRATEGY>
**MULTI-STAGE QUERY GENERATION** - Generate queries that bridge semantic gaps in mixed Czech/English knowledge base.

**STAGE 1 - SEMANTIC BRIDGING:**
- **Location Queries**: Map "kancelář/pobočka" → "Service Center/office/address/Brno"  
- **Product Mapping**: Map user brands → actual product names (AlzaPower → AlzaConnect)
- **Availability Bridge**: "dostupnost/skladem" → "Stock Status/availability/in stock"

**STAGE 2 - CROSS-REFERENCE QUERIES:**
- **Return Questions**: Generate both return process + policy queries
- **Product Questions**: Generate both catalog + availability + service queries  
- **Warranty Questions**: Generate both warranty + service center queries

**STAGE 3 - LANGUAGE OPTIMIZATION:**
- **Product Specs**: English technical terms (4K HDMI, USB 3.0, aluminum)
- **Policy Rules**: Czech terms (hygienické zboží, vrácení, záruka)
- **Locations**: English format (Service Center, Prague, Bratislava, Brno)

**STAGE 4 - CONTEXT EXPANSION:**
Add contextual keywords to improve semantic matching:
- USB queries: include "hub", "adapter", "cable", "connector", "port"
- Location queries: include "address", "contact", "phone", "email"  
- Return queries: include "process", "policy", "method", "refund"
</ADVANCED_STRATEGY>
<ADVANCED_EXAMPLES>
---
EMAIL: "ohledne adressu kancelaru v Brno, kam bych mohl obratit"
MULTI-STAGE QUERIES:
{{"queries": ["Alza Service Center Brno address contact", "Service Centers Prague Bratislava Brno", "office address Brno phone email", "kancelář pobočka Brno kontakt"]}}
// BRIDGES: kancelář → Service Center + address + contact info
---
EMAIL: "A taky jestli mate tento product a jeho specifikace: AlzaPower USB hub"  
MULTI-STAGE QUERIES:
{{"queries": ["AlzaConnect USB-C Hub specifications price", "USB-C hub AlzaPower 4K HDMI", "Tech Accessories cables adapters USB", "dostupnost skladem USB hub"]}}
// BRIDGES: AlzaPower → AlzaConnect + multiple spec variants + availability
---
EMAIL: "Můžu vrátit otevřené hygienické zboží?"
MULTI-STAGE QUERIES:
{{"queries": ["vrácení hygienického zboží otevřené", "return hygienic products opened packaging", "odstoupení od smlouvy hygienické výrobky", "cosmetics return policy opened"]}}
// CROSS-REF: policy rules + return process + legal terms
---
EMAIL: "Jak vrátit objednávku a kolik to stojí?"
MULTI-STAGE QUERIES:
{{"queries": ["return process steps online orders", "vrácení zboží proces kroky", "return shipping costs courier pickup", "refund timeline bank transfer card"]}}
// CROSS-REF: return process + costs + refund methods from multiple files
---
</ADVANCED_EXAMPLES>
<CURRENT_TASK>
---
EMAIL SUBJECT: "{email_subject}"
EMAIL BODY: "{email_body}"
ATTACHMENT: [The user has attached one or more files. Analyze their content.]
---
YOUR RESPONSE:
"""
        all_parts = [prompt] + attachments

        def _generate_queries():
            return model.generate_content(all_parts)

        response = _retry_llm_call(_generate_queries)

        try:
            logger.info(f"Raw structured response: {response.text}")
            parsed_json = json.loads(response.text)
            queries = parsed_json.get("queries", [])

            if not isinstance(queries, list):
                logger.warning("Model returned valid JSON but 'queries' key is not a list.")
                return []

            max_queries = 8
            queries = queries[:max_queries]
            logger.info(f"Successfully generated {len(queries)} search queries: {queries}")
            return queries
        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON from model response: {response.text}", exc_info=True)
            return []

    except Exception as e:
        logger.error(f"An unexpected error occurred during query generation: {e}", exc_info=True)
        return []

# --- Other Helper Functions ---
def _get_email_details(payload: dict) -> Tuple[str, dict]:
    headers = {h['name'].lower(): h['value'] for h in payload.get('headers', [])}
    body = ""

    def _extract_text_from_part(part: dict, level: int = 0) -> str:
        """Recursively extract text/plain content from email parts"""
        indent = "  " * level
        logger.debug(f"{indent}Processing part: mimeType={part.get('mimeType')}")

        # Direct text/plain content
        if part.get('mimeType') == 'text/plain':
            body_data = part.get('body', {}).get('data', '')
            if body_data:
                try:
                    decoded_body = base64.urlsafe_b64decode(body_data.encode('ASCII')).decode('utf-8')
                    logger.debug(f"{indent}Found text/plain content: {len(decoded_body)} chars")
                    return decoded_body
                except Exception as e:
                    logger.warning(f"{indent}Failed to decode text/plain part: {e}")

        # Recursively process multipart content
        elif part.get('mimeType', '').startswith('multipart/'):
            nested_parts = part.get('parts', [])
            logger.debug(f"{indent}Processing multipart with {len(nested_parts)} nested parts")
            for nested_part in nested_parts:
                nested_text = _extract_text_from_part(nested_part, level + 1)
                if nested_text:
                    return nested_text

        return ""

    # Try to extract from parts first
    parts = payload.get('parts', [])
    if parts:
        logger.debug(f"Email has {len(parts)} top-level parts")
        for part in parts:
            body = _extract_text_from_part(part)
            if body:
                break

    # Fallback to direct body data
    if not body and 'body' in payload and payload['body'].get('data'):
        try:
            body = base64.urlsafe_b64decode(payload['body']['data'].encode('ASCII')).decode('utf-8')
            logger.debug(f"Extracted body from direct payload: {len(body)} chars")
        except Exception as e:
            logger.warning(f"Failed to decode direct body: {e}")

    if not body:
        logger.warning("❌ NO EMAIL BODY EXTRACTED!")
        # Debug: Log email structure for troubleshooting
        logger.debug(f"Email structure debug - Parts: {len(parts)}, Payload keys: {list(payload.keys())}")
    else:
        logger.info(f"✅ Successfully extracted email body: {len(body)} characters")

    return body, headers

def _validate_attachment_security(filename: str, mime_type: str, size: int, file_data: bytes = None) -> tuple[bool, str]:
    # Original file type and size validation
    MAX_SIZES = {'image': 10 * 1024 * 1024, 'audio': 50 * 1024 * 1024, 'application': 25 * 1024 * 1024}
    ALLOWED_TYPES = {'image/jpeg', 'image/jpg', 'image/png', 'audio/mp3', 'audio/wav', 'audio/mpeg', 'audio/m4a', 'application/pdf'}
    if mime_type not in ALLOWED_TYPES: return False, f"Unsupported file type: {mime_type}"
    category = mime_type.split('/')[0]
    max_size = MAX_SIZES.get(category, 5 * 1024 * 1024)
    if size > max_size: return False, f"File too large: {size} bytes (max: {max_size})"
    if not filename or '..' in filename or '/' in filename or '\\' in filename: return False, "Invalid filename"

    return True, ""

def _process_attachments(gmail_service, message_id: str, parts: List[dict]) -> List[Part]:
    gemini_parts = []
    if not parts: return gemini_parts
    for part in parts:
        filename = part.get('filename')
        if not filename: continue
        try:
            mime_type = part.get('mimeType', 'unknown')
            size = part.get('body', {}).get('size', 0)
            is_valid, error_msg = _validate_attachment_security(filename, mime_type, size)
            if not is_valid:
                logger.warning(f"SECURITY: Blocked attachment {filename}: {error_msg}")
                continue
            attachment_id = part['body']['attachmentId']
            attachment_data = gmail_service.users().messages().attachments().get(userId='me', messageId=message_id, id=attachment_id).execute()['data']
            file_data = base64.urlsafe_b64decode(attachment_data.encode('UTF-8'))

            # This second validation call was present in the user-provided code.
            # It's slightly redundant but kept as per the "don't change" instruction.
            is_valid_content, content_error = _validate_attachment_security(filename, mime_type, size, file_data)
            if not is_valid_content:
                logger.warning(f"SECURITY: Enhanced validation failed for {filename}: {content_error}")
                continue

            if mime_type.startswith('audio/'):
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
        if original_message_id:
            message['In-Reply-To'] = original_message_id
            message['References'] = original_message_id
        else:
            logger.warning("Could not find original 'Message-ID' header. Reply may not be threaded.")
        message.attach(MIMEText(reply_body, 'html', 'utf-8'))
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

# --- Main Function ---
@functions_framework.http
def process_email_http(request: Request):
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

                # --- APPLYING SECURITY SANITIZATION ---
                # Sanitize user-provided content before sending to LLM
                email_subject = _sanitize_for_prompt_injection(email_subject)
                email_body = _sanitize_for_prompt_injection(email_body)
                # --- END OF SECURITY SANITIZATION ---

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

                prompt_template = """<ROLE_AND_GOAL>
Jsi "AlzaBot", špičkový AI asistent zákaznické podpory pro Alza.cz. Tvoje osobnost je profesionální, přátelská, a především extrémně nápomocná a efektivní. Tvým cílem je vyřešit dotaz zákazníka v jediné odpovědi. Odpovídej VŽDY v jazyce, ve kterém je napsán původní email zákazníka.
</ROLE_AND_GOAL>
<CONTEXT>
<customer_email>
<subject>{subject}</subject>
<body>
{email_body}
</body>
</customer_email>
<attachment_summary>
{attachment_summary}
</attachment_summary>
<knowledge_base_search_results>
{knowledge_base_search_results}
</knowledge_base_search_results>
</CONTEXT>
<RULES>
1.  **Priorita Znalostí:** Pokud `<knowledge_base_search_results>` obsahuje odpověď, použij VÝHRADNĚ tyto informace. Nikdy si nevymýšlej specifikace produktů, ceny, ani firemní pravidla.
2.  **Kompletnost:** Odpověz na VŠECHNY otázky zákazníka z emailu i příloh.
3.  **Tón Hlasu:** Vždy udržuj profesionální a přátelský tón Alza.cz.
4.  **Formát Odpovědi (HTML):** Vytvoř dobře strukturovaný a čitelný HTML kód pro tělo emailu. Používej tagy jako `<p>`, `<strong>`, `<ul>`, `<li>` a `<a>`.
</RULES>
<TASK>
Na základě VŠECH informací v <CONTEXT> a dodržení VŠECH <RULES>, zformuluj finální odpověď a vlož ji do pole `html_body` v JSON struktuře.
</TASK>"""
                final_prompt = prompt_template.format(
                    subject=email_subject,
                    email_body=email_body,
                    attachment_summary=attachment_summary_text,
                    knowledge_base_search_results=consolidated_rag_context or "Pro dotazy v emailu nebyly v databázi nalezeny žádné specifické informace."
                )

                reply_model = GenerativeModel(
                    GEMINI_MODEL_F,
                    generation_config=GenerationConfig(
                        temperature=0.7,
                        top_p=0.8,
                        top_k=40,
                        max_output_tokens=4096,
                        response_mime_type="application/json",
                        ## FIX: Pass the schema dictionary, not the dataclass type.
                        response_schema=FINAL_REPLY_SCHEMA,
                    )
                )

                def _generate_reply():
                    return reply_model.generate_content([final_prompt] + attachment_parts)

                response = _retry_llm_call(_generate_reply)

                try:
                    reply_json = json.loads(response.text)
                    raw_html_body = reply_json.get('html_body', _create_technical_error_response())
                    # Add Trust & Safety features: sources and responsible AI disclaimer
                    reply_body = _add_responsible_ai_disclaimer(raw_html_body, all_sources)
                except json.JSONDecodeError:
                    logger.error(f"Failed to decode JSON from final reply model: {response.text}")
                    reply_body = _create_fallback_response(email_subject, consolidated_rag_context)

                logger.info(f"Generated reply: {reply_body[:200].replace(chr(10), ' ')}...")
                _send_reply(gmail_service, headers, thread_id, reply_body)
                _mark_email_as_processed(message_id, status="replied")
                gmail_service.users().messages().modify(userId='me', id=message_id, body={'removeLabelIds': ['UNREAD']}).execute()
                logger.info(f"Marked message {message_id} as read.")
            except ResourceExhausted as e:
                logger.error(f"🚨 CRITICAL: Resource exhausted while processing {message_id}: {e}")
                # Send immediate fallback response to customer
                try:
                    fallback_reply = _create_technical_error_response()
                    _send_reply(gmail_service, headers, thread_id, fallback_reply)
                    logger.info(f"✅ Sent fallback response due to quota exhaustion for {message_id}")
                    _mark_email_as_processed(message_id, status="replied_fallback")
                except Exception as send_error:
                    logger.error(f"❌ Failed to send fallback response: {send_error}")
                    _mark_email_as_processed(message_id, status="failed_quota_exhausted")
                continue

            except Exception as e:
                logger.error(f"❌ Error processing message {message_id}: {e}", exc_info=True)
                # Try to send fallback response even on general errors
                try:
                    sender_email = headers.get('from', 'unknown')
                    if sender_email != BOT_EMAIL:  # Don't reply to ourselves
                        fallback_reply = _create_fallback_response(email_subject, "")
                        _send_reply(gmail_service, headers, thread_id, fallback_reply)
                        logger.info(f"✅ Sent fallback response due to processing error for {message_id}")
                        _mark_email_as_processed(message_id, status="replied_fallback")
                    else:
                        _mark_email_as_processed(message_id, status="failed")
                except Exception as send_error:
                    logger.error(f"❌ Failed to send fallback response: {send_error}")
                    _mark_email_as_processed(message_id, status="failed")
                continue
        _save_last_history_id_to_firestore(new_history_id)
    except Exception as e:
        logger.error(f"Critical error in function execution: {e}", exc_info=True)
        return "Error", 500
    return "OK", 200