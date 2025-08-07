"""
Gmail API operations and email processing module.
"""
import base64
import logging
import os
import re
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Tuple, Dict
from googleapiclient.errors import HttpError
from vertexai.generative_models import Part

from .config import storage_client, GCS_BUCKET_NAME, BOT_EMAIL

logger = logging.getLogger(__name__)

def _get_email_details(payload: dict) -> Tuple[str, dict]:
    """Extract email body and headers from Gmail API payload."""
    headers = {h['name'].lower(): h['value'] for h in payload.get('headers', [])}
    body = ""

    def _extract_text_from_part(part: dict, level: int = 0) -> str:
        """Recursively extract text/plain content from email parts"""
        indent = "  " * level
        logger.debug(f"{indent}Processing part: mimeType={part.get('mimeType')}")
        
        if part.get('mimeType') == 'multipart/alternative' or part.get('mimeType') == 'multipart/mixed':
            logger.debug(f"{indent}Processing multipart with {len(part.get('parts', []))} nested parts")
            for nested_part in part.get('parts', []):
                result = _extract_text_from_part(nested_part, level + 1)
                if result:
                    return result
        
        elif part.get('mimeType') == 'text/plain':
            part_body = part.get('body', {})
            if 'data' in part_body:
                try:
                    decoded_body = base64.urlsafe_b64decode(part_body['data']).decode('utf-8')
                    logger.debug(f"{indent}Found text/plain content: {len(decoded_body)} chars")
                    return decoded_body
                except Exception as e:
                    logger.warning(f"Failed to decode text/plain part: {e}")
        
        return ""

    # Extract body from parts
    if payload.get('parts'):
        logger.debug(f"Email has {len(payload['parts'])} top-level parts")
        for part in payload['parts']:
            body = _extract_text_from_part(part)
            if body:
                break
    elif payload.get('body', {}).get('data'):
        # Handle single-part message
        try:
            body = base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8')
        except Exception as e:
            logger.warning(f"Failed to decode single-part message: {e}")

    if not body.strip():
        logger.warning("No email body content found")
    else:
        logger.info(f"✅ Successfully extracted email body: {len(body)} characters")

    return body, headers

def _process_attachments(gmail_service, message_id: str, parts: List[dict]) -> List[Part]:
    """Process email attachments and return Gemini Parts."""
    from .security import _validate_attachment_security
    
    gemini_parts = []
    if not parts:
        return gemini_parts
        
    for part in parts:
        filename = part.get('filename')
        if not filename:
            continue
            
        try:
            mime_type = part.get('mimeType', 'application/octet-stream')
            size = part.get('body', {}).get('size', 0)
            
            # Security validation
            is_valid, error_msg = _validate_attachment_security(filename, mime_type, size)
            if not is_valid:
                logger.warning(f"SECURITY: Blocked attachment {filename}: {error_msg}")
                continue
                
            # Download attachment
            attachment_id = part['body']['attachmentId']
            attachment_data = gmail_service.users().messages().attachments().get(
                userId='me', messageId=message_id, id=attachment_id
            ).execute()['data']
            file_data = base64.urlsafe_b64decode(attachment_data.encode('UTF-8'))

            # Enhanced validation with file content
            is_valid_content, content_error = _validate_attachment_security(
                filename, mime_type, size, file_data
            )
            if not is_valid_content:
                logger.warning(f"SECURITY: Enhanced validation failed for {filename}: {content_error}")
                continue
            
            # Handle audio files (upload to storage)
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
                # Handle images/PDFs (direct to Gemini)
                gemini_parts.append(Part.from_data(file_data, mime_type=mime_type))
                logger.info(f"Processed inline attachment: {filename} ({size} bytes)")
                
        except Exception as e:
            logger.error(f"Failed to process attachment {filename}: {e}")
            
    return gemini_parts

def _send_reply(gmail_service, headers: dict, thread_id: str, reply_body: str):
    """Send reply email in the same thread."""
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
            # Handle References chain for proper threading
            original_references = headers.get('references', '')
            if original_references:
                message['References'] = f"{original_references} {original_message_id}"
            else:
                message['References'] = original_message_id
        else:
            logger.warning("No original message-id found for threading")
            
        message.attach(MIMEText(reply_body, 'html', 'utf-8'))
        encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
        create_message = {'raw': encoded_message, 'threadId': thread_id}
        gmail_service.users().messages().send(userId='me', body=create_message).execute()
        logger.info(f"Successfully sent reply to thread {thread_id}")
        
    except HttpError as error:
        logger.error(f"An error occurred while sending email: {error}")
        raise