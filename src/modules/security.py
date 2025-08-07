"""
Security validation and sanitization module.
"""
import re
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

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

def _validate_attachment_security(filename: str, mime_type: str, size: int, file_data: bytes = None) -> Tuple[bool, str]:
    """Validate attachment security based on type, size, and content."""
    # File type and size validation
    MAX_SIZES = {'image': 10 * 1024 * 1024, 'audio': 50 * 1024 * 1024, 'application': 25 * 1024 * 1024}
    ALLOWED_TYPES = {
        'image/jpeg', 'image/jpg', 'image/png',
        'audio/mp3', 'audio/mpeg', 'audio/wav', 'audio/wave', 'audio/x-wav',
        'audio/m4a', 'audio/mp4', 'audio/aac', 'audio/x-m4a',
        'audio/flac', 'audio/x-flac',
        'audio/ogg', 'audio/oga', 'audio/vorbis',
        'audio/aiff', 'audio/x-aiff', 'audio/aif',
        'application/pdf'
    }
    
    if mime_type not in ALLOWED_TYPES:
        return False, f"Unsupported file type: {mime_type}"
        
    category = mime_type.split('/')[0]
    max_size = MAX_SIZES.get(category, 5 * 1024 * 1024)
    if size > max_size:
        return False, f"File too large: {size} bytes (max: {max_size})"
        
    if not filename or '..' in filename or '/' in filename or '\\' in filename:
        return False, "Invalid filename"
    
    return True, ""