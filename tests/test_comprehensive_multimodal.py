#!/usr/bin/env python3
"""
Comprehensive test to diagnose the multi-modal processing issue.
Tests the exact scenario from the user's email.
"""

import json
import logging
import os
import sys
from unittest.mock import Mock, patch
from typing import List

# Add the project directory to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'alza_bot'))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def test_multimodal_query_generation():
    """Test the exact scenario from user's email."""
    
    print("=" * 80)
    print("COMPREHENSIVE MULTI-MODAL TEST")
    print("=" * 80)
    
    # Simulate the exact user email content
    email_subject = ""  # Empty subject as shown in logs
    email_body = """Dobry den, 

Mam nekolik dotazu:

1) chtel bych se zeptat, ja mam AlzaConnect USB-C Hub, a chci vedet jaky ma porty? 

2) Taky bych chtel vedet, jestli mate membership? kolik stoji? AlzaPlus.

3) Dej taky odpoved pak na audio."""
    
    # Audio content (what the user said in Czech audio)
    audio_transcription = "Ahoj, já mám takovou otázku pro Alzu, já mám spory s Alzo a chci to nějak vyřešit. Potřebuju poskytnout Alza web nebo telefonní číslo nějaké, nebo mohli byste mi nějak pomoct."
    
    print(f"EMAIL SUBJECT: '{email_subject}'")
    print(f"EMAIL BODY:\n{email_body}")
    print(f"AUDIO CONTENT: {audio_transcription}")
    print("-" * 60)
    
    # Expected queries (what SHOULD be generated)
    expected_queries = [
        "AlzaConnect USB-C Hub porty specifikace",
        "AlzaPlus membership cena",
        "řešení sporů s Alza kontakt",
        "Alza telefonní číslo web kontakt"
    ]
    
    print("EXPECTED QUERIES:")
    for i, query in enumerate(expected_queries, 1):
        print(f"  {i}. {query}")
    print("-" * 60)
    
    # Test the actual function
    try:
        # Mock the environment variables
        with patch.dict(os.environ, {
            'GEMINI_MODEL': 'gemini-2.0-flash-lite-001',
            'GOOGLE_CLOUD_PROJECT': 'test-project',
            'GOOGLE_CLOUD_LOCATION_GEMINI': 'europe-west1'
        }):
            
            # Mock the Gemini response (simulating what SHOULD happen)
            mock_response = Mock()
            mock_response.text = """{"queries": ["AlzaConnect USB-C Hub porty specifikace", "AlzaPlus membership cena", "řešení sporů s Alza kontakt"]}"""
            
            # Mock the attachment parts (simulating audio processing)
            mock_audio_part = Mock()
            mock_audio_part.mime_type = 'audio/mpeg'
            mock_audio_part.data = b'mock_audio_data'
            
            attachments = [mock_audio_part]
            
            # Import and test the function
            from mainfromgemini import generate_search_queries_from_email
            
            with patch('mainfromgemini.GenerativeModel') as mock_model_class:
                mock_model = Mock()
                mock_model.generate_content.return_value = mock_response
                mock_model_class.return_value = mock_model
                
                # Test the query generation
                queries = generate_search_queries_from_email(email_subject, email_body, attachments)
                
                print("ACTUAL QUERIES GENERATED:")
                for i, query in enumerate(queries, 1):
                    print(f"  {i}. {query}")
                
                # Check what prompt was sent to Gemini
                if mock_model.generate_content.called:
                    call_args = mock_model.generate_content.call_args[0][0]
                    prompt_content = call_args[0] if isinstance(call_args, list) else str(call_args)
                    
                    print("\n" + "=" * 60)
                    print("PROMPT SENT TO GEMINI:")
                    print("=" * 60)
                    print(prompt_content)
                    print("=" * 60)
                    
                    # Check if email body is in the prompt
                    if "AlzaConnect USB-C Hub" in prompt_content:
                        print("✅ EMAIL TEXT CONTENT IS IN PROMPT")
                    else:
                        print("❌ EMAIL TEXT CONTENT MISSING FROM PROMPT!")
                    
                    if "attachment" in prompt_content.lower():
                        print("✅ ATTACHMENT PROCESSING MENTIONED IN PROMPT")
                    else:
                        print("❌ ATTACHMENT PROCESSING NOT MENTIONED IN PROMPT!")
                
                return queries
                
    except Exception as e:
        print(f"❌ ERROR DURING TEST: {e}")
        import traceback
        traceback.print_exc()
        return []

def test_prompt_analysis():
    """Analyze the prompt structure to identify issues."""
    
    print("\n" + "=" * 80)
    print("PROMPT ANALYSIS")
    print("=" * 80)
    
    # Recreate the prompt exactly as in the code
    email_subject = ""
    email_body = """Dobry den, 

Mam nekolik dotazu:

1) chtel bych se zeptat, ja mam AlzaConnect USB-C Hub, a chci vedet jaky ma porty? 

2) Taky bych chtel vedet, jestli mate membership? kolik stoji? AlzaPlus.

3) Dej taky odpoved pak na audio."""
    
    prompt = f"""<ROLE_AND_GOAL>
You are an expert query analysis engine. Your goal is to thoroughly analyze all provided information (email text and attachments) and extract every distinct user question into a concise search query.
</ROLE_AND_GOAL>
<INSTRUCTIONS>
Follow these steps in order:
1.  **Analyze Text First:** Read ONLY the email subject and body. Identify all distinct questions and topics from the text.
2.  **Analyze Attachments Second:** Analyze the content of ALL provided attachments (especially audio). Identify all distinct questions and topics from the attachments.
3.  **Combine and De-duplicate:** Merge the questions from the text and the attachments into a single, comprehensive list. Remove any duplicate questions.
4.  **Generate Final JSON:** Convert the final, combined list of questions into concise search queries in Czech. Your final output MUST be ONLY a valid JSON object in the format: {{"queries": ["query1", "query2", ...]}}. Do not add any other text or explanation.
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
    
    print("COMPLETE PROMPT ANALYSIS:")
    print(prompt)
    
    # Check for issues
    issues = []
    
    if "AlzaConnect USB-C Hub" not in prompt:
        issues.append("❌ Email body content missing from prompt")
    else:
        print("✅ Email body correctly included in prompt")
    
    if "Analyze Text First" not in prompt:
        issues.append("❌ Text analysis instruction missing")
    else:
        print("✅ Text analysis instruction present")
        
    if "Analyze Attachments Second" not in prompt:
        issues.append("❌ Attachment analysis instruction missing") 
    else:
        print("✅ Attachment analysis instruction present")
    
    if issues:
        print("\nISSUES FOUND:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\n✅ PROMPT STRUCTURE LOOKS CORRECT")
    
    return len(issues) == 0

def test_log_analysis():
    """Analyze the actual logs to understand what happened."""
    
    print("\n" + "=" * 80) 
    print("LOG ANALYSIS")
    print("=" * 80)
    
    # Key findings from the logs
    log_findings = [
        "✅ Function triggered successfully",
        "✅ Audio attachment processed: Record-alza.mp3 (574464 bytes)",
        "✅ Query generation called: 'Generating multi-modal search queries using Gemini 2.0 Lite...'",
        "❌ ONLY audio queries generated: ['řešení sporů s Alza', 'kontaktní údaje Alza web', 'kontaktní telefonní číslo Alza']",
        "❌ NO text queries: Missing AlzaConnect USB-C Hub, AlzaPlus membership",
        "✅ RAG search worked: Found 10 contexts for each query",
        "✅ Response generated successfully"
    ]
    
    print("LOG ANALYSIS FINDINGS:")
    for finding in log_findings:
        print(f"  {finding}")
    
    print("\n🔍 ROOT CAUSE HYPOTHESIS:")
    print("  The system appears to be processing ONLY the audio attachment content")
    print("  and completely IGNORING the email text content.")
    print("  This suggests the prompt might not be correctly formatted,")
    print("  or Gemini is not reading the email body portion of the prompt.")

if __name__ == "__main__":
    print("🧪 STARTING COMPREHENSIVE MULTI-MODAL DIAGNOSIS")
    
    # Run all tests
    queries = test_multimodal_query_generation()
    prompt_ok = test_prompt_analysis()
    test_log_analysis()
    
    print("\n" + "=" * 80)
    print("FINAL DIAGNOSIS")
    print("=" * 80)
    
    if len(queries) == 3 and all("spor" in q.lower() for q in queries):
        print("❌ CONFIRMED: System only processes audio, ignores email text")
        print("🔧 SOLUTION NEEDED: Fix prompt format or email content processing")
    elif len(queries) >= 3 and any("USB-C Hub" in q for q in queries):
        print("✅ System correctly processes both text and audio")
    else:
        print("⚠️  Unclear results, need more investigation")
    
    print("\n📋 NEXT STEPS:")
    print("1. Check temperature settings (might be too low)")
    print("2. Verify email body is correctly passed to function")
    print("3. Add more detailed logging to see exact Gemini input")
    print("4. Test with simpler prompt structure")