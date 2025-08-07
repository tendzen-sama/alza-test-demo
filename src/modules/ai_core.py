"""
AI core module for query generation, RAG, and response synthesis.
"""
import json
import logging
import time
import random
from typing import List, Tuple
from google.api_core.exceptions import ResourceExhausted
from vertexai import rag
from vertexai.generative_models import GenerativeModel, GenerationConfig, Part

from .config import RAG_CORPUS_NAME, GEMINI_MODEL, LLM_RANKER, GEMINI_MODEL_F

logger = logging.getLogger(__name__)

# Search queries schema for structured output
SEARCH_QUERIES_SCHEMA = {
    "type": "object",
    "properties": {
        "queries": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of search queries"
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

def get_rag_corpus_name():
    """Get RAG corpus name from available corpora."""
    try:
        corpora = rag.list_corpora()
        for corpus in corpora:
            if corpus.display_name == RAG_CORPUS_NAME:
                return corpus.name
        logger.warning(f"RAG corpus '{RAG_CORPUS_NAME}' not found")
        return None
    except Exception as e:
        logger.warning(f"Failed to list RAG corpora: {e}")
        return None

def _retry_llm_call(api_call, max_retries: int = 3, base_delay: float = 1.0):
    """Retry LLM API calls with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return api_call()
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            logger.warning(f"LLM call failed (attempt {attempt + 1}/{max_retries}), retrying in {delay:.1f}s: {e}")
            time.sleep(delay)
    return None

def get_rag_context(query: str, max_length: int = 3000, max_retries: int = 3) -> Tuple[str, List]:
    """Get RAG context with intelligent quota handling."""
    corpus_name = get_rag_corpus_name()
    if not corpus_name:
        return "", []
    
    def _make_rag_call():
        rag_config = rag.RagRetrievalConfig(
            top_k=3,
            ranking=rag.Ranking(
                llm_ranker=rag.LlmRanker(model_name=LLM_RANKER)
            ) if LLM_RANKER else None
        )
        
        return rag.retrieval_query(
            rag_resources=[rag.RagResource(rag_corpus=corpus_name)],
            text=query.strip(),
            rag_retrieval_config=rag_config
        )
    
    for attempt in range(max_retries):
        try:
            logger.info(f"RAG searching with reranking for: '{query}' (attempt {attempt+1}/{max_retries})")
            response = _make_rag_call()
            return _process_rag_response(response, query)
            
        except ResourceExhausted as e:
            if "textembedding-gecko" in str(e):
                wait_time = min(15 + attempt * 5, 30)
                logger.warning(f"🚨 QUOTA EXHAUSTED on attempt {attempt+1}: {e}")
                logger.warning(f"⏳ Quota exhausted - waiting {wait_time}s for quota recovery...")
                time.sleep(wait_time)
            else:
                raise e
        except Exception as e:
            logger.error(f"RAG query failed on attempt {attempt+1}: {e}")
            if attempt == max_retries - 1:
                return "", []
    
    return "", []

def _process_rag_response(response, query: str) -> Tuple[str, List]:
    """Process successful RAG response with source information for citations."""
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

    return "", []

def generate_search_queries_from_email(email_subject: str, email_body: str, attachments: List[Part]) -> List[str]:
    """Generate search queries from email using structured LLM output."""
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

def generate_final_reply(email_subject: str, email_body: str, attachment_summary: str, 
                        consolidated_rag_context: str, attachment_parts: List[Part]) -> str:
    """Generate final AI reply for customer support emails."""
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
        attachment_summary=attachment_summary,
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
            response_schema=FINAL_REPLY_SCHEMA,
        )
    )

    def _generate_reply():
        return reply_model.generate_content([final_prompt] + attachment_parts)

    response = _retry_llm_call(_generate_reply)
    return response