import logging
import time
import json
from typing import Dict, Callable, Any

from vertexai import rag
from vertexai.generative_models import GenerativeModel
from google.api_core.exceptions import ResourceExhausted

# --- IMPROVED: Smart Retries with Reasonable Delays ---
def call_with_retry(api_call: Callable[[], Any], max_retries: int = 8, initial_delay: float = 2.0) -> Any:
    """
    Calls an API with intelligent retry strategy - much more reasonable delays.
    """
    # Reasonable delay progression instead of exponential madness
    delay_schedule = [2, 5, 10, 15, 30, 45, 60, 60]  # Max 60 seconds, not 900+
    
    for attempt in range(max_retries):
        try:
            return api_call()
        except Exception as e:
            # Check if it's a rate limit error
            is_rate_limit_error = (
                isinstance(e, ResourceExhausted) or 
                isinstance(e.__cause__, ResourceExhausted) or
                "429" in str(e) or
                "ResourceExhausted" in str(e) or
                "Quota exceeded" in str(e)
            )

            if is_rate_limit_error and attempt < max_retries - 1:
                # Use predefined delay schedule instead of exponential backoff
                delay = delay_schedule[min(attempt, len(delay_schedule) - 1)]
                
                # Add small jitter to prevent thundering herd
                import random
                jitter = random.uniform(0, min(3, delay * 0.1))
                actual_delay = delay + jitter
                
                logging.warning(f"🚨 API call failed with ResourceExhausted (attempt {attempt + 1}/{max_retries}). Retrying in {actual_delay:.1f} seconds...")
                time.sleep(actual_delay)
            else:
                logging.error(f"❌ Max retries reached or non-retriable error. Total attempts: {attempt + 1}")
                raise e
    return None

def get_rag_response(question: str, rag_corpus_path: str, generation_model: GenerativeModel, num_chunks: int) -> Dict:
    logging.info(f"Step 1: Retrieving context for question: '{question}'")
    contexts = []
    try:
        rag_api_call = lambda: rag.retrieval_query(
            rag_resources=[rag.RagResource(rag_corpus=rag_corpus_path)],
            text=question
        )
        response = call_with_retry(rag_api_call)

        all_contexts = response.contexts.contexts
        top_contexts = all_contexts[:num_chunks]
        contexts = [context.text for context in top_contexts]
        logging.info(f"Successfully retrieved {len(contexts)} context chunk(s).")
    except Exception as e:
        logging.error(f"Failed to retrieve RAG context after retries: {e}", exc_info=True)
        return {"answer": "Error during context retrieval.", "contexts": []}

    if not contexts:
        logging.warning("No context was retrieved from the RAG corpus.")
        return {"answer": "I do not have enough information to answer this question.", "contexts": []}

    logging.info("Step 2: Generating an answer based on the retrieved context.")
    combined_context = "\n\n---\n\n".join(contexts)
    prompt = f"""<CONTEXT>{combined_context}</CONTEXT><INSTRUCTIONS>Based ONLY on the information provided in the <CONTEXT> above, answer the following question. If the context does not contain the answer, state that you do not have enough information.</INSTRUCTIONS><QUESTION>{question}</QUESTION>"""

    answer = "Error during answer generation."
    try:
        gen_api_call = lambda: generation_model.generate_content(prompt)
        answer_response = call_with_retry(gen_api_call)
        answer = answer_response.text
        logging.info("Successfully generated an answer.")
    except Exception as e:
        logging.error(f"Failed to generate answer from context after retries: {e}", exc_info=True)

    return {"answer": answer, "contexts": contexts}

def run_multi_faceted_evaluation(question: str, answer: str, context: str, ground_truth: str, eval_model: GenerativeModel) -> Dict:
    logging.info("Evaluating generated answer with multi-faceted metrics using Gemini...")
    prompt = f"""<ROLE>
You are an expert, impartial, and strict evaluator for a Retrieval-Augmented Generation (RAG) system. Your task is to meticulously evaluate the system's performance based *only* on the provided data. Do not use any external knowledge.
</ROLE>

<EVALUATION_CRITERIA>
You will provide a score from the set [0.0, 0.25, 0.5, 0.75, 1.0] and a detailed reasoning for each of the following three metrics.

1.  **Context Relevance:**
    - **Description:** Measures how well the retrieved 'Context' addresses the 'Question'.
    - **Scoring Rubric:**
        - **1.0 (Excellent):** The context contains all the necessary information to directly and completely answer the question, with little to no irrelevant information.
        - **0.75 (Good):** The context contains the necessary information but also includes some extra, mildly distracting information.
        - **0.5 (Fair):** The context is on the correct topic but is missing specific details required to fully answer the question.
        - **0.25 (Poor):** The context is only tangentially related to the question (e.g., mentions a keyword but in the wrong context).
        - **0.0 (Irrelevant):** The context is completely irrelevant to the question.

2.  **Answer Faithfulness (Groundedness):**
    - **Description:** Measures if the 'Answer' is factually supported *only* by the provided 'Context'. This is a strict check for hallucinations.
    - **Scoring Rubric:**
        - **1.0 (Perfectly Grounded):** Every single claim, fact, or figure in the 'Answer' is directly and explicitly supported by the 'Context'.
        - **0.75 (Mostly Grounded):** The 'Answer' is almost entirely based on the context but may include a minor, harmless inference or rephrasing.
        - **0.5 (Partially Grounded):** The 'Answer' includes a significant claim or detail that is not present in the context (a clear hallucination).
        - **0.25 (Mostly Ungrounded):** The core claim of the 'Answer' is not supported by the context, though it may share some keywords.
        - **0.0 (Contradictory/Ungrounded):** The 'Answer' directly contradicts the context or is entirely fabricated.

3.  **Answer Correctness:**
    - **Description:** Measures how well the 'Answer' matches the ideal 'Ground Truth' answer.
    - **Scoring Rubric:**
        - **1.0 (Perfect):** The 'Answer' is semantically identical to the 'Ground Truth', conveying the exact same information.
        - **0.75 (Mostly Correct):** The 'Answer' is correct but omits a minor detail present in the 'Ground Truth' (e.g., answers 'Log in' when the ground truth is 'Log in and go to My Orders').
        - **0.5 (Partially Correct):** The 'Answer' gets the main idea right but misses significant information or contains a notable inaccuracy.
        - **0.25 (Mostly Incorrect):** The 'Answer' has a small element of truth but the overall message is wrong or misleading.
        - **0.0 (Completely Incorrect):** The 'Answer' is factually wrong compared to the 'Ground Truth'.
</EVALUATION_CRITERIA>

<INPUT_DATA>
**Question:**
---
{question}
---

**Context:**
---
{context if context else "No context was provided."}
---

**Answer to Evaluate:**
---
{answer}
---

**Ground Truth:**
---
{ground_truth}
---
</INPUT_DATA>

<TASK>
Carefully analyze the INPUT_DATA against the EVALUATION_CRITERIA. Provide a step-by-step reasoning for each metric before concluding with the final JSON object. Your output must be a single JSON object with the keys: "context_relevance_score", "context_relevance_reasoning", "faithfulness_score", "faithfulness_reasoning", "correctness_score", "correctness_reasoning".
</TASK>
"""

    eval_result = {
        "context_relevance_score": 0.0, "context_relevance_reasoning": "Evaluation failed.",
        "faithfulness_score": 0.0, "faithfulness_reasoning": "Evaluation failed.",
        "correctness_score": 0.0, "correctness_reasoning": "Evaluation failed."
    }
    try:
        eval_api_call = lambda: eval_model.generate_content(prompt)
        eval_response = call_with_retry(eval_api_call)
        eval_result = json.loads(eval_response.text)
        logging.info(f"Evaluation result: {eval_result}")
    except Exception as e:
        logging.error(f"Failed to perform model-based evaluation after retries: {e}", exc_info=True)

    return eval_result