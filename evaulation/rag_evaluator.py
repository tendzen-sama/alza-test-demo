import logging
import time
import json
from typing import Dict, Callable, Any

from vertexai import rag
from vertexai.generative_models import GenerativeModel
from google.api_core.exceptions import ResourceExhausted

# --- BEST PRACTICE 1: Reactive Retries with Exponential Backoff ---
def call_with_retry(api_call: Callable[[], Any], max_retries: int = 5, initial_delay: float = 2.0) -> Any:
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            return api_call()
        except ResourceExhausted as e:
            logging.warning(f"API call failed with ResourceExhausted (attempt {attempt + 1}/{max_retries}). Retrying in {delay:.1f} seconds...")
            if attempt == max_retries - 1:
                logging.error("Max retries reached. Failing.")
                raise e
            time.sleep(delay)
            delay *= 2.0
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
You are an expert, impartial evaluator for a Retrieval-Augmented Generation (RAG) system. Your task is to evaluate the system's performance based on the provided data.
</ROLE>

<EVALUATION_CRITERIA>
You will provide a score from 0.0 to 1.0 and a brief reasoning for each of the following three metrics:

1.  **Context Relevance:**
    - **Description:** How relevant is the retrieved 'Context' to the 'Question'?
    - **1.0:** The context directly contains the information needed to answer the question.
    - **0.5:** The context is on the same general topic but does not contain the specific information needed.
    - **0.0:** The context is completely irrelevant to the question.

2.  **Answer Faithfulness (Groundedness):**
    - **Description:** Is the 'Answer' factually supported by the 'Context'? Do not use external knowledge.
    - **1.0:** Every claim in the answer is directly and explicitly supported by the context.
    - **0.5:** The answer makes some claims that are not supported by the context.
    - **0.0:** The answer is completely unsupported by or contradicts the context.

3.  **Answer Correctness:**
    - **Description:** How well does the generated 'Answer' match the ideal 'Ground Truth' answer?
    - **1.0:** The answer is perfectly aligned with the ground truth, conveying the same information.
    - **0.5:** The answer is partially correct but misses key information or contains minor inaccuracies compared to the ground truth.
    - **0.0:** The answer is completely incorrect compared to the ground truth.
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
Provide your evaluation in the specified JSON format.
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