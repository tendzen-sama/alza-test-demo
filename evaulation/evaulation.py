import os
import pandas as pd
import json
import logging
import time
from typing import List, Dict, Callable, Any

from dotenv import load_dotenv
import vertexai
from vertexai import rag
from vertexai.generative_models import GenerativeModel, GenerationConfig
from google.api_core.exceptions import ResourceExhausted

# --- Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

# --- Environment Variables ---
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("RAG_CORPUS_REGION")
RAG_CORPUS_ID = os.getenv("RAG_CORPUS_ID")
GENERATION_MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-1.5-flash-001")
EVALUATION_MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-1.5-flash-001")

# --- Client Initialization ---
logging.info(f"Initializing Vertex AI in project '{PROJECT_ID}' at location '{LOCATION}'...")
vertexai.init(project=PROJECT_ID, location=LOCATION)


# --- BEST PRACTICE 1: Reactive Retries with Exponential Backoff ---
def call_with_retry(api_call: Callable[[], Any], max_retries: int = 5, initial_delay: float = 2.0) -> Any:
    """
    Calls a function and retries with exponential backoff if a ResourceExhausted error occurs.
    """
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


def get_rag_response_two_step(question: str, rag_corpus_path: str, generation_model: GenerativeModel) -> Dict:
    """
    Performs a robust, two-step RAG process with retry logic.
    """
    logging.info(f"Step 1: Retrieving context for question: '{question}'")

    contexts = []
    try:
        rag_api_call = lambda: rag.retrieval_query(
            rag_resources=[rag.RagResource(rag_corpus=rag_corpus_path)],
            text=question
        )
        response = call_with_retry(rag_api_call)

        all_contexts = response.contexts.contexts
        top_contexts = all_contexts[:2]
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


def run_gemini_evaluation(question: str, answer: str, context: str, ground_truth: str, eval_model: GenerativeModel) -> Dict:
    """
    Uses a Gemini model to perform model-based evaluation with retry logic.
    """
    logging.info("Evaluating generated answer for groundedness using Gemini...")
    prompt = f"""<ROLE>You are an expert evaluator. Your task is to determine if the provided 'Answer' is factually supported by the 'Context'. Do not use any external knowledge.</ROLE><EVALUATION_CRITERIA>- **Groundedness Score (0.0 to 1.0):** - **1.0:** Every piece of information in the 'Answer' is directly and explicitly supported by the 'Context'. - **0.5:** The 'Answer' is related to the 'Context' but makes claims or includes details not present in the 'Context'. - **0.0:** The 'Answer' is completely unrelated to or contradicts the 'Context'.</EVALUATION_CRITERIA><INPUT_DATA>**Context:**\n---\n{context if context else "No context was provided."}\n---\n\n**Question:**\n---\n{question}\n---\n\n**Answer to Evaluate:**\n---\n{answer}\n---</INPUT_DATA><TASK>Carefully compare the 'Answer to Evaluate' against the 'Context'. Provide a 'score' and 'reasoning' in the specified JSON format.</TASK>"""

    eval_result = {"score": 0.0, "reasoning": "Evaluation failed due to a persistent API error."}
    try:
        eval_api_call = lambda: eval_model.generate_content(prompt)
        eval_response = call_with_retry(eval_api_call)
        eval_result = json.loads(eval_response.text)
        logging.info(f"Evaluation result: {eval_result}")
    except Exception as e:
        logging.error(f"Failed to perform model-based evaluation after retries: {e}", exc_info=True)

    return eval_result


if __name__ == "__main__":
    if not all([PROJECT_ID, LOCATION, RAG_CORPUS_ID]):
        raise ValueError("Please set PROJECT_ID, RAG_CORPUS_REGION, and RAG_CORPUS_ID in your .env file.")

    logging.info("Initializing generative models...")
    generation_model = GenerativeModel(GENERATION_MODEL_NAME)
    evaluation_model = GenerativeModel(
        EVALUATION_MODEL_NAME,
        generation_config=GenerationConfig(
            response_mime_type="application/json",
            response_schema={
                "type": "object",
                "properties": {
                    "score": {"type": "number", "description": "A groundedness score from 0.0 to 1.0."},
                    "reasoning": {"type": "string", "description": "A brief explanation for the given score."}
                },
                "required": ["score", "reasoning"]
            },
            temperature=0.0
        )
    )

    rag_corpus_path = f"projects/{PROJECT_ID}/locations/{LOCATION}/ragCorpora/{RAG_CORPUS_ID}"
    logging.info(f"Using RAG Corpus Path: {rag_corpus_path}")

    golden_dataset_file = "golden_dataset.jsonl"
    logging.info(f"Loading golden dataset from '{golden_dataset_file}'...")
    with open(golden_dataset_file, 'r', encoding='utf-8') as f:
        questions = [json.loads(line) for line in f]

    results = []
    logging.info(f"Starting evaluation for {len(questions)} questions...")
    for i, item in enumerate(questions):
        question = item['question']
        ground_truth = item['ground_truth']
        logging.info(f"\n--- Processing item {i+1}/{len(questions)}: '{question}' ---")

        rag_response = get_rag_response_two_step(question, rag_corpus_path, generation_model)
        answer = rag_response['answer']
        contexts = rag_response['contexts']
        full_context_str = "\n\n---\n\n".join(contexts)

        eval_scores = run_gemini_evaluation(question, answer, full_context_str, ground_truth, evaluation_model)

        results.append({
            "question": question,
            "answer": answer,
            "ground_truth": ground_truth,
            "retrieved_context": full_context_str,
            "groundedness_score": eval_scores.get("score"),
            "groundedness_reasoning": eval_scores.get("reasoning")
        })

        # --- BEST PRACTICE 2: Proactive Rate Limiting (Throttling) ---
        # Add a delay to avoid overwhelming the API's per-minute quota.
        logging.info(f"--- Pausing for 2 seconds to respect API rate limits ---")
        time.sleep(6)

    results_df = pd.DataFrame(results)
    print("\n\n--- EVALUATION RESULTS ---")
    print(results_df)

    results_df.to_markdown("EVALUATION_RESULTS.md", index=False)
    logging.info("\nEvaluation complete. Results saved to EVALUATION_RESULTS.md")