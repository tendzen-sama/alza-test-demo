import os
import pandas as pd
import json
import logging
import time
import yaml
from datetime import datetime

from dotenv import load_dotenv
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig

# Import the core logic from our new module
import rag_evaluator_without_rank


# --- Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

# --- BEST PRACTICE: Load all configuration from a YAML file ---
with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)

# --- Environment & Config Variables ---
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("RAG_CORPUS_REGION")
RAG_CORPUS_ID = os.getenv("RAG_CORPUS_ID")

# --- BEST PRACTICE: Caching Logic ---
def load_cache(cache_path_without_rank):
    if os.path.exists(cache_path_without_rank):
        with open(cache_path_without_rank, 'r') as f:
            return json.load(f)
    return {}

def save_cache(cache_path_without_rank, cache):
    with open(cache_path_without_rank, 'w') as f:
        json.dump(cache, f, indent=2)

def generate_report(results_df, output_folder):
    """Generates a comprehensive Markdown and JSONL report."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path = os.path.join(output_folder, f"EVALUATION_RESULTS_{timestamp}.md")
    jsonl_path = os.path.join(output_folder, f"EVALUATION_RESULTS_{timestamp}.jsonl")

    os.makedirs(output_folder, exist_ok=True)

    # --- Generate Markdown Report ---
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# RAG System Evaluation Report\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # --- Summary Statistics ---
        f.write("## Summary Statistics\n\n")
        summary = {
            "Average Context Relevance": results_df['context_relevance_score'].mean(),
            "Average Faithfulness": results_df['faithfulness_score'].mean(),
            "Average Correctness": results_df['correctness_score'].mean()
        }
        summary_df = pd.DataFrame([summary])
        f.write(summary_df.to_markdown(index=False))
        f.write("\n\n---\n\n")

        # --- Detailed Results ---
        f.write("## Detailed Results\n\n")
        f.write(results_df.to_markdown(index=False))

    logging.info(f"Markdown report saved to {md_path}")

    # --- Generate JSONL Report ---
    results_df.to_json(jsonl_path, orient='records', lines=True, force_ascii=False)
    logging.info(f"JSONL report saved to {jsonl_path}")


if __name__ == "__main__":
    if not all([PROJECT_ID, LOCATION, RAG_CORPUS_ID]):
        raise ValueError("Please set PROJECT_ID, RAG_CORPUS_REGION, and RAG_CORPUS_ID in your .env file.")

    logging.info(f"Initializing Vertex AI in project '{PROJECT_ID}' at location '{LOCATION}'...")
    vertexai.init(project=PROJECT_ID, location=LOCATION)

    logging.info("Initializing generative models...")
    generation_model = GenerativeModel(config['generation_model_name'])
    evaluation_model = GenerativeModel(
        config['evaluation_model_name'],
        generation_config=GenerationConfig(
            response_mime_type="application/json",
            response_schema={
                "type": "object",
                "properties": {
                    "context_relevance_score": {"type": "number"}, "context_relevance_reasoning": {"type": "string"},
                    "faithfulness_score": {"type": "number"}, "faithfulness_reasoning": {"type": "string"},
                    "correctness_score": {"type": "number"}, "correctness_reasoning": {"type": "string"}
                },
                "required": ["context_relevance_score", "faithfulness_score", "correctness_score"]
            },
            temperature=0.0
        )
    )

    rag_corpus_path = f"projects/{PROJECT_ID}/locations/{LOCATION}/ragCorpora/{RAG_CORPUS_ID}"
    logging.info(f"Using RAG Corpus Path: {rag_corpus_path}")

    logging.info(f"Loading golden dataset from '{config['golden_dataset_path']}'...")
    with open(config['golden_dataset_path'], 'r', encoding='utf-8') as f:
        questions = [json.loads(line) for line in f]

    cache = load_cache(config['cache_path_without_rank'])
    results = []

    logging.info(f"Starting evaluation for {len(questions)} questions...")
    for i, item in enumerate(questions):
        question = item['question']
        ground_truth = item['ground_truth']
        logging.info(f"\n--- Processing item {i+1}/{len(questions)}: '{question}' ---")

        if question in cache:
            logging.info("Result found in cache. Skipping API calls.")
            results.append(cache[question])
        else:
            rag_response = rag_evaluator_without_rank.get_rag_response(question, rag_corpus_path, generation_model, config['num_context_chunks'])
            answer = rag_response['answer']
            contexts = rag_response['contexts']
            full_context_str = "\n\n---\n\n".join(contexts)

            eval_scores = rag_evaluator_without_rank.run_multi_faceted_evaluation(question, answer, full_context_str, ground_truth, evaluation_model)

            current_result = {
                "question": question,
                "answer": answer,
                "ground_truth": ground_truth,
                "retrieved_context": full_context_str,
                **eval_scores # Neatly merge the evaluation scores dictionary
            }
            results.append(current_result)
            cache[question] = current_result # Save new result to cache

        # Intelligent rate limiting - longer pause after errors
        if 'error' in str(results[-1]).lower():
            delay = config['rate_limit_delay'] * 2  # Double delay after errors
            logging.info(f"⚠️ Error detected, extended pause: {delay} seconds")
        else:
            delay = config['rate_limit_delay']
            logging.info(f"✅ Success, standard pause: {delay} seconds")
        
        time.sleep(delay)

    save_cache(config['cache_path_without_rank'], cache)
    results_df = pd.DataFrame(results)
    generate_report(results_df, config['output_folder'])

    print("\n\n--- FINAL EVALUATION SUMMARY ---")
    print(results_df[['context_relevance_score', 'faithfulness_score', 'correctness_score']].mean().to_frame('Average Score'))
    logging.info("\nEvaluation complete.")