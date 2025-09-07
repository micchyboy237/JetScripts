import numpy as np
import re
import json
from typing import List, Dict, Any
from jet.file.utils import save_file
from helpers import (
    setup_config, initialize_mlx, generate_embeddings,
    load_validation_data, generate_ai_response,
    load_json_data, SearchResult, SimpleVectorStore, DATA_DIR, DOCS_PATH
)


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[Dict[str, Any]]:
    """Chunk text into overlapping segments with metadata."""
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if chunk:
            chunks.append({
                "text": chunk,
                "chunk_id": len(chunks) + 1,
                "start_char": i,
                "end_char": i + len(chunk)
            })
    logger.debug(f"Created {len(chunks)} text chunks")
    return chunks


def generate_propositions(chunk: Dict[str, Any], mlx, model: str = "llama-3.2-3b-instruct-4bit") -> List[str]:
    """Generate propositions from a text chunk."""
    system_prompt = "Convert the provided text into concise propositions, each representing a single fact or idea. List each proposition on a new line."
    user_prompt = f"Text to convert into propositions:\n\n{chunk['text']}"
    response = mlx.chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        model=model,
        temperature=0
    )
    raw_propositions = response["choices"][0]["message"]["content"].strip().split(
        '\n')
    clean_propositions = []
    for prop in raw_propositions:
        cleaned = re.sub(r'^\s*(\d+\.|\-|\*)\s*', '', prop).strip()
        if cleaned and len(cleaned) > 10:
            clean_propositions.append(cleaned)
    return clean_propositions


def evaluate_proposition(proposition: str, original_text: str, mlx, model: str = "llama-3.2-3b-instruct-4bit") -> Dict[str, int]:
    """Evaluate proposition quality."""
    system_prompt = "Evaluate the proposition based on accuracy, clarity, completeness, and conciseness on a scale of 0-10. Return a JSON object with these metrics."
    user_prompt = f"Proposition: {proposition}\nOriginal Text: {original_text}"
    response = mlx.chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        model=model,
        temperature=0
    )
    try:
        scores = json.loads(response["choices"][0]
                            ["message"]["content"].strip())
        return scores
    except json.JSONDecodeError:
        return {
            "accuracy": 5,
            "clarity": 5,
            "completeness": 5,
            "conciseness": 5
        }


def process_document_into_propositions(chunks: List[Dict[str, Any]], mlx, quality_thresholds: Dict[str, int] = None, model: str = "llama-3.2-3b-instruct-4bit") -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Process document into propositions with quality filtering."""
    if quality_thresholds is None:
        quality_thresholds = {
            "accuracy": 7,
            "clarity": 7,
            "completeness": 7,
            "conciseness": 7
        }
    all_propositions = []
    logger.debug("Generating propositions from chunks...")
    for i, chunk in enumerate(chunks):
        logger.debug(f"Processing chunk {i+1}/{len(chunks)}...")
        chunk_propositions = generate_propositions(chunk, mlx, model)
        logger.debug(f"Generated {len(chunk_propositions)} propositions")
        for prop in chunk_propositions:
            proposition_data = {
                "text": prop,
                "source_chunk_id": chunk["metadata"]["chunk_id"],
                "source_text": chunk["text"]
            }
            all_propositions.append(proposition_data)
    logger.debug("\nEvaluating proposition quality...")
    quality_propositions = []
    for i, prop in enumerate(all_propositions):
        if i % 10 == 0:
            logger.debug(
                f"Evaluating proposition {i+1}/{len(all_propositions)}...")
        scores = evaluate_proposition(
            prop["text"], prop["source_text"], mlx, model)
        prop["quality_scores"] = scores
        passes_quality = True
        for metric, threshold in quality_thresholds.items():
            if scores.get(metric, 0) < threshold:
                passes_quality = False
                break
        if passes_quality:
            quality_propositions.append(prop)
        else:
            logger.debug(
                f"Proposition failed quality check: {prop['text'][:50]}...")
    logger.debug(
        f"\nRetained {len(quality_propositions)}/{len(all_propositions)} propositions after quality filtering")
    return chunks, quality_propositions


def build_vector_stores(chunks: List[Dict[str, Any]], propositions: List[Dict[str, Any]], embed_func) -> tuple[SimpleVectorStore, SimpleVectorStore]:
    """Build vector stores for chunks and propositions."""
    chunk_store = SimpleVectorStore()
    chunk_texts = [chunk["text"] for chunk in chunks]
    logger.debug(f"Creating embeddings for {len(chunk_texts)} chunks...")
    chunk_embeddings = generate_embeddings(chunk_texts, embed_func, logger)
    chunk_metadata = [{"chunk_id": chunk["metadata"]["chunk_id"],
                       "type": "chunk"} for chunk in chunks]
    chunk_store.add_items(chunk_texts, chunk_embeddings, chunk_metadata)
    prop_store = SimpleVectorStore()
    prop_texts = [prop["text"] for prop in propositions]
    logger.debug(f"Creating embeddings for {len(prop_texts)} propositions...")
    prop_embeddings = generate_embeddings(prop_texts, embed_func, logger)
    prop_metadata = [
        {
            "type": "proposition",
            "source_chunk_id": prop["source_chunk_id"],
            "quality_scores": prop["quality_scores"]
        }
        for prop in propositions
    ]
    prop_store.add_items(prop_texts, prop_embeddings, prop_metadata)
    return chunk_store, prop_store


def retrieve_from_store(query: str, vector_store: SimpleVectorStore, embed_func, k: int = 5) -> List[Dict[str, Any]]:
    """Retrieve from vector store."""
    query_embedding = embed_func(query)
    results = vector_store.search(query_embedding, top_k=k)
    return results


def compare_retrieval_approaches(query: str, chunk_store: SimpleVectorStore, prop_store: SimpleVectorStore, embed_func, k: int = 5) -> Dict[str, Any]:
    """Compare proposition-based and chunk-based retrieval."""
    logger.debug(f"\n=== Query: {query} ===")
    logger.debug("\nRetrieving with proposition-based approach...")
    prop_results = retrieve_from_store(query, prop_store, embed_func, k)
    logger.debug("Retrieving with chunk-based approach...")
    chunk_results = retrieve_from_store(query, chunk_store, embed_func, k)
    logger.debug("\n=== Proposition-Based Results ===")
    for i, result in enumerate(prop_results):
        logger.debug(
            f"{i+1}) {result['text']} (Score: {result['similarity']:.4f})")
    logger.debug("\n=== Chunk-Based Results ===")
    for i, result in enumerate(chunk_results):
        truncated_text = result['text'][:150] + \
            "..." if len(result['text']) > 150 else result['text']
        logger.debug(
            f"{i+1}) {truncated_text} (Score: {result['similarity']:.4f})")
    return {
        "query": query,
        "proposition_results": prop_results,
        "chunk_results": chunk_results
    }


def evaluate_responses(query: str, prop_response: str, chunk_response: str, reference_answer: str = None, mlx=None, model: str = "llama-3.2-3b-instruct-4bit") -> str:
    """Evaluate proposition-based and chunk-based responses."""
    system_prompt = "You are an objective evaluator. Compare the two responses to the query and provide a concise evaluation. If a reference answer is provided, use it to assess accuracy and completeness."
    user_prompt = f"Query: {query}\n\nProposition-Based Response:\n{prop_response}\n\nChunk-Based Response:\n{chunk_response}"
    if reference_answer:
        user_prompt += f"\n\nReference Answer:\n{reference_answer}"
    response = mlx.chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        model=model,
        temperature=0
    )
    return response["choices"][0]["message"]["content"]


def run_proposition_chunking_evaluation(chunks: List[Dict[str, Any]], test_queries: List[str], embed_func, mlx, reference_answers: List[str] = None, model: str = "llama-3.2-3b-instruct-4bit") -> Dict[str, Any]:
    """Run proposition chunking evaluation."""
    logger.debug("=== Starting Proposition Chunking Evaluation ===\n")
    chunks, propositions = process_document_into_propositions(
        chunks, mlx, model=model)
    chunk_store, prop_store = build_vector_stores(
        chunks, propositions, embed_func)
    results = []
    for i, query in enumerate(test_queries):
        logger.debug(f"\n\n=== Testing Query {i+1}/{len(test_queries)} ===")
        logger.debug(f"Query: {query}")
        retrieval_results = compare_retrieval_approaches(
            query, chunk_store, prop_store, embed_func)
        logger.debug("\nGenerating response from proposition-based results...")
        prop_response = generate_ai_response(query, "You are a helpful AI assistant. Answer the question based on the provided context. If the context is insufficient, acknowledge the limitation.",
                                             retrieval_results["proposition_results"], mlx, logger, model=model)
        logger.debug("Generating response from chunk-based results...")
        chunk_response = generate_ai_response(
            query, "You are a helpful AI assistant. Answer the question based on the provided context. If the context is insufficient, acknowledge the limitation.", retrieval_results["chunk_results"], mlx, logger, model=model)
        reference = None
        if reference_answers and i < len(reference_answers):
            reference = reference_answers[i]
        logger.debug("\nEvaluating responses...")
        evaluation = evaluate_responses(
            query, prop_response, chunk_response, reference, mlx, model)
        query_result = {
            "query": query,
            "proposition_results": retrieval_results["proposition_results"],
            "chunk_results": retrieval_results["chunk_results"],
            "proposition_response": prop_response,
            "chunk_response": chunk_response,
            "reference_answer": reference,
            "evaluation": evaluation
        }
        results.append(query_result)
        logger.debug("\n=== Proposition-Based Response ===")
        logger.debug(prop_response)
        logger.debug("\n=== Chunk-Based Response ===")
        logger.debug(chunk_response)
        logger.debug("\n=== Evaluation ===")
        logger.debug(evaluation)
    logger.debug("\n\n=== Generating Overall Analysis ===")
    overall_analysis = generate_overall_analysis(results, mlx, model)
    logger.debug("\n" + overall_analysis)
    return {
        "results": results,
        "overall_analysis": overall_analysis,
        "proposition_count": len(propositions),
        "chunk_count": len(chunks)
    }


def generate_overall_analysis(results: List[Dict[str, Any]], mlx, model: str = "llama-3.2-3b-instruct-4bit") -> str:
    """Generate overall analysis of retrieval approaches."""
    evaluations_summary = ""
    for i, result in enumerate(results):
        evaluations_summary += f"Query {i+1}: {result['query']}\n"
        evaluations_summary += f"Evaluation Summary: {result['evaluation'][:200]}...\n\n"
    system_prompt = "Provide an overall analysis of the performance of proposition-based versus chunk-based retrieval based on the provided summaries."
    user_prompt = f"Evaluations Summary:\n{evaluations_summary}"
    response = mlx.chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        model=model,
        temperature=0
    )
    return response["choices"][0]["message"]["content"]


script_dir, generated_dir, log_file, logger = setup_config(__file__)
mlx, embed_func = initialize_mlx(logger)
formatted_texts, original_chunks = load_json_data(DOCS_PATH, logger)
logger.info("Loaded pre-chunked data from DOCS_PATH")
test_queries = [
    "What are the main ethical concerns in AI development?",
]
reference_answers = [
    "The main ethical concerns in AI development include bias and fairness, privacy, transparency, accountability, safety, and the potential for misuse or harmful applications.",
]
evaluation_results = run_proposition_chunking_evaluation(
    chunks=original_chunks,
    test_queries=test_queries,
    embed_func=embed_func,
    mlx=mlx,
    reference_answers=reference_answers
)
save_file(evaluation_results, f"{generated_dir}/evaluation_results.json")
logger.info(
    f"Saved evaluation results to {generated_dir}/evaluation_results.json")
logger.debug("\n=== OVERALL ANALYSIS ===\n")
logger.debug(evaluation_results["overall_analysis"])
logger.info("\n\n[DONE]", bright=True)
