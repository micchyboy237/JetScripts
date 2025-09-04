from datetime import datetime
import os
import numpy as np
import json
from typing import List, Dict, Any
from jet.file.utils import save_file
from helpers import (
    setup_config, initialize_mlx, generate_embeddings,
    load_validation_data, generate_ai_response,
    load_json_data, SearchResult, SimpleVectorStore, DATA_DIR, DOCS_PATH
)


def chunk_text(text: str, n: int, overlap: int) -> List[str]:
    """Chunk text into overlapping segments."""
    chunks = []
    for i in range(0, len(text), n - overlap):
        chunks.append(text[i:i + n])
    return chunks


def get_user_feedback(query: str, response: str, relevance: int, quality: int, comments: str = "") -> Dict[str, Any]:
    """Collect user feedback for a query response."""
    return {
        "query": query,
        "response": response,
        "relevance": int(relevance),
        "quality": int(quality),
        "comments": comments,
        "timestamp": datetime.now().isoformat()
    }


def store_feedback(feedback: Dict[str, Any], feedback_file: str = "feedback_data.json") -> None:
    """Store feedback in a JSON file."""
    with open(feedback_file, "a") as f:
        json.dump(feedback, f)
        f.write("\n")


def load_feedback_data(feedback_file: str = "feedback_data.json") -> List[Dict[str, Any]]:
    """Load feedback data from a JSON file."""
    feedback_data = []
    try:
        with open(feedback_file, "r") as f:
            for line in f:
                if line.strip():
                    feedback_data.append(json.loads(line.strip()))
    except FileNotFoundError:
        logger.debug(
            "No feedback data file found. Starting with empty feedback.")
    return feedback_data


def process_document(chunks: List[Dict[str, Any]], embed_func) -> tuple[List[str], SimpleVectorStore]:
    """Process document chunks and store in vector store."""
    logger.debug("Processing chunks...")
    text_chunks = [chunk["text"] for chunk in chunks]
    logger.debug(f"Created {len(text_chunks)} text chunks")
    chunk_embeddings = generate_embeddings(text_chunks, embed_func, logger)
    store = SimpleVectorStore()
    for i, (chunk, embedding) in enumerate(zip(text_chunks, chunk_embeddings)):
        store.add_item(
            text=chunk,
            embedding=embedding,
            metadata={
                "index": i,
                "source": chunks[i]["metadata"]["doc_index"],
                "relevance_score": 1.0,
                "feedback_count": 0
            }
        )
    logger.debug(f"Added {len(text_chunks)} chunks to the vector store")
    return text_chunks, store


def assess_feedback_relevance(query: str, doc_text: str, feedback: Dict[str, Any], mlx, model: str = "llama-3.2-3b-instruct-4bit") -> bool:
    """Assess if feedback is relevant to the query and document."""
    system_prompt = "Determine if the feedback is relevant to the query and document. Answer with 'yes' or 'no'."
    user_prompt = f"Query: {query}\nDocument: {doc_text[:500]}\nFeedback Query: {feedback['query']}\nFeedback Response: {feedback['response'][:200]}"
    response = mlx.chat(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        model=model,
        temperature=0
    )
    answer = response["choices"][0]["message"]["content"].lower()
    return 'yes' in answer


def adjust_relevance_scores(query: str, results: List[Dict[str, Any]], feedback_data: List[Dict[str, Any]], mlx, model: str = "llama-3.2-3b-instruct-4bit") -> List[Dict[str, Any]]:
    """Adjust document relevance scores based on feedback history."""
    if not feedback_data:
        return results
    logger.debug("Adjusting relevance scores based on feedback history...")
    for i, result in enumerate(results):
        document_text = result["text"]
        relevant_feedback = []
        for feedback in feedback_data:
            if assess_feedback_relevance(query, document_text, feedback, mlx, model):
                relevant_feedback.append(feedback)
        if relevant_feedback:
            avg_relevance = sum(f['relevance']
                                for f in relevant_feedback) / len(relevant_feedback)
            modifier = 0.5 + (avg_relevance / 5.0)
            original_score = result["similarity"]
            adjusted_score = original_score * modifier
            result["original_similarity"] = original_score
            result["similarity"] = adjusted_score
            result["relevance_score"] = adjusted_score
            result["feedback_applied"] = True
            result["feedback_count"] = len(relevant_feedback)
            logger.debug(
                f"  Document {i+1}: Adjusted score from {original_score:.4f} to {adjusted_score:.4f} based on {len(relevant_feedback)} feedback(s)")
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results


def fine_tune_index(current_store: SimpleVectorStore, chunks: List[str], feedback_data: List[Dict[str, Any]], embed_func) -> SimpleVectorStore:
    """Fine-tune the vector store with high-quality feedback."""
    logger.debug("Fine-tuning index with high-quality feedback...")
    good_feedback = [f for f in feedback_data if f['relevance']
                     >= 4 and f['quality'] >= 4]
    if not good_feedback:
        logger.debug("No high-quality feedback found for fine-tuning.")
        return current_store
    new_store = SimpleVectorStore()
    for i in range(len(current_store.texts)):
        new_store.add_item(
            text=current_store.texts[i],
            embedding=current_store.vectors[i],
            metadata=current_store.metadata[i].copy()
        )
    for feedback in good_feedback:
        enhanced_text = f"Question: {feedback['query']}\nAnswer: {feedback['response']}"
        embedding = embed_func(enhanced_text)
        new_store.add_item(
            text=enhanced_text,
            embedding=embedding,
            metadata={
                "type": "feedback_enhanced",
                "query": feedback["query"],
                "relevance_score": 1.2,
                "feedback_count": 1,
                "original_feedback": feedback
            }
        )
        logger.debug(
            f"Added enhanced content from feedback: {feedback['query'][:50]}...")
    logger.debug(
        f"Fine-tuned index now has {len(new_store.texts)} items (original: {len(chunks)})")
    return new_store


def rag_with_feedback_loop(query: str, vector_store: SimpleVectorStore, feedback_data: List[Dict[str, Any]], embed_func, mlx, k: int = 5, model: str = "meta-llama/Llama-3.2-3B-Instruct") -> Dict[str, Any]:
    """Run RAG with feedback loop."""
    logger.debug(f"\n=== Processing query with feedback-enhanced RAG ===")
    logger.debug(f"Query: {query}")
    query_embedding = embed_func(query)
    results = vector_store.search(query_embedding, top_k=k)
    adjusted_results = adjust_relevance_scores(
        query, results, feedback_data, mlx, model)
    system_prompt = "You are a helpful AI assistant. Answer the user's question based only on the provided context. If you cannot find the answer in the context, state that you don't have enough information."
    response = generate_ai_response(
        query, system_prompt, adjusted_results, mlx, logger, model=model)
    result = {
        "query": query,
        "retrieved_documents": adjusted_results,
        "response": response
    }
    logger.debug("\n=== Response ===")
    logger.debug(response)
    return result


def full_rag_workflow(chunks: List[Dict[str, Any]], query: str, embed_func, mlx, feedback_data: List[Dict[str, Any]] = None, feedback_file: str = "feedback_data.json", fine_tune: bool = False) -> Dict[str, Any]:
    """Execute full RAG workflow with feedback."""
    if feedback_data is None:
        feedback_data = load_feedback_data(feedback_file)
        logger.debug(
            f"Loaded {len(feedback_data)} feedback entries from {feedback_file}")
    text_chunks, vector_store = process_document(chunks, embed_func)
    if fine_tune and feedback_data:
        vector_store = fine_tune_index(
            vector_store, text_chunks, feedback_data, embed_func)
    result = rag_with_feedback_loop(
        query, vector_store, feedback_data, embed_func, mlx)
    logger.debug(
        "\n=== Would you like to provide feedback on this response? ===")
    logger.debug("Rate relevance (1-5, with 5 being most relevant):")
    relevance = input()
    logger.debug("Rate quality (1-5, with 5 being highest quality):")
    quality = input()
    logger.debug("Any comments? (optional, press Enter to skip)")
    comments = input()
    feedback = get_user_feedback(
        query=query,
        response=result["response"],
        relevance=int(relevance),
        quality=int(quality),
        comments=comments
    )
    store_feedback(feedback, feedback_file)
    logger.debug("Feedback recorded. Thank you!")
    return result


def calculate_similarity(text1: str, text2: str, embed_func) -> float:
    """Calculate cosine similarity between two texts."""
    embedding1 = embed_func(text1)
    embedding2 = embed_func(text2)
    vec1 = np.squeeze(np.array(embedding1))
    vec2 = np.squeeze(np.array(embedding2))
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0
    similarity = np.dot(vec1, vec2) / \
        (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return similarity


def evaluate_feedback_loop(chunks: List[Dict[str, Any]], test_queries: List[str], embed_func, mlx, reference_answers: List[str] = None) -> Dict[str, Any]:
    """Evaluate the impact of feedback loop on RAG performance."""
    logger.debug("=== Evaluating Feedback Loop Impact ===")
    temp_feedback_file = "temp_evaluation_feedback.json"
    feedback_data = []
    logger.debug("\n=== ROUND 1: NO FEEDBACK ===")
    round1_results = []
    for i, query in enumerate(test_queries):
        logger.debug(f"\nQuery {i+1}: {query}")
        text_chunks, vector_store = process_document(chunks, embed_func)
        result = rag_with_feedback_loop(
            query, vector_store, [], embed_func, mlx)
        round1_results.append(result)
        if reference_answers and i < len(reference_answers):
            similarity_to_ref = calculate_similarity(
                result["response"], reference_answers[i], embed_func)
            relevance = max(1, min(5, int(similarity_to_ref * 5)))
            quality = max(1, min(5, int(similarity_to_ref * 5)))
            feedback = get_user_feedback(
                query=query,
                response=result["response"],
                relevance=relevance,
                quality=quality,
                comments=f"Synthetic feedback based on reference similarity: {similarity_to_ref:.2f}"
            )
            feedback_data.append(feedback)
            store_feedback(feedback, temp_feedback_file)
    logger.debug("\n=== ROUND 2: WITH FEEDBACK ===")
    round2_results = []
    text_chunks, vector_store = process_document(chunks, embed_func)
    vector_store = fine_tune_index(
        vector_store, text_chunks, feedback_data, embed_func)
    for i, query in enumerate(test_queries):
        logger.debug(f"\nQuery {i+1}: {query}")
        result = rag_with_feedback_loop(
            query, vector_store, feedback_data, embed_func, mlx)
        round2_results.append(result)
    comparison = compare_results(
        test_queries, round1_results, round2_results, reference_answers, mlx)
    if os.path.exists(temp_feedback_file):
        os.remove(temp_feedback_file)
    return {
        "round1_results": round1_results,
        "round2_results": round2_results,
        "comparison": comparison
    }


def compare_results(queries: List[str], round1_results: List[Dict[str, Any]], round2_results: List[Dict[str, Any]], reference_answers: List[str] = None, mlx=None, model: str = "llama-3.2-3b-instruct-4bit") -> List[Dict[str, Any]]:
    """Compare results between rounds with and without feedback."""
    logger.debug("\n=== COMPARING RESULTS ===")
    comparisons = []
    for i, (query, r1, r2) in enumerate(zip(queries, round1_results, round2_results)):
        comparison_prompt = f"Round 1 Response (No Feedback):\n{r1['response']}\n\nRound 2 Response (With Feedback):\n{r2['response']}"
        if reference_answers and i < len(reference_answers):
            comparison_prompt += f"\n\nReference Answer:\n{reference_answers[i]}"
        system_prompt = "Compare the two responses and provide a concise analysis of improvements or differences."
        response = mlx.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": comparison_prompt}
            ],
            model=model,
            temperature=0
        )
        comparisons.append({
            "query": query,
            "analysis": response["choices"][0]["message"]["content"]
        })
        logger.debug(f"\nQuery {i+1}: {query}")
        logger.debug(
            f"Analysis: {response['choices'][0]['message']['content'][:200]}...")
    return comparisons


script_dir, generated_dir, log_file, logger = setup_config(__file__)
mlx, embed_func = initialize_mlx(logger)
formatted_texts, original_chunks = load_json_data(DOCS_PATH, logger)
logger.info("Loaded pre-chunked data from DOCS_PATH")
test_queries = [
    "What is an isekai anime?",
]
reference_answers = [
    "An isekai anime is a genre of Japanese animation where the protagonist is transported, reincarnated, or trapped in a parallel or fantasy world. These stories typically focus on the character adapting to their new environment, often gaining special powers or abilities, and embarking on adventures in that alternate realm.",
]
evaluation_results = evaluate_feedback_loop(
    original_chunks, test_queries, embed_func, mlx, reference_answers
)
save_file(evaluation_results, f"{generated_dir}/evaluation_results.json")
logger.info(
    f"Saved evaluation results to {generated_dir}/evaluation_results.json")
comparisons = evaluation_results['comparison']
logger.debug("\n=== FEEDBACK IMPACT ANALYSIS ===\n")
for i, comparison in enumerate(comparisons):
    logger.debug(f"Query {i+1}: {comparison['query']}")
    logger.debug(f"\nAnalysis of feedback impact:")
    logger.debug(comparison['analysis'])
    logger.debug("\n" + "-"*50 + "\n")
round_responses = [evaluation_results[f'round{round_num}_results']
                   for round_num in range(1, len(evaluation_results) - 1)]
response_lengths = [[len(r["response"]) for r in round]
                    for round in round_responses]
logger.debug("\nResponse length comparison (proxy for completeness):")
avg_lengths = [sum(lengths) / len(lengths) for lengths in response_lengths]
for round_num, avg_len in enumerate(avg_lengths, start=1):
    logger.debug(f"Round {round_num}: {avg_len:.1f} chars")
if len(avg_lengths) > 1:
    changes = [(avg_lengths[i] - avg_lengths[i-1]) / avg_lengths[i-1]
               * 100 for i in range(1, len(avg_lengths))]
    for round_num, change in enumerate(changes, start=2):
        logger.debug(
            f"Change from Round {round_num-1} to Round {round_num}: {change:.1f}%")
logger.info("\n\n[DONE]", bright=True)
