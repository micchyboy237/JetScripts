import json
import numpy as np
import re
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any
from jet.file.utils import save_file
from helpers import (
    setup_config, initialize_mlx, generate_embeddings,
    load_validation_data, generate_ai_response,
    load_json_data, SearchResult, SimpleVectorStore, DATA_DIR, DOCS_PATH
)


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict[str, Any]]:
    """Chunk text into overlapping segments with metadata."""
    chunks = []
    for i in range(0, len(text), chunk_size - chunk_overlap):
        chunk = text[i:i + chunk_size]
        if chunk:
            chunk_data = {
                "text": chunk,
                "metadata": {
                    "start_char": i,
                    "end_char": i + len(chunk)
                }
            }
            chunks.append(chunk_data)
    logger.debug(f"Created {len(chunks)} text chunks")
    return chunks


def clean_text(text: str) -> str:
    """Clean text by normalizing whitespace."""
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('\\t', ' ').replace('\\n', ' ')
    return ' '.join(text.split())


def create_bm25_index(chunks: List[Dict[str, Any]]) -> BM25Okapi:
    """Create BM25 index for text chunks."""
    texts = [chunk["text"] for chunk in chunks]
    tokenized_docs = [text.split() for text in texts]
    bm25 = BM25Okapi(tokenized_docs)
    logger.debug(f"Created BM25 index with {len(texts)} documents")
    return bm25


def bm25_search(bm25: BM25Okapi, chunks: List[Dict[str, Any]], query: str, k: int = 5) -> List[Dict[str, Any]]:
    """Perform BM25 search."""
    query_tokens = query.split()
    scores = bm25.get_scores(query_tokens)
    results = []
    for i, score in enumerate(scores):
        metadata = chunks[i].get("metadata", {}).copy()
        metadata["index"] = i
        results.append({
            "text": chunks[i]["text"],
            "metadata": metadata,
            "bm25_score": float(score)
        })
    results.sort(key=lambda x: x["bm25_score"], reverse=True)
    return results[:k]


def fusion_retrieval(query: str, chunks: List[Dict[str, Any]], vector_store: SimpleVectorStore, bm25_index: BM25Okapi, embed_func, k: int = 5, alpha: float = 0.5) -> List[Dict[str, Any]]:
    """Perform fusion retrieval combining vector and BM25 scores."""
    logger.debug(f"Performing fusion retrieval for query: {query}")
    epsilon = 1e-8
    query_embedding = embed_func(query)
    vector_results = vector_store.search(query_embedding, top_k=len(chunks))
    bm25_results = bm25_search(bm25_index, chunks, query, k=len(chunks))
    vector_scores_dict = {result["metadata"]["index"]: result["similarity"] for result in vector_results}
    bm25_scores_dict = {result["metadata"]["index"]: result["bm25_score"] for result in bm25_results}
    all_docs = [{"text": text, "metadata": meta}
                for text, meta in zip(vector_store.texts, vector_store.metadata)]
    combined_results = []
    for i, doc in enumerate(all_docs):
        vector_score = vector_scores_dict.get(i, 0.0)
        bm25_score = bm25_scores_dict.get(i, 0.0)
        combined_results.append({
            "text": doc["text"],
            "metadata": doc["metadata"],
            "vector_score": vector_score,
            "bm25_score": bm25_score,
            "index": i
        })
    vector_scores = np.array([doc["vector_score"] for doc in combined_results])
    bm25_scores = np.array([doc["bm25_score"] for doc in combined_results])
    norm_vector_scores = (vector_scores - np.min(vector_scores)) / \
        (np.max(vector_scores) - np.min(vector_scores) + epsilon)
    norm_bm25_scores = (bm25_scores - np.min(bm25_scores)) / \
        (np.max(bm25_scores) - np.min(bm25_scores) + epsilon)
    combined_scores = alpha * norm_vector_scores + \
        (1 - alpha) * norm_bm25_scores
    for i, score in enumerate(combined_scores):
        combined_results[i]["combined_score"] = float(score)
    combined_results.sort(key=lambda x: x["combined_score"], reverse=True)
    top_results = combined_results[:k]
    logger.debug(
        f"Retrieved {len(top_results)} documents with fusion retrieval")
    return top_results


def process_document(chunks: List[Dict[str, Any]], embed_func) -> tuple[List[Dict[str, Any]], SimpleVectorStore, BM25Okapi]:
    """Process document chunks into vector store and BM25 index."""
    chunk_texts = [chunk["text"] for chunk in chunks]
    logger.debug("Creating embeddings for chunks...")
    embeddings = generate_embeddings(chunk_texts, embed_func, logger)
    vector_store = SimpleVectorStore()
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        vector_store.add_item(
            text=chunk["text"],
            embedding=embedding,
            metadata={**chunk.get("metadata", {}), "index": i}
        )
    logger.debug(f"Added {len(chunks)} items to vector store")
    bm25_index = create_bm25_index(chunks)
    return chunks, vector_store, bm25_index


def answer_with_fusion_rag(query: str, chunks: List[Dict[str, Any]], vector_store: SimpleVectorStore, bm25_index: BM25Okapi, embed_func, mlx, k: int = 5, alpha: float = 0.5, model: str = "llama-3.2-1b-instruct-4bit") -> Dict[str, Any]:
    """Answer query using fusion RAG."""
    retrieved_docs = fusion_retrieval(
        query, chunks, vector_store, bm25_index, embed_func, k, alpha)
    response = generate_ai_response(
        query,
        "You are a helpful AI assistant. Answer the question based on the provided context. If the context is insufficient, acknowledge the limitation.",
        retrieved_docs,
        mlx,
        logger,
        model=model
    )
    return {
        "query": query,
        "retrieved_documents": retrieved_docs,
        "response": response
    }


def vector_only_rag(query: str, vector_store: SimpleVectorStore, embed_func, mlx, k: int = 5, model: str = "llama-3.2-1b-instruct-4bit") -> Dict[str, Any]:
    """Answer query using vector-only RAG."""
    query_embedding = embed_func(query)
    retrieved_docs = vector_store.search(query_embedding, top_k=k)
    response = generate_ai_response(
        query,
        "You are a helpful AI assistant. Answer the question based on the provided context. If the context is insufficient, acknowledge the limitation.",
        retrieved_docs,
        mlx,
        logger,
        model=model
    )
    return {
        "query": query,
        "retrieved_documents": retrieved_docs,
        "response": response
    }


def bm25_only_rag(query: str, chunks: List[Dict[str, Any]], bm25_index: BM25Okapi, mlx, k: int = 5, model: str = "llama-3.2-1b-instruct-4bit") -> Dict[str, Any]:
    """Answer query using BM25-only RAG."""
    retrieved_docs = bm25_search(bm25_index, chunks, query, k=k)
    response = generate_ai_response(
        query,
        "You are a helpful AI assistant. Answer the question based on the provided context. If the context is insufficient, acknowledge the limitation.",
        retrieved_docs,
        mlx,
        logger,
        model=model
    )
    return {
        "query": query,
        "retrieved_documents": retrieved_docs,
        "response": response
    }


def compare_retrieval_methods(query: str, chunks: List[Dict[str, Any]], vector_store: SimpleVectorStore, bm25_index: BM25Okapi, embed_func, mlx, k: int = 5, alpha: float = 0.5, reference_answer: str = None, model: str = "llama-3.2-1b-instruct-4bit") -> Dict[str, Any]:
    """Compare vector-only, BM25, and fusion retrieval methods."""
    logger.debug(f"\n=== Comparing retrieval methods for query: {query} ===\n")
    logger.debug("\nRunning vector-only RAG...")
    vector_result = vector_only_rag(
        query, vector_store, embed_func, mlx, k, model)
    logger.debug("\nRunning BM25-only RAG...")
    bm25_result = bm25_only_rag(query, chunks, bm25_index, mlx, k, model)
    logger.debug("\nRunning fusion RAG...")
    fusion_result = answer_with_fusion_rag(
        query, chunks, vector_store, bm25_index, embed_func, mlx, k, alpha, model)
    logger.debug("\nComparing responses...")
    comparison = evaluate_responses(
        query,
        vector_result["response"],
        bm25_result["response"],
        fusion_result["response"],
        reference_answer,
        mlx,
        model
    )
    return {
        "query": query,
        "vector_result": vector_result,
        "bm25_result": bm25_result,
        "fusion_result": fusion_result,
        "comparison": comparison
    }


def evaluate_responses(query: str, vector_response: str, bm25_response: str, fusion_response: str, reference_answer: str = None, mlx=None, model: str = "llama-3.2-1b-instruct-4bit") -> str:
    """Evaluate responses from different retrieval methods."""
    system_prompt = "You are an objective evaluator. Compare the three responses to the query and provide a concise evaluation. If a reference answer is provided, use it to assess accuracy and completeness."
    user_prompt = f"Query: {query}\n\nVector-only Response:\n{vector_response}\n\nBM25 Response:\n{bm25_response}\n\nFusion Response:\n{fusion_response}"
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


def evaluate_fusion_retrieval(chunks: List[Dict[str, Any]], test_queries: List[str], embed_func, mlx, reference_answers: List[str] = None, k: int = 5, alpha: float = 0.5, model: str = "llama-3.2-1b-instruct-4bit") -> Dict[str, Any]:
    """Evaluate fusion retrieval against vector-only and BM25."""
    logger.debug("=== EVALUATING FUSION RETRIEVAL ===\n")
    chunks, vector_store, bm25_index = process_document(chunks, embed_func)
    results = []
    for i, query in enumerate(test_queries):
        logger.debug(f"\n\n=== Evaluating Query {i+1}/{len(test_queries)} ===")
        logger.debug(f"Query: {query}")
        reference = None
        if reference_answers and i < len(reference_answers):
            reference = reference_answers[i]
        comparison = compare_retrieval_methods(
            query,
            chunks,
            vector_store,
            bm25_index,
            embed_func,
            mlx,
            k=k,
            alpha=alpha,
            reference_answer=reference,
            model=model
        )
        results.append(comparison)
        logger.debug("\n=== Vector-based Response ===")
        logger.debug(comparison["vector_result"]["response"])
        logger.debug("\n=== BM25 Response ===")
        logger.debug(comparison["bm25_result"]["response"])
        logger.debug("\n=== Fusion Response ===")
        logger.debug(comparison["fusion_result"]["response"])
        logger.debug("\n=== Comparison ===")
        logger.debug(comparison["comparison"])
    overall_analysis = generate_overall_analysis(results, mlx, model)
    return {
        "results": results,
        "overall_analysis": overall_analysis
    }


def generate_overall_analysis(results: List[Dict[str, Any]], mlx, model: str = "llama-3.2-1b-instruct-4bit") -> str:
    """Generate overall analysis of retrieval methods."""
    evaluations_summary = ""
    for i, result in enumerate(results):
        evaluations_summary += f"Query {i+1}: {result['query']}\n"
        evaluations_summary += f"Comparison Summary: {result['comparison'][:200]}...\n\n"
    system_prompt = "Provide an overall analysis of the performance of vector-only, BM25, and fusion retrieval methods based on the provided summaries."
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
    "What are the main applications of transformer models in natural language processing?"
]
reference_answers = [
    "Transformer models have revolutionized natural language processing with applications including machine translation, text summarization, question answering, sentiment analysis, and text generation. They excel at capturing long-range dependencies in text and have become the foundation for models like BERT, GPT, and T5.",
]
k = 5
alpha = 0.5
evaluation_results = evaluate_fusion_retrieval(
    chunks=original_chunks,
    test_queries=test_queries,
    embed_func=embed_func,
    mlx=mlx,
    reference_answers=reference_answers,
    k=k,
    alpha=alpha
)
save_file(evaluation_results, f"{generated_dir}/evaluation_results.json")
logger.info(
    f"Saved evaluation results to {generated_dir}/evaluation_results.json")
logger.debug("\n=== OVERALL ANALYSIS ===\n")
logger.debug(evaluation_results["overall_analysis"])
logger.info("\n\n[DONE]", bright=True)
