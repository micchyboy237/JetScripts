import numpy as np
import re
from typing import List, Dict, Any
from jet.file.utils import save_file
from helpers import (
    setup_config, initialize_mlx, generate_embeddings,
    load_validation_data, generate_ai_response, evaluate_ai_response,
    load_json_data, SearchResult, SimpleVectorStore, DATA_DIR, DOCS_PATH
)


def chunk_text(text: str, n: int, overlap: int) -> List[str]:
    """Chunk text into overlapping segments."""
    chunks = []
    for i in range(0, len(text), n - overlap):
        chunks.append(text[i:i + n])
    return chunks


def process_document(chunks: List[Dict[str, Any]], embed_func) -> SimpleVectorStore:
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
            metadata=chunks[i]["metadata"]
        )
    logger.debug(f"Added {len(text_chunks)} chunks to the vector store")
    return store


def rerank_with_llm(query: str, results: List[Dict[str, Any]], mlx, top_n: int = 3, model: str = "llama-3.2-3b-instruct-4bit") -> List[Dict[str, Any]]:
    """Rerank search results using LLM scoring."""
    logger.debug(f"Reranking {len(results)} documents...")
    system_prompt = "You are an AI assistant. Score the relevance of the document to the query from 0 to 10, where 10 is highly relevant. Provide only the score."
    scored_results = []
    for i, result in enumerate(results):
        if i % 5 == 0:
            logger.debug(f"Scoring document {i+1}/{len(results)}...")
        user_prompt = f"Query: {query}\nDocument: {result['text']}"
        response = mlx.chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model=model,
            temperature=0
        )
        score_text = response["choices"][0]["message"]["content"].strip()
        score_match = re.search(r'\b(10|[0-9])\b', score_text)
        if score_match:
            score = float(score_match.group(1))
        else:
            logger.debug(
                f"Warning: Could not extract score from response: '{score_text}', using similarity score instead")
            score = result["similarity"] * 10
        scored_results.append({
            "text": result["text"],
            "metadata": result["metadata"],
            "similarity": result["similarity"],
            "relevance_score": score
        })
    reranked_results = sorted(
        scored_results, key=lambda x: x["relevance_score"], reverse=True)
    return reranked_results[:top_n]


def rerank_with_keywords(query: str, results: List[Dict[str, Any]], top_n: int = 3) -> List[Dict[str, Any]]:
    """Rerank search results using keyword-based scoring."""
    keywords = [word.lower() for word in query.split() if len(word) > 3]
    scored_results = []
    for result in results:
        document_text = result["text"].lower()
        base_score = result["similarity"] * 0.5
        keyword_score = 0
        for keyword in keywords:
            if keyword in document_text:
                keyword_score += 0.1
                first_position = document_text.find(keyword)
                if first_position < len(document_text) / 4:
                    keyword_score += 0.1
                frequency = document_text.count(keyword)
                keyword_score += min(0.05 * frequency, 0.2)
        final_score = base_score + keyword_score
        scored_results.append({
            "text": result["text"],
            "metadata": result["metadata"],
            "similarity": result["similarity"],
            "relevance_score": final_score
        })
    reranked_results = sorted(
        scored_results, key=lambda x: x["relevance_score"], reverse=True)
    return reranked_results[:top_n]


def rag_with_reranking(query: str, vector_store: SimpleVectorStore, embed_func, mlx, reranking_method: str = "llm", top_n: int = 3, model: str = "llama-3.2-3b-instruct-4bit") -> Dict[str, Any]:
    """Run RAG with reranking."""
    query_embedding = embed_func(query)
    initial_results = vector_store.search(query_embedding, top_k=10)
    if reranking_method == "llm":
        reranked_results = rerank_with_llm(
            query, initial_results, mlx, top_n=top_n, model=model)
    elif reranking_method == "keywords":
        reranked_results = rerank_with_keywords(
            query, initial_results, top_n=top_n)
    else:
        reranked_results = initial_results[:top_n]
    context = "\n\n===\n\n".join([result["text"]
                                 for result in reranked_results])
    system_prompt = "You are a helpful AI assistant. Answer the user's question based only on the provided context. If you cannot find the answer in the context, state that you don't have enough information."
    response = generate_ai_response(
        query, system_prompt, reranked_results, mlx, logger, model=model)
    return {
        "query": query,
        "reranking_method": reranking_method,
        "initial_results": initial_results[:top_n],
        "reranked_results": reranked_results,
        "context": context,
        "response": response
    }


def evaluate_reranking(query: str, standard_results: Dict[str, Any], reranked_results: Dict[str, Any], reference_answer: str, mlx, model: str = "llama-3.2-3b-instruct-4bit") -> str:
    """Evaluate standard and reranked responses."""
    standard_score, standard_text = evaluate_ai_response(
        query, standard_results["response"], reference_answer, mlx, logger, model)
    reranked_score, reranked_text = evaluate_ai_response(
        query, reranked_results["response"], reference_answer, mlx, logger, model)
    return {
        "standard": {"score": standard_score, "evaluation": standard_text},
        "reranked": {"score": reranked_score, "evaluation": reranked_text}
    }


script_dir, generated_dir, log_file, logger = setup_config(__file__)
mlx, embed_func = initialize_mlx(logger)
formatted_texts, original_chunks = load_json_data(DOCS_PATH, logger)
logger.info("Loaded pre-chunked data from DOCS_PATH")
vector_store = process_document(original_chunks, embed_func)
validation_data = load_validation_data(f"{DATA_DIR}/val.json", logger)
query = validation_data[0]['question']
reference_answer = validation_data[0]['ideal_answer']
logger.debug("Comparing retrieval methods...")
logger.debug("\n=== STANDARD RETRIEVAL ===")
standard_results = rag_with_reranking(
    query, vector_store, embed_func, mlx, reranking_method="none")
logger.debug(f"\nQuery: {query}")
logger.debug(f"\nResponse:\n{standard_results['response']}")
save_file(standard_results, f"{generated_dir}/standard_results.json")
logger.info(f"Saved standard results to {generated_dir}/standard_results.json")
logger.debug("\n=== LLM-BASED RERANKING ===")
llm_results = rag_with_reranking(
    query, vector_store, embed_func, mlx, reranking_method="llm")
logger.debug(f"\nQuery: {query}")
logger.debug(f"\nResponse:\n{llm_results['response']}")
save_file(llm_results, f"{generated_dir}/llm_results.json")
logger.info(f"Saved LLM reranking results to {generated_dir}/llm_results.json")
logger.debug("\n=== KEYWORD-BASED RERANKING ===")
keyword_results = rag_with_reranking(
    query, vector_store, embed_func, mlx, reranking_method="keywords")
logger.debug(f"\nQuery: {query}")
logger.debug(f"\nResponse:\n{keyword_results['response']}")
save_file(keyword_results, f"{generated_dir}/keyword_results.json")
logger.info(
    f"Saved keyword reranking results to {generated_dir}/keyword_results.json")
evaluation = evaluate_reranking(
    query, standard_results, llm_results, reference_answer, mlx)
logger.debug("\n=== EVALUATION RESULTS ===")
logger.debug(evaluation)
save_file(evaluation, f"{generated_dir}/evaluation.json")
logger.info(f"Saved evaluation results to {generated_dir}/evaluation.json")
logger.info("\n\n[DONE]", bright=True)
