import json
import numpy as np
import os
import re
import requests
from typing import List, Dict, Any
from urllib.parse import quote_plus
from jet.file.utils import save_file
from helpers import (
    setup_config, initialize_mlx, generate_embeddings,
    load_validation_data, generate_ai_response,
    load_json_data, SearchResult, SimpleVectorStore, DATA_DIR, DOCS_PATH
)


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
    """Chunk text into overlapping segments with metadata."""
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk_text = text[i:i + chunk_size]
        if chunk_text:
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "start_pos": i,
                    "end_pos": i + len(chunk_text),
                    "source_type": "document"
                }
            })
    logger.debug(f"Created {len(chunks)} text chunks")
    return chunks


def process_document(pages: List[Dict[str, Any]], embed_func, chunk_size: int = 1000, chunk_overlap: int = 200) -> SimpleVectorStore:
    """Process document into chunks and create vector store."""
    all_chunks = []
    for page in pages:
        page_chunks = chunk_text(page["text"], chunk_size, chunk_overlap)
        for chunk in page_chunks:
            chunk["metadata"].update(page["metadata"])
        all_chunks.extend(page_chunks)
    logger.debug("Creating embeddings for chunks...")
    chunk_texts = [chunk["text"] for chunk in all_chunks]
    chunk_embeddings = generate_embeddings(chunk_texts, embed_func, logger)
    vector_store = SimpleVectorStore()
    for i, chunk in enumerate(all_chunks):
        vector_store.add_item(
            text=chunk["text"],
            embedding=chunk_embeddings[i],
            metadata=chunk["metadata"]
        )
    logger.debug(f"Vector store created with {len(all_chunks)} chunks")
    return vector_store


def evaluate_document_relevance(query: str, document: str, mlx, model: str = "llama-3.2-1b-instruct-4bit") -> float:
    """Evaluate document relevance to query."""
    system_prompt = "You are an objective evaluator. Rate the relevance of the document to the query on a scale from 0 to 1, where 0 is irrelevant and 1 is highly relevant. Return only the numerical score."
    user_prompt = f"Query: {query}\n\nDocument: {document}"
    try:
        response = mlx.chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model=model,
            temperature=0,
            max_tokens=5
        )
        score_text = response["choices"][0]["message"]["content"].strip()
        score_match = re.search(r'(\d+(\.\d+)?)', score_text)
        if score_match:
            return float(score_match.group(1))
        return 0.5
    except Exception as e:
        logger.debug(f"Error evaluating document relevance: {e}")
        return 0.5


def duck_duck_go_search(query: str, num_results: int = 3) -> tuple[str, List[Dict[str, str]]]:
    """Perform web search using DuckDuckGo API."""
    encoded_query = quote_plus(query)
    url = f"https://api.duckduckgo.com/?q={encoded_query}&format=json"
    try:
        response = requests.get(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        })
        data = response.json()
        results_text = ""
        sources = []
        if data.get("AbstractText"):
            results_text += f"{data['AbstractText']}\n\n"
            sources.append({
                "title": data.get("AbstractSource", "Wikipedia"),
                "url": data.get("AbstractURL", "")
            })
        for topic in data.get("RelatedTopics", [])[:num_results]:
            if "Text" in topic and "FirstURL" in topic:
                results_text += f"{topic['Text']}\n\n"
                sources.append({
                    "title": topic.get("Text", "").split(" - ")[0],
                    "url": topic.get("FirstURL", "")
                })
        return results_text, sources
    except Exception as e:
        logger.debug(f"Error performing web search: {e}")
        return "Failed to retrieve search results.", []


def rewrite_search_query(query: str, mlx, model: str = "llama-3.2-1b-instruct-4bit") -> str:
    """Rewrite query for optimized web search."""
    system_prompt = "You are a helpful assistant. Rewrite the query to optimize it for a web search, making it more specific and concise."
    try:
        response = mlx.chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Original query: {query}\n\nRewritten query:"}
            ],
            model=model,
            temperature=0.3,
            max_tokens=50
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.debug(f"Error rewriting search query: {e}")
        return query


def perform_web_search(query: str, mlx, model: str = "llama-3.2-1b-instruct-4bit") -> tuple[str, List[Dict[str, str]]]:
    """Perform web search with rewritten query."""
    rewritten_query = rewrite_search_query(query, mlx, model)
    logger.debug(f"Rewritten search query: {rewritten_query}")
    results_text, sources = duck_duck_go_search(rewritten_query)
    return results_text, sources


def refine_knowledge(text: str, mlx, model: str = "llama-3.2-1b-instruct-4bit") -> str:
    """Refine text to be concise and relevant."""
    system_prompt = "You are a helpful assistant. Refine the provided text to make it concise, clear, and relevant, removing any redundant or irrelevant information."
    try:
        response = mlx.chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Text to refine:\n\n{text}"}
            ],
            model=model,
            temperature=0.3
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.debug(f"Error refining knowledge: {e}")
        return text


def crag_process(query: str, vector_store: SimpleVectorStore, embed_func, mlx, k: int = 3, model: str = "llama-3.2-1b-instruct-4bit") -> Dict[str, Any]:
    """Run CRAG pipeline."""
    logger.debug(f"\n=== Processing query with CRAG: {query} ===\n")
    logger.debug("Retrieving initial documents...")
    query_embedding = embed_func(query)
    retrieved_docs = vector_store.search(query_embedding, top_k=k)
    logger.debug("Evaluating document relevance...")
    relevance_scores = []
    for doc in retrieved_docs:
        score = evaluate_document_relevance(query, doc["text"], mlx, model)
        relevance_scores.append(score)
        doc["relevance"] = score
        logger.debug(f"Document scored {score:.2f} relevance")
    max_score = max(relevance_scores) if relevance_scores else 0
    best_doc_idx = relevance_scores.index(
        max_score) if relevance_scores else -1
    sources = []
    final_knowledge = ""
    if max_score > 0.7:
        logger.debug(
            f"High relevance ({max_score:.2f}) - Using document directly")
        best_doc = retrieved_docs[best_doc_idx]["text"]
        final_knowledge = best_doc
        sources.append({"title": "Document", "url": ""})
    elif max_score < 0.3:
        logger.debug(
            f"Low relevance ({max_score:.2f}) - Performing web search")
        web_results, web_sources = perform_web_search(query, mlx, model)
        final_knowledge = refine_knowledge(web_results, mlx, model)
        sources.extend(web_sources)
    else:
        logger.debug(
            f"Medium relevance ({max_score:.2f}) - Combining document with web search")
        best_doc = retrieved_docs[best_doc_idx]["text"]
        refined_doc = refine_knowledge(best_doc, mlx, model)
        web_results, web_sources = perform_web_search(query, mlx, model)
        refined_web = refine_knowledge(web_results, mlx, model)
        final_knowledge = f"From document:\n{refined_doc}\n\nFrom web search:\n{refined_web}"
        sources.append({"title": "Document", "url": ""})
        sources.extend(web_sources)
    logger.debug("Generating final response...")
    sources_text = "\n".join([f"- {source['title']}: {source['url']}" if source['url']
                             else f"- {source['title']}" for source in sources])
    response = generate_ai_response(
        query,
        f"You are a helpful AI assistant. Answer the question based on the provided knowledge and cite sources where applicable.\n\nKnowledge:\n{final_knowledge}\n\nSources:\n{sources_text}",
        retrieved_docs,
        mlx,
        logger,
        model=model,
        temperature=0.2
    )
    return {
        "query": query,
        "response": response,
        "retrieved_docs": retrieved_docs,
        "relevance_scores": relevance_scores,
        "max_relevance": max_score,
        "final_knowledge": final_knowledge,
        "sources": sources
    }


def evaluate_crag_response(query: str, response: str, reference_answer: str = None, mlx=None, model: str = "llama-3.2-1b-instruct-4bit") -> str:
    """Evaluate CRAG response."""
    system_prompt = "You are an objective evaluator. Assess the response for accuracy, completeness, and relevance to the query. If a reference answer is provided, compare against it. Provide a concise evaluation."
    user_prompt = f"Query: {query}\n\nResponse: {response}"
    if reference_answer:
        user_prompt += f"\n\nReference Answer: {reference_answer}"
    try:
        response = mlx.chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model=model,
            temperature=0
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        logger.debug(f"Error evaluating response: {e}")
        return f"Evaluation failed: {str(e)}"


def compare_crag_vs_standard_rag(query: str, vector_store: SimpleVectorStore, embed_func, mlx, reference_answer: str = None, model: str = "llama-3.2-1b-instruct-4bit") -> Dict[str, Any]:
    """Compare CRAG and standard RAG approaches."""
    logger.debug("\n=== Running CRAG ===")
    crag_result = crag_process(
        query, vector_store, embed_func, mlx, model=model)
    crag_response = crag_result["response"]
    logger.debug("\n=== Running standard RAG ===")
    query_embedding = embed_func(query)
    retrieved_docs = vector_store.search(query_embedding, top_k=3)
    standard_response = generate_ai_response(
        query,
        "You are a helpful AI assistant. Answer the question based on the provided context. If the context is insufficient, acknowledge the limitation.",
        retrieved_docs,
        mlx,
        logger,
        model=model,
        temperature=0.2
    )
    logger.debug("\n=== Evaluating CRAG response ===")
    crag_eval = evaluate_crag_response(
        query, crag_response, reference_answer, mlx, model)
    logger.debug("\n=== Evaluating standard RAG response ===")
    standard_eval = evaluate_crag_response(
        query, standard_response, reference_answer, mlx, model)
    logger.debug("\n=== Comparing approaches ===")
    comparison = compare_responses(
        query, crag_response, standard_response, reference_answer, mlx, model)
    return {
        "query": query,
        "crag_response": crag_response,
        "standard_response": standard_response,
        "reference_answer": reference_answer,
        "crag_evaluation": crag_eval,
        "standard_evaluation": standard_eval,
        "comparison": comparison
    }


def compare_responses(query: str, crag_response: str, standard_response: str, reference_answer: str = None, mlx=None, model: str = "llama-3.2-1b-instruct-4bit") -> str:
    """Compare CRAG and standard RAG responses."""
    system_prompt = "You are an objective evaluator. Compare the CRAG and standard RAG responses to the query, assessing their accuracy, completeness, and relevance. If a reference answer is provided, use it to evaluate correctness. Provide a concise comparison."
    user_prompt = f"Query: {query}\n\nCRAG Response: {crag_response}\n\nStandard RAG Response: {standard_response}"
    if reference_answer:
        user_prompt += f"\n\nReference Answer: {reference_answer}"
    try:
        response = mlx.chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model=model,
            temperature=0
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.debug(f"Error comparing responses: {e}")
        return f"Error comparing responses: {str(e)}"


def run_crag_evaluation(pages: List[Dict[str, Any]], test_queries: List[str], embed_func, mlx, reference_answers: List[str] = None, model: str = "llama-3.2-1b-instruct-4bit") -> Dict[str, Any]:
    """Run evaluation of CRAG vs standard RAG."""
    vector_store = process_document(pages, embed_func)
    results = []
    for i, query in enumerate(test_queries):
        logger.debug(
            f"\n\n===== Evaluating Query {i+1}/{len(test_queries)} =====")
        logger.debug(f"Query: {query}")
        reference = None
        if reference_answers and i < len(reference_answers):
            reference = reference_answers[i]
        result = compare_crag_vs_standard_rag(
            query, vector_store, embed_func, mlx, reference, model)
        results.append(result)
        logger.debug("\n=== Comparison ===")
        logger.debug(result["comparison"])
    overall_analysis = generate_overall_analysis(results, mlx, model)
    return {
        "results": results,
        "overall_analysis": overall_analysis
    }


def generate_overall_analysis(results: List[Dict[str, Any]], mlx, model: str = "llama-3.2-1b-instruct-4bit") -> str:
    """Generate overall analysis of CRAG vs standard RAG."""
    evaluations_summary = ""
    for i, result in enumerate(results):
        evaluations_summary += f"Query {i+1}: {result['query']}\n"
        evaluations_summary += f"Comparison summary: {result['comparison'][:200]}...\n\n"
    system_prompt = "Provide an overall analysis of the performance of CRAG versus standard RAG based on the provided summaries."
    user_prompt = f"Evaluations Summary:\n{evaluations_summary}"
    try:
        response = mlx.chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model=model,
            temperature=0
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.debug(f"Error generating overall analysis: {e}")
        return f"Error generating overall analysis: {str(e)}"


script_dir, generated_dir, log_file, logger = setup_config(__file__)
mlx, embed_func = initialize_mlx(logger)
formatted_texts, original_chunks = load_json_data(DOCS_PATH, logger)
logger.info("Loaded pre-chunked data from DOCS_PATH")
# Adapt chunks to match expected page structure
pages = [
    {
        "text": chunk["text"],
        "metadata": {
            "source": "AI_Information.pdf",
            "page": i + 1,
            "source_type": "document"
        }
    }
    for i, chunk in enumerate(original_chunks)
]
test_queries = [
    "How does machine learning differ from traditional programming?"
]
reference_answers = [
    "Machine learning differs from traditional programming by having computers learn patterns from data rather than following explicit instructions. In traditional programming, developers write specific rules for the computer to follow, while in machine learning, algorithms learn from examples to make predictions or decisions."
]
evaluation_results = run_crag_evaluation(
    pages=pages,
    test_queries=test_queries,
    embed_func=embed_func,
    mlx=mlx,
    reference_answers=reference_answers
)
save_file(evaluation_results, f"{generated_dir}/evaluation_results.json")
logger.info(
    f"Saved evaluation results to {generated_dir}/evaluation_results.json")
logger.debug("\n=== Overall Analysis of CRAG vs Standard RAG ===")
logger.debug(evaluation_results["overall_analysis"])
logger.info("\n\n[DONE]", bright=True)
