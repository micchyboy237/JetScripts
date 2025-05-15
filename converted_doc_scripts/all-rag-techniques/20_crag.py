from jet.llm.mlx.base import MLX
from jet.llm.utils.embeddings import get_embedding_function
from jet.logger import CustomLogger
from typing import List, Dict, Tuple, Any
from urllib.parse import quote_plus
from helper_functions import extract_text_from_pdf
import pypdf
import json
import numpy as np
import os
import re
import requests
import time

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(
    script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")
DATA_DIR = os.path.join(script_dir, "data")
os.makedirs(DATA_DIR, exist_ok=True)
logger.info("Initializing MLX and embedding function")
mlx = MLX()
embed_func = get_embedding_function("mxbai-embed-large")


def chunk_text(text, chunk_size=1000, overlap=200):
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


class SimpleVectorStore:
    def __init__(self):
        self.vectors = []
        self.texts = []
        self.metadata = []

    def add_item(self, text, embedding, metadata=None):
        self.vectors.append(np.array(embedding))
        self.texts.append(text)
        self.metadata.append(metadata or {})

    def add_items(self, items, embeddings):
        for i, (item, embedding) in enumerate(zip(items, embeddings)):
            self.add_item(
                text=item["text"],
                embedding=embedding,
                metadata=item.get("metadata", {})
            )

    def similarity_search(self, query_embedding, k=5):
        if not self.vectors:
            return []
        query_vector = np.array(query_embedding).flatten()
        similarities = []
        for i, vector in enumerate(self.vectors):
            vector = vector.flatten()
            dot_product = np.dot(query_vector, vector)
            query_norm = np.linalg.norm(query_vector)
            vector_norm = np.linalg.norm(vector)
            if query_norm == 0 or vector_norm == 0:
                similarity = 0.0
            else:
                similarity = dot_product / (query_norm * vector_norm)
            similarities.append((i, similarity))
        similarities.sort(key=lambda x: -float('inf')
                          if np.isnan(x[1]) else x[1], reverse=True)
        results = []
        for i in range(min(k, len(similarities))):
            idx, score = similarities[i]
            results.append({
                "text": self.texts[idx],
                "metadata": self.metadata[idx],
                "similarity": float(score)
            })
        return results


def create_embeddings(texts):
    input_texts = texts if isinstance(texts, list) else [texts]
    embeddings = embed_func(input_texts)
    if isinstance(texts, str):
        return embeddings[0]
    return embeddings


def process_document(pdf_path, chunk_size=1000, chunk_overlap=200):
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text, chunk_size, chunk_overlap)
    logger.debug("Creating embeddings for chunks...")
    chunk_texts = [chunk["text"] for chunk in chunks]
    chunk_embeddings = create_embeddings(chunk_texts)
    vector_store = SimpleVectorStore()
    vector_store.add_items(chunks, chunk_embeddings)
    logger.debug(f"Vector store created with {len(chunks)} chunks")
    return vector_store


def evaluate_document_relevance(query, document, model="llama-3.2-1b-instruct-4bit"):
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


def duck_duck_go_search(query, num_results=3):
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


def rewrite_search_query(query, model="llama-3.2-1b-instruct-4bit"):
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


def perform_web_search(query, model="llama-3.2-1b-instruct-4bit"):
    rewritten_query = rewrite_search_query(query, model)
    logger.debug(f"Rewritten search query: {rewritten_query}")
    results_text, sources = duck_duck_go_search(rewritten_query)
    return results_text, sources


def refine_knowledge(text, model="llama-3.2-1b-instruct-4bit"):
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


def crag_process(query, vector_store, k=3, model="llama-3.2-1b-instruct-4bit"):
    logger.debug(f"\n=== Processing query with CRAG: {query} ===\n")
    logger.debug("Retrieving initial documents...")
    query_embedding = create_embeddings(query)
    retrieved_docs = vector_store.similarity_search(query_embedding, k=k)
    logger.debug("Evaluating document relevance...")
    relevance_scores = []
    for doc in retrieved_docs:
        score = evaluate_document_relevance(query, doc["text"], model)
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
        sources.append({
            "title": "Document",
            "url": ""
        })
    elif max_score < 0.3:
        logger.debug(
            f"Low relevance ({max_score:.2f}) - Performing web search")
        web_results, web_sources = perform_web_search(query, model)
        final_knowledge = refine_knowledge(web_results, model)
        sources.extend(web_sources)
    else:
        logger.debug(
            f"Medium relevance ({max_score:.2f}) - Combining document with web search")
        best_doc = retrieved_docs[best_doc_idx]["text"]
        refined_doc = refine_knowledge(best_doc, model)
        web_results, web_sources = perform_web_search(query, model)
        refined_web = refine_knowledge(web_results, model)
        final_knowledge = f"From document:\n{refined_doc}\n\nFrom web search:\n{refined_web}"
        sources.append({
            "title": "Document",
            "url": ""
        })
        sources.extend(web_sources)
    logger.debug("Generating final response...")
    response = generate_response(query, final_knowledge, sources, model)
    return {
        "query": query,
        "response": response,
        "retrieved_docs": retrieved_docs,
        "relevance_scores": relevance_scores,
        "max_relevance": max_score,
        "final_knowledge": final_knowledge,
        "sources": sources
    }


def generate_response(query, knowledge, sources, model="llama-3.2-1b-instruct-4bit"):
    sources_text = ""
    for source in sources:
        title = source.get("title", "Unknown Source")
        url = source.get("url", "")
        if url:
            sources_text += f"- {title}: {url}\n"
        else:
            sources_text += f"- {title}\n"
    system_prompt = "You are a helpful AI assistant. Answer the question based on the provided knowledge and cite sources where applicable."
    user_prompt = f"Knowledge:\n{knowledge}\n\nSources:\n{sources_text}\n\nQuestion: {query}"
    try:
        response = mlx.chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model=model,
            temperature=0.2
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.debug(f"Error generating response: {e}")
        return f"I apologize, but I encountered an error while generating a response to your query: '{query}'. The error was: {str(e)}"


def evaluate_crag_response(query, response, reference_answer=None, model="llama-3.2-1b-instruct-4bit"):
    system_prompt = """You are an objective evaluator. Evaluate the response to the query based on accuracy, completeness, and relevance. If a reference answer is provided, use it to assess correctness. Return a JSON object with:
- overall_score (0-1): Overall quality of the response
- accuracy (0-1): Correctness of information
- completeness (0-1): How comprehensively the query is answered
- relevance (0-1): How well the response addresses the query
- summary: Brief explanation of the evaluation"""
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
            # response_format={"type": "json_object"},
            temperature=0
        )
        evaluation = response["choices"][0]["message"]["content"]
        return evaluation
    except Exception as e:
        logger.debug(f"Error evaluating response: {e}")
        return {
            "error": str(e),
            "overall_score": 0,
            "summary": "Evaluation failed due to an error."
        }


def compare_crag_vs_standard_rag(query, vector_store, reference_answer=None, model="llama-3.2-1b-instruct-4bit"):
    logger.debug("\n=== Running CRAG ===")
    crag_result = crag_process(query, vector_store, model=model)
    crag_response = crag_result["response"]
    logger.debug("\n=== Running standard RAG ===")
    query_embedding = create_embeddings(query)
    retrieved_docs = vector_store.similarity_search(query_embedding, k=3)
    combined_text = "\n\n".join([doc["text"] for doc in retrieved_docs])
    standard_sources = [{"title": "Document", "url": ""}]
    standard_response = generate_response(
        query, combined_text, standard_sources, model)
    logger.debug("\n=== Evaluating CRAG response ===")
    crag_eval = evaluate_crag_response(
        query, crag_response, reference_answer, model)
    logger.debug("\n=== Evaluating standard RAG response ===")
    standard_eval = evaluate_crag_response(
        query, standard_response, reference_answer, model)
    logger.debug("\n=== Comparing approaches ===")
    comparison = compare_responses(
        query, crag_response, standard_response, reference_answer, model)
    return {
        "query": query,
        "crag_response": crag_response,
        "standard_response": standard_response,
        "reference_answer": reference_answer,
        "crag_evaluation": crag_eval,
        "standard_evaluation": standard_eval,
        "comparison": comparison
    }


def compare_responses(query, crag_response, standard_response, reference_answer=None, model="llama-3.2-1b-instruct-4bit"):
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


def run_crag_evaluation(pdf_path, test_queries, reference_answers=None, model="llama-3.2-1b-instruct-4bit"):
    vector_store = process_document(pdf_path)
    results = []
    for i, query in enumerate(test_queries):
        logger.debug(
            f"\n\n===== Evaluating Query {i+1}/{len(test_queries)} =====")
        logger.debug(f"Query: {query}")
        reference = None
        if reference_answers and i < len(reference_answers):
            reference = reference_answers[i]
        result = compare_crag_vs_standard_rag(
            query, vector_store, reference, model)
        results.append(result)
        logger.debug("\n=== Comparison ===")
        logger.debug(result["comparison"])
    overall_analysis = generate_overall_analysis(results, model)
    return {
        "results": results,
        "overall_analysis": overall_analysis
    }


def generate_overall_analysis(results, model="llama-3.2-1b-instruct-4bit"):
    evaluations_summary = ""
    for i, result in enumerate(results):
        evaluations_summary += f"Query {i+1}: {result['query']}\n"
        if 'crag_evaluation' in result and 'overall_score' in result['crag_evaluation']:
            crag_score = result['crag_evaluation'].get('overall_score', 'N/A')
            evaluations_summary += f"CRAG score: {crag_score}\n"
        if 'standard_evaluation' in result and 'overall_score' in result['standard_evaluation']:
            std_score = result['standard_evaluation'].get(
                'overall_score', 'N/A')
            evaluations_summary += f"Standard RAG score: {std_score}\n"
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


pdf_path = os.path.join(DATA_DIR, "AI_Information.pdf")
test_queries = [
    "How does machine learning differ from traditional programming?",
]
reference_answers = [
    "Machine learning differs from traditional programming by having computers learn patterns from data rather than following explicit instructions. In traditional programming, developers write specific rules for the computer to follow, while in machine learning, algorithms learn from examples to make predictions or decisions.",
]
evaluation_results = run_crag_evaluation(
    pdf_path, test_queries, reference_answers)
logger.debug("\n=== Overall Analysis of CRAG vs Standard RAG ===")
logger.debug(evaluation_results["overall_analysis"])
logger.info("\n\n[DONE]", bright=True)
