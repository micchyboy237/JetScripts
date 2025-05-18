import json
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from typing import List, Dict, Any
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
                    "end_pos": i + len(chunk_text)
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


def generate_hypothetical_document(query: str, mlx, desired_length: int = 1000, model: str = "llama-3.2-1b-instruct-4bit") -> str:
    """Generate a hypothetical document for the query."""
    system_prompt = "You are a helpful AI assistant. Generate a detailed document that answers the given question comprehensively."
    user_prompt = f"Question: {query}\n\nGenerate a document that fully answers this question:"
    response = mlx.chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        model=model,
        temperature=0.1
    )
    return response["choices"][0]["message"]["content"]


def hyde_rag(query: str, vector_store: SimpleVectorStore, embed_func, mlx, k: int = 5, should_generate_response: bool = True, model: str = "llama-3.2-1b-instruct-4bit") -> Dict[str, Any]:
    """Run HyDE RAG pipeline."""
    logger.debug(f"\n=== Processing query with HyDE: {query} ===\n")
    logger.debug("Generating hypothetical document...")
    hypothetical_doc = generate_hypothetical_document(query, mlx, model=model)
    logger.debug(
        f"Generated hypothetical document of {len(hypothetical_doc)} characters")
    logger.debug("Creating embedding for hypothetical document...")
    hypothetical_embedding = embed_func(hypothetical_doc)
    logger.debug(f"Retrieving {k} most similar chunks...")
    retrieved_chunks = vector_store.search(hypothetical_embedding, top_k=k)
    results = {
        "query": query,
        "hypothetical_document": hypothetical_doc,
        "retrieved_chunks": retrieved_chunks
    }
    if should_generate_response:
        logger.debug("Generating final response...")
        response = generate_ai_response(
            query,
            "You are a helpful AI assistant. Answer the question based on the provided context. If the context is insufficient, acknowledge the limitation.",
            retrieved_chunks,
            mlx,
            logger,
            model=model,
            max_tokens=500
        )
        results["response"] = response
    return results


def standard_rag(query: str, vector_store: SimpleVectorStore, embed_func, mlx, k: int = 5, should_generate_response: bool = True, model: str = "llama-3.2-1b-instruct-4bit") -> Dict[str, Any]:
    """Run standard RAG pipeline."""
    logger.debug(f"\n=== Processing query with Standard RAG: {query} ===\n")
    logger.debug("Creating embedding for query...")
    query_embedding = embed_func(query)
    logger.debug(f"Retrieving {k} most similar chunks...")
    retrieved_chunks = vector_store.search(query_embedding, top_k=k)
    results = {
        "query": query,
        "retrieved_chunks": retrieved_chunks
    }
    if should_generate_response:
        logger.debug("Generating final response...")
        response = generate_ai_response(
            query,
            "You are a helpful AI assistant. Answer the question based on the provided context. If the context is insufficient, acknowledge the limitation.",
            retrieved_chunks,
            mlx,
            logger,
            model=model,
            max_tokens=500
        )
        results["response"] = response
    return results


def compare_approaches(query: str, vector_store: SimpleVectorStore, embed_func, mlx, reference_answer: str = None, model: str = "llama-3.2-1b-instruct-4bit") -> Dict[str, Any]:
    """Compare HyDE and standard RAG approaches."""
    hyde_result = hyde_rag(query, vector_store, embed_func, mlx, model=model)
    hyde_response = hyde_result["response"]
    standard_result = standard_rag(
        query, vector_store, embed_func, mlx, model=model)
    standard_response = standard_result["response"]
    comparison = compare_responses(
        query, hyde_response, standard_response, reference_answer, mlx, model)
    return {
        "query": query,
        "hyde_response": hyde_response,
        "hyde_hypothetical_doc": hyde_result["hypothetical_document"],
        "standard_response": standard_response,
        "reference_answer": reference_answer,
        "comparison": comparison
    }


def compare_responses(query: str, hyde_response: str, standard_response: str, reference: str = None, mlx=None, model: str = "llama-3.2-1b-instruct-4bit") -> str:
    """Compare HyDE and standard RAG responses."""
    system_prompt = "You are an objective evaluator. Compare the two responses to the query and provide a concise evaluation. If a reference answer is provided, use it to assess accuracy and completeness."
    user_prompt = f"Query: {query}\n\nHyDE Response:\n{hyde_response}\n\nStandard RAG Response:\n{standard_response}"
    if reference:
        user_prompt += f"\n\nReference Answer:\n{reference}"
    response = mlx.chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        model=model,
        temperature=0
    )
    return response["choices"][0]["message"]["content"]


def run_evaluation(pages: List[Dict[str, Any]], test_queries: List[str], embed_func, mlx, reference_answers: List[str] = None, chunk_size: int = 1000, chunk_overlap: int = 200, model: str = "llama-3.2-1b-instruct-4bit") -> Dict[str, Any]:
    """Run evaluation of HyDE vs standard RAG."""
    vector_store = process_document(
        pages, embed_func, chunk_size, chunk_overlap)
    results = []
    for i, query in enumerate(test_queries):
        logger.debug(
            f"\n\n===== Evaluating Query {i+1}/{len(test_queries)} =====")
        logger.debug(f"Query: {query}")
        reference = None
        if reference_answers and i < len(reference_answers):
            reference = reference_answers[i]
        result = compare_approaches(
            query, vector_store, embed_func, mlx, reference, model)
        results.append(result)
    overall_analysis = generate_overall_analysis(results, mlx, model)
    return {
        "results": results,
        "overall_analysis": overall_analysis
    }


def generate_overall_analysis(results: List[Dict[str, Any]], mlx, model: str = "llama-3.2-1b-instruct-4bit") -> str:
    """Generate overall analysis of RAG approaches."""
    evaluations_summary = ""
    for i, result in enumerate(results):
        evaluations_summary += f"Query {i+1}: {result['query']}\n"
        evaluations_summary += f"Comparison summary: {result['comparison'][:200]}...\n\n"
    system_prompt = "Provide an overall analysis of the performance of HyDE versus standard RAG based on the provided summaries."
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


def visualize_results(query: str, hyde_result: Dict[str, Any], standard_result: Dict[str, Any]):
    """Visualize comparison of HyDE and standard RAG results."""
    fig, axs = plt.subplots(1, 3, figsize=(20, 6))
    axs[0].text(0.5, 0.5, f"Query:\n\n{query}",
                horizontalalignment='center', verticalalignment='center',
                fontsize=12, wrap=True)
    axs[0].axis('off')
    hypothetical_doc = hyde_result["hypothetical_document"]
    shortened_doc = hypothetical_doc[:500] + \
        "..." if len(hypothetical_doc) > 500 else hypothetical_doc
    axs[1].text(0.5, 0.5, f"Hypothetical Document:\n\n{shortened_doc}",
                horizontalalignment='center', verticalalignment='center',
                fontsize=10, wrap=True)
    axs[1].axis('off')
    hyde_chunks = [chunk["text"][:100] +
                   "..." for chunk in hyde_result["retrieved_chunks"]]
    std_chunks = [chunk["text"][:100] +
                  "..." for chunk in standard_result["retrieved_chunks"]]
    comparison_text = "Retrieved by HyDE:\n\n"
    for i, chunk in enumerate(hyde_chunks):
        comparison_text += f"{i+1}. {chunk}\n\n"
    comparison_text += "\nRetrieved by Standard RAG:\n\n"
    for i, chunk in enumerate(std_chunks):
        comparison_text += f"{i+1}. {chunk}\n\n"
    axs[2].text(0.5, 0.5, comparison_text,
                horizontalalignment='center', verticalalignment='center',
                fontsize=8, wrap=True)
    axs[2].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, 'hyde_vs_standard_rag_comparison.png'))


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
            "page": i + 1
        }
    }
    for i, chunk in enumerate(original_chunks)
]
vector_store = process_document(pages, embed_func)
query = "What are the main ethical considerations in artificial intelligence development?"
hyde_result = hyde_rag(query, vector_store, embed_func, mlx)
logger.debug("\n=== HyDE Response ===")
logger.debug(hyde_result["response"])
standard_result = standard_rag(query, vector_store, embed_func, mlx)
logger.debug("\n=== Standard RAG Response ===")
logger.debug(standard_result["response"])
visualize_results(query, hyde_result, standard_result)
test_queries = [
    "How does neural network architecture impact AI performance?"
]
reference_answers = [
    "Neural network architecture significantly impacts AI performance through factors like depth (number of layers), width (neurons per layer), connectivity patterns, and activation functions. Different architectures like CNNs, RNNs, and Transformers are optimized for specific tasks such as image recognition, sequence processing, and natural language understanding respectively."
]
evaluation_results = run_evaluation(
    pages=pages,
    test_queries=test_queries,
    embed_func=embed_func,
    mlx=mlx,
    reference_answers=reference_answers
)
save_file(evaluation_results, f"{generated_dir}/evaluation_results.json")
logger.info(
    f"Saved evaluation results to {generated_dir}/evaluation_results.json")
logger.debug("\n=== OVERALL ANALYSIS ===")
logger.debug(evaluation_results["overall_analysis"])
logger.info("\n\n[DONE]", bright=True)
