import json
import numpy as np
import os
import pickle
import re
from typing import List, Dict, Any
from jet.file.utils import save_file
from helpers import (
    setup_config, initialize_mlx, generate_embeddings,
    load_validation_data, generate_ai_response,
    load_json_data, SearchResult, SimpleVectorStore, DATA_DIR, DOCS_PATH, LLM_MODEL
)


def chunk_text(text: str, metadata: Dict[str, Any], chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
    """Chunk text into overlapping segments with metadata."""
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk_text = text[i:i + chunk_size]
        if chunk_text and len(chunk_text.strip()) > 50:
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "chunk_index": len(chunks),
                "start_char": i,
                "end_char": i + len(chunk_text),
                "is_summary": False
            })
            chunks.append({
                "text": chunk_text,
                "metadata": chunk_metadata
            })
    return chunks


def generate_page_summary(page_text: str, mlx, model: str = LLM_MODEL) -> str:
    """Generate summary for a page."""
    max_tokens = 6000
    truncated_text = page_text[:max_tokens] if len(
        page_text) > max_tokens else page_text
    system_prompt = "You are a helpful AI assistant. Summarize the provided text concisely, capturing the main ideas and key points."
    response = ""
    for chunk in mlx.stream_chat(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Please summarize this text:\n\n{truncated_text}"}
        ],
        model=model,
        temperature=0.3
    ):
        content = chunk["choices"][0]["message"]["content"]
        response += content
        logger.success(content, flush=True)
    return response


def process_document_hierarchically(pages: List[Dict[str, Any]], embed_func, mlx, chunk_size: int = 1000, chunk_overlap: int = 200, model: str = LLM_MODEL) -> tuple[SimpleVectorStore, SimpleVectorStore]:
    """Process document hierarchically into summaries and detailed chunks."""
    logger.debug("Generating page summaries...")
    summaries = []
    for i, page in enumerate(pages):
        logger.debug(f"Summarizing page {i+1}/{len(pages)}...")
        summary_text = generate_page_summary(page["text"], mlx, model)
        summary_metadata = page["metadata"].copy()
        summary_metadata.update({"is_summary": True})
        summaries.append({
            "text": summary_text,
            "metadata": summary_metadata
        })
    detailed_chunks = []
    for page in pages:
        page_chunks = chunk_text(
            page["text"], page["metadata"], chunk_size, chunk_overlap)
        detailed_chunks.extend(page_chunks)
    logger.debug(f"Created {len(detailed_chunks)} detailed chunks")
    logger.debug("Creating embeddings for summaries...")
    summary_texts = [summary["text"] for summary in summaries]
    summary_embeddings = generate_embeddings(summary_texts, embed_func, logger)
    logger.debug("Creating embeddings for detailed chunks...")
    chunk_texts = [chunk["text"] for chunk in detailed_chunks]
    chunk_embeddings = generate_embeddings(chunk_texts, embed_func, logger)
    summary_store = SimpleVectorStore()
    detailed_store = SimpleVectorStore()
    for i, summary in enumerate(summaries):
        summary_store.add_item(
            text=summary["text"],
            embedding=summary_embeddings[i],
            metadata=summary["metadata"]
        )
    for i, chunk in enumerate(detailed_chunks):
        detailed_store.add_item(
            text=chunk["text"],
            embedding=chunk_embeddings[i],
            metadata=chunk["metadata"]
        )
    logger.debug(
        f"Created vector stores with {len(summaries)} summaries and {len(detailed_chunks)} chunks")
    return summary_store, detailed_store


def retrieve_hierarchically(query: str, summary_store: SimpleVectorStore, detailed_store: SimpleVectorStore, embed_func, k_summaries: int = 3, k_chunks: int = 5) -> List[Dict[str, Any]]:
    """Retrieve chunks hierarchically using summaries."""
    logger.debug(f"Performing hierarchical retrieval for query: {query}")
    query_embedding = embed_func(query)
    summary_results = summary_store.search(query_embedding, top_k=k_summaries)
    logger.debug(f"Retrieved {len(summary_results)} relevant summaries")
    relevant_pages = [result["metadata"]["page"] for result in summary_results]

    def page_filter(metadata):
        return metadata["page"] in relevant_pages
    detailed_results = detailed_store.search(
        query_embedding, top_k=k_chunks * len(relevant_pages), filter_func=page_filter)
    logger.debug(
        f"Retrieved {len(detailed_results)} detailed chunks from relevant pages")
    for result in detailed_results:
        page = result["metadata"]["page"]
        matching_summaries = [
            s for s in summary_results if s["metadata"]["page"] == page]
        if matching_summaries:
            result["summary"] = matching_summaries[0]["text"]
    return detailed_results


def hierarchical_rag(query: str, pages: List[Dict[str, Any]], embed_func, mlx, chunk_size: int = 1000, chunk_overlap: int = 200, k_summaries: int = 3, k_chunks: int = 5, regenerate: bool = False, model: str = LLM_MODEL) -> Dict[str, Any]:
    """Run hierarchical RAG pipeline."""
    source = pages[0]["metadata"]["source"] if pages else "document"
    summary_store_file = os.path.join(
        DATA_DIR, f"{os.path.basename(source)}_summary_store.pkl")
    detailed_store_file = os.path.join(
        DATA_DIR, f"{os.path.basename(source)}_detailed_store.pkl")
    if regenerate or not os.path.exists(summary_store_file) or not os.path.exists(detailed_store_file):
        logger.debug("Processing document and creating vector stores...")
        summary_store, detailed_store = process_document_hierarchically(
            pages, embed_func, mlx, chunk_size, chunk_overlap, model)
        with open(summary_store_file, 'wb') as f:
            pickle.dump(summary_store, f)
        with open(detailed_store_file, 'wb') as f:
            pickle.dump(detailed_store, f)
    else:
        logger.debug("Loading existing vector stores...")
        with open(summary_store_file, 'rb') as f:
            summary_store = pickle.load(f)
        with open(detailed_store_file, 'rb') as f:
            detailed_store = pickle.load(f)
    retrieved_chunks = retrieve_hierarchically(
        query, summary_store, detailed_store, embed_func, k_summaries, k_chunks)
    response = generate_ai_response(
        query,
        "You are a helpful AI assistant. Answer the question based on the provided context. If the context is insufficient, acknowledge the limitation.",
        retrieved_chunks,
        mlx,
        logger,
        model=model
    )
    return {
        "query": query,
        "response": response,
        "retrieved_chunks": retrieved_chunks,
        "summary_count": len(summary_store.texts),
        "detailed_count": len(detailed_store.texts)
    }


def standard_rag(query: str, pages: List[Dict[str, Any]], embed_func, mlx, chunk_size: int = 1000, chunk_overlap: int = 200, k: int = 15, model: str = LLM_MODEL) -> Dict[str, Any]:
    """Run standard RAG pipeline."""
    chunks = []
    for page in pages:
        page_chunks = chunk_text(
            page["text"], page["metadata"], chunk_size, chunk_overlap)
        chunks.extend(page_chunks)
    logger.debug(f"Created {len(chunks)} chunks for standard RAG")
    store = SimpleVectorStore()
    logger.debug("Creating embeddings for chunks...")
    texts = [chunk["text"] for chunk in chunks]
    embeddings = generate_embeddings(texts, embed_func, logger)
    for i, chunk in enumerate(chunks):
        store.add_item(
            text=chunk["text"],
            embedding=embeddings[i],
            metadata=chunk["metadata"]
        )
    query_embedding = embed_func(query)
    retrieved_chunks = store.search(query_embedding, top_k=k)
    logger.debug(f"Retrieved {len(retrieved_chunks)} chunks with standard RAG")
    response = generate_ai_response(
        query,
        "You are a helpful AI assistant. Answer the question based on the provided context. If the context is insufficient, acknowledge the limitation.",
        retrieved_chunks,
        mlx,
        logger,
        model=model
    )
    return {
        "query": query,
        "response": response,
        "retrieved_chunks": retrieved_chunks
    }


def compare_approaches(query: str, pages: List[Dict[str, Any]], embed_func, mlx, reference_answer: str = None, model: str = LLM_MODEL) -> Dict[str, Any]:
    """Compare hierarchical and standard RAG approaches."""
    logger.debug(f"\n=== Comparing RAG approaches for query: {query} ===")
    logger.debug("\nRunning hierarchical RAG...")
    hierarchical_result = hierarchical_rag(
        query, pages, embed_func, mlx, model=model)
    hier_response = hierarchical_result["response"]
    logger.debug("\nRunning standard RAG...")
    standard_result = standard_rag(query, pages, embed_func, mlx, model=model)
    std_response = standard_result["response"]
    comparison = compare_responses(
        query, hier_response, std_response, reference_answer, mlx, model)
    return {
        "query": query,
        "hierarchical_response": hier_response,
        "standard_response": std_response,
        "reference_answer": reference_answer,
        "comparison": comparison,
        "hierarchical_chunks_count": len(hierarchical_result["retrieved_chunks"]),
        "standard_chunks_count": len(standard_result["retrieved_chunks"])
    }


def compare_responses(query: str, hierarchical_response: str, standard_response: str, reference: str = None, mlx=None, model: str = LLM_MODEL) -> str:
    """Compare hierarchical and standard RAG responses."""
    system_prompt = "You are an objective evaluator. Compare the two responses to the query and provide a concise evaluation. If a reference answer is provided, use it to assess accuracy and completeness."
    user_prompt = f"Query: {query}\n\nHierarchical RAG Response:\n{hierarchical_response}\n\nStandard RAG Response:\n{standard_response}"
    if reference:
        user_prompt += f"\n\nReference Answer:\n{reference}"
    response = ""
    for chunk in mlx.stream_chat(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        model=model,
        temperature=0
    ):
        content = chunk["choices"][0]["message"]["content"]
        response += content
        logger.success(content, flush=True)
    return response


def run_evaluation(pages: List[Dict[str, Any]], test_queries: List[str], embed_func, mlx, reference_answers: List[str] = None, model: str = LLM_MODEL) -> Dict[str, Any]:
    """Run evaluation of hierarchical vs standard RAG."""
    results = []
    for i, query in enumerate(test_queries):
        logger.debug(f"Query: {query}")
        reference = None
        if reference_answers and i < len(reference_answers):
            reference = reference_answers[i]
        result = compare_approaches(
            query, pages, embed_func, mlx, reference, model)
        results.append(result)
    overall_analysis = generate_overall_analysis(results, mlx, model)
    return {
        "results": results,
        "overall_analysis": overall_analysis
    }


def generate_overall_analysis(results: List[Dict[str, Any]], mlx, model: str = LLM_MODEL) -> str:
    """Generate overall analysis of RAG approaches."""
    evaluations_summary = ""
    for i, result in enumerate(results):
        evaluations_summary += f"Query {i+1}: {result['query']}\n"
        evaluations_summary += f"Hierarchical chunks: {result['hierarchical_chunks_count']}, Standard chunks: {result['standard_chunks_count']}\n"
        evaluations_summary += f"Comparison summary: {result['comparison'][:200]}...\n\n"
    system_prompt = "Provide an overall analysis of the performance of hierarchical versus standard RAG based on the provided summaries."
    user_prompt = f"Evaluations Summary:\n{evaluations_summary}"
    response = ""
    for chunk in mlx.stream_chat(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        model=model,
        temperature=0
    ):
        content = chunk["choices"][0]["message"]["content"]
        response += content
        logger.success(content, flush=True)
    return response


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
query = "What are the key applications of transformer models in natural language processing?"
result = hierarchical_rag(
    query=query,
    pages=pages,
    embed_func=embed_func,
    mlx=mlx
)
logger.debug("\n=== Response ===")
logger.debug(result["response"])
test_queries = [
    "How do transformers handle sequential data compared to RNNs?"
]
reference_answers = [
    "Transformers handle sequential data differently from RNNs by using self-attention mechanisms instead of recurrent connections. This allows transformers to process all tokens in parallel rather than sequentially, capturing long-range dependencies more efficiently and enabling better parallelization during training. Unlike RNNs, transformers don't suffer from vanishing gradient problems with long sequences."
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
