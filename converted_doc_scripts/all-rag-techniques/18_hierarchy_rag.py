from jet.llm.mlx.base import MLX
from jet.llm.utils.embeddings import get_embedding_function
from jet.logger import CustomLogger
import pypdf
import json
import numpy as np
import os
import pickle
import re

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


def extract_text_from_pdf(pdf_path):
    logger.debug(f"Extracting text from {pdf_path}...")
    pages = []
    with open(pdf_path, "rb") as file:
        reader = pypdf.PdfReader(file)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text = page.extract_text() or ""
            if len(text.strip()) > 50:
                pages.append({
                    "text": text,
                    "metadata": {
                        "source": pdf_path,
                        "page": page_num + 1
                    }
                })
    logger.debug(f"Extracted {len(pages)} pages with content")
    return pages


def chunk_text(text, metadata, chunk_size=1000, overlap=200):
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


class SimpleVectorStore:
    def __init__(self):
        self.vectors = []
        self.texts = []
        self.metadata = []

    def add_item(self, text, embedding, metadata=None):
        self.vectors.append(np.array(embedding))
        self.texts.append(text)
        self.metadata.append(metadata or {})

    def similarity_search(self, query_embedding, k=5, filter_func=None):
        if not self.vectors:
            return []
        query_vector = np.array(query_embedding).flatten()
        similarities = []
        for i, vector in enumerate(self.vectors):
            vector = vector.flatten()
            if filter_func and not filter_func(self.metadata[i]):
                continue
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
    if not texts:
        return []
    return embed_func(texts)


def generate_page_summary(page_text, model="mlx-community/Llama-3.2-3B-Instruct-4bit"):
    max_tokens = 6000
    truncated_text = page_text[:max_tokens] if len(
        page_text) > max_tokens else page_text
    system_prompt = "You are a helpful AI assistant. Summarize the provided text concisely, capturing the main ideas and key points."
    response = mlx.chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Please summarize this text:\n\n{truncated_text}"}
        ],
        model=model,
        temperature=0.3
    )
    return response["choices"][0]["message"]["content"]


def process_document_hierarchically(pdf_path, chunk_size=1000, chunk_overlap=200, model="mlx-community/Llama-3.2-3B-Instruct-4bit"):
    pages = extract_text_from_pdf(pdf_path)
    logger.debug("Generating page summaries...")
    summaries = []
    for i, page in enumerate(pages):
        logger.debug(f"Summarizing page {i+1}/{len(pages)}...")
        summary_text = generate_page_summary(page["text"], model)
        summary_metadata = page["metadata"].copy()
        summary_metadata.update({"is_summary": True})
        summaries.append({
            "text": summary_text,
            "metadata": summary_metadata
        })
    detailed_chunks = []
    for page in pages:
        page_chunks = chunk_text(
            page["text"],
            page["metadata"],
            chunk_size,
            chunk_overlap
        )
        detailed_chunks.extend(page_chunks)
    logger.debug(f"Created {len(detailed_chunks)} detailed chunks")
    logger.debug("Creating embeddings for summaries...")
    summary_texts = [summary["text"] for summary in summaries]
    summary_embeddings = create_embeddings(summary_texts)
    logger.debug("Creating embeddings for detailed chunks...")
    chunk_texts = [chunk["text"] for chunk in detailed_chunks]
    chunk_embeddings = create_embeddings(chunk_texts)
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


def retrieve_hierarchically(query, summary_store, detailed_store, k_summaries=3, k_chunks=5):
    logger.debug(f"Performing hierarchical retrieval for query: {query}")
    query_embedding = create_embeddings(query)
    summary_results = summary_store.similarity_search(
        query_embedding,
        k=k_summaries
    )
    logger.debug(f"Retrieved {len(summary_results)} relevant summaries")
    relevant_pages = [result["metadata"]["page"] for result in summary_results]

    def page_filter(metadata):
        return metadata["page"] in relevant_pages
    detailed_results = detailed_store.similarity_search(
        query_embedding,
        k=k_chunks * len(relevant_pages),
        filter_func=page_filter
    )
    logger.debug(
        f"Retrieved {len(detailed_results)} detailed chunks from relevant pages")
    for result in detailed_results:
        page = result["metadata"]["page"]
        matching_summaries = [
            s for s in summary_results if s["metadata"]["page"] == page]
        if matching_summaries:
            result["summary"] = matching_summaries[0]["text"]
    return detailed_results


def generate_response(query, retrieved_chunks, model="mlx-community/Llama-3.2-3B-Instruct-4bit"):
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks):
        page_num = chunk["metadata"]["page"]
        context_parts.append(f"[Page {page_num}]: {chunk['text']}")
    context = "\n\n".join(context_parts)
    system_prompt = "You are a helpful AI assistant. Answer the question based on the provided context. If the context is insufficient, acknowledge the limitation."
    user_prompt = f"Context:\n\n{context}\n\nQuestion: {query}"
    response = mlx.chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        model=model,
        temperature=0.2
    )
    return response["choices"][0]["message"]["content"]


def hierarchical_rag(query, pdf_path, chunk_size=1000, chunk_overlap=200,
                     k_summaries=3, k_chunks=5, regenerate=False, model="mlx-community/Llama-3.2-3B-Instruct-4bit"):
    summary_store_file = os.path.join(
        DATA_DIR, f"{os.path.basename(pdf_path)}_summary_store.pkl")
    detailed_store_file = os.path.join(
        DATA_DIR, f"{os.path.basename(pdf_path)}_detailed_store.pkl")
    if regenerate or not os.path.exists(summary_store_file) or not os.path.exists(detailed_store_file):
        logger.debug("Processing document and creating vector stores...")
        summary_store, detailed_store = process_document_hierarchically(
            pdf_path, chunk_size, chunk_overlap, model
        )
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
        query, summary_store, detailed_store, k_summaries, k_chunks
    )
    response = generate_response(query, retrieved_chunks, model)
    return {
        "query": query,
        "response": response,
        "retrieved_chunks": retrieved_chunks,
        "summary_count": len(summary_store.texts),
        "detailed_count": len(detailed_store.texts)
    }


def standard_rag(query, pdf_path, chunk_size=1000, chunk_overlap=200, k=15, model="mlx-community/Llama-3.2-3B-Instruct-4bit"):
    pages = extract_text_from_pdf(pdf_path)
    chunks = []
    for page in pages:
        page_chunks = chunk_text(
            page["text"],
            page["metadata"],
            chunk_size,
            chunk_overlap
        )
        chunks.extend(page_chunks)
    logger.debug(f"Created {len(chunks)} chunks for standard RAG")
    store = SimpleVectorStore()
    logger.debug("Creating embeddings for chunks...")
    texts = [chunk["text"] for chunk in chunks]
    embeddings = create_embeddings(texts)
    for i, chunk in enumerate(chunks):
        store.add_item(
            text=chunk["text"],
            embedding=embeddings[i],
            metadata=chunk["metadata"]
        )
    query_embedding = create_embeddings(query)
    retrieved_chunks = store.similarity_search(query_embedding, k=k)
    logger.debug(f"Retrieved {len(retrieved_chunks)} chunks with standard RAG")
    response = generate_response(query, retrieved_chunks, model)
    return {
        "query": query,
        "response": response,
        "retrieved_chunks": retrieved_chunks
    }


def compare_approaches(query, pdf_path, reference_answer=None, model="mlx-community/Llama-3.2-3B-Instruct-4bit"):
    logger.debug(f"\n=== Comparing RAG approaches for query: {query} ===")
    logger.debug("\nRunning hierarchical RAG...")
    hierarchical_result = hierarchical_rag(query, pdf_path, model=model)
    hier_response = hierarchical_result["response"]
    logger.debug("\nRunning standard RAG...")
    standard_result = standard_rag(query, pdf_path, model=model)
    std_response = standard_result["response"]
    comparison = compare_responses(
        query, hier_response, std_response, reference_answer, model)
    return {
        "query": query,
        "hierarchical_response": hier_response,
        "standard_response": std_response,
        "reference_answer": reference_answer,
        "comparison": comparison,
        "hierarchical_chunks_count": len(hierarchical_result["retrieved_chunks"]),
        "standard_chunks_count": len(standard_result["retrieved_chunks"])
    }


def compare_responses(query, hierarchical_response, standard_response, reference=None, model="mlx-community/Llama-3.2-3B-Instruct-4bit"):
    system_prompt = "You are an objective evaluator. Compare the two responses to the query and provide a concise evaluation. If a reference answer is provided, use it to assess accuracy and completeness."
    user_prompt = f"Query: {query}\n\nHierarchical RAG Response:\n{hierarchical_response}\n\nStandard RAG Response:\n{standard_response}"
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


def run_evaluation(pdf_path, test_queries, reference_answers=None, model="mlx-community/Llama-3.2-3B-Instruct-4bit"):
    results = []
    for i, query in enumerate(test_queries):
        logger.debug(f"Query: {query}")
        reference = None
        if reference_answers and i < len(reference_answers):
            reference = reference_answers[i]
        result = compare_approaches(query, pdf_path, reference, model)
        results.append(result)
    overall_analysis = generate_overall_analysis(results, model)
    return {
        "results": results,
        "overall_analysis": overall_analysis
    }


def generate_overall_analysis(results, model="mlx-community/Llama-3.2-3B-Instruct-4bit"):
    evaluations_summary = ""
    for i, result in enumerate(results):
        evaluations_summary += f"Query {i+1}: {result['query']}\n"
        evaluations_summary += f"Hierarchical chunks: {result['hierarchical_chunks_count']}, Standard chunks: {result['standard_chunks_count']}\n"
        evaluations_summary += f"Comparison summary: {result['comparison'][:200]}...\n\n"
    system_prompt = "Provide an overall analysis of the performance of hierarchical versus standard RAG based on the provided summaries."
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


pdf_path = os.path.join(DATA_DIR, "AI_Information.pdf")
query = "What are the key applications of transformer models in natural language processing?"
result = hierarchical_rag(query, pdf_path)
logger.debug("\n=== Response ===")
logger.debug(result["response"])
test_queries = [
    "How do transformers handle sequential data compared to RNNs?"
]
reference_answers = [
    "Transformers handle sequential data differently from RNNs by using self-attention mechanisms instead of recurrent connections. This allows transformers to process all tokens in parallel rather than sequentially, capturing long-range dependencies more efficiently and enabling better parallelization during training. Unlike RNNs, transformers don't suffer from vanishing gradient problems with long sequences."
]
evaluation_results = run_evaluation(
    pdf_path=pdf_path,
    test_queries=test_queries,
    reference_answers=reference_answers
)
logger.debug("\n=== OVERALL ANALYSIS ===")
logger.debug(evaluation_results["overall_analysis"])
logger.info("\n\n[DONE]", bright=True)
